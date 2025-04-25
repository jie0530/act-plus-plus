import rospy,sys,os,time
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import h5py
import Robot
from finger_state_handle import ModbusClient
import math
from sensor_msgs.msg import PointCloud2
import ros_numpy
from typing import Dict, Optional
from functools import partial
import threading
sys.path.append('/home/wsco/jie_ws/src/d3roma/')
from inference_d3roma import D3RoMa
from utils_d3roma.camera import Realsense

class DataRecorder:
    def __init__(self, camera_name_topic_dict, pcl_type):
        self.pcl_type = pcl_type
        # 与机器人控制器建立连接，连接成功返回一个机器人对象
        self.robot = Robot.RPC('192.168.58.2')
        # self.robot.LoggerInit()
        # self.robot.SetLoggerLevel(4)
        
        # rospy.init_node('pointcloud_publisher', anonymous=True)
        self.pub_raw = rospy.Publisher('raw_pcl', PointCloud2, queue_size=10)
        self.pub_pred = rospy.Publisher('pred_pcl', PointCloud2, queue_size=10)

        # 实例化 ModbusClient 1是左手，2是右手
        self.modbus_client = ModbusClient(serial_port_name="/dev/ttyUSB0", slave_id=1)  # 替换为实际的串口名称
        # 连接设备
        if not self.modbus_client.connect():
            print("Failed to connect to Modbus device")
            return

        # 定义相机序列号映射
        self.camera_serials = camera_name_topic_dict
        
        # 初始化多个RealSense相机
        self.cameras = {}
        # from aloha_scripts.jie_record_scripts.realsense import RealSenseRGBDCamera
        sys.path.append('/home/wsco/jie_ws/src/act-plus-plus/aloha_scripts/jie_record_scripts/')
        from realsense import RealSenseRGBDCamera
        for cam_name, serial in self.camera_serials.items():
            try:
                self.cameras[cam_name] = RealSenseRGBDCamera(serial=serial)
                # 预热相机
                for _ in range(10):
                    self.cameras[cam_name].get_rgbd_image()
                print(f"Camera {cam_name} (Serial: {serial}) initialization finished.")
            except Exception as e:
                print(f"Failed to initialize camera {cam_name} (Serial: {serial}): {e}")
        
        # 为每个相机创建数据缓存
        self.rgb_frames = {cam_name: None for cam_name in self.cameras}
        self.depth_aligned_frames = {cam_name: None for cam_name in self.cameras}
        self.cloud_pred = {cam_name: None for cam_name in self.cameras}
        self.cloud_raw = {cam_name: None for cam_name in self.cameras}
        
        # 初始化D3RoMa
        self.camera_config_right = Realsense.default_real("d435_right")
        self.camera_config_high = Realsense.default_real("d435_high")
        overrides = [
            # uncomment if you choose variant left+right+raw
            # "task=eval_ldm_mixed",
            # "task.resume_pretrained=experiments/ldm_sf-mixed.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.nossi.my_ddpm1000.SceneFlow_Dreds_HssdIsaacStd.180x320.cond7-raw+left+right.w0.0/epoch_0199",
            
            # uncomment if you choose variant rgb+raw
            "task=eval_ldm_mixed_rgb+raw",
            # "task.resume_pretrained=experiments/ldm_sf-241212.2.dep4.lr3e-05.v_prediction.nossi.scaled_linear.randn.ddpm1000.Dreds_HssdIsaacStd_ClearPose.180x320.rgb+raw.w0.0/epoch_0056",
            "task.resume_pretrained=/home/wsco/jie_ws/src/d3roma/experiments/ldm/epoch_0056",
            # rest of the configurations
            "task.eval_num_batch=1",
            "task.image_size=[360,640]", 
            # "task.image_size=[180,320]",
            "task.eval_batch_size=1",
            "task.num_inference_rounds=1",
            "task.num_inference_timesteps=5", "task.num_intermediate_images=1",
            "task.write_pcd=true"
        ]
        self.d3roma_right = D3RoMa(overrides, self.camera_config_right, variant="rgb+raw")
        self.d3roma_high = D3RoMa(overrides, self.camera_config_high, variant="rgb+raw")
        
        # 修改相机工作线程
        self.running = True  # 添加控制标志
        self.camera_threads = {}
        for cam_name in self.cameras:
            self.camera_threads[cam_name] = threading.Thread(
                target=self.camera_worker,
                args=(cam_name,)
            )
            self.camera_threads[cam_name].start()
    
    def camera_worker(self, cam_name):
        while self.running:  # 使用控制标志
            try:
                # 获取特定相机的RGB-D图像
                rgb_frame, depth_aligned = self.cameras[cam_name].get_rgbd_image()
                
                if cam_name == 'cam_right' or cam_name == 'cam_high':
                    # D3RoMa处理
                    if cam_name == 'cam_right':
                        pred_depth = self.d3roma_right.infer_with_rgb_raw(rgb_frame, depth_aligned) #单独运行0.38s,一起运行0.76s
                    else:
                        pred_depth = self.d3roma_high.infer_with_rgb_raw(rgb_frame, depth_aligned)  #单独运行0.38s,一起运行0.76s
                    if cam_name == 'cam_right':
                        extrinsics = self.cameras[cam_name].cam_right_extrinsics()
                        # intrinsics = self.cameras[cam_name].cam_right_intrinsics()
                        intrinsics = self.camera_config_right.K.arr
                    else:
                        extrinsics = self.cameras[cam_name].cam_high_extrinsics()
                        # intrinsics = self.cameras[cam_name].cam_high_intrinsics()
                        intrinsics = self.camera_config_high.K.arr
                    
                    timestamp = time.time()
                    if self.pcl_type == 'raw':
                        self.cloud_raw[cam_name] = self.cameras[cam_name].rgbd_to_pointcloud(rgb_frame, depth_aligned, intrinsics, extrinsics, downsample_factor=1, 
                                # fname=f"{self.output_dir}/raw_{timestamp}.ply"
                                )
                    else:
                        self.cloud_pred[cam_name] = self.cameras[cam_name].rgbd_to_pointcloud(rgb_frame, pred_depth, intrinsics, extrinsics, downsample_factor=1, 
                                # fname=f"{self.output_dir}/pred_{timestamp}.ply"
                                )
                
                self.rgb_frames[cam_name] = rgb_frame
                self.depth_aligned_frames[cam_name] = depth_aligned
            except Exception as e:
                print(f"Error in camera worker for {cam_name}: {e}")
                if self.running:  # 只有在正常运行时才等待重试
                    time.sleep(1)
                
    def close(self):
        """关闭所有相机线程和资源"""
        print("正在关闭相机线程...")
        self.running = False  # 设置标志以停止所有线程
        
        # 等待所有线程结束
        for cam_name, thread in self.camera_threads.items():
            print(f"等待 {cam_name} 线程结束...")
            thread.join(timeout=5)  # 等待最多5秒
            if thread.is_alive():
                print(f"警告: {cam_name} 线程未能正常结束")
        
        # 关闭所有相机
        for cam_name, camera in self.cameras.items():
            print(f"关闭 {cam_name} 相机...")
            try:
                camera.pipeline.stop()
            except Exception as e:
                print(f"关闭 {cam_name} 时出错: {e}")
        
        print("所有相机已关闭")

    def _get_base_obs(self, include_joints=True, include_fingers=True, num_fingers=6, pcl_type='raw'):
        """基础观测函数，处理共同的逻辑
        Args:
            include_joints: 是否包含关节数据
            include_fingers: 是否包含手指数据
            num_fingers: 需要的手指数量
            include_pcl: 是否包含点云数据
        """
        # if any(value is None for value in self.image_buffer.values()):
        #     print("No images received yet")
        #     return False
        if pcl_type == 'raw':
            if self.cloud_raw['cam_high'] is None:
                print("No pcl received yet")
                return
        else:
            if self.cloud_pred['cam_high'] is None:
                print("No pcl received yet")
                return
        obs = {'qpos': [], 'images': {}, 'depth_images': {}}
        obs['pointcloud'] = {'xyz': [], 'rgb': []}
        
        # 获取关节位置
        if include_joints:
            ret = self.robot.GetActualJointPosRadian()
            if ret[0] != 0:
                rospy.logwarn(f"Error retrieving joint positions: {ret[0]}")
                return False
            joint_positions = ret[1]
            obs['qpos'].extend(list(joint_positions[:6]))
            print(f"Joint positions: {joint_positions}")

        # 获取手指状态
        if include_fingers:
            fingers_status = self.modbus_client.get_finger_status()
            norm_fingers_status = [finger_status/100.0 for finger_status in fingers_status]
            obs['qpos'].extend(norm_fingers_status[:num_fingers])
            print(f"Finger status: {fingers_status}")
            print(f"Norm Finger status: {norm_fingers_status}")

        # 数据
        cam_name = 'cam_high' #HardCode
        # 提取点云xyz和rgb数据
        if pcl_type == 'raw':
            cloud_raw = self.cloud_raw[cam_name]
            cloud_raw_xyz = np.asarray(cloud_raw.points, dtype=np.float32)
            cloud_raw_rgb = self.cameras[cam_name].open3d_rgb_to_rgb_packed(cloud_raw.colors)
            obs['pointcloud']['xyz'].extend(cloud_raw_xyz)
            obs['pointcloud']['rgb'].extend(cloud_raw_rgb)
        elif pcl_type == 'pred':
            cloud_pred = self.cloud_pred[cam_name]
            cloud_pred_xyz = np.asarray(cloud_pred.points, dtype=np.float32)
            cloud_pred_rgb = self.cameras[cam_name].open3d_rgb_to_rgb_packed(cloud_pred.colors)
            obs['pointcloud']['xyz'].extend(cloud_pred_xyz)
            obs['pointcloud']['rgb'].extend(cloud_pred_rgb)

        # try:
            # # 处理图像数据
            # for cam_name in self.image_topics.keys():
            #     if self.image_buffer[cam_name][0].shape != self.image_shape:
            #         raise Exception(f"image shape {self.image_buffer[cam_name][0].shape} error, require {self.image_shape}")
            #     obs['images'][cam_name] = self.image_buffer[cam_name][0]
            
            # # 处理深度图像数据
            # for cam_name in self.depth_image_topics.keys():
            #     if self.depth_image_buffer[cam_name][0].shape != self.depth_image_shape:
            #         raise Exception(f"image shape {self.depth_image_buffer[cam_name][0].shape} error, require {self.depth_image_shape}")
            #     obs['depth_images'][cam_name] = self.depth_image_buffer[cam_name][0]
        # except Exception as e:
        #     print(e)
        #     return False
            
        return obs

    def get_obs(self, pcl_type='raw'):  # action_dim=12
        return self._get_base_obs(include_joints=True, include_fingers=True, num_fingers=6)

    def get_obs_arm_gripper(self, pcl_type='raw'):  # action_dim=7
        return self._get_base_obs(include_joints=True, include_fingers=True, num_fingers=1, pcl_type=pcl_type)

    def get_obs_arm(self, pcl_type='raw'):  # action_dim=6
        return self._get_base_obs(include_joints=True, include_fingers=False)

    def get_obs_hand(self, pcl_type='raw'):  # action_dim=6
        return self._get_base_obs(include_joints=False, include_fingers=True, num_fingers=6)

    def control_finger(self, action_list):
        int_action_list = [int(i * 100) if 0 <= i <= 100 else 0 for i in action_list] # 手指动作恢复到原来的比例
        self.modbus_client.set_finger_status(int_action_list)
        
    def control_gripper(self, action):
        if action < 0.1: # 如果动作小于0.1，则open
            self.modbus_client.set_finger_status([0, 0, 0, 0, 0, 0])
        else: # 否则，close
            self.modbus_client.set_finger_status([50, 50, 60, 60, 70, 70])
    
    def control_arm(self, target_qpos):
        # 将每个弧度值转换为角度值
        target_qpos_in_degrees = [math.degrees(rad) for rad in target_qpos]
        tool = 0  # 工具坐标系编号
        user = 0  # 工件坐标系编号
        ret2 = self.robot.MoveJ(target_qpos_in_degrees, tool, user, vel=100)  # 关节空间运动
        print("关节空间运动点1:错误码", ret2)
    
    def control_arm_finger(self, action_list):
        self.control_arm(action_list[:6])
        self.control_finger(action_list[6:])
        return
    
    def control_arm_gripper(self, action_list):
        self.control_arm(action_list[:6])
        # self.control_gripper(action_list[6:])
        self.control_gripper(action_list[6:7])
        return
    
if __name__ == '__main__':
    rospy.init_node('data_recorder')
    camera_name_topic_dict = {'cam_high':"/cam_high/color/image_raw",
                            # 'cam_left':"/cam_left/color/image_raw",
                            # 'cam_right':"/cam_right/color/image_raw"
                            }
    recorder = DataRecorder(camera_name_topic_dict)
    while True:
        recorder.get_obs()
        rospy.sleep(0.1)