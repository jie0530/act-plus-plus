import rospy,sys,os,time
import cv2
import numpy as np
import h5py
from record_constants import DT, TASK_CONFIGS 
from utils import get_xyz_points
import Robot
from record_finger_state_handle import ModbusClient
import pyrealsense2 as rs
import threading
import queue
# 添加模块所在目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('/home/wsco/jie_ws/src/act-plus-plus/aloha_scripts/jie_record_scripts'))))
# 然后再导入
from inference_d3roma import D3RoMa
from utils_d3roma.camera import Realsense
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
import ros_numpy
import time

class DataRecorder:
    def __init__(self,DT, max_timesteps,camera_name_topic_dict, dataset_dir, dataset_name, overwrite, debug=False):

        self.dataset_dir = dataset_dir
        self.overwrite = overwrite
        self.debug = debug

        self.make_file_path(self.dataset_dir,dataset_name,self.overwrite)
        self.max_timesteps=max_timesteps
        self.is_collecting_data = False

        # 与机器人控制器建立连接，连接成功返回一个机器人对象
        self.robot = Robot.RPC('192.168.58.2')
        # self.robot.LoggerInit()
        # self.robot.SetLoggerLevel(4)
        
        # rospy.init_node('pointcloud_publisher', anonymous=True)
        self.pub_raw = rospy.Publisher('raw_pcl', PointCloud2, queue_size=10)
        self.pub_pred = rospy.Publisher('pred_pcl', PointCloud2, queue_size=10)

        # # 实例化 ModbusClient 1是左手，2是右手
        # self.modbus_client = ModbusClient(serial_port_name="/dev/ttyUSB0", slave_id=1)  # 替换为实际的串口名称
        # # 连接设备
        # if not self.modbus_client.connect():
        #     print("Failed to connect to Modbus device")
        #     return
        self.right_action_gripper_msg:Float32MultiArray = None
        self.right_action_gripper_sub = rospy.Subscriber('/right_finger_status', Float32MultiArray, lambda msg: setattr(self, 'right_action_gripper_msg', msg))
        
        # 定义相机序列号映射
        self.camera_serials = camera_name_topic_dict
        
        # 初始化多个RealSense相机
        self.cameras = {}
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
        
        # 使用定时器定时保存图像，每0.05秒保存一次
        self.timer = rospy.Timer(rospy.Duration(DT), self.timer_callback)
        self.record_data_dict = {}
        self.output_dir = "/home/wsco/jie_ws/infer_results"
        
        
        # 修改相机工作线程
        self.camera_threads = {}
        for cam_name in self.cameras:
            self.camera_threads[cam_name] = threading.Thread(
                target=self.camera_worker,
                args=(cam_name,)
            )
            self.camera_threads[cam_name].start()
    
    def camera_worker(self, cam_name):
        while True:
            try:
                if not self.is_collecting_data:
                    time.sleep(0.01)
                    continue
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
                    self.cloud_raw[cam_name] = self.cameras[cam_name].rgbd_to_pointcloud(rgb_frame, depth_aligned, intrinsics, extrinsics, downsample_factor=1, 
                            # fname=f"{self.output_dir}/raw_{timestamp}.ply"
                            )
                    self.cloud_pred[cam_name] = self.cameras[cam_name].rgbd_to_pointcloud(rgb_frame, pred_depth, intrinsics, extrinsics, downsample_factor=1, 
                            # fname=f"{self.output_dir}/pred_{timestamp}.ply"
                            )
                
                self.rgb_frames[cam_name] = rgb_frame
                self.depth_aligned_frames[cam_name] = depth_aligned
            except Exception as e:
                print(f"Error in camera worker for {cam_name}: {e}")
                time.sleep(1)  # 出错时等待较长时间再重试
                
            
    def make_file_path(self,dataset_dir,dataset_name,overwrite):
                # saving dataset
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
        self.dataset_path = os.path.join(dataset_dir, dataset_name)
        if os.path.isfile(self.dataset_path) and not overwrite:
            print(f'Dataset already exist at \n{self.dataset_path}\nHint: set overwrite to True.')
            exit()

    def stop(self):
        self.is_collecting_data = False

    def restart_collecting(self,bool):
        self.make_file_path(self.dataset_dir,f'episode_{get_auto_index(self.dataset_dir)}',self.overwrite)
        self.record_data_dict = {}
        self.is_collecting_data = True
        """
        For each timestep:
            observations
                - images
                    - cam_high          (640, 360, 3) 'uint8'
                    - cam_left_wrist    (640, 360, 3) 'uint8'
                    - cam_right_wrist   (640, 360, 3) 'uint8'
                - qpos                  (24,)         'float64'
                - qvel                  (24,)         'float64'
                - pointcloud
                    - xyz               (N, 3)        'float32'
                    - rgb               (N, 3)        'float32'
            
            action                  (7,)         'float64'
        """
        self.record_data_dict = {
            '/observations/qpos': [],
            # '/observations/qvel': [],
            # '/observations/effort': [], 
            '/action': [],
            '/compress_len': [],
        }
        self.finish_step = 0
        for cam_name in self.camera_serials.keys():
            self.record_data_dict[f'/observations/images/{cam_name}'] = []
            self.record_data_dict[f'/observations/depth_images/{cam_name}'] = []
            if cam_name == 'cam_right' or cam_name == 'cam_high':
                self.record_data_dict[f'/observations/pointcloud_pred/{cam_name}/xyz'] = []
                self.record_data_dict[f'/observations/pointcloud_pred/{cam_name}/rgb'] = []
                self.record_data_dict[f'/observations/pointcloud_raw/{cam_name}/xyz'] = []
                self.record_data_dict[f'/observations/pointcloud_raw/{cam_name}/rgb'] = []
        print("is_collecting_data",self.is_collecting_data )
        print("record_data_dict",self.record_data_dict )
 
    def timer_callback(self, event):
        # if self.cloud_pred['cam_right'] is None or self.cloud_raw['cam_right'] is None:
        #     return
        if self.cloud_pred['cam_high'] is None or self.cloud_raw['cam_high'] is None:
            return
        time1 = time.time()
        if self.is_collecting_data:
            obs={}
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            try:
                # 获取关节位置
                ret = self.robot.GetActualJointPosRadian()
                if ret[0] != 0:
                    rospy.logwarn(f"Error retrieving joint positions: {ret[0]}")
                    return False
                joint_positions = ret[1]
                # print(f"Joint positions: {joint_positions}")

                # 获取手指状态
                # fingers_status = self.modbus_client.get_finger_status()
                # norm_fingers_status = [finger_status/100.0 for finger_status in fingers_status]
                # obs['qpos'] = list(joint_positions[:6]) + list(norm_fingers_status[:6])
                obs['qpos'] = list(joint_positions[:6]) + list(self.right_action_gripper_msg.data[:6])
                # obs['qpos'] = list(joint_positions[:6]) + list([0]*6)
                
                # print(f"Finger status: {fingers_status}")
                # print(f"Norm Finger status: {norm_fingers_status}")
                
                # obs['action'] = list(joint_positions[:6]) + list(norm_fingers_status[:6])
                obs['action'] = list(joint_positions[:6]) + list(self.right_action_gripper_msg.data[:6])
                
                # obs['action'] = list(joint_positions[:6]) + list([0]*6)
                
                #print(f"gripper:{gripper}")
                obs['images'] ={}
                obs['depth_images'] ={}
                obs['pointcloud_pred'] ={}
                obs['pointcloud_raw'] ={}
                
                # 处理每个相机的数据
                for cam_name in self.camera_serials:
                    obs['images'][cam_name] = (self.rgb_frames[cam_name], time.time())
                    obs['depth_images'][cam_name] = self.depth_aligned_frames[cam_name]
                    # 分别存储预测和原始点云数据
                    cloud_pred = self.cloud_pred[cam_name]
                    cloud_raw = self.cloud_raw[cam_name]
                    
                    if cam_name == 'cam_right' or cam_name == 'cam_high':
                        # 提取xyz和rgb数据
                        cloud_pred_xyz = np.asarray(cloud_pred.points, dtype=np.float32)
                        cloud_pred_rgb = self.cameras[cam_name].open3d_rgb_to_rgb_packed(cloud_pred.colors)
                        cloud_raw_xyz = np.asarray(cloud_raw.points, dtype=np.float32)
                        cloud_raw_rgb = self.cameras[cam_name].open3d_rgb_to_rgb_packed(cloud_raw.colors)

                        # 转换坐标系
                        cloud_pred_xyz = self.transform_camera_to_base_frame(cloud_pred_xyz)
                        cloud_raw_xyz = self.transform_camera_to_base_frame(cloud_raw_xyz)    

                        if self.debug:
                            # 发布点云数据到ROS话题
                            cloud_array = self.cameras[cam_name].merge_xyz_rgb(cloud_raw.points, cloud_raw.colors)
                            pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
                                # cloud_array, rospy.Time.now(), "cam_right_color_optical_frame"
                                cloud_array, rospy.Time.now(), "base_link"
                            )
                            self.pub_raw.publish(pointcloud_msg)  
                        
                        timestamp = time.time()
                        # 保存原始点云可视化
                        # vis_path_raw = os.path.join(self.output_dir, f'raw_cloud_{timestamp}.png')
                        # self.visualize_pointcloud(cloud_raw_xyz, cloud_raw_rgb, vis_path_raw)
                        
                        # 保存预测点云可视化
                        # vis_path_pred = os.path.join(self.output_dir, f'pred_cloud_{timestamp}.png')
                        # self.visualize_pointcloud(cloud_pred_xyz, cloud_pred_rgb, vis_path_pred)

                        # 将点云以图片可视化
                        # vis_path_raw_filtered = os.path.join(self.output_dir, f'raw_cloud_filtered_{timestamp}.png')
                        # self.visualize_pointcloud(raw_xyz, raw_rgb, vis_path_raw_filtered)                        
                        
                        pred_xyz = cloud_pred_xyz[:, :3]
                        pred_rgb = cloud_pred_rgb[:, :3]
                        raw_xyz = cloud_raw_xyz[:, :3]
                        raw_rgb = cloud_raw_rgb[:, :3]
                        
                        obs['pointcloud_pred'][cam_name] = {
                            'xyz': pred_xyz,
                            'rgb': pred_rgb,
                            # 'timestamp': time.time()
                        }
                        obs['pointcloud_raw'][cam_name] = {
                            'xyz': raw_xyz,
                            'rgb': raw_rgb,
                            # 'timestamp': time.time()
                        }

            except Exception as e:
                print(e)
                return
                
            if len(self.record_data_dict['/observations/qpos']) >= self.max_timesteps:
                self.finish_step +=1
                if (self.finish_step % 100) == 0:
                    print(f"finish step : {self.finish_step}")
                return 
                
            # 记录机器人状态和动作
            self.record_data_dict['/observations/qpos'].append(obs['qpos'])
            # self.record_data_dict['/observations/qvel'].append(obs['qvel'])
            # self.record_data_dict['/observations/effort'].append(obs['effort'])
            self.record_data_dict['/action'].append(obs['action'])
            
            # 记录点云数据
            for cam_name in self.camera_serials:
                # 只记录一个全局相机点云
                if cam_name == 'cam_right' or cam_name == 'cam_high':
                    # 记录预测点云数据
                    self.record_data_dict[f'/observations/pointcloud_pred/{cam_name}/xyz'].append(
                        obs['pointcloud_pred'][cam_name]['xyz'])
                    self.record_data_dict[f'/observations/pointcloud_pred/{cam_name}/rgb'].append(
                        obs['pointcloud_pred'][cam_name]['rgb'])
                    # 记录原始点云数据
                    self.record_data_dict[f'/observations/pointcloud_raw/{cam_name}/xyz'].append(
                        obs['pointcloud_raw'][cam_name]['xyz'])
                    self.record_data_dict[f'/observations/pointcloud_raw/{cam_name}/rgb'].append(
                        obs['pointcloud_raw'][cam_name]['rgb'])
                # 记录RGB图像数据
                result, encoded_image = cv2.imencode('.jpg', obs['images'][cam_name][0], encode_param)
                self.record_data_dict[f'/observations/images/{cam_name}'].append(encoded_image)
                # 记录深度图像数据
                # result, encoded_image = cv2.imencode('.jpg', obs['depth_images'][cam_name][0], encode_param)
                # self.record_data_dict[f'/observations/depth_images/{cam_name}'].append(encoded_image)
                self.record_data_dict[f'/observations/depth_images/{cam_name}'].append(
                    obs['depth_images'][cam_name])
                
            step = len(self.record_data_dict['/observations/qpos'])
            if (step % 100) == 0:
                print(f"step : {step}")
            self.finish_step = 0
        time2 = time.time()
        # print(f"time1: {time1}, time2: {time2}, timer_callback_cost: {time2 - time1}")
    
    def transform_camera_to_base_frame(self, xyz):
        """
        将相机坐标系的点云转换到base坐标系
        """
        # 创建变换矩阵（根据实际相机安装位置调整）
        transform = np.array([
            [1, 0, 0],   # x轴方向
            [0, -1, 0],  # y轴方向（翻转）
            [0, 0, -1]   # z轴方向（翻转）
        ])
    
        # 应用变换
        transformed_xyz = xyz @ transform.T
        return transformed_xyz
        
    def save_succ(self):
        try:
            if len(self.record_data_dict['/observations/qpos']) >= self.max_timesteps:
                self.is_collecting_data = False
                self.save_data(self.record_data_dict,self.max_timesteps,self.camera_serials.keys(), self.dataset_path)
            else:
                print("数据采集失败", len(self.record_data_dict['/observations/qpos']))
        except Exception as e:
            print(e)
            return False 

    def save_data(self,data_dict,max_timesteps,camera_names,dataset_path): 
        sys.stdout.flush()
        rospy.sleep(1)
        user_input = input("文件已修改，是否保存？ (y/N) ").strip().lower()
        if not ( user_input == "y" or user_input == "yes"):
            return 
        print("开始保存文件")

        # rgb图像压缩
        compressed_len = []
        for cam_name in camera_names: 
            compressed_len.append([])
            for i in range(max_timesteps): 
                len_encode = data_dict[f'/observations/images/{cam_name}'][i].shape[0]
                compressed_len[-1].append(len_encode) 
        compressed_len = np.array(compressed_len)
        data_dict['/compress_len'] = compressed_len
        padded_size = compressed_len.max()
        for cam_name in camera_names:
            compressed_image_list = data_dict[f'/observations/images/{cam_name}']
            padded_compressed_image_list = []
            for compressed_image in compressed_image_list:
                padded_compressed_image = np.zeros(padded_size, dtype='uint8')
                image_len = len(compressed_image)
                # padded_compressed_image[:image_len] = compressed_image # two camera
                padded_compressed_image[:image_len] = compressed_image[:,0] # three camera
                padded_compressed_image_list.append(padded_compressed_image)
            data_dict[f'/observations/images/{cam_name}'] = padded_compressed_image_list
        print("images成功保存文件到数组")
        
        # 深度图像数据处理 - 不进行压缩，但需要统一数据格式
        for cam_name in camera_names:
            depth_image_list = data_dict[f'/observations/depth_images/{cam_name}']
            # 打印调试信息
            print(f"Camera {cam_name} depth data length: {len(depth_image_list)}")
            if len(depth_image_list) > 0:
                print(f"First depth image shape: {depth_image_list[0].shape}")
     
            # 将列表转换为numpy数组，确保统一的形状 (max_timesteps, height, width)
            # depth_array = np.array([frame[0] for frame in depth_image_list], dtype=np.float32)
            # depth_array = np.stack(depth_image_list, axis=0)  # 使用stack而不是array
            depth_array = np.stack([depth_img for depth_img in depth_image_list], axis=0)
            print(f"Final depth array shape: {depth_array.shape}")
            data_dict[f'/observations/depth_images/{cam_name}'] = depth_array
        print("depth_images成功保存文件到数组")
        
        # 点云数据填充（预测点云和原始点云）
        camera_names_pcl = ['cam_high'] #'cam_right', 
        for point_cloud_type in ['pred', 'raw']:
            pcd_data = {}
            for cam_name in camera_names_pcl:
                pcd_buffer = {
                    'xyz': data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/xyz'],
                    'rgb': data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/rgb']
                }
                
                # 找到最大点数
                max_pcd_n = max(len(xyz) for xyz in pcd_buffer['xyz'])
                
                # 填充到最大点数
                padded_pcd_xyz, padded_pcd_rgb, padding_mask = [], [], []
                for xyz, rgb in zip(pcd_buffer['xyz'], pcd_buffer['rgb']):
                    padded_pcd_xyz.append(
                        np.concatenate([
                            xyz,
                            np.zeros((max_pcd_n - len(xyz), 3), dtype=xyz.dtype)
                        ], axis=0)
                    )
                    padded_pcd_rgb.append(
                        np.concatenate([
                            rgb,
                            np.zeros((max_pcd_n - len(rgb), 3), dtype=rgb.dtype)
                        ], axis=0)
                    )
                    padding_mask.append(
                        np.concatenate([
                            np.ones(len(xyz), dtype=bool),
                            np.zeros(max_pcd_n - len(xyz), dtype=bool)
                        ], axis=0)
                    )
                
                pcd_data[cam_name] = {
                    'xyz': np.stack(padded_pcd_xyz, axis=0),  # (T, N_max, 3)
                    'rgb': np.stack(padded_pcd_rgb, axis=0),  # (T, N_max, 3)
                    'padding_mask': np.stack(padding_mask, axis=0)  # (T, N_max)
                }
                
                # 更新数据字典
                data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/xyz'] = pcd_data[cam_name]['xyz']
                data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/rgb'] = pcd_data[cam_name]['rgb']
                data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/padding_mask'] = pcd_data[cam_name]['padding_mask']
                
        print("pointcloud成功保存文件到数组")    

        # root 是文件的根组，HDF5 文件类似于一个文件系统，文件中可以包含组（文件夹）和数据集（文件）等层次结构。
        # rdcc_nbytes=1024**2*2 设置数据缓存的大小,这里为 2MB
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            # 设置文件级属性，attrs 是 HDF5 文件的属性集合。
            root.attrs['compress'] = True
            root.attrs['sim'] = False
            
            # 创建观测组
            obs = root.create_group('observations')
            
            # 创建并保存图像数据
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                      chunks=(1, padded_size))
            
            # 创建并保存深度图像数据
            depth_image = obs.create_group('depth_images')
            for cam_name in camera_names:
                depth_data = data_dict[f'/observations/depth_images/{cam_name}']
                print(f"Saving depth data for {cam_name}, shape: {depth_data.shape}")
                _ = depth_image.create_dataset(cam_name, data=depth_data, dtype='float32',
                                                chunks=(1, 360, 640), compression="gzip") #两种写法都可以
                # _ = depth_image.create_dataset(cam_name, (max_timesteps, 360, 640), dtype='float32',
                #                           chunks=(1, 360, 640), compression="gzip")  # 可选：使用无损压缩
            
            # 创建并保存点云数据
            for point_cloud_type in ['pred', 'raw']:
                pointcloud = obs.create_group(f'pointcloud_{point_cloud_type}')
                for cam_name in camera_names_pcl:
                    xyz_list = data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/xyz']
                    rgb_list = data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/rgb']
                    
                    # # 每隔一定帧数保存一次可视化
                    # save_interval = 10  # 每10帧保存一次
                    # for frame_idx in range(0, len(xyz_list), save_interval):
                    #     vis_path = os.path.join(
                    #         self.output_dir, 
                    #         f'{point_cloud_type}_cloud_frame{frame_idx}.png'
                    #     )
                    #     self.visualize_pointcloud(xyz_list[frame_idx], rgb_list[frame_idx], vis_path)
            
                    cam_group = pointcloud.create_group(cam_name)
                    _ = cam_group.create_dataset('xyz', 
                        data=data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/xyz'])
                    _ = cam_group.create_dataset('rgb', 
                        data=data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/rgb'])
                    _ = cam_group.create_dataset('padding_mask', 
                        data=data_dict[f'/observations/pointcloud_{point_cloud_type}/{cam_name}/padding_mask'])
                    print(f"Saving pointcloud_{point_cloud_type}/{cam_name}")
                    
            # 创建并保存其他数据
            _ = obs.create_dataset('qpos', (max_timesteps, 12))
            # _ = obs.create_dataset('qvel', (max_timesteps, 12))
            # _ = obs.create_dataset('effort', (max_timesteps, 12))
            _ = root.create_dataset('action', (max_timesteps, 12))
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            _ = root.create_dataset('depth_compress_len', (len(camera_names), max_timesteps))

            # 保存所有数据
            for name, array in data_dict.items():
                # print("name",name) 
                # print("array",array)
                # 省略号 ... 在NumPy数组的上下文中被用作一个占位符，表示要选取数组中的多个冒号切片。
                # 具体来说，如果数组是多维的，... 会自动扩展为足够数量的冒号，以选择数组的所有维度。
                # root[name][...] = array 
                if 'depth_images' in name:
                    # 确保深度数据的形状正确
                    root[name][...] = array.reshape(max_timesteps, 360, 640)
                    # continue
                else:
                    root[name][...] = array
                
        print("save " + dataset_path + ' finish \n')

def get_auto_index(dataset_dir, dataset_name_prefix = '', data_suffix = 'hdf5'):
    max_idx = 1000
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir)
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'{dataset_name_prefix}episode_{i}.{data_suffix}')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def get_recorder(task_name="aloha_ros",set_episode_idx=None):
    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    max_timesteps = task_config['episode_len']
    camera_names = task_config['camera_names']

    if set_episode_idx is not None:
        episode_idx = set_episode_idx
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    return DataRecorder(DT, max_timesteps, camera_names, dataset_dir, dataset_name, overwrite),max_timesteps*DT

if __name__ == '__main__':
    import argparse
    rospy.init_node('record_data_node', anonymous=True) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', action='store',default="aloha_ros", type=str, help='Task name.', required=False)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', default=None, required=False)
    args = vars(parser.parse_args())
    recorder = get_recorder(args['task_name'])
    recorder.restart_collecting(True)
    while not rospy.is_shutdown():
        time.sleep(1)