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
from utils import get_xyz_points
def fix_image(cv_image,cv_type=cv2.COLOR_BGR2RGB,target_height = 480,target_width = 640):
    # 将图像从BGR转换为RGB
    cv_image_rgb = cv_image
    if cv_type:
        cv_image_rgb = cv2.cvtColor(cv_image, cv_type)
    
    # 获取图像尺寸
    height, width = cv_image_rgb.shape[:2] 
    
    # 调整图像尺寸
    if height > target_height or width > target_width:
        # 图像太大，需要裁剪
        startx = width//2 - target_width//2
        starty = height//2 - target_height//2    
        cv_image_rgb = cv_image_rgb[starty:starty+target_height, startx:startx+target_width]
    elif height < target_height or width < target_width:
        # 图像太小，需要填充
        delta_w = target_width - width
        delta_h = target_height - height
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = [0, 0, 0] # 黑色填充
        cv_image_rgb = cv2.copyMakeBorder(cv_image_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return cv_image_rgb

class DataRecorder:
    def __init__(self, camera_name_topic_dict, depth_camera_name_topic_dict):

        # 定义了一个实例变量 joint_msg，并使用类型注解指定了该变量的类型应该是 JointState。类型注解帮助开发者理解变量预期的数据类型，虽然它不会在Python运行时强制类型检查。这里将 joint_msg 初始化为 None，意味着在收到任何关节状态消息前，这个变量没有具体的值
        # self.joint_msg:JointState = None
        # # setattr(self, 'joint_msg', msg) 是设置当前类实例的 joint_msg 属性为接收到的消息
        # # 当 /frcobot/joint_states 话题上有新的 JointState 消息发布时，这个订阅者的回调函数会自动更新 self.joint_msg 的值为这个新消息。这是实现机器人软件组件之间通信的有效方式，特别是在需要实时响应外部传感器数据或状态信息时。
        # self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, lambda msg: setattr(self, 'joint_msg', msg))  
        # # self.left_action_gripper_msg:Int32MultiArray = None
        # # self.left_action_gripper_sub = rospy.Subscriber('/left_finger_status', Int32MultiArray, lambda msg: setattr(self, 'left_action_gripper_msg', msg))
        # self.right_action_gripper_msg:Int32MultiArray = None
        # self.right_action_gripper_sub = rospy.Subscriber('/right_finger_status', Int32MultiArray, lambda msg: setattr(self, 'right_action_gripper_msg', msg))

        self.is_collecting_data = False

        # 创建CV Bridge
        self.bridge = CvBridge()
        
        # 订阅图像topic
        self.image_shape = (480, 640, 3)
        self.image_topics:dict = camera_name_topic_dict
        self.image_sub = {}
        self.image_buffer = {}
        
        # 订阅深度图像topic
        self.depth_image_shape = (480, 640)
        self.depth_image_topics:dict = depth_camera_name_topic_dict
        self.depth_image_sub = {}
        self.depth_image_buffer = {}
        
        # 订阅点云topic
        self.point_cloud_data = None
        self.point_cloud_sub = rospy.Subscriber('/fused_pcd', PointCloud2, self.pointcloud_callback)
        rospy.wait_for_message('/fused_pcd', PointCloud2)
        print("Subscribed to fused_pcd topic")

        # 与机器人控制器建立连接，连接成功返回一个机器人对象
        self.robot = Robot.RPC('192.168.58.2')
        self.robot.LoggerInit()
        self.robot.SetLoggerLevel(4)
        
        # 实例化 ModbusClient 1是左手，2是右手
        self.modbus_client = ModbusClient(serial_port_name="/dev/ttyUSB0", slave_id=1)  # 替换为实际的串口名称

        # 连接设备
        if not self.modbus_client.connect():
            print("Failed to connect to Modbus device")
            return

        # callback(data) 是 make_callback 返回的回调函数，它会在接收到图像消息时执行。
        def make_callback(cam_name, bridge, image_buffer, image_shape):
            def callback(data):
                # print(f"Received image from {cam_name}")
                try:
                    # 将接收到的图像转换为OpenCV图像格式
                    cv_image = bridge.imgmsg_to_cv2(data, "rgb8")
                    image_buffer[cam_name] = (fix_image(cv_image, None, image_shape[0], image_shape[1]), data.header.stamp.to_sec())
                except CvBridgeError as e:
                    rospy.logerr(e)
            return callback

        for cam_name, cam_topic in self.image_topics.items():
            self.image_buffer[cam_name] = None
            # 使用make_callback函数创建callback
            print(f"Subscribing to {cam_name} topic: {cam_topic}")
            callback = make_callback(cam_name, self.bridge, self.image_buffer, self.image_shape)
            self.image_sub[cam_name] = rospy.Subscriber(cam_topic, Image, callback)
            print(f"Subscribed to {cam_name} topic: {cam_topic}")

        def depth_make_callback(cam_name, bridge, depth_image_buffer, image_shape):
            def depth_callback(data):
                try:
                    # 将接收到的图像转换为OpenCV图像格式
                    depth_img = bridge.imgmsg_to_cv2(data, "32FC1")
                    depth_image_buffer[cam_name] = (fix_image(depth_img, None, image_shape[0], image_shape[1]), data.header.stamp.to_sec())
                except CvBridgeError as e:
                    rospy.logerr(e)
            return depth_callback

        for cam_name, cam_topic in self.depth_image_topics.items():
            self.depth_image_buffer[cam_name] = None
            # 使用make_callback函数创建callback
            depth_callback = depth_make_callback(cam_name, self.bridge, self.depth_image_buffer, self.depth_image_shape)
            self.depth_image_sub[cam_name] = rospy.Subscriber(cam_topic, Image, depth_callback)
            print(f"Subscribed to {cam_name} topic: {cam_topic}")
            
 
 
    def pointcloud_callback(self, pcd_msg: PointCloud2):
        # self.point_cloud_data = {}
        # stamp = pcd_msg.header.stamp.secs + pcd_msg.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msg)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self.point_cloud_data = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            # "stamp": np.array([stamp]),
        }  

    def _get_base_obs(self, include_joints=True, include_fingers=True, num_fingers=6, include_pcl=True):
        """基础观测函数，处理共同的逻辑
        Args:
            include_joints: 是否包含关节数据
            include_fingers: 是否包含手指数据
            num_fingers: 需要的手指数量
            include_pcl: 是否包含点云数据
        """
        if any(value is None for value in self.image_buffer.values()):
            print("No images received yet")
            return False

        obs = {'qpos': [], 'images': {}, 'depth_images': {}}
        if include_pcl:
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

        try:
            # 处理图像数据
            for cam_name in self.image_topics.keys():
                if self.image_buffer[cam_name][0].shape != self.image_shape:
                    raise Exception(f"image shape {self.image_buffer[cam_name][0].shape} error, require {self.image_shape}")
                obs['images'][cam_name] = self.image_buffer[cam_name][0]
            
            # 处理深度图像数据
            for cam_name in self.depth_image_topics.keys():
                if self.depth_image_buffer[cam_name][0].shape != self.depth_image_shape:
                    raise Exception(f"image shape {self.depth_image_buffer[cam_name][0].shape} error, require {self.depth_image_shape}")
                obs['depth_images'][cam_name] = self.depth_image_buffer[cam_name][0]

            # 处理点云数据
            if include_pcl:
                if self.point_cloud_data is not None:
                    obs['pointcloud']['xyz'].extend(self.point_cloud_data['xyz'])
                    obs['pointcloud']['rgb'].extend(self.point_cloud_data['rgb'])
            
        except Exception as e:
            print(e)
            return False
            
        return obs

    def get_obs(self, include_pcl=True):  # action_dim=12
        return self._get_base_obs(include_joints=True, include_fingers=True, num_fingers=6, include_pcl=include_pcl)

    def get_obs_arm_gripper(self, include_pcl=True):  # action_dim=7
        return self._get_base_obs(include_joints=True, include_fingers=True, num_fingers=1, include_pcl=include_pcl)

    def get_obs_arm(self, include_pcl=True):  # action_dim=6
        return self._get_base_obs(include_joints=True, include_fingers=False, include_pcl=include_pcl)

    def get_obs_hand(self, include_pcl=True):  # action_dim=6
        return self._get_base_obs(include_joints=False, include_fingers=True, num_fingers=6, include_pcl=include_pcl)
    
    

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