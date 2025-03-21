import rospy,sys,os,time
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray, Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import h5py
from constants import DT, TASK_CONFIGS 
from sensor_msgs.msg import PointCloud2
import ros_numpy
from utils import get_xyz_points
from typing import Dict, Optional
from functools import partial

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
    def __init__(self,DT, max_timesteps,camera_name_topic_dict,  depth_camera_name_topic_dict, pointcloud_name_topic_dict, dataset_dir, dataset_name, overwrite):

        self.dataset_dir = dataset_dir
        self.overwrite = overwrite

        self.make_file_path(self.dataset_dir,dataset_name,self.overwrite)
        self.max_timesteps=max_timesteps
        # 定义了一个实例变量 joint_msg，并使用类型注解指定了该变量的类型应该是 JointState。类型注解帮助开发者理解变量预期的数据类型，虽然它不会在Python运行时强制类型检查。这里将 joint_msg 初始化为 None，意味着在收到任何关节状态消息前，这个变量没有具体的值
        self.joint_msg:JointState = None
        # setattr(self, 'joint_msg', msg) 是设置当前类实例的 joint_msg 属性为接收到的消息
        # 当 /frcobot/joint_states 话题上有新的 JointState 消息发布时，这个订阅者的回调函数会自动更新 self.joint_msg 的值为这个新消息。这是实现机器人软件组件之间通信的有效方式，特别是在需要实时响应外部传感器数据或状态信息时。
        self.joint_states_sub = rospy.Subscriber('/joint_states', JointState, lambda msg: setattr(self, 'joint_msg', msg))  
        # self.left_action_gripper_msg:Int32MultiArray = None
        # self.left_action_gripper_sub = rospy.Subscriber('/left_finger_status', Int32MultiArray, lambda msg: setattr(self, 'left_action_gripper_msg', msg))
        self.right_action_gripper_msg:Float32MultiArray = None
        self.right_action_gripper_sub = rospy.Subscriber('/right_finger_status', Float32MultiArray, lambda msg: setattr(self, 'right_action_gripper_msg', msg))

        # 添加点云话题订阅
        # self.pointcloud_msg = None
        # self.pointcloud_sub = rospy.Subscriber('/cam_left/depth/color/points', PointCloud2, self.pointcloud_callback)
        # self.point_cloud_topics: Optional[Dict[str, str]] = None,

        self.is_collecting_data = False

        # 创建CV Bridge
        self.bridge = CvBridge()
        
        # 订阅图像topic
        self.image_shape = (480, 640, 3)
        self.image_topics:dict = camera_name_topic_dict
        self.image_sub = {}
        self.image_buffer = {}
        # callback(data) 是 make_callback 返回的回调函数，它会在接收到图像消息时执行。
        def make_callback(cam_name, bridge, image_buffer, image_shape):
            def callback(data):
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
            callback = make_callback(cam_name, self.bridge, self.image_buffer, self.image_shape)
            self.image_sub[cam_name] = rospy.Subscriber(cam_topic, Image, callback)
            
        # 订阅深度图像
        self.depth_image_shape = (480, 640)
        self.depth_image_topics:dict = depth_camera_name_topic_dict
        self.depth_image_sub = {}
        self.depth_image_buffer = {}
        self.depth_point_cloud_data = {}
        # callback(data) 是 make_callback 返回的回调函数，它会在接收到图像消息时执行。
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
        
        # 初始化点云数据缓存
        self.point_cloud_data: Dict[str, Optional[Dict[str, np.array]]] = {
            k: None for k in pointcloud_name_topic_dict
        }
        # 订阅点云话题
        self._point_cloud_subs = {
            k: rospy.Subscriber(
                v, PointCloud2, partial(self.pointcloud_callback, cam_name=k)
            )
            for k, v in pointcloud_name_topic_dict.items()
        }
        for topic in pointcloud_name_topic_dict.values():
            rospy.wait_for_message(topic, PointCloud2)
        
 
        # 使用定时器定时保存图像，每0.02秒保存一次
        self.timer = rospy.Timer(rospy.Duration(DT), self.timer_callback)
        self.record_data_dict = {}
    
    def pointcloud_callback(self, pcd_msg: PointCloud2, cam_name:str):
        # stamp = pcd_msg.header.stamp.secs + pcd_msg.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msg)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self.point_cloud_data[cam_name] = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            # "stamp": np.array([stamp]),
        }
        # print(f"Received point_cloud_data")
        
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
                    - cam_high          (480, 640, 3) 'uint8'
                    - cam_left_wrist    (480, 640, 3) 'uint8'
                    - cam_right_wrist   (480, 640, 3) 'uint8'
                - qpos                  (24,)         'float64'
                - qvel                  (24,)         'float64'
                - pointcloud
                    - xyz               (N, 3)        'float32'
                    - rgb               (N, 3)        'float32'
            
            action                  (7,)         'float64'
        """
        self.record_data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            # '/observations/effort': [], 
            '/action': [],
            '/compress_len': [],
        }
        self.finish_step = 0
        for cam_name in self.image_topics.keys():
            self.record_data_dict[f'/observations/images/{cam_name}'] = []
        for cam_name in self.depth_image_topics.keys():
            self.record_data_dict[f'/observations/depth_images/{cam_name}'] = []
        for cam_name in self.point_cloud_data.keys():
            self.record_data_dict[f'/observations/pointcloud/{cam_name}/xyz'] = []
            self.record_data_dict[f'/observations/pointcloud/{cam_name}/rgb'] = []
        print("is_collecting_data",self.is_collecting_data )
        print("record_data_dict",self.record_data_dict )
 
    def timer_callback(self, event):

        if self.is_collecting_data:
            obs={}
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            try:
                # obs['qpos'] = list(self.joint_msg.position[:6]) + list(self.joint_msg.position[16:22]) + list(self.left_action_gripper_msg.data[:6]) + list(self.right_action_gripper_msg.data[:6])
                obs['qpos'] = list(self.joint_msg.position[:6]) + list([0]*6)
                # obs['qpos'] = list(self.joint_msg.position[:6]) + list(self.right_action_gripper_msg.data[:6])
                try:
                    # obs['qvel'] = list(self.joint_msg.position[:6]) + list(self.joint_msg.position[16:22]) + list([0]*6) + list([0]*6) # TODO gripper实际速度值;
                    # obs['effort'] = list(self.joint_msg.position[:6]) + list(self.joint_msg.position[16:22]) + list([0]*6) + list([0]*6)## TODO gripper实际力值;
                    obs['qvel'] = list(self.joint_msg.position[:6]) + list([0]*6) # TODO gripper实际速度值;
                    # obs['effort'] = list(self.joint_msg.position[:6]) + list(self.joint_msg.position[16:22]) + list([0]*6) + list([0]*6)## TODO gripper实际力值;
                    #print(self.joint_msg.effort,self.joint_msg.effort[-3])
                except Exception as e:
                    obs['qvel'] = []
                    # obs['effort'] = [] 
                # gripper = list(self.left_action_gripper_msg.data) + list(self.right_action_gripper_msg.data)
                # action_arm = list(self.joint_msg.position[:6]) + list(self.joint_msg.position[16:22])
                # gripper = list(self.right_action_gripper_msg.data)
                gripper = list([0]*6)
                
                action_arm = list(self.joint_msg.position[:6])
                obs['action'] = list(action_arm) + list(gripper)
                #print(f"gripper:{gripper}")
                obs['images'] ={}
                obs['depth_images'] ={}
                obs['pointcloud'] ={}
                
                # 处理点云数据
                for cam_name in self.point_cloud_data.keys():
                    obs[f'pointcloud/{cam_name}/xyz'] = self.point_cloud_data[cam_name]['xyz']
                    obs[f'pointcloud/{cam_name}/rgb'] = self.point_cloud_data[cam_name]['rgb']

                # 拼接所有self.image_buffer内的图片到all_img
                all_img = None
                all_stamp = []
                for cam_name in self.image_topics.keys():
                    if self.image_buffer[cam_name][0].shape != self.image_shape:
                        raise Exception(f"image shape { self.image_buffer[cam_name][0].shape} error , require {self.image_shape} ")
                    obs['images'][cam_name] = self.image_buffer[cam_name][0]
                    if all_img is None:
                        # 如果all_img还没有任何图像，直接使用第一个图像
                        all_img = self.image_buffer[cam_name][0]
                        all_stamp.append(self.image_buffer[cam_name][1])
                    else:
                        # 水平拼接图像
                        all_img = np.hstack((all_img, self.image_buffer[cam_name][0]))
                        all_stamp.append(self.image_buffer[cam_name][1])
                
                for cam_name in self.depth_image_topics.keys():
                    if self.depth_image_buffer[cam_name][0].shape != self.depth_image_shape:
                        raise Exception(f"image shape { self.depth_image_buffer[cam_name][0].shape} error , require {self.depth_image_shape} ")
                    obs['depth_images'][cam_name] = self.depth_image_buffer[cam_name][0]                
                
                # 显示图像
                # cv2.imshow("all_img", cv2.cvtColor(all_img, cv2.COLOR_RGB2BGR))
                # #print("img stamp:" ,all_stamp)
                # cv2.waitKey(10)  # 给时间处理GUI事件，参数1表示等待1毫秒
            except Exception as e:
                print(e)
                return
            if len(self.record_data_dict['/observations/qpos']) >= self.max_timesteps:
                self.finish_step +=1
                if (self.finish_step % 100) == 0:
                    print(f"finish step : {self.finish_step}")
                return 
            self.record_data_dict['/observations/qpos'].append(obs['qpos'])
            self.record_data_dict['/observations/qvel'].append(obs['qvel'])
            # self.record_data_dict['/observations/effort'].append(obs['effort'])
            self.record_data_dict['/action'].append(obs['action'])
            for cam_name in self.point_cloud_data.keys():
                self.record_data_dict[f'/observations/pointcloud/{cam_name}/xyz'].append(obs[f'pointcloud/{cam_name}/xyz'])
                self.record_data_dict[f'/observations/pointcloud/{cam_name}/rgb'].append(obs[f'pointcloud/{cam_name}/rgb'])
            for cam_name in self.image_topics.keys():
                result, encoded_image = cv2.imencode('.jpg', obs['images'][cam_name], encode_param)
                self.record_data_dict[f'/observations/images/{cam_name}'].append(encoded_image)
            for cam_name in self.depth_image_topics.keys():
                result, encoded_image = cv2.imencode('.jpg', obs['depth_images'][cam_name], encode_param)
                self.record_data_dict[f'/observations/depth_images/{cam_name}'].append(encoded_image)
            step =len(self.record_data_dict['/observations/qpos'])
            if (step % 100) == 0:
                print(f"step : {step}")
            self.finish_step = 0
            
    def save_succ(self):
        try:
            if len(self.record_data_dict['/observations/qpos']) >= self.max_timesteps:
                self.is_collecting_data = False
                self.save_data(self.record_data_dict,self.max_timesteps,self.image_topics.keys(), 
                               self.depth_image_topics.keys(), self.point_cloud_data.keys(),self.dataset_path)
            else:
                print("数据采集失败", len(self.record_data_dict['/observations/qpos']))
        except Exception as e:
            print(e)
            return False 

    def save_data(self,data_dict,max_timesteps,camera_names,depth_camera_names,pointcloud_names,dataset_path): 
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
        
        # 深度图像压缩
        depth_compressed_len = []
        for cam_name in depth_camera_names: 
            depth_compressed_len.append([])
            for i in range(max_timesteps): 
                len_encode = data_dict[f'/observations/depth_images/{cam_name}'][i].shape[0]
                depth_compressed_len[-1].append(len_encode) 
        depth_compressed_len = np.array(depth_compressed_len)
        data_dict['/depth_compress_len'] = depth_compressed_len
        depth_padded_size = depth_compressed_len.max()
        for cam_name in depth_camera_names:
            depth_compressed_image_list = data_dict[f'/observations/depth_images/{cam_name}']
            depth_padded_compressed_image_list = []
            for depth_compressed_image in depth_compressed_image_list:
                depth_padded_compressed_image = np.zeros(depth_padded_size, dtype='uint8')
                image_len = len(depth_compressed_image)
                # padded_compressed_image[:image_len] = compressed_image # two camera
                depth_padded_compressed_image[:image_len] = depth_compressed_image[:,0] # three camera
                depth_padded_compressed_image_list.append(depth_padded_compressed_image)
            data_dict[f'/observations/depth_images/{cam_name}'] = depth_padded_compressed_image_list
        print("depth_images成功保存文件到数组")
        
        
        # 点云数据压缩
        pcd_data = {}
        pcd_buffer = {
            'xyz': [],
            'rgb': [],
            'padding_mask': []
        }
        for cam_name in pointcloud_names:
            pcd_buffer['xyz'] = data_dict[f'/observations/pointcloud/{cam_name}/xyz']
            pcd_buffer['rgb'] = data_dict[f'/observations/pointcloud/{cam_name}/rgb']
            # find the max number of points
            max_pcd_n = max(
                len(pcd_buffer['xyz'][i]) for i in range(len(pcd_buffer['xyz']))
            )
            # pad to the max number of points
            padded_pcd_xyz, padded_pcd_rgb, padding_mask = [], [], []
            for xyz, rgb in zip(pcd_buffer['xyz'], pcd_buffer['rgb']):
                padded_pcd_xyz.append(
                    np.concatenate(
                        [
                            xyz,
                            np.zeros((max_pcd_n - len(xyz), 3), dtype=xyz.dtype),
                        ],
                        axis=0,
                    )
                )
                padded_pcd_rgb.append(
                    np.concatenate(
                        [
                            rgb,
                            np.zeros((max_pcd_n - len(rgb), 3), dtype=rgb.dtype),
                        ],
                        axis=0,
                    )
                )
                padding_mask.append(
                    np.concatenate(
                        [
                            np.ones(len(xyz), dtype=bool),
                            np.zeros(max_pcd_n - len(xyz), dtype=bool),
                        ],
                        axis=0,
                    )
                )
            pcd_data = {
                'xyz': np.stack(padded_pcd_xyz, axis=0),  # (T, N_max, 3)
                'rgb': np.stack(padded_pcd_rgb, axis=0),  # (T, N_max, 3)
                'padding_mask': np.stack(padding_mask, axis=0),  # (T, N_max)
            }
            data_dict[f'/observations/pointcloud/{cam_name}/xyz'] = pcd_data['xyz']
            data_dict[f'/observations/pointcloud/{cam_name}/rgb'] = pcd_data['rgb']
            data_dict[f'/observations/pointcloud/{cam_name}/padding_mask'] = pcd_data['padding_mask']
        print("pointcloud成功保存文件到数组")    

        # root 是文件的根组，HDF5 文件类似于一个文件系统，文件中可以包含组（文件夹）和数据集（文件）等层次结构。
        # rdcc_nbytes=1024**2*2 设置数据缓存的大小,这里为 2MB
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
            # 设置文件级属性，attrs 是 HDF5 文件的属性集合。
            root.attrs['compress'] = True
            root.attrs['sim'] = False
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_names:
                _ = image.create_dataset(cam_name, (max_timesteps, padded_size), dtype='uint8',
                                        chunks=(1, padded_size), )
            depth_image = obs.create_group('depth_images')
            for cam_name in depth_camera_names:
                _ = depth_image.create_dataset(cam_name, (max_timesteps, depth_padded_size), dtype='uint8',
                                        chunks=(1, depth_padded_size), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            _ = obs.create_dataset('qpos', (max_timesteps, 12))
            _ = obs.create_dataset('qvel', (max_timesteps, 12))
            # _ = obs.create_dataset('effort', (max_timesteps, 12))
            _ = root.create_dataset('action', (max_timesteps, 12))
            _ = root.create_dataset('compress_len', (len(camera_names), max_timesteps))
            _ = root.create_dataset('depth_compress_len', (len(depth_camera_names), max_timesteps))

            # 创建点云数据组
            pointcloud = obs.create_group('pointcloud')
            for cam_name in pointcloud_names:
                _ = pointcloud.create_dataset(f"{cam_name}/xyz", data=data_dict[f'/observations/pointcloud/{cam_name}/xyz'])
                _ = pointcloud.create_dataset(f"{cam_name}/rgb", data=data_dict[f'/observations/pointcloud/{cam_name}/rgb'])
                _ = pointcloud.create_dataset(f"{cam_name}/padding_mask", 
                                              data=data_dict[f'/observations/pointcloud/{cam_name}/padding_mask']) 


            for name, array in data_dict.items():
                # print("name",name) 
                # print("array",array)
                # 省略号 ... 在NumPy数组的上下文中被用作一个占位符，表示要选取数组中的多个冒号切片。
                # 具体来说，如果数组是多维的，... 会自动扩展为足够数量的冒号，以选择数组的所有维度。
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
    depth_camera_names = task_config['depth_camera_names']
    pointcloud_names = task_config['pointcloud_names']

    if set_episode_idx is not None:
        episode_idx = set_episode_idx
    else:
        episode_idx = get_auto_index(dataset_dir)
    overwrite = True

    dataset_name = f'episode_{episode_idx}'
    print(dataset_name + '\n')
    return DataRecorder(DT, max_timesteps, camera_names, depth_camera_names, pointcloud_names, dataset_dir, dataset_name, overwrite),max_timesteps*DT

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