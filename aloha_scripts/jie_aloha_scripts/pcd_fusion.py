from typing import Dict, Optional, List
from functools import partial

import rospy
from sensor_msgs.msg import JointState, PointCloud2
import numpy as np
import ros_numpy
import tf2_ros
from geometry_msgs.msg import TransformStamped

try:
    import fpsample
except ImportError:
    print(
        "[WARNING] fpsample not found. PCDFusionPublisher won't work if set use_fps=True"
    )

from utils import get_xyz_points


def merge_xyz_rgb(xyz, rgb):
    # 将点云的空间坐标(xyz)和颜色信息(rgb)合并成一个结构化数组
    # 将RGB颜色值打包成一个32位的浮点数
    # 用于创建ROS点云消息
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = np.asarray(rgb, dtype=np.uint8)

    rgb_packed = np.asarray(
        (rgb[:, 0].astype(np.uint32) << 16)
        | (rgb[:, 1].astype(np.uint32) << 8)
        | rgb[:, 2].astype(np.uint32),
        dtype=np.uint32,
    ).view(np.float32)

    structured_array = np.zeros(
        xyz.shape[0],
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.float32),
        ],
    )
    structured_array["x"] = xyz[:, 0]
    structured_array["y"] = xyz[:, 1]
    structured_array["z"] = xyz[:, 2]
    structured_array["rgb"] = rgb_packed

    return structured_array


class PCDFusionPublisher:

    def __init__(
        # 设置点云下采样参数
        # 配置订阅的话题（关节状态和点云数据）
        # 初始化ROS节点和发布器/订阅器
        # 设置空间裁剪范围
        # 初始化机器人运动学模型
        self,
        downsample_N: int,
        use_fps: bool,
        publish_freq: int,
        spatial_cutoff: List[float],
        fps_h: int = 7,
        point_cloud_topics: Optional[Dict[str, str]] = None,
    ):
        point_cloud_topics = point_cloud_topics or {
            'cam_high':"/cam_high/depth/color/points",
            # 'cam_left':"/cam_left/depth/color/points",
            'cam_right':"/cam_right/depth/color/points"
        }
        self._xmin, self._xmax = spatial_cutoff[0], spatial_cutoff[1]
        self._ymin, self._ymax = spatial_cutoff[2], spatial_cutoff[3]
        self._zmin, self._zmax = spatial_cutoff[4], spatial_cutoff[5]
        self._downsample_N = downsample_N
        self._use_fps = use_fps
        self._fps_h = fps_h

        self._point_cloud_data: Dict[str, Optional[Dict[str, np.array]]] = {
            k: None for k in point_cloud_topics
        }

        # ros node initialization
        rospy.init_node("fused_pcd_publisher_jetson", anonymous=True)
        self._rate = rospy.Rate(publish_freq)

        self._point_cloud_subs = {
            k: rospy.Subscriber(
                v, PointCloud2, partial(self._update_pointcloud_callback, name=k)
            )
            for k, v in point_cloud_topics.items()
        }
        self._fused_pcd_pub = rospy.Publisher(
            "/fused_pcd", PointCloud2, queue_size=1
        )

        # 为每个相机添加单独的发布器
        self._individual_pcd_pubs = {
            camera_name: rospy.Publisher(
                f'/individual_pcd_{camera_name}', 
                PointCloud2, 
                queue_size=1
            )
            for camera_name in point_cloud_topics.keys()
        }

        for topic in point_cloud_topics.values():
            rospy.wait_for_message(topic, PointCloud2)

        # 添加 TF 相关初始化
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        
        # 相机帧 ID
        self._camera_frames = {
            # 'cam_high': 'cam_high_link',
            # 'cam_left': 'cam_left_link',
            # 'cam_right': 'cam_right_link'
            'cam_high': 'cam_high_depth_optical_frame',
            # 'cam_left': 'cam_left_depth_optical_frame',
            'cam_right': 'cam_right_depth_optical_frame'
        }

    def _update_pointcloud_callback(self, pcd_msg: PointCloud2, name: str):
        stamp = pcd_msg.header.stamp.secs + pcd_msg.header.stamp.nsecs * 1e-9
        pcd = ros_numpy.point_cloud2.pointcloud2_to_array(pcd_msg)
        pcd_xyz, pcd_xyz_mask = get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        self._point_cloud_data[name] = {
            "xyz": pcd_xyz,
            "rgb": pcd_rgb,
            "stamp": np.array([stamp]),
        }

    def _get_transform_matrix(self, target_frame, source_frame):
        """获取从source_frame到target_frame的4x4变换矩阵"""
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rospy.Time(0)  # 获取最新的转换
            )
            
            # 创建4x4变换矩阵
            matrix = np.eye(4)
            
            # 设置平移部分
            matrix[0, 3] = transform.transform.translation.x
            matrix[1, 3] = transform.transform.translation.y
            matrix[2, 3] = transform.transform.translation.z
            
            # 从四元数获取旋转矩阵
            q = transform.transform.rotation
            xx = q.x * q.x
            xy = q.x * q.y
            xz = q.x * q.z
            xw = q.x * q.w
            yy = q.y * q.y
            yz = q.y * q.z
            yw = q.y * q.w
            zz = q.z * q.z
            zw = q.z * q.w
            
            matrix[0, 0] = 1 - 2 * (yy + zz)
            matrix[0, 1] = 2 * (xy - zw)
            matrix[0, 2] = 2 * (xz + yw)
            matrix[1, 0] = 2 * (xy + zw)
            matrix[1, 1] = 1 - 2 * (xx + zz)
            matrix[1, 2] = 2 * (yz - xw)
            matrix[2, 0] = 2 * (xz - yw)
            matrix[2, 1] = 2 * (yz + xw)
            matrix[2, 2] = 1 - 2 * (xx + yy)
            
            return matrix
            
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn(f"无法获取从 {source_frame} 到 {target_frame} 的转换: {str(e)}")
            return None

    def _process_single_camera_pointcloud(self, camera_name: str, pcd_data: Dict):
        """处理单个相机的点云数据"""
        # 获取相机到base_link的转换矩阵
        transform = self._get_transform_matrix(
            "base_link",
            self._camera_frames[camera_name]
        )
        
        if transform is None:
            rospy.logwarn(f"跳过 {camera_name} 的点云数据，因为无法获取转换关系")
            return None, None
            
        xyz = pcd_data["xyz"]
        rgb = pcd_data["rgb"]
        
        # 添加齐次坐标
        xyz_homo = np.concatenate(
            [xyz, np.ones((xyz.shape[0], 1))], 
            axis=-1
        )  # (N_points, 4)
        
        # 转换到base_link坐标系
        transformed_xyz = (transform @ xyz_homo.T).T[..., :3]
        
        # 空间裁剪
        x_mask = np.logical_and(
            transformed_xyz[:, 0] >= self._xmin, 
            transformed_xyz[:, 0] <= self._xmax
        )
        y_mask = np.logical_and(
            transformed_xyz[:, 1] >= self._ymin, 
            transformed_xyz[:, 1] <= self._ymax
        )
        z_mask = np.logical_and(
            transformed_xyz[:, 2] >= self._zmin, 
            transformed_xyz[:, 2] <= self._zmax
        )
        cutoff_mask = np.logical_and(x_mask, np.logical_and(y_mask, z_mask))
        
        transformed_xyz = transformed_xyz[cutoff_mask]
        rgb = rgb[cutoff_mask]
        
        # 降采样
        if len(transformed_xyz) > self._downsample_N:
            if self._use_fps:
                sampling_idx = fpsample.bucket_fps_kdline_sampling(
                    transformed_xyz, 
                    n_samples=self._downsample_N, 
                    h=self._fps_h
                )
            else:
                sampling_idx = np.random.permutation(len(transformed_xyz))[
                    : self._downsample_N
                ]
            transformed_xyz = transformed_xyz[sampling_idx]
            rgb = rgb[sampling_idx]
            
        return transformed_xyz, rgb

    def _fused_pcd_pub_callback(self):
        """融合点云数据并发布"""
        if not all(self._point_cloud_data.values()):
            return
            
        transformed_pcd_xyz, pcd_rgb = [], []
        
        # 处理每个相机的点云数据
        for camera_name, pcd_data in self._point_cloud_data.items():
            # 处理单个相机的点云
            transformed_xyz, rgb = self._process_single_camera_pointcloud(
                camera_name, 
                pcd_data
            )
            
            if transformed_xyz is None:
                continue
                
            # 发布单个相机的处理后点云
            cloud_array = merge_xyz_rgb(transformed_xyz, rgb)
            pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
                cloud_array, rospy.Time.now(), "base_link"
            )
            self._individual_pcd_pubs[camera_name].publish(pointcloud_msg)
            
            # 收集用于融合的点云数据
            transformed_pcd_xyz.append(transformed_xyz)
            pcd_rgb.append(rgb)

        if not transformed_pcd_xyz:  # 如果没有有效的点云数据
            return

        # 合并所有点云
        fused_pcd_xyz = np.concatenate(transformed_pcd_xyz, axis=0)
        fused_pcd_rgb = np.concatenate(pcd_rgb, axis=0)
        
        # 发布融合后的点云
        cloud_array = merge_xyz_rgb(fused_pcd_xyz, fused_pcd_rgb)
        pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(
            cloud_array, rospy.Time.now(), "base_link"
        )
        self._fused_pcd_pub.publish(pointcloud_msg)
        self._rate.sleep()

    def run(self):
        while not rospy.is_shutdown():
            self._fused_pcd_pub_callback()
        rospy.spin()
