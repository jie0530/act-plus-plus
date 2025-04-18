'''
RealSense Camera.
'''

import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from record_constants import * 
from scipy.spatial.transform import Rotation as R
import cv2
import ros_numpy

class RealSenseRGBDCamera:
    '''
    RealSense RGB-D Camera.
    '''
    def __init__(
        self, 
        serial, 
        frame_rate = 30, 
        resolution = (640, 360),
        # resolution = (180, 320),
        align = True,
        **kwargs
    ):
        '''
        Initialization.

        Parameters:
        - serial: str, required, the serial number of the realsense device;
        - frame_rate: int, optional, default: 15, the framerate of the realsense camera;
        - resolution: (int, int), optional, default: (1280, 720), the resolution of the realsense camera;
        - align: bool, optional, default: True, whether align the frameset with the RGB image.
        '''
        super(RealSenseRGBDCamera, self).__init__()
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.serial = serial
        # =============== Support L515 Camera ============== #
        self.is_radar = str.isalpha(serial[0])
        # print(f"self.is_radar: {self.is_radar}") # False
        depth_resolution = (1024, 768) if self.is_radar else resolution
        if self.is_radar:
            frame_rate = max(frame_rate, 30)
            self.depth_scale = 4000
        else:
            self.depth_scale = 1000
        # ================================================== #
        self.config.enable_device(self.serial)
        self.config.enable_stream(rs.stream.depth, depth_resolution[0], depth_resolution[1], rs.format.z16, frame_rate)
        self.config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.rgb8, frame_rate)
        self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.with_align = align
    def get_rgb_image(self):
        '''
        Get the RGB image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
        return color_image

    def get_depth_image(self):
        '''
        Get the depth image from the camera.
        '''
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / self.depth_scale
        return depth_image

    def get_rgbd_image(self):
        '''
        Get the RGB image along with the depth image from the camera.
        '''
        frameset = self.pipeline.wait_for_frames()
        if self.with_align:
            frameset = self.align.process(frameset)
        color_image = np.asanyarray(frameset.get_color_frame().get_data()).astype(np.uint8)
        depth_image = np.asanyarray(frameset.get_depth_frame().get_data()).astype(np.float32) / self.depth_scale
        return color_image, depth_image
    
    def transform_pointcloud(self,pcd):
        # 平移向量
        translation = np.array([-0.769, 0.436, 0.265])
        # 四元数
        quaternion = np.array([0.031, 0.833, -0.553, -0.002])

        # 使用 scipy 库将四元数转换为旋转矩阵
        rotation = R.from_quat(quaternion)
        rotation_matrix = rotation.as_matrix()

        # 构建变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix
        transform_matrix[:3, 3] = translation

        # 应用变换矩阵到点云
        transformed_pcd = pcd.transform(transform_matrix)

        return transformed_pcd
    
    # rosrun tf2_ros static_transform_publisher -0.769 0.436 0.265 0.031 0.833 -0.553 -0.002 base_link cam_right_color_optical_frame     
    def cam_extrinsic(self):
        return np.array([[ -0.99807128,  0.04940125, -0.03759308, -0.769],
                    [0.05382232,  0.38686651,  -0.92056367,  0.436     ],
                    [-0.03093349,  -0.9208115,  -0.38877924,  0.265     ],
                    [ 0.,          0.,          0.,          1.        ]])
    
    
    def cam_intrinsics(self):        
        return np.array([
            [604.988525390625, 0, 325.60302734375, 0],
            [0, 604.2501831054688, 251.7237548828125, 0],
            [0, 0, 1, 0]
        ])
        # return CAM_INTRINSICS
    
    def open3d_rgb_to_rgb_packed(self, open3d_rgb):
        rgb = np.asarray(open3d_rgb)
        # 将颜色信息转换为 32 位无符号整数
        colors_uint32 = (rgb * 255).astype(np.uint32)
        # r = (colors_uint32[:, 0] << 16)
        # g = (colors_uint32[:, 1] << 8)
        # b = colors_uint32[:, 2]
        # rgb_packed = r | g | b
        return colors_uint32
    
    
    def rgbd_to_pointcloud(
        self, 
        color:np.ndarray, 
        depth:np.ndarray, 
        intrinsic:np.ndarray, 
        # extrinsic:np.ndarray=np.eye(4), 
        downsample_factor:float=1,
        fname:str=None,
        voxel_size:float=0.005
    ):
        depth = depth.astype(np.float32)

        # downsample image
        color = cv2.resize(color, (int(640 / downsample_factor), int(480 / downsample_factor))).astype(np.int8)
        depth = cv2.resize(depth, (int(640 / downsample_factor), int(480 / downsample_factor)))

        # if pcl_type == "raw":
        #     depth /= 1000.0  # from millimeters to meters
        # depth[depth < self.min_depth_m] = 0
        # depth[depth > self.max_depth_m] = 0
        

        rgbd_image = o3d.geometry.RGBDImage()
        rgbd_image = rgbd_image.create_from_color_and_depth(o3d.geometry.Image(color),
            o3d.geometry.Image(depth), depth_scale=1.0, convert_rgb_to_intensity=False)

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic()
        intrinsic_o3d.set_intrinsics(int(640 / downsample_factor), int(480 / downsample_factor), 
            intrinsic[0, 0] / downsample_factor, intrinsic[1, 1] / downsample_factor, 
            intrinsic[0, 2] / downsample_factor, intrinsic[1, 2] / downsample_factor)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic_o3d) #, extrinsic=extrinsic)
        if fname is not None:
            o3d.io.write_point_cloud(fname, pcd)
        pcd = pcd.voxel_down_sample(voxel_size) # 下采样
        # 在保存点云前添加坐标系变换
        pcd.transform(self.cam_extrinsic())
        if fname is not None:
            o3d.io.write_point_cloud(fname, pcd)
        return pcd 
    
    
    def create_point_cloud(self, colors, depths, voxel_size = 0.005):
        """
        color, depth => point cloud
        """
        h, w = depths.shape
        # fx, fy = self.cam_intrinsics[0, 0], self.cam_intrinsics[1, 1]
        # cx, cy = self.cam_intrinsics[0, 2], self.cam_intrinsics[1, 2]
        fx = CAM_INTRINSICS[0][0]
        fy = CAM_INTRINSICS[0][1]
        cx = CAM_INTRINSICS[0][2]
        cy = CAM_INTRINSICS[0][3]

        colors = o3d.geometry.Image(colors.astype(np.uint8))
        depths = o3d.geometry.Image(depths.astype(np.float32))

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width = w, height = h, fx = fx, fy = fy, cx = cx, cy = cy
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colors, depths, depth_scale = 1.0, convert_rgb_to_intensity = False
        )
        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera_intrinsics)
        cloud = cloud.voxel_down_sample(voxel_size) # 下采样
        
        # # 将点云转换到base_link坐标系
        # xyz = [-0.783746, 0.437297, 0.246427+0.018]
        # rpy = [-1.96669+1.5+1.5, 0.0312856+23, 3.09404-90-5]
        # # 转换为弧度
        # rpy = np.radians(rpy)

        # # 转换为齐次变换矩阵
        # camera_to_base = self.xyz_rpy_to_homogeneous_matrix(xyz, rpy)
        # cloud.transform(camera_to_base)
        
        points = np.array(cloud.points).astype(np.float32)
        colors = np.array(cloud.colors).astype(np.float32)

        # 只取workspace范围内的点
        # x_mask = ((points[:, 0] >= WORKSPACE_MIN[0]) & (points[:, 0] <= WORKSPACE_MAX[0]))
        # y_mask = ((points[:, 1] >= WORKSPACE_MIN[1]) & (points[:, 1] <= WORKSPACE_MAX[1]))
        # z_mask = ((points[:, 2] >= WORKSPACE_MIN[2]) & (points[:, 2] <= WORKSPACE_MAX[2]))
        # mask = (x_mask & y_mask & z_mask)
        # points = points[mask]
        # colors = colors[mask]
        # imagenet normalization
        # colors = (colors - IMG_MEAN) / IMG_STD
        # final cloud
        cloud_final = np.concatenate([points, colors], axis = -1).astype(np.float32)
        return cloud_final
    

    def xyz_rpy_to_homogeneous_matrix(self, xyz, rpy):
        """
        将 xyz 平移向量和 rpy 旋转角度转换为 4x4 的齐次变换矩阵
        :param xyz: 包含 x, y, z 坐标的列表或数组
        :param rpy: 包含滚转、俯仰、偏航角度的列表或数组，单位为弧度
        :return: 4x4 的齐次变换矩阵
        """
        # 创建旋转对象
        rotation = R.from_euler('xyz', rpy)
        # 获取旋转矩阵
        rotation_matrix = rotation.as_matrix()
        # 创建 4x4 的齐次变换矩阵
        homogeneous_matrix = np.eye(4)
        # 将旋转矩阵赋值给齐次变换矩阵的左上角 3x3 子矩阵
        homogeneous_matrix[:3, :3] = rotation_matrix
        # 将平移向量赋值给齐次变换矩阵的最后一列的前三个元素
        homogeneous_matrix[:3, 3] = xyz
        return homogeneous_matrix

    def merge_xyz_rgb(self, xyz, rgb):
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

    def pcl_splite(self, pcd):
        pcd_xyz, pcd_xyz_mask = self.get_xyz_points(pcd, remove_nans=True)
        pcd = ros_numpy.point_cloud2.split_rgb_field(pcd)
        pcd_rgb = np.zeros(pcd.shape + (3,), dtype=np.uint8)
        pcd_rgb[..., 0] = pcd["r"]
        pcd_rgb[..., 1] = pcd["g"]
        pcd_rgb[..., 2] = pcd["b"]
        pcd_rgb = pcd_rgb[pcd_xyz_mask]
        
        
    def get_xyz_points(self, cloud_array, remove_nans=True, dtype=np.float32):
        """Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        """
        mask = None
        # remove crap points
        if remove_nans:
            # 将xyz的nan值去除
            mask = (
                np.isfinite(cloud_array["x"])
                & np.isfinite(cloud_array["y"])
                & np.isfinite(cloud_array["z"])
            )
            cloud_array = cloud_array[mask]

        # pull out x, y, and z values
        points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        points[..., 0] = cloud_array["x"]
        points[..., 1] = cloud_array["y"]
        points[..., 2] = cloud_array["z"]

        return points, mask    