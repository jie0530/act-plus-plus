import time

import numpy as np
import rospy
from .utils import PointCloudVisualizer


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


if __name__ == "__main__":


    pcd_viz = PointCloudVisualizer()

    while not rospy.is_shutdown():
        # latest_pcd = robot.last_pointcloud
        pcd_xyz, pcd_rgb = latest_pcd["xyz"], latest_pcd["rgb"]

        if isinstance(pcd_rgb, np.ndarray):
            pcd_viz(latest_pcd["xyz"], latest_pcd["rgb"])
