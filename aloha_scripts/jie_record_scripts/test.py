import numpy as np
from scipy.spatial.transform import Rotation as R
def transform_pointcloud():
    # 平移向量
    translation = np.array([-0.769, 0.426, 0.266])
    quaternion = np.array([0.067, 0.831, -0.552, 0.022])
    
    translation = np.array([-0.819, -0.609, 0.463])
    quaternion = np.array([0.916, -0.011, -0.011, -0.401])

    # 使用 scipy 库将四元数转换为旋转矩阵
    rotation = R.from_quat(quaternion)
    rotation_matrix = rotation.as_matrix()

    # 构建变换矩阵
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation

 
    print(np.array(transform_matrix))
    return transform_matrix

transform_pointcloud()