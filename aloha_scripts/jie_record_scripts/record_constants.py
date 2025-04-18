### Task parameters
import pathlib
import os

DATA_DIR = os.path.join('/'.join(__file__.split('/')[:-2]),'data')
TASK_CONFIGS = { 
    # by sdk
    'aloha_pick_fruit10':{
        'dataset_dir': DATA_DIR + '/aloha_pick_fruit10',
        'train_ratio': 0.92,
        'episode_len': 160,
        # 'camera_names': ['cam_high', 'cam_left', 'cam_right']
        'camera_names': {
                        # 'cam_left':"236522072295",
                        # 'cam_wrist':"236522071143",
                        'cam_right':"236522072295"
                        },
    },              
}


import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# tcp normalization and gripper width normalization
TRANS_MIN, TRANS_MAX = np.array([-0.5, -0.5, 0]), np.array([0.5, 0.5, 1.0]) 
MAX_GRIPPER_WIDTH = 0.11 # meter

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# camera intrinsic matrix
CAM_INTRINSICS = np.array([604.988525390625, 604.2501831054688, 325.60302734375, 251.7237548828125]),

# safe workspace in base coordinate
SAFE_EPS = 0.002
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.0])
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 0.02 # meter



IMAGE_SHAPE = (640, 360, 3)
HISTORY_QPOS_LEN_SCALE = 0.4

### ALOHA fixed constants
DT = 0.05
FPS = 20
JOINT_NAMES = ["r_L1", "r_L2", "r_L3", "r_L4", "r_L5", "r_L6","l_L1", "l_L2", "l_L3", "l_L4", "l_L5", "l_L6"]
#START_ARM_POSE = [0.,-0.605,0.074,0.,1.479,0., 0.02239, -0.02239]
START_ARM_POSE = [0.0, -1.29, 0.97, 0.0, 0.91, 0.]
# START_ARM_POSE = [-0.00153, -1.782, 1.557, 0.0092, 0.813, 0.0107, 0.07]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path
