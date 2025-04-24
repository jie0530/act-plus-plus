### Task parameters
import pathlib
import os

DATA_DIR = '/home/wsco/jie_ws/datasets'
TASK_CONFIGS = { 
    # by sdk
    'aloha_pick_fruit10':{
        'dataset_dir': DATA_DIR + '/aloha_pick_fruit10',
        'train_ratio': 0.92,
        'episode_len': 180,
        # 'camera_names': ['cam_high', 'cam_left', 'cam_right']
        'camera_names': {
                        'cam_high':"915112071851",
                        'cam_wrist':"236522071143",
                        # 'cam_right':"236522072295"
                        },
    },            
    'aloha_pick_trans_bottle':{
        'dataset_dir': DATA_DIR + '/aloha_pick_trans_bottle',
        'train_ratio': 0.92,
        'episode_len': 180,
        # 'camera_names': ['cam_high', 'cam_left', 'cam_right']
        'camera_names': {
                        'cam_high':"915112071851",
                        'cam_wrist':"236522071143",
                        # 'cam_right':"236522072295"
                        },
    },              
}


import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406])
IMG_STD = np.array([0.229, 0.224, 0.225])

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.9, -0.5, 0])
WORKSPACE_MAX = np.array([-0.4, 0.5, 1.0])

# camera intrinsic matrix
CAM_INTRINSICS = np.array([604.988525390625, 604.2501831054688, 325.60302734375, 251.7237548828125]),

IMAGE_SHAPE = (640, 360, 3)
HISTORY_QPOS_LEN_SCALE = 0.4

### ALOHA fixed constants
DT = 0.05
FPS = 20
