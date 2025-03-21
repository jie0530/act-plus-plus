### Task parameters
import pathlib
import os

DATA_DIR = os.path.join('/'.join(__file__.split('/')[:-2]),'data')
TASK_CONFIGS = { 
    'aloha_pick_fruit9':{
        'dataset_dir': DATA_DIR + '/aloha_pick_fruit9',
        'train_ratio': 0.92,
        'episode_len': 100,
        # 'camera_names': ['cam_high', 'cam_left', 'cam_right']
        'camera_names': {
                        'cam_high':"/cam_high/color/image_raw",
                        'cam_left':"/cam_left/color/image_raw",
                        'cam_right':"/cam_right/color/image_raw"
                        },
        'depth_camera_names': {
            'cam_high':"/cam_high/aligned_depth_to_color/image_raw",
            'cam_left':"/cam_left/aligned_depth_to_color/image_raw",
            'cam_right':"/cam_right/aligned_depth_to_color/image_raw"
        },
        'pointcloud_names': {
            'cam_high':"/cam_high/depth/color/points",
            'cam_left':"/cam_left/depth/color/points",
            'cam_right':"/cam_right/depth/color/points"
        },
    },
    # elevator
    'aloha_mobile_elevator':{
        'dataset_dir': DATA_DIR + '/aloha_mobile_elevator',
        'train_ratio': 0.99,
        'episode_len': 8500,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    'aloha_mobile_elevator_truncated':{
        'dataset_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2', # 1200
            DATA_DIR + '/aloha_mobile_elevator_button', # 800
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_mobile_elevator_truncated',
            DATA_DIR + '/aloha_mobile_elevator_2',
        ],
        'sample_weights': [3, 3, 2],
        'train_ratio': 0.99, # ratio of train data from the first dataset_dir
        'episode_len': 2250,
        'camera_names': ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
    },
    # fruit platter
    'aloha_ros_4cam_0322_compressed':{
        'dataset_dir': DATA_DIR + '/aloha_ros_4cam_0322_compressed',
        'train_ratio': 0.9,
        'episode_len': 300,
        'camera_names': ['cam_1','cam_2','cam_3']#, 'cam_4']
    },
    # fruit platter
    'desktop_fruit_platter':{
        'dataset_dir': DATA_DIR + '/aloha_ros_pan_compressed',
        'train_ratio': 0.95,
        'episode_len': 1000,
        'camera_names': ['cam_1','cam_2','cam_3']#, 'cam_4']
    },
    'desktop_fruit_platter_co_train':{
        'dataset_dir': [
            DATA_DIR + '/aloha_ros_pan_compressed',
            DATA_DIR + '/aloha_ros_bot_shoes_compressed', # 800
            DATA_DIR + '/aloha_ros_bot_shoes_single_compressed', # 400
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/aloha_ros_pan_compressed',
            DATA_DIR + '/aloha_ros_bot_shoes_compressed', 
        ],
        'sample_weights': [4, 2, 2],
        'train_ratio': 0.9, # ratio of train data from the first dataset_dir
        'episode_len': 300,
        'camera_names': ['cam_1','cam_2','cam_3']#, 'cam_4']
    },
    'grape_put_plate_compressed':{
        'dataset_dir':  DATA_DIR + '/grape_put_plate_compressed', # 600
        'train_ratio': 0.95, # ratio of train data from the first dataset_dir
        'episode_len': 300,
        'camera_names': ['cam_1','cam_2','cam_3']#, 'cam_4']
    },
    'fruit_put_plate_compressed':{
        'dataset_dir':  DATA_DIR + '/fruit_put_plate_compressed', # 2000
        'train_ratio': 0.94, # ratio of train data from the first dataset_dir
        'episode_len': 2000,
        'camera_names': ['cam_1','cam_2','cam_3']#, 'cam_4']
    },
    'fruit_put_plate2_compressed':{
        'dataset_dir':  DATA_DIR + '/fruit_put_plate2_compressed', # 1000
        'train_ratio': 0.90, # ratio of train data from the first dataset_dir
        'episode_len': 1000,
        'camera_names': ['cam_1','cam_2']#,'cam_3', 'cam_4']
    },
    'fruit_put_plate3_compressed':{
        'dataset_dir':  DATA_DIR + '/fruit_put_plate3_compressed', # 1000
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 400,
        'camera_names': ['cam_1','cam_2']#,'cam_3', 'cam_4']
    },
    'fruit_put_plate4_compressed':{
        'dataset_dir':  DATA_DIR + '/fruit_put_plate4_compressed', # 1000
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 400,
        'camera_names': ['cam_1','cam_2']#,'cam_3', 'cam_4']
    },
    'pick_place_two_shoes_compressed':{
        'dataset_dir':  DATA_DIR + '/pick_place_two_shoes_compressed', # 1000
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 700,
        'camera_names': ['cam_1','cam_2']#,'cam_3', 'cam_4']
    },
    'pick_place_two_shoes_2_compressed':{
        'dataset_dir':  DATA_DIR + '/pick_place_two_shoes_2_compressed', # 1000
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 600,
        'camera_names': ['cam_1','cam_2']#,'cam_3', 'cam_4']
    },
    'pick_place_two_shoes_keyborad_compressed':{
        'dataset_dir':  DATA_DIR + '/pick_place_two_shoes_keyborad_compressed', # 1000
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 1200,
        'camera_names': ['cam_1','cam_2']#,'cam_3', 'cam_4']
    },
    'pick_place_two_shoes_keyborad_co_tarin':{
        'dataset_dir': [
            DATA_DIR + '/pick_place_two_shoes_keyborad_compressed',
            DATA_DIR + '/pick_place_two_shoes_short_keyborad_compressed', # 800 
        ], # only the first dataset_dir is used for val
        'stats_dir': [
            DATA_DIR + '/pick_place_two_shoes_keyborad_compressed',
            DATA_DIR + '/pick_place_two_shoes_short_keyborad_compressed', # 800 
        ],
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 1200,
        'camera_names': ['cam_1']#,'cam_2','cam_3', 'cam_4']
    },
    'pick_place_cloth':{
        'dataset_dir':  DATA_DIR + '/cloth', # 1000
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 2200,
        'camera_names': ['usbcam_arm']#,'usbcam_outside']
    },
    # pick_table_cloth_in_washer_3cam_sharp_finger_25hz
    'pick_table_cloth_in_washer_3cam_sharp_finger_25hz':{
        'dataset_dir':  DATA_DIR + '/pick_table_cloth_in_washer_3cam_sharp_finger_25hz',
        'train_ratio': 0.92, # ratio of train data from the first dataset_dir
        'episode_len': 2200,
        'camera_names': ['usbcam_arm']#,'usbcam_outside
    }
}

CAM_ROS_TOPIC = {
    'cam_1':"/berxel_lower/berxel_lower/color/image_raw",
    'cam_2':"/berxel_arm/berxel_arm/color/image_raw",
    'cam_3':"/berxel_left/berxel_left/color/image_raw",
    'cam_4':"/berxel_right/berxel_right/color/image_raw",
    'usbcam_arm':"/usb_cam/arm",
    'usbcam_outside':"/usb_cam/outside",
}

CAM_ROS_TOPIC2 = {
    'cam_high':"/cam_high/color/image_raw",
    'cam_left':"/cam_left/color/image_raw",
    'cam_right':"/cam_right/color/image_raw",
}

IMAGE_SHAPE = (480, 640, 3)
HISTORY_QPOS_LEN_SCALE = 0.4

### ALOHA fixed constants
DT = 0.05
FPS = 20
JOINT_NAMES = ["r_L1", "r_L2", "r_L3", "r_L4", "r_L5", "r_L6","l_L1", "l_L2", "l_L3", "l_L4", "l_L5", "l_L6"]
#START_ARM_POSE = [0.,-0.605,0.074,0.,1.479,0., 0.02239, -0.02239]
START_ARM_POSE = [0.0, -1.29, 0.97, 0.0, 0.91, 0.]
# START_ARM_POSE = [-0.00153, -1.782, 1.557, 0.0092, 0.813, 0.0107, 0.07]

XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path

# Left finger position limits (qpos[7]), right_finger = -1 * left_finger
# MASTER_GRIPPER_POSITION_OPEN = 0.02417
# MASTER_GRIPPER_POSITION_CLOSE = 0.01244
PUPPET_GRIPPER_POSITION_OPEN = 0.05800
PUPPET_GRIPPER_POSITION_CLOSE = 0.01844

# Gripper joint limits (qpos[6])
# MASTER_GRIPPER_JOINT_OPEN = -0.8
# MASTER_GRIPPER_JOINT_CLOSE = -1.65
PUPPET_GRIPPER_JOINT_OPEN = 0.057 * 2
PUPPET_GRIPPER_JOINT_CLOSE = 0.021 * 2

############################ Helper functions ############################

# MASTER_GRIPPER_POSITION_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_POSITION_CLOSE) / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_POSITION_NORMALIZE_FN = lambda x: x # (x - PUPPET_GRIPPER_POSITION_CLOSE) / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)
# MASTER_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE) + MASTER_GRIPPER_POSITION_CLOSE
PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN = lambda x: x # x * (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE) + PUPPET_GRIPPER_POSITION_CLOSE
# MASTER2PUPPET_POSITION_FN = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(MASTER_GRIPPER_POSITION_NORMALIZE_FN(x))

# MASTER_GRIPPER_JOINT_NORMALIZE_FN = lambda x: (x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE)
PUPPET_GRIPPER_JOINT_NORMALIZE_FN = lambda x: x # (x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE)
# MASTER_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN = lambda x: x #x * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
# MASTER2PUPPET_JOINT_FN = lambda x: PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(MASTER_GRIPPER_JOINT_NORMALIZE_FN(x))

# MASTER_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (MASTER_GRIPPER_POSITION_OPEN - MASTER_GRIPPER_POSITION_CLOSE)
PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN = lambda x: x / (PUPPET_GRIPPER_POSITION_OPEN - PUPPET_GRIPPER_POSITION_CLOSE)

# MASTER_POS2JOINT = lambda x: MASTER_GRIPPER_POSITION_NORMALIZE_FN(x) * (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE) + MASTER_GRIPPER_JOINT_CLOSE
# MASTER_JOINT2POS = lambda x: MASTER_GRIPPER_POSITION_UNNORMALIZE_FN((x - MASTER_GRIPPER_JOINT_CLOSE) / (MASTER_GRIPPER_JOINT_OPEN - MASTER_GRIPPER_JOINT_CLOSE))
PUPPET_POS2JOINT = lambda x: PUPPET_GRIPPER_POSITION_NORMALIZE_FN(x) * (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE) + PUPPET_GRIPPER_JOINT_CLOSE
PUPPET_JOINT2POS = lambda x: PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN((x - PUPPET_GRIPPER_JOINT_CLOSE) / (PUPPET_GRIPPER_JOINT_OPEN - PUPPET_GRIPPER_JOINT_CLOSE))

# MASTER_GRIPPER_JOINT_MID = (MASTER_GRIPPER_JOINT_OPEN + MASTER_GRIPPER_JOINT_CLOSE)/2
