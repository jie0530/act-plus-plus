### Task parameters
import pathlib
import os

DATA_DIR = '/home/wsco/jie_ws/datasets'
TASK_CONFIGS = { 
    'aloha_pick_fruit10':{
        'dataset_dir': DATA_DIR + '/aloha_pick_fruit10',
        # 'dataset_dir': '/home/wsco/jie_ws/src/act-plus-plus/aloha_scripts/data/aloha_pick_fruit10',
        'train_ratio': 0.92,
        'episode_len': 180,
        # 'camera_names': ['cam_high', 'cam_left', 'cam_right']
        'camera_names': {
                        'cam_high':"915112071851",
                        # 'cam_wrist':"236522071143",
                        # 'cam_right':"236522072295"
                        },
    },            
    'aloha_pick_trans_bottle':{
        'dataset_dir': DATA_DIR + '/aloha_pick_trans_bottle',
        # 'dataset_dir': '/home/wsco/jie_ws/src/act-plus-plus/aloha_scripts/data/aloha_pick_trans_bottle',
        'train_ratio': 0.92,
        'episode_len': 180,
        # 'camera_names': ['cam_high', 'cam_left', 'cam_right']
        'camera_names': {
                        'cam_high':"915112071851",
                        # 'cam_wrist':"236522071143",
                        # 'cam_right':"236522072295"
                        },
    },
}
