import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms

from constants import FPS
# from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, postprocess_base_action # helper functions
from policy import ACTPolicy, CNNMLPPolicy, DiffusionPolicy
from visualize_episodes import save_videos

from detr.models.latent_model import Latent_Model_Transformer
from ros_sub_data import DataRecorder
import rospy
import h5py

# from sim_env import BOX_POSE

import IPython
e = IPython.embed

def get_auto_index(dataset_dir):
    max_idx = 1000
    for i in range(max_idx+1):
        if not os.path.isfile(os.path.join(dataset_dir, f'qpos_{i}.npy')):
            return i
    raise Exception(f"Error getting auto index, or more than {max_idx} episodes")

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_steps = args['num_steps']
    eval_every = args['eval_every']
    validate_every = args['validate_every']
    save_every = args['save_every']
    resume_ckpt_path = args['resume_ckpt_path']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim or task_name == 'all':
        from constants_org import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from constants import TASK_CONFIGS
        # from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 12
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'vq': args['use_vq'],
                         'vq_class': args['vq_class'],
                         'vq_dim': args['vq_dim'],
                         'action_dim': 12,#org:16
                         'no_encoder': args['no_encoder'],
                         }
    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'action_dim': 12,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    actuator_config = {
        'actuator_network_dir': args['actuator_network_dir'],
        'history_len': args['history_len'],
        'future_len': args['future_len'],
        'prediction_len': args['prediction_len'],
    }

    config = {
        'num_steps': num_steps,##act是num_epochs
        'eval_every': eval_every,#act没有用到
        'validate_every': validate_every,#act没有用到
        'save_every': save_every,#act没有用到
        'ckpt_dir': ckpt_dir,
        'resume_ckpt_path': resume_ckpt_path,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'load_pretrain': args['load_pretrain'],#act没有用到
        'actuator_config': actuator_config,#act没有用到
    }

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    config_path = os.path.join(ckpt_dir, 'config.pkl')
    expr_name = ckpt_dir.split('/')[-1]
    # WandB**（全称：Weights & Biases）是一个为机器学习和深度学习项目提供可视化、追踪和协作功能的工具平台。
    # 它帮助研究人员和开发者在训练模型时进行高效的实验管理和结果可视化。
    # if not is_eval:
    #     wandb.init(project="mobile-aloha2", reinit=True, entity="mobile-aloha2", name=expr_name)
    #     wandb.config.update(config)
    with open(config_path, 'wb') as f:
        pickle.dump(config, f) ## 将config对象序列化并写入文件
    if is_eval:
        # ckpt_names = [f'policy_last.ckpt']
        ckpt_names = [f'policy_best.ckpt']
        for ckpt_name in ckpt_names:
            eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
        exit()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'Diffusion':
        policy = DiffusionPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'Diffusion':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(obs, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(obs['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    #diffusion策略有效
    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True, num_rollouts=50):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'
    #以下三个变量act没使用
    vq = config['policy_config']['vq']
    actuator_config = config['actuator_config']
    use_actuator_net = actuator_config['actuator_network_dir'] is not None


    # To store the actions
    inference_actions = []
    ground_truth_actions = []  # Assuming ground truth actions are available or can be extracted


    sub_data = DataRecorder(camera_names)
    while True:
        # obs = sub_data.get_obs_no_finger()
        obs = sub_data.get_obs()
        if obs is False:
            rospy.sleep(0.1)
            continue
        else: break

    for i in range(5):
        # load policy and stats
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        policy = make_policy(policy_class, policy_config)
        loading_status = policy.deserialize(torch.load(ckpt_path))
        print(loading_status)
        policy.cuda()
        policy.eval()
        if vq:
            vq_dim = config['policy_config']['vq_dim']
            vq_class = config['policy_config']['vq_class']
            latent_model = Latent_Model_Transformer(vq_dim, vq_dim, vq_class)
            latent_model_ckpt_path = os.path.join(ckpt_dir, 'latent_model_last.ckpt')
            latent_model.deserialize(torch.load(latent_model_ckpt_path))
            latent_model.eval()
            latent_model.cuda()
            print(f'Loaded policy from: {ckpt_path}, latent model from: {latent_model_ckpt_path}')
        else:
            print(f'Loaded: {ckpt_path}')
        stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)
        # if use_actuator_net:
        #     prediction_len = actuator_config['prediction_len']
        #     future_len = actuator_config['future_len']
        #     history_len = actuator_config['history_len']
        #     actuator_network_dir = actuator_config['actuator_network_dir']

        #     from act.train_actuator_network import ActuatorNetwork
        #     actuator_network = ActuatorNetwork(prediction_len)
        #     actuator_network_path = os.path.join(actuator_network_dir, 'actuator_net_last.ckpt')
        #     loading_status = actuator_network.load_state_dict(torch.load(actuator_network_path))
        #     actuator_network.eval()
        #     actuator_network.cuda()
        #     print(f'Loaded actuator network from: {actuator_network_path}, {loading_status}')

        #     actuator_stats_path  = os.path.join(actuator_network_dir, 'actuator_net_stats.pkl')
        #     with open(actuator_stats_path, 'rb') as f:
        #         actuator_stats = pickle.load(f)
            
        #     actuator_unnorm = lambda x: x * actuator_stats['commanded_speed_std'] + actuator_stats['commanded_speed_std']
        #     actuator_norm = lambda x: (x - actuator_stats['observed_speed_mean']) / actuator_stats['observed_speed_mean']
        #     def collect_base_action(all_actions, norm_episode_all_base_actions):
        #         post_processed_actions = post_process(all_actions.squeeze(0).cpu().numpy())
        #         norm_episode_all_base_actions += actuator_norm(post_processed_actions[:, -2:]).tolist()

        pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
        if policy_class == 'Diffusion':
            post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']
        else:
            post_process = lambda a: a * stats['action_std'] + stats['action_mean']

        query_frequency = policy_config['num_queries']
        if temporal_agg:
            query_frequency = 1
            num_queries = policy_config['num_queries']

        max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

        ### evaluation loop
        if temporal_agg:
            # all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 16]).cuda() #org
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        # qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        # qpos_history_raw = np.zeros((max_timesteps, state_dim))
        # image_list = [] # for visualization

        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        
        #初始化订阅观测类，订阅joints, hand, images
        # print(f"camera_name : {camera_names}")
        # sub_data = DataRecorder(camera_names)
        # camera_name_topic_dict = {'cam_high':"/cam_high/color/image_raw",
        #                         # 'cam_left':"/cam_left/color/image_raw",
        #                         'cam_right':"/cam_right/color/image_raw"}
        # sub_data = DataRecorder(camera_names)
        # while True:
        #     # obs = sub_data.get_obs_no_finger()
        #     obs = sub_data.get_obs()
        #     if obs is False:
        #         rospy.sleep(0.1)
        #         continue
        #     else: break
        
        with torch.inference_mode():
            time0 = time.time()
            for t in range(max_timesteps):#max_timesteps
                time1 = time.time()

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                # obs = sub_data.get_obs_no_finger()
                obs = sub_data.get_obs()
                qpos_numpy = np.array(obs['qpos'])
                # qpos_history_raw[t] = qpos_numpy
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                # qpos_history[:, t] = qpos
                # curr_image = obs['img_all']
                curr_image = get_image(obs, camera_names, rand_crop_resize=(config['policy_class'] == 'Diffusion'))

                # if t == 0:
                #     # warm up
                #     for _ in range(10):
                #         policy(qpos, curr_image)
                #     print('network warm up done')
                #     time1 = time.time()

                ### query policy
                time3 = time.time()
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                    if temporal_agg:
                        print(f"t: {t}")
                        # print(f"all_actions: {all_actions}")
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                        # if t % query_frequency == query_frequency - 1:
                        #     # zero out base actions to avoid overshooting
                        #     raw_action[0, -2:] = 0
                elif config['policy_class'] == "Diffusion":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                        # if use_actuator_net:
                        #     collect_base_action(all_actions, norm_episode_all_base_actions)
                    raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                    all_actions = raw_action.unsqueeze(0)
                    # if use_actuator_net:
                    #     collect_base_action(all_actions, norm_episode_all_base_actions)
                else:
                    raise NotImplementedError
                # print('query policy: ', time.time() - time3)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                print(f"action: {action}")
                inference_actions.append(action)

                # if use_actuator_net:
                #     assert(not temporal_agg)
                #     if t % prediction_len == 0:
                #         offset_start_ts = t + history_len
                #         actuator_net_in = np.array(norm_episode_all_base_actions[offset_start_ts - history_len: offset_start_ts + future_len])
                #         actuator_net_in = torch.from_numpy(actuator_net_in).float().unsqueeze(dim=0).cuda()
                #         pred = actuator_network(actuator_net_in)
                #         base_action_chunk = actuator_unnorm(pred.detach().cpu().numpy()[0])
                #     base_action = base_action_chunk[t % prediction_len]
                # else:

                ### step the environment
                # sub_data.control_arm(action)
                sub_data.control_arm_finger(action)
                rospy.sleep(0.1)
    # After the inference loop in eval_bc
    dataset_path = '/home/wsco/jie_ws/src/act-plus-plus/aloha_scripts/data/'+ task_name + '/episode_0.hdf5'
    with h5py.File(dataset_path, 'r') as root:
        ground_truth_actions = root['/action'][()]
    visualize_actions(inference_actions, ground_truth_actions, 6)

def visualize_actions(inference_actions, ground_truth_actions, rows_per_plot=3):
    # Convert lists to numpy arrays for easier plotting
    inference_actions = np.array(inference_actions)
    ground_truth_actions = np.array(ground_truth_actions)
    
    # Assuming both inference and ground truth actions have the same dimensions
    num_timesteps = inference_actions.shape[0]
    num_joints = inference_actions.shape[1]
    
    # Calculate the number of plots needed
    num_plots = (num_joints + rows_per_plot - 1) // rows_per_plot  # Ceiling division
    
    # Plot comparison for each joint or action
    for plot_idx in range(num_plots):
        start_idx = plot_idx * rows_per_plot
        end_idx = min((plot_idx + 1) * rows_per_plot, num_joints)
        num_rows = end_idx - start_idx
        
        fig, axs = plt.subplots(num_rows, 1, figsize=(10, 5 * num_rows))
        
        for i in range(num_rows):
            joint_idx = start_idx + i
            axs[i].plot(range(num_timesteps), inference_actions[:, joint_idx], label='Inference Action')
            axs[i].plot(range(num_timesteps), ground_truth_actions[:, joint_idx], label='Ground Truth Action', linestyle='dashed')
            axs[i].set_title(f'Action Comparison for Joint {joint_idx}')
            axs[i].set_xlabel('Time Step')
            axs[i].set_ylabel('Action Value')
            axs[i].legend()
        
        plt.tight_layout()
        plt.savefig(f"action_comparison_part{plot_idx + 1}.png")
        print(f'Saved action comparison plot to: action_comparison_part{plot_idx + 1}.png')
        plt.close()

# def visualize_actions(inference_actions, ground_truth_actions, plot_path):
#     # Convert lists to numpy arrays for easier plotting
#     inference_actions = np.array(inference_actions)
#     ground_truth_actions = np.array(ground_truth_actions)
    
#     # Assuming both inference and ground truth actions have the same dimensions
#     num_timesteps = inference_actions.shape[0]
    
#     # Plot comparison for each joint or action
#     fig, axs = plt.subplots(inference_actions.shape[1], 1, figsize=(10, 5 * inference_actions.shape[1]))
    
#     for i in range(inference_actions.shape[1]):
#         axs[i].plot(range(num_timesteps), inference_actions[:, i], label='Inference Action')
#         axs[i].plot(range(num_timesteps), ground_truth_actions[:, i], label='Ground Truth Action', linestyle='dashed')
#         axs[i].set_title(f'Action Comparison for Joint {i}')
#         axs[i].set_xlabel('Time Step')
#         axs[i].set_ylabel('Action Value')
#         axs[i].legend()
    
#     plt.tight_layout()
#     plt.savefig(plot_path)
#     print(f'Saved action comparison plot to: {plot_path}')
#     plt.close()

if __name__ == '__main__':
    rospy.init_node('eval_arm_hand')
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_steps', action='store', type=int, help='num_steps', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--load_pretrain', action='store_true', default=False)#预训练权重默认不加载
    parser.add_argument('--eval_every', action='store', type=int, default=500, help='eval_every', required=False)
    parser.add_argument('--validate_every', action='store', type=int, default=500, help='validate_every', required=False)
    parser.add_argument('--save_every', action='store', type=int, default=500, help='save_every', required=False)
    parser.add_argument('--resume_ckpt_path', action='store', type=str, help='resume_ckpt_path', required=False)
    parser.add_argument('--skip_mirrored_data', action='store_true')
    parser.add_argument('--actuator_network_dir', action='store', type=str, help='actuator_network_dir', required=False)
    parser.add_argument('--history_len', action='store', type=int)
    parser.add_argument('--future_len', action='store', type=int)
    parser.add_argument('--prediction_len', action='store', type=int)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_vq', action='store_true')
    parser.add_argument('--vq_class', action='store', type=int, help='vq_class')
    parser.add_argument('--vq_dim', action='store', type=int, help='vq_dim')
    parser.add_argument('--no_encoder', action='store_true')
    
    main(vars(parser.parse_args()))
