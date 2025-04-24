import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from itertools import repeat
from tqdm import tqdm
from einops import rearrange
import wandb
import time
from torchvision import transforms

from utils_arm_gripper_pcl import load_data # data functions
from utils_arm_gripper_pcl import compute_dict_mean, set_seed
from policy import ACTPolicy, DiffusionPolicy

from detr.models.latent_model import Latent_Model_Transformer
from constants import TASK_CONFIGS

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
    pcl_type = args['pcl_type']

    task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    depth_camera_names = task_config.get('depth_camera_names', None)
    pointcloud_names = task_config.get('pointcloud_names', None)
    stats_dir = task_config.get('stats_dir', None)
    sample_weights = task_config.get('sample_weights', None)
    train_ratio = task_config.get('train_ratio', 0.99)
    name_filter = task_config.get('name_filter', lambda n: True)

    # fixed parameters
    state_dim = 7
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
                         'action_dim': 7,#org:16
                         'no_encoder': args['no_encoder'],
                         }
    elif policy_class == 'Diffusion':

        policy_config = {'lr': args['lr'],
                         'camera_names': camera_names,
                         'depth_camera_names': depth_camera_names,
                         'pointcloud_names': pointcloud_names,
                         'action_dim': 7,
                         'observation_horizon': 1,
                         'action_horizon': 8,
                         'prediction_horizon': args['chunk_size'],
                         'num_queries': args['chunk_size'],
                         'num_inference_timesteps': 10,
                         'ema_power': 0.75,
                         'vq': False,
                         }
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
        'pcl_type': pcl_type,
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

    # train_dataloader, val_dataloader, stats, _ = load_data(dataset_dir, name_filter, camera_names, batch_size_train, batch_size_val, args['chunk_size'], args['skip_mirrored_data'], config['load_pretrain'], policy_class, stats_dir, sample_weights, train_ratio, depth_camera_names, pointcloud_names)
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,                    # 必需的位置参数
        name_filter,                    # 必需的位置参数
        camera_names,                   # 必需的位置参数
        pcl_type,                      # 必需的位置参数
        batch_size_train,              # 必需的位置参数
        batch_size_val,                # 必需的位置参数
        args['chunk_size'],            # 必需的位置参数
        skip_mirrored_data=args['skip_mirrored_data'],    # 关键字参数
        load_pretrain=config['load_pretrain'],            # 关键字参数
        policy_class=policy_class,                        # 关键字参数
        stats_dir_l=stats_dir,                           # 关键字参数
        sample_weights=sample_weights,                    # 关键字参数
        train_ratio=train_ratio,                         # 关键字参数
    )

    # save dataset stats
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_step, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ step{best_step}')
    # wandb.finish()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
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


def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
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

def forward_pass(data, policy):
    qpos_data, action_data, is_pad, pointcloud_data = data 
    qpos_data, action_data, is_pad = qpos_data.cuda(), action_data.cuda(), is_pad.cuda()

    # 将 pointcloud_data 字典中的张量移动到 CUDA 设备
    pointcloud_data = {
        'xyz': pointcloud_data['xyz'].cuda(),
        'rgb': pointcloud_data['rgb'].cuda()
    }

    return policy(qpos_data, action_data, is_pad, pointcloud=pointcloud_data)


def train_bc(train_dataloader, val_dataloader, config):
    num_steps = config['num_steps']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    eval_every = config['eval_every']
    validate_every = config['validate_every']
    save_every = config['save_every']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    if config['load_pretrain']:
        loading_status = policy.deserialize(torch.load(os.path.join('/home/wsco/jie_ws/src/act-plus-plus/trainings/', 'policy_best.ckpt')))
        print(f'loaded! {loading_status}')
    if config['resume_ckpt_path'] is not None:
        loading_status = policy.deserialize(torch.load(config['resume_ckpt_path']))
        print(f'Resume policy from: {config["resume_ckpt_path"]}, Status: {loading_status}')
    policy.cuda()
    # 根据策略类创建优化器
    optimizer = make_optimizer(policy_class, policy)

    #add
    train_history = []
    validation_history = []
    
    min_val_loss = np.inf
    best_ckpt_info = None
    
    train_dataloader = repeater(train_dataloader)
    for step in tqdm(range(num_steps+1)):
        # validation
        if step % validate_every == 0:
            print('validating')

            with torch.inference_mode():
                policy.eval()
                validation_dicts = []
                # 从数据加载器中获取数据，并将其传递给forward_pass函数。forward_pass函数执行模型的前向传播
                for batch_idx, data in enumerate(val_dataloader):
                    forward_dict = forward_pass(data, policy)
                    validation_dicts.append(forward_dict)
                    if batch_idx > 50:
                        break

                validation_summary = compute_dict_mean(validation_dicts)
                validation_history.append(validation_summary)

                epoch_val_loss = validation_summary['loss']
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (step, min_val_loss, deepcopy(policy.serialize()))
            for k in list(validation_summary.keys()):
                validation_summary[f'val_{k}'] = validation_summary.pop(k)            
            # wandb.log(validation_summary, step=step)
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in validation_summary.items():
                summary_string += f'{k}: {v.item():.3f} '
            print(summary_string)
                
        # evaluation
        if (step > 0) and (step % eval_every == 0):
            # first save then eval
            ckpt_name = f'policy_step_{step}_seed_{seed}.ckpt'
            ckpt_path = os.path.join(ckpt_dir, ckpt_name)
            torch.save(policy.serialize(), ckpt_path)
            # success, _ = eval_bc(config, ckpt_name, save_episode=True, num_rollouts=10)
            # wandb.log({'success': success}, step=step)

        # training
        policy.train()
        optimizer.zero_grad()
        data = next(train_dataloader)
        # 将当前批次的数据data和模型policy作为输入。forward_pass函数执行模型的前向传播，并返回一个字典forward_dict
        forward_dict = forward_pass(data, policy)
        # backward
        # 从forward_dict字典中提取损失值，并将其存储在变量loss中。这个损失值用于衡量模型预测与实际标签之间的差异，是模型优化的目标。
        loss = forward_dict['loss']
        # 执行反向传播。它根据损失值计算模型中所有可训练参数的梯度。这些梯度指示了如何调整参数以最小化损失值。
        loss.backward()
        # 使用优化器更新模型的参数。optimizer是预先定义好的，它根据计算出的梯度来调整模型的参数，以期望在下一次迭代中减少损失值。
        optimizer.step()
        
        # for batch_idx, data in enumerate(train_dataloader):
        #     forward_dict = forward_pass(data, policy)
        #     # backward
        #     loss = forward_dict['loss']
        #     loss.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #     train_history.append(detach_dict(forward_dict))
        # epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*step:(batch_idx+1)*(step+1)])
        # epoch_train_loss = epoch_summary['loss']
        # print(f'Train loss: {epoch_train_loss:.5f}')
        # summary_string = ''
        # for k, v in epoch_summary.items():
        #     summary_string += f'{k}: {v.item():.3f} '
        # print(summary_string)
        
        # wandb.log(forward_dict, step=step) # not great, make training 1-2% slower

        if step % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_step_{step}_seed_{seed}.ckpt')
            torch.save(policy.serialize(), ckpt_path)
            # plot_history(train_history, validation_history, step, ckpt_dir, seed)


    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.serialize(), ckpt_path)

    best_step, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_step_{best_step}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at step {best_step}')

    # save training curves
    # plot_history(train_history, validation_history, num_steps, ckpt_dir, seed)


    return best_ckpt_info

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')

def repeater(data_loader):
    epoch = 0
    for loader in repeat(data_loader):
        for data in loader:
            yield data
        print(f'Epoch {epoch} done')
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--pcl_type', action='store', type=str, help='pcl_type', required=True)
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
