import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from jie_aloha_scripts.constants import DT

import IPython
e = IPython.embed

def exponential_moving_average(data, alpha=0.1):
    """使用指数加权平均法平滑数据"""
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # 初始化第一个数据点
    for i in range(1, len(data)):
        smoothed_data[i] = alpha * data[i] + (1 - alpha) * smoothed_data[i-1]
    return smoothed_data

def smooth_and_save(dataset_dir, dataset_name, alpha=0.1):
    """对原始数据集中的qpos和action进行平滑处理并保存，保留其他数据不变"""
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()
    
    with h5py.File(dataset_path, 'r+') as f:  # 'r+' 模式可以读取和修改文件
        # 加载原始数据
        qpos = f['/observations/qpos'][()]
        action = f['/action'][()]
        
        # 对qpos和action数据进行平滑
        qpos_smooth = np.array([exponential_moving_average(qpos[:, i], alpha) for i in range(qpos.shape[1])]).T
        action_smooth = np.array([exponential_moving_average(action[:, i], alpha) for i in range(action.shape[1])]).T
        
        # 保存平滑后的数据到同一文件中
        if '/observations/qpos' in f:
            del f['/observations/qpos']  # 如果已有平滑数据，则先删除
        if '/action' in f:
            del f['/action']  # 如果已有平滑数据，则先删除
            
        f.create_dataset('/observations/qpos', data=qpos_smooth)
        f.create_dataset('/action', data=action_smooth)
        
        print(f"Saved smoothed data to: {dataset_path}")

def main(args):
    dataset_dir = args['dataset_dir']
    episode_idx = args['episode_idx']
    alpha = 0.08  # 指数加权平均的平滑系数，值越大，越强调当前数据点
    
    # If the --all flag is set, iterate over all available datasets
    if args['all']:
        dataset_names = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]  # Assuming datasets are in .hdf5 format
        print(f"Visualizing datasets: {dataset_names}")
        for dataset_name in dataset_names:
            dataset_name = dataset_name.rsplit('.', 1)[0]
            print(f"Visualizing dataset: {dataset_name}")
            smooth_and_save(dataset_dir, dataset_name, alpha)
    else:
        dataset_name = f'episode_{episode_idx}'
        smooth_and_save(dataset_dir, dataset_name, alpha)
        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=False)
    parser.add_argument('--all', action='store_true', help='Visualize all datasets instead of one episode')
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))
