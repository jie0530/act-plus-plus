import numpy as np
import torch
import os
import h5py
import pickle
import fnmatch
import cv2
from time import time
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import IPython
e = IPython.embed

def flatten_list(l):
    return [item for sublist in l for item in sublist]

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path_list, camera_names, norm_stats, episode_ids, episode_len, chunk_size, policy_class, depth_camera_names=None, pointcloud_names=None, use_depth=False, use_pcd=False):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_path_list = dataset_path_list
        self.camera_names = camera_names
        self.depth_camera_names = depth_camera_names if depth_camera_names is not None else []
        self.pointcloud_names = pointcloud_names if pointcloud_names is not None else []
        self.use_depth = use_depth
        self.use_pcd = use_pcd
        
        self.norm_stats = norm_stats
        self.episode_len = episode_len
        self.chunk_size = chunk_size
        self.cumulative_len = np.cumsum(self.episode_len)
        self.max_episode_len = max(episode_len)
        self.policy_class = policy_class
        if self.policy_class == 'Diffusion':
            self.augment_images = True
        else:
            self.augment_images = False
        self.transformations = None
        self.__getitem__(0) # initialize self.is_sim and self.transformations
        self.is_sim = False

    # def __len__(self):
    #     return sum(self.episode_len)

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len[episode_index])
        episode_id = self.episode_ids[episode_index]
        return episode_id, start_ts

    def __getitem__(self, index):
        episode_id, start_ts = self._locate_transition(index)
        dataset_path = self.dataset_path_list[episode_id]
        try:
            with h5py.File(dataset_path, 'r') as root:
                try:
                    is_sim = root.attrs['sim']
                except:
                    is_sim = False
                compressed = root.attrs.get('compress', False)
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:  
                    action = root['/action'][:,:7]
                original_action_shape = action.shape
                episode_len = original_action_shape[0]
                
                # 获取观测数据
                qpos = root['/observations/qpos'][:, :7]
                qpos = qpos[start_ts]
                qvel = root['/observations/qvel'][:, :7]
                qvel = qvel[start_ts]
                
                # 获取图像数据
                image_dict = dict()
                for cam_name in self.camera_names:
                    image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                
                # 获取深度图数据
                depth_dict = dict()
                if self.use_depth:
                    for cam_name in self.depth_camera_names:
                        depth_dict[cam_name] = root[f'/observations/depth_images/{cam_name}'][start_ts]
                
                # 获取点云数据
                pointcloud_data = None
                if self.use_pcd and '/observations/pointcloud' in root:
                    pointcloud_data = {}
                    for pc_name in self.pointcloud_names:
                        pc_path = f'/observations/pointcloud/{pc_name}'
                        if pc_path in root:
                            pointcloud_data[pc_name] = {
                                'xyz': root[f'{pc_path}/xyz'][start_ts],
                                'rgb': root[f'{pc_path}/rgb'][start_ts]
                            }
                
                if compressed:
                    for cam_name in image_dict.keys():
                        decompressed_image = cv2.imdecode(image_dict[cam_name], 1)
                        image_dict[cam_name] = np.array(decompressed_image)
                
                    if self.use_depth:
                        for cam_name in depth_dict.keys():
                            decompressed_depth = cv2.imdecode(depth_dict[cam_name], 1)
                            depth_dict[cam_name] = np.array(decompressed_depth)
                
                # 获取动作数据
                if is_sim:
                    action = action[start_ts:]
                    action_len = episode_len - start_ts
                else:
                    action = action[max(0, start_ts - 1):]
                    action_len = episode_len - max(0, start_ts - 1)

            padded_action = np.zeros((self.max_episode_len, original_action_shape[1]), dtype=np.float32)
            padded_action[:action_len] = action
            is_pad = np.zeros(self.max_episode_len)
            is_pad[action_len:] = 1

            padded_action = padded_action[:self.chunk_size]
            is_pad = is_pad[:self.chunk_size]

            # 处理相机图像
            all_cam_images = []
            for cam_name in self.camera_names:
                all_cam_images.append(image_dict[cam_name])
            all_cam_images = np.stack(all_cam_images, axis=0)

            if self.use_depth:
                # construct depth images
                all_depth_images = []

                for cam_name in self.depth_camera_names:
                    depth_img = depth_dict[cam_name]
                    if len(depth_img.shape) == 3:
                        depth_img = depth_img[:,:,0]
                    all_depth_images.append(depth_img)
                all_depth_images = np.stack(all_depth_images, axis=0)
                all_depth_images = np.expand_dims(all_depth_images, axis=1)

            # construct observations
            image_data = torch.from_numpy(all_cam_images)
            depth_data = torch.from_numpy(all_depth_images).float() if self.use_depth else None
            qpos_data = torch.from_numpy(qpos).float()
            action_data = torch.from_numpy(padded_action).float()
            is_pad = torch.from_numpy(is_pad).bool()

            # 获取点云数据
            if pointcloud_data is not None:
                for pc_name in pointcloud_data:
                    pointcloud_data[pc_name]['xyz'] = torch.from_numpy(pointcloud_data[pc_name]['xyz']).float()
                    pointcloud_data[pc_name]['rgb'] = torch.from_numpy(pointcloud_data[pc_name]['rgb']).float()
            # else:
                # pointcloud_data = {
                #     'cam_high': {
                #         'xyz': torch.zeros((2000, 3), dtype=torch.float32),
                #         'rgb': torch.zeros((2000, 3), dtype=torch.float32)
                #     }
                # }
            # channel last to channel first for images
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # augmentation
            if self.transformations is None:
                print('Initializing transformations')
                original_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
                    transforms.Resize(original_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                ]

            if self.augment_images:
                for transform in self.transformations:
                    image_data = transform(image_data)
                    if depth_data is not None:
                        if not isinstance(transform, transforms.ColorJitter):
                            depth_data = transform(depth_data)

            # normalize image and change dtype to float
            image_data = image_data / 255.0
            if depth_data is not None:
                depth_data = (depth_data - depth_data.min()) / (depth_data.max() - depth_data.min() + 1e-6)

            if self.policy_class == 'Diffusion':
                # normalize to [-1, 1]
                action_data = ((action_data - self.norm_stats["action_min"]) / (self.norm_stats["action_max"] - self.norm_stats["action_min"])) * 2 - 1
            else:
                # normalize to mean 0 std 1
                action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

            qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        except:
            print(f'Error loading {dataset_path} in __getitem__')
            quit()
        if self.use_depth:
            if self.use_pcd:
                return image_data, qpos_data, action_data, is_pad, depth_data, pointcloud_data
            else:
                return image_data, qpos_data, action_data, is_pad, depth_data
        else:
            if self.use_pcd:
                return image_data, qpos_data, action_data, is_pad, pointcloud_data
            else:
                return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                # qpos = root['/observations/qpos'][()] #org
                qpos = root['/observations/qpos'][:,:7] #只取手臂数据训练
                # qpos = root['/observations/qpos'][:,6:] #只取手臂数据训练
                # qvel = root['/observations/qvel'][()]
                if '/base_action' in root:
                    base_action = root['/base_action'][()]
                    base_action = preprocess_base_action(base_action)
                    action = np.concatenate([root['/action'][()], base_action], axis=-1)
                else:
                    # action = root['/action'][()] #org
                    action = root['/action'][:,:7] #只取手臂数据训练
                    # action = root['/action'][:,6:] #只取手臂数据训练
                    # dummy_base_action = np.zeros([action.shape[0], 2])
                    # action = np.concatenate([action, dummy_base_action], axis=-1)
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
        all_episode_len.append(len(qpos))
    all_qpos_data = torch.cat(all_qpos_data, dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)

    # normalize action data
    action_mean = all_action_data.mean(dim=[0]).float()
    action_std = all_action_data.std(dim=[0]).float()
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0]).float()
    qpos_std = all_qpos_data.std(dim=[0]).float()
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    action_min = all_action_data.min(dim=0).values.float()
    action_max = all_action_data.max(dim=0).values.float()

    eps = 0.0001
    stats = {"action_mean": action_mean.numpy(), "action_std": action_std.numpy(),
             "action_min": action_min.numpy() - eps,"action_max": action_max.numpy() + eps,
             "qpos_mean": qpos_mean.numpy(), "qpos_std": qpos_std.numpy(),
             "example_qpos": qpos}

    return stats, all_episode_len

def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files

def BatchSampler(batch_size, episode_len_l, sample_weights):
    sample_probs = np.array(sample_weights) / np.sum(sample_weights) if sample_weights is not None else None
    sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in episode_len_l])
    while True:
        batch = []
        for _ in range(batch_size):
            episode_idx = np.random.choice(len(episode_len_l), p=sample_probs)
            step_idx = np.random.randint(sum_dataset_len_l[episode_idx], sum_dataset_len_l[episode_idx + 1])
            batch.append(step_idx)
        yield batch

def load_data(dataset_dir_l, name_filter, camera_names, batch_size_train, batch_size_val, chunk_size, skip_mirrored_data=False, load_pretrain=False, policy_class=None, stats_dir_l=None, sample_weights=None, train_ratio=0.99, depth_camera_names=None, pointcloud_names=None, use_depth=False, use_pcd=False):
    if type(dataset_dir_l) == str:
        dataset_dir_l = [dataset_dir_l]
    dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data) for dataset_dir in dataset_dir_l]
    num_episodes_0 = len(dataset_path_list_list[0])
    dataset_path_list = flatten_list(dataset_path_list_list)
    dataset_path_list = [n for n in dataset_path_list if name_filter(n)]
    num_episodes_l = [len(dataset_path_list) for dataset_path_list in dataset_path_list_list]
    num_episodes_cumsum = np.cumsum(num_episodes_l)

    # obtain train test split on dataset_dir_l[0]
    shuffled_episode_ids_0 = np.random.permutation(num_episodes_0)
    train_episode_ids_0 = shuffled_episode_ids_0[:int(train_ratio * num_episodes_0)]
    val_episode_ids_0 = shuffled_episode_ids_0[int(train_ratio * num_episodes_0):]
    train_episode_ids_l = [train_episode_ids_0] + [np.arange(num_episodes) + num_episodes_cumsum[idx] for idx, num_episodes in enumerate(num_episodes_l[1:])]
    val_episode_ids_l = [val_episode_ids_0]
    train_episode_ids = np.concatenate(train_episode_ids_l)
    val_episode_ids = np.concatenate(val_episode_ids_l)
    print(f'\n\nData from: {dataset_dir_l}\n- Train on {[len(x) for x in train_episode_ids_l]} episodes\n- Test on {[len(x) for x in val_episode_ids_l]} episodes\n\n')

    # obtain normalization stats for qpos and action
    # if load_pretrain:
    #     with open(os.path.join('/home/zfu/interbotix_ws/src/act/ckpts/pretrain_all', 'dataset_stats.pkl'), 'rb') as f:
    #         norm_stats = pickle.load(f)
    #     print('Loaded pretrain dataset stats')
    _, all_episode_len = get_norm_stats(dataset_path_list)
    train_episode_len_l = [[all_episode_len[i] for i in train_episode_ids] for train_episode_ids in train_episode_ids_l]
    val_episode_len_l = [[all_episode_len[i] for i in val_episode_ids] for val_episode_ids in val_episode_ids_l]
    train_episode_len = flatten_list(train_episode_len_l)
    val_episode_len = flatten_list(val_episode_len_l)
    if stats_dir_l is None:
        stats_dir_l = dataset_dir_l
    elif type(stats_dir_l) == str:
        stats_dir_l = [stats_dir_l]
    norm_stats, _ = get_norm_stats(flatten_list([find_all_hdf5(stats_dir, skip_mirrored_data) for stats_dir in stats_dir_l]))
    print(f'Norm stats from: {stats_dir_l}')

    batch_sampler_train = BatchSampler(batch_size_train, train_episode_len_l, sample_weights)
    batch_sampler_val = BatchSampler(batch_size_val, val_episode_len_l, None)

    # print(f'train_episode_len: {train_episode_len}, val_episode_len: {val_episode_len}, train_episode_ids: {train_episode_ids}, val_episode_ids: {val_episode_ids}')

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, train_episode_ids, train_episode_len, chunk_size, policy_class, depth_camera_names, pointcloud_names, use_depth, use_pcd)
    val_dataset = EpisodicDataset(dataset_path_list, camera_names, norm_stats, val_episode_ids, val_episode_len, chunk_size, policy_class, depth_camera_names, pointcloud_names, use_depth, use_pcd)
    train_num_workers = (8 if os.getlogin() == 'zfu' else 16) if train_dataset.augment_images else 2
    val_num_workers = 8 if train_dataset.augment_images else 2
    print(f'Augment images: {train_dataset.augment_images}, train_num_workers: {train_num_workers}, val_num_workers: {val_num_workers}')
    train_dataloader = DataLoader(train_dataset, batch_sampler=batch_sampler_train, pin_memory=True, num_workers=train_num_workers, prefetch_factor=2)
    val_dataloader = DataLoader(val_dataset, batch_sampler=batch_sampler_val, pin_memory=True, num_workers=val_num_workers, prefetch_factor=2)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

def calibrate_linear_vel(base_action, c=None):
    if c is None:
        c = 0.0 # 0.19
    v = base_action[..., 0]
    w = base_action[..., 1]
    base_action = base_action.copy()
    base_action[..., 0] = v - c * w
    return base_action

def smooth_base_action(base_action):
    return np.stack([
        np.convolve(base_action[:, i], np.ones(5)/5, mode='same') for i in range(base_action.shape[1])
    ], axis=-1).astype(np.float32)

def preprocess_base_action(base_action):
    # base_action = calibrate_linear_vel(base_action)
    base_action = smooth_base_action(base_action)

    return base_action

def postprocess_base_action(base_action):
    linear_vel, angular_vel = base_action
    linear_vel *= 1.0
    angular_vel *= 1.0
    # angular_vel = 0
    # if np.abs(linear_vel) < 0.05:
    #     linear_vel = 0
    return np.array([linear_vel, angular_vel])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
