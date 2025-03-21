import os
import numpy as np
import cv2
import h5py
import argparse

import matplotlib.pyplot as plt
from jie_aloha_scripts.constants import DT

import IPython
e = IPython.embed

# JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
# STATE_NAMES = JOINT_NAMES + ["gripper"]
# BASE_STATE_NAMES = ["linear_vel", "angular_vel"]
# JOINT_NAMES = ["r_L1", "r_L2", "r_L3", "r_L4", "r_L5", "r_L6","l_L1", "l_L2", "l_L3", "l_L4", "l_L5", "l_L6"]
JOINT_NAMES = ["j1", "j2", "j3", "j4", "j5", "j6"]
GRIPPER_NAME = ["thumb_cmc", "thumb_mcp", "index", "middle", "ring", "little",]
# STATE_NAMES = JOINT_NAMES + ["right_gripper_1"] + ["left_gripper"] 
# STATE_NAMES = JOINT_NAMES
STATE_NAMES = JOINT_NAMES + GRIPPER_NAME

def load_hdf5(dataset_dir, dataset_name):
    dataset_path = os.path.join(dataset_dir, dataset_name + '.hdf5')
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        compressed = root.attrs.get('compress', False)
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        if 'effort' in root.keys():
            effort = root['/observations/effort'][()]
        else:
            effort = None
        action = root['/action'][()]
        # base_action = root['/base_action'][()]
        image_dict = dict()
        for cam_name in root[f'/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]
        
        # 检查是否存在深度图像数据
        depth_image_dict = dict()
        if '/observations/depth_images' in root:
            for cam_name in root[f'/observations/depth_images/'].keys():
                depth_image_dict[cam_name] = root[f'/observations/depth_images/{cam_name}'][()]
        else:
            print("Warning: No depth images found in dataset")
        
        pcd_data_dict = dict()
        if '/observations/pointcloud' in root:
            for cam_name in root[f'/observations/pointcloud/'].keys():
                pcd_data_dict[cam_name] = dict()
                pcd_data_dict[cam_name]['xyz'] = root[f'/observations/pointcloud/{cam_name}/xyz'][()]
                pcd_data_dict[cam_name]['rgb'] = root[f'/observations/pointcloud/{cam_name}/rgb'][()]
                pcd_data_dict[cam_name]['padding_mask'] = root[f'/observations/pointcloud/{cam_name}/padding_mask'][()]
        else:
            print("Warning: No point cloud data found in dataset")
            
        if compressed:
            compress_len = root['/compress_len'][()]
            if '/depth_compress_len' in root:
                depth_compress_len = root['/depth_compress_len'][()]
            else:
                depth_compress_len = None

    # hdf5文件中压缩图的存储方式是cam_name: 帧数*压缩图 [n_frames* compressed_image]
    # 解压时先根据cam_name取出某个相机的所有数据，然后把每帧的压缩图解压，再把解压图拼起来
    if compressed:
        for cam_id, cam_name in enumerate(image_dict.keys()):
            # un-pad and uncompress
            padded_compressed_image_list = image_dict[cam_name]
            image_list = []
            for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                image_len = int(compress_len[cam_id, frame_id])
                compressed_image = padded_compressed_image
                image = cv2.imdecode(compressed_image, 1)
                image_list.append(image)
            image_dict[cam_name] = image_list
            
        # 只在有深度图像数据时进行解压
        if depth_image_dict and depth_compress_len is not None:
            for cam_id, cam_name in enumerate(depth_image_dict.keys()):
                # un-pad and uncompress
                padded_compressed_image_list = depth_image_dict[cam_name]
                depth_image_list = []
                for frame_id, padded_compressed_image in enumerate(padded_compressed_image_list): # [:1000] to save memory
                    depth_image_len = int(depth_compress_len[cam_id, frame_id])
                    depth_compressed_image = padded_compressed_image
                    depth_image = cv2.imdecode(depth_compressed_image, 1)
                    depth_image_list.append(depth_image)
                depth_image_dict[cam_name] = depth_image_list

    return qpos, qvel, effort, action, image_dict, depth_image_dict, pcd_data_dict

def main(args):
    dataset_dir = args['dataset_dir']
    task_name = args['task_name']
    vis_data_dir = '/home/wsco/jie_ws/src/act-plus-plus/aloha_scripts/data/vis_data/' + task_name
    if not os.path.exists(vis_data_dir):
        try:
            os.makedirs(vis_data_dir)
            print(f"文件夹 {vis_data_dir} 创建成功")
        except Exception as e:
            print(f"创建文件夹 {vis_data_dir} 时发生错误: {e}")

    # If the --all flag is set, iterate over all available datasets
    if args['all']:
        dataset_names = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]  # Assuming datasets are in .hdf5 format
        print(f"Visualizing datasets: {dataset_names}")
        for dataset_name in dataset_names:
            dataset_name = dataset_name.rsplit('.', 1)[0]
            print(f"Visualizing dataset: {dataset_name}")
            # data_path_complete = True
            qpos, qvel, effort, action, image_dict, depth_image_dict, pcd_data_dict = load_hdf5(dataset_dir, dataset_name)
            save_videos(image_dict, DT, video_path=os.path.join(vis_data_dir, dataset_name + '_video.mp4'))
            if depth_image_dict:  # 只在有深度图像数据时保存
                save_videos(depth_image_dict, DT, video_path=os.path.join(vis_data_dir, dataset_name + '_depth_video.mp4'))
            visualize_joints(qpos, action, plot_path=os.path.join(vis_data_dir, dataset_name + '_qpos.png'))
            if pcd_data_dict:
                # Decompress point cloud data
                for cam_name in pcd_data_dict.keys():
                    decompressed_pcd = decompress_pointcloud(pcd_data_dict[cam_name])
                    # Save point cloud video
                    save_pointcloud_video(decompressed_pcd, os.path.join(vis_data_dir, dataset_name + '_' + cam_name + '_pcd_video.mp4'))
    else:
        episode_idx = args['episode_idx']
        ismirror = args['ismirror']
        if ismirror:
            dataset_name = f'mirror_episode_{episode_idx}'
        else:
            dataset_name = f'episode_{episode_idx}'

        qpos, qvel, effort, action, image_dict, depth_image_dict, pcd_data_dict = load_hdf5(dataset_dir, dataset_name)
        print('hdf5 loaded!!')
        save_videos(image_dict, DT, video_path=os.path.join(vis_data_dir, dataset_name + '_video.mp4'))
        if depth_image_dict:  # 只在有深度图像数据时保存
            save_videos(depth_image_dict, DT, video_path=os.path.join(vis_data_dir, dataset_name + '_depth_video.mp4'))
        visualize_joints(qpos, action, plot_path=os.path.join(vis_data_dir, dataset_name + '_qpos.png'))
        
        if pcd_data_dict:
            for cam_name in pcd_data_dict.keys():
                decompressed_pcd = decompress_pointcloud(pcd_data_dict[cam_name])
                # Save point cloud video
                save_pointcloud_video(decompressed_pcd, os.path.join(vis_data_dir, dataset_name + '_' + cam_name + '_pcd_video.mp4'))
        # visualize_single(effort, 'effort', plot_path=os.path.join(dataset_dir, dataset_name + '_effort.png'))
        # visualize_single(action - qpos, 'tracking_error', plot_path=os.path.join(dataset_dir, dataset_name + '_error.png'))
        # visualize_base(base_action, plot_path=os.path.join(dataset_dir, dataset_name + '_base_action.png'))
        # visualize_timestamp(t_list, dataset_path) # TODO addn timestamp back

def decompress_pointcloud(pcd_data):
    decompressed_pcd = {}
    xyz = pcd_data['xyz']
    rgb = pcd_data['rgb']
    mask = pcd_data['padding_mask']
    decompressed_xyz = [xyz[i][mask[i]] for i in range(len(xyz))]
    decompressed_rgb = [rgb[i][mask[i]] for i in range(len(rgb))]
    decompressed_pcd = {'xyz': decompressed_xyz, 'rgb': decompressed_rgb}
    return decompressed_pcd

def save_pointcloud_video(pcd_data, video_path, fps=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))  # Adjust resolution as needed
    # TODO: 下采样
    for i in range(10): # len(pcd_data['xyz'])
        xyz = pcd_data['xyz'][i]
        rgb = pcd_data['rgb'][i]
        img = np.zeros((480, 640, 3), dtype=np.uint8)  # Create a blank image

        # Convert point cloud to image
        for j in range(len(xyz)):
            x, y, z = xyz[j]
            r, g, b = rgb[j]
            # Map 3D points to 2D image plane (simple projection)
            u, v = int(x * 100 + 320), int(y * 100 + 240)  # Adjust scaling and offset
            if 0 <= u < 640 and 0 <= v < 480:
                img[v, u] = (b, g, r)  # OpenCV uses BGR format

        out.write(img)

    out.release()
    print(f'Saved point cloud video to: {video_path}')



def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


def visualize_joints(qpos_list, command_list, plot_path=None, ylim=None, label_overwrite=None):
    if label_overwrite:
        label1, label2 = label_overwrite
    else:
        label1, label2 = 'State', 'Command'

    qpos = np.array(qpos_list) # ts, dim
    command = np.array(command_list)
    num_ts, num_dim = qpos.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(qpos[:, dim_idx], label=label1)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # plot arm command
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(command[:, dim_idx], label=label2)
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved qpos plot to: {plot_path}')
    plt.close()

def visualize_single(efforts_list, label, plot_path=None, ylim=None, label_overwrite=None):
    efforts = np.array(efforts_list) # ts, dim
    num_ts, num_dim = efforts.shape
    h, w = 2, num_dim
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(w, h * num_figs))

    # plot joint state
    all_names = [name + '_left' for name in STATE_NAMES] + [name + '_right' for name in STATE_NAMES]
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(efforts[:, dim_idx], label=label)
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    if ylim:
        for dim_idx in range(num_dim):
            ax = axs[dim_idx]
            ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()

def visualize_base(readings, plot_path=None):
    readings = np.array(readings) # ts, dim
    num_ts, num_dim = readings.shape
    num_figs = num_dim
    fig, axs = plt.subplots(num_figs, 1, figsize=(8, 2 * num_dim))

    # plot joint state
    all_names = BASE_STATE_NAMES
    for dim_idx in range(num_dim):
        ax = axs[dim_idx]
        ax.plot(readings[:, dim_idx], label='raw')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(20)/20, mode='same'), label='smoothed_20')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(10)/10, mode='same'), label='smoothed_10')
        ax.plot(np.convolve(readings[:, dim_idx], np.ones(5)/5, mode='same'), label='smoothed_5')
        ax.set_title(f'Joint {dim_idx}: {all_names[dim_idx]}')
        ax.legend()

    # if ylim:
    #     for dim_idx in range(num_dim):
    #         ax = axs[dim_idx]
    #         ax.set_ylim(ylim)

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved effort plot to: {plot_path}')
    plt.close()


def visualize_timestamp(t_list, dataset_path):
    plot_path = dataset_path.replace('.pkl', '_timestamp.png')
    h, w = 4, 10
    fig, axs = plt.subplots(2, 1, figsize=(w, h*2))
    # process t_list
    t_float = []
    for secs, nsecs in t_list:
        t_float.append(secs + nsecs * 10E-10)
    t_float = np.array(t_float)

    ax = axs[0]
    ax.plot(np.arange(len(t_float)), t_float)
    ax.set_title(f'Camera frame timestamps')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    ax = axs[1]
    ax.plot(np.arange(len(t_float)-1), t_float[:-1] - t_float[1:])
    ax.set_title(f'dt')
    ax.set_xlabel('timestep')
    ax.set_ylabel('time (sec)')

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'Saved timestamp plot to: {plot_path}')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', action='store', type=str, help='Dataset dir.', required=True)
    parser.add_argument('--episode_idx', action='store', type=int, help='Episode index.', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='Task name.', required=False)
    parser.add_argument('--all', action='store_true', help='Visualize all datasets instead of one episode')
    parser.add_argument('--ismirror', action='store_true')
    main(vars(parser.parse_args()))
