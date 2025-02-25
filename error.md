1. ImportError: cannot import name 'cached_download' from 'huggingface_hub'
`将huggingface-hub卸载然后重新安装huggingface-hub-0.24.6`

2. AttributeError: module 'wandb' has no attribute 'init'
`pip install wandb`

3. RuntimeError: torch.cat(): expected a non-empty list of Tensors
   `修改constants.py的第5行左右DATA_DIR变量的值, 设置为自己下载的路径即可，如下：`
    `DATA_DIR = '/home/lin/aloha/act-plus-plus/data'`

4. Error loading /home/wsco/jie_ws/src/act-plus-plus/data/sim_transfer_cube_scripted/episode_24.hdf5 in __getitem__
   `剔除文件，运行指令重新录制数据集`
   `python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/sim_transfer_cube_scripted --num_episodes 10`


5. 验证环境是否正常，可以把数据集录制和train代码跑一遍，步骤[4]


6. 记一处错误：`ros_record_data.py`文件的220行`padded_compressed_image[:image_len] = compressed_image`，由于compressed_image图像shape发生改变，导致复制出现错误，之前`compressed_image.shape = (18868,1)`是个二维数组, 现在是`(18868，)`，是个一维数组，


7. 输入的维度改为6后，The size of tensor a (8) must match the size of tensor b (22) at non-singleton dimension 0
   qpos和action维度修改的不对

8. DataLoader worker (pid(s) 1620196) exited unexpectedly
   RuntimeError: DataLoader worker (pid 1620196) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.