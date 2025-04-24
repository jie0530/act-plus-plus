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
   [2025.3.12更新]目前发现是使用[3]个相机和[1~2]个相机录制数据集时，压缩图像的维度发生了变化。


7. 输入的维度改为6后，The size of tensor a (8) must match the size of tensor b (22) at non-singleton dimension 0
   qpos和action维度修改的不对

8. DataLoader worker (pid(s) 1620196) exited unexpectedly
   RuntimeError: DataLoader worker (pid 1620196) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

9. QObject::moveToThread: Current thread (0x55fe321ad760) is not the object's
   [原因]：opencv-python和pyqt5冲突是因为在opencv-python4.2.0以上的版本，将opencv将低版本到4.2.0以下
   [解决]
   ```
   #卸载原版本
   pip uninstall opencv-python
   #安装指定版本
   pip install opencv-python==4.1.2.30
   ```

10. Cannot mix incompatible Qt library (version 0x50c08) with this library (version 0x50c05)
   [原因]CoppeliaSim启动涉及到了QT的部分
   [解决]将LD_LIBRARY_PATH和QT_QPA_PLATFORM_PLUGIN_PATH注释掉，关掉所有的terminal或者source一下，重新运行即可解决原来的问题

11. RuntimeError: Trying to resize storage that is not resizable
   [原因]点云数据维度不一致
   [解决]数据加载时进行处理，调整到一致的维度


12. 某个模块找不到，可以尝试导入路径：[sys.path.append('/home/wsco/jie_ws/src/d3roma/')]
    