1. 特定参数下训练2000和5000steps，文件夹为trainings_2000和trainings_5000，训练集和验证集比例为0.99
   具体参数：
   ```
   python3 imitate_episodes.py \
    --task_name aloha_mobile_pick_fruit \
    --ckpt_dir trainings \
    --policy_class ACT \
    --kl_weight 1 --chunk_size 10 \
    --hidden_dim 512 --batch_size 1 \
    --dim_feedforward 3200  --lr 1e-5 \
    --seed 0 --num_steps 2000
   ```

2. 特定参数下训练2000和5000steps，文件夹为trainings_20_2000和trainings_20_5000，且将训练集和验证集比例改为0.92, 单场景训练，数据集50
   python3 imitate_episodes_single_arm_hand.py \
   --task_name aloha_mobile_pick_fruit \
   --ckpt_dir trainings_20_2000 \
   --policy_class ACT \
   --kl_weight 10 --chunk_size 20 \
   --hidden_dim 512 --batch_size 8 \
   --dim_feedforward 3200  --lr 1e-5 \
   --seed 0 --num_steps 2000



   图像太抖，手指影响---[简化问题]移除手端视角图像，移除手指维度
   -  在utils.py中将修改读取的action的维度
   diffusion策略尝试---[策略调整]改为diffusion策略
   给人感觉是预测的太短了---[代码调试]



3. 移除手指维度尝试，特定参数下训练2000和5000steps，文件夹为trainings_6_20_2000和trainings_6_20_5000，训练集和验证集比例为0.92

4. crop与smooth操作都要对原数据集进行处理，否则会报错
   
[简化环境进行测试]
5. 用op控制手指抓取，50个数据集，训练5000steps，action会向下执行，但是对物体感知不敏感；
   分析原因：可视化出机械臂与手指数据的均值和方差，发现手指数据方差很大。且有数据有抖动；
   - 对数据进行平滑处理，平滑处理后，再进行训练[平滑后训练改变不大，附上平滑效果图]
   - 将手指简化为夹爪[夹爪改为键盘控制，重新训练80个数据]

6. 相机视角移动与数据集有差异时,推理效果很差,很难复现之前动作
7. [仅改变]参数chunk_size增加效果显著,单机械臂时动作效果很好
8. 将灵巧手简化为一维开合夹爪，推理效果很好