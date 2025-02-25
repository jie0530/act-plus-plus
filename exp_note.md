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
   