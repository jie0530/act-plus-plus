#!/bin/bash

# 启动机器人TF发布节点
python3 robot_base_to_tool_tf_pub2.py &

# 等待1秒确保第一个节点已经启动
sleep 1

# 启动点云融合节点
python run_pcd_fusion_publisher.py --spatial_cutoff -1.2 -0.5 -0.5 0.5 0.0 0.1 --downsample_N 4096 --publish_freq 20 --use_fps --fps_h 5 &


# 等待用户按Ctrl+C
echo "所有节点已启动。按 Ctrl+C 停止..."
wait