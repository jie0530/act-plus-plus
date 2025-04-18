1. 手部信息：
   [动捕系统]roslaunch vrpn_client_ros sample.launch server:=192.168.1.113
   [灵巧手控制] cd ~/jie_ws/src/stark-serialport-example/python/modbus_example && python3 test_left_op_avatar.py

2. 相机：scam
3. 机械臂： cd ~/jie_ws/src/fairino-python-sdk-master/linux/examples && python3 robot_state_pub.py


~/jie_ws/src/stark-serialport-example/python/modbus_example$ python3 ctrl_hand.py


python run_pcd_fusion_publisher.py --spatial_cutoff -1.2 -0.5 -0.5 0.5 0.0 0.1 --downsample_N 4096 --publish_freq 20 --use_fps --fps_h 5

python3 robot_base_to_tool_tf_pub2.py