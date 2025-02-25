import ros_record_data
import rospy
import sys
import time
import select
from std_msgs.msg import Int32

esipode_collection_flag = True

def callback(msg):
    global esipode_collection_flag
    print("收到停止信号[0], msg.data = %d", msg.data)
    esipode_collection_flag = msg.data

def main():
    global esipode_collection_flag
    rospy.init_node('ros_action_capture_record_node', anonymous=True)
    t1 = rospy.Time.now()
    print("开始采集")
    recorder,max_t = ros_record_data.get_recorder("aloha_mobile_pick_fruit2")
    # rospy.Subscriber("/stop_collection", Int32, callback)

    # 循环，每次重新开始收集
    continue_collection = True
    while continue_collection and not rospy.is_shutdown():
        # rospy.spin()
        print("重新开始收集")
        recorder.restart_collecting(True)
        sleep_t = 1 + max_t - (rospy.Time.now() - t1).to_sec()
        print("sleep_t",sleep_t,max_t)
        # if sleep_t >0:
        # while esipode_collection_flag and len(recorder.record_data_dict['/observations/qpos']) < recorder.max_timesteps:
        while len(recorder.record_data_dict['/observations/qpos']) < recorder.max_timesteps:
            rospy.sleep(0.1)
            # print("sleep")
            # # 增加停止单次收集的键盘控制
            # if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            #     line = input()
            #     if line.strip().lower() == 'stop':
            #         print("收集停止")
            #         break
        recorder.save_succ()
        recorder.stop() 
        sys.stdout.flush()
        rospy.sleep(1)
        user_input = input("继续采集？ (y/N) ").strip().lower()
        if not ( user_input == "y" or user_input == "yes"):
            continue_collection = False 
        rospy.sleep(rospy.Duration(0.1))
        # esipode_collection_flag = True # 重置标志位

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
