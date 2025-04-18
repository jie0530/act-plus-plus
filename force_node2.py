#!/usr/bin/env python3

import rospy
import serial
from pymodbus.client import ModbusSerialClient  # pymodbus 3.x 客户端
from std_msgs.msg import String
from time import sleep
from geometry_msgs.msg import WrenchStamped

# 串口配置
SERIAL_PORT = "/dev/ttyUSB2"  # 你的 USB 转串口设备路径
BAUD_RATE = 9600              # 设置与传感器的串口通信波特率，通常为 9600
TIMEOUT = 10                   # 串口通信超时（单位：秒）

# Modbus 配置（RTU 协议）
MODBUS_SLAVE_ID = 1           # 传感器的 Modbus 从设备地址（可以在传感器手册中找到）
REGISTER_ADDRESS = 40001-40001          # Modbus 寄存器地址（从 40001 开始）
REGISTER_COUNT = 1            # 读取寄存器数量

# ROS 话题发布者
pub = rospy.Publisher('/force_data_1', WrenchStamped, queue_size=10)

def read_sensor_data():
    """
    通过 Modbus RTU 从传感器读取数据
    """
    # 创建 Modbus RTU 客户端
    client = ModbusSerialClient(method='RTU', port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=TIMEOUT,
                                stopbits=1, bytesize=8, parity='N')

    # 连接到传感器
    if not client.connect():
        rospy.logerr("无法连接到 Modbus 传感器")
        return None

    # 读取 Modbus 寄存器数据
    result = client.read_holding_registers(REGISTER_ADDRESS, REGISTER_COUNT, slave=MODBUS_SLAVE_ID)

    # 关闭 Modbus 连接
    client.close()

    if result.isError():
        rospy.logerr("Modbus 错误类型: {}".format(result))
        return None

    # 获取寄存器中的原始值
    raw_value = result.registers[0]

    # 由于数据是单极性（0～65535），直接返回原始值
    # 这里假设寄存器的值在有效范围内
    if raw_value >= 0 and raw_value <= 65535:
        sensor_data = raw_value
        return sensor_data
    else:
        rospy.logerr("传感器数据无效，超出范围")
        return None

def create_wrench_message(sensor_data):
    """
    根据读取到的传感器数据，创建并返回 WrenchStamped 消息
    """
    # 假设我们从传感器得到的原始数据代表力的 x, y, z 分量
    wrench_msg = WrenchStamped()
    wrench_msg.header.stamp = rospy.Time.now()
    wrench_msg.header.frame_id = "base_link"  # 根据需求设置frame_id

    # 将传感器数据赋值给力的三个分量
    wrench_msg.wrench.force.x = sensor_data  # 假设原始数据是单一的，可以根据实际需要修改
    #wrench_msg.wrench.force.y = 0.0
    #wrench_msg.wrench.force.z = 0.0

    # 力矩部分默认为0
    #wrench_msg.wrench.torque.x = 0.0
    #wrench_msg.wrench.torque.y = 0.0
    #wrench_msg.wrench.torque.z = 0.0

    return wrench_msg

def sensor_callback(event):
    """
    回调函数，读取传感器数据并通过 ROS 发布
    """
    # 从传感器读取数据
    sensor_data = read_sensor_data()

    if sensor_data is not None:
        # 创建 WrenchStamped 消息
        wrench_msg = create_wrench_message(sensor_data)

        # 发布消息到话题
        pub.publish(wrench_msg)

        # 在 ROS 日志中打印数据
        rospy.loginfo("已发布力传感器数据: {}".format(sensor_data))

def main():
    """
    ROS 节点入口
    """
    # 初始化 ROS 节点
    rospy.init_node('modbus_serial_sender2', anonymous=True)

    # 设置定时器回调函数，每隔 1 秒读取一次力传感器数据并发布
    rospy.Timer(rospy.Duration(1), sensor_callback)

    # 保持节点运行
    rospy.spin()

if __name__ == '__main__':
    main()

