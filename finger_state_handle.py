#!/usr/bin/env python
import os
import logging

import sys
sys.path.append('/home/wsco/jie_ws/src/stark-serialport-example/python/modbus_example/')
from modbus_client_utils import *
from queue import Queue

class ModbusClient:
    def __init__(self, serial_port_name, slave_id=1, baudrate=BaudRate.baudrate_115200):
        self.serial_port_name = serial_port_name
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.client = None
        self.device = None
        filename = os.path.basename(__file__).split(".")[0]
        # print(f"{filename}.log")
        StarkSDK.init(isModbus=True, log_level=logging.INFO, log_file_name=f"{filename}.log")

    def connect(self):
        """连接 Modbus 设备并初始化"""
        # shutdown_event = setup_shutdown_event()
        StarkSDK.set_error_callback(lambda error: SKLog.error(f"Error: {error.message}"))

        ports = serial_ports()
        SKLog.info(f"serial_ports: {ports}")
        if len(ports) == 0:
            SKLog.error("No serial ports found")
            return False

        self.client = client_connect(port=self.serial_port_name, baudrate=self.baudrate)
        if self.client is None:
            SKLog.error("Failed to open modbus serial port")
            return False

        # 创建 Modbus 设备实例
        self.device = StarkDevice.create_device(self.slave_id, f"{self.serial_port_name}_{self.slave_id}")

        # 设置 Modbus 设备的读写寄存器回调
        self.device.set_write_registers_callback(
            lambda register_address, values: client_write_registers(
                self.client, register_address, values=values, slave=self.slave_id
            )
        )
        self.device.set_read_holding_registers_callback(
            lambda register_address, count: client_read_holding_registers(
                self.client, register_address, count, slave=self.slave_id
            )
        )
        self.device.set_read_input_registers_callback(
            lambda register_address, count: client_read_input_registers(
                self.client, register_address, count, slave=self.slave_id
            )
        )
        
        return True

    def get_finger_status(self):
        result_queue = Queue()  # 创建一个队列用于存储结果

        def save_finger_position(positions):
            result_queue.put(positions)  # 将结果放入队列

        self.device.get_finger_positions(
                lambda positions: save_finger_position(positions))

        # 等待队列中出现结果
        status = result_queue.get()
        # print(f"Finger status: {status}")
        return status
    def set_finger_status(self, action_list):   
        self.device.set_finger_positions([action_list[0], action_list[1], action_list[2], action_list[3], action_list[4], action_list[5]])


if __name__ == "__main__":
    # 灵巧手的设备ID  1是左手，2是右手
    client = ModbusClient("/dev/ttyUSB0", slave_id=1)
    client.connect()
    while True:
        finger_status = client.get_finger_status()
        print(f"Finger status: {finger_status}")
        # client.set_finger_status([60,60,100,100,100,100])
        client.set_finger_status([0, 0, 0, 0, 0, 0])
        