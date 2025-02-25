import os
import subprocess
import sys

# 串口设备名
SERIAL_PORT = "/dev/ttyUSB0"
# 串口占用的进程组
GROUP_NAME = "dialout"

def check_serial_port_usage():
    """检查串口设备是否被占用，并返回占用串口的进程ID"""
    try:
        # 使用 lsof 命令查找占用该串口的进程
        result = subprocess.check_output(["sudo", "lsof", SERIAL_PORT], stderr=subprocess.STDOUT)
        result = result.decode("utf-8")

        # 提取 PID
        lines = result.strip().split("\n")
        pids = []
        for line in lines[1:]:
            parts = line.split()
            pids.append(parts[1])  # 第二列是PID
        return pids
    except subprocess.CalledProcessError as e:
        # 如果没有进程占用串口，lsof 会返回非零状态，捕获异常
        print(f"No process found using {SERIAL_PORT}")
        return []

def kill_process(pid):
    """杀死占用串口的进程"""
    try:
        print(f"Killing process with PID: {pid}")
        subprocess.run(["sudo", "kill", "-9", pid], check=True)
        print(f"Successfully killed process with PID {pid}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to kill process with PID {pid}: {e}")

def check_and_fix_serial_port_permissions():
    """检查串口权限，并提示用户添加到 dialout 组"""
    # 检查当前用户是否在 dialout 组中
    try:
        result = subprocess.check_output("groups", shell=True).decode("utf-8")
        if GROUP_NAME not in result:
            print(f"Your user is not in the '{GROUP_NAME}' group. Adding you to the group...")
            subprocess.run(f"sudo usermod -aG {GROUP_NAME} $USER", shell=True, check=True)
            print(f"You have been added to the '{GROUP_NAME}' group. Please log out and log back in.")
        else:
            print(f"User is already in the '{GROUP_NAME}' group.")
    except subprocess.CalledProcessError as e:
        print(f"Error checking or modifying groups: {e}")
        sys.exit(1)

def main():
    # 检查串口是否被占用
    pids = check_serial_port_usage()
    
    # 如果有进程占用串口，尝试杀死它
    if pids:
        for pid in pids:
            kill_process(pid)

    # 检查串口权限，确保用户有权限访问
    # check_and_fix_serial_port_permissions()

    print("Serial port management completed.")

if __name__ == "__main__":
    main()
