# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:02:31 2020

@author: admin
"""

import sys
import time

sys.path.insert(0, r'F:\xx')  # 可以确保先搜索这个路径,用户需要按需替换
# 否则Python解释器可能找不到 jkrc 模块
import jkrc
robot = jkrc.RC("20.20.10.189")  # 返回一个机器人对象,此IP需要替换成自己使用的机器人IP
robot.login()  # 登录
time.sleep(1)
print(robot.get_sdk_version()[1])
src = robot.get_joint_position()[1]
print("source: {}".format(src))
sample_path = [src]
flag = 0
timeout = False

print("请开始操作机械臂")
time1 = time.time()
# ret = robot.joint_move(joint_pos=[], move_mode=0, is_block=True, speed=0.05)
while robot.is_in_pos()[1] or not timeout:
    while not robot.is_in_pos()[1]:
        sample_path.append(robot.get_joint_position()[1])
        # time.sleep(0.008)
        flag = 1
    time2 = time.time()
    if flag == 0:
        if (time2-time1) > 10:
            print("time diff:{}".format(time2-time1))
            timeout = True
            break
    else:
        print("not move now")
        time1 = time2
        flag = 0
print("sample path length:", len(sample_path))

print("开始运动至初始点")
ret = robot.joint_move(joint_pos=src, move_mode=0, is_block=True, speed=0.05)
if ret[0] != 0:
    print("error code: {}".format(ret[0]))
print("已回到原点")

time.sleep(5)
print("开始复现sample路径")
ret = robot.servo_move_enable(True)
for item in sample_path:
    ret = robot.servo_j(joint_pos=item, move_mode=False)
    # if ret[0] != 0:
    #     print("error code: {}".format(ret[0]))
    time.sleep(0.1)
robot.servo_move_enable(False)

robot.logout()  # 登出
