import random
import time

'''
此文件包含了一些关于随机数的工具方法：
1、随机选人
'''


def rand_selection(names=list(), subjects=list(), showTime=True):
    
    names_indices = list(range(len(names)))
    random.shuffle(names_indices)
    for i, index in enumerate(names_indices):
        print(f"{subjects[i]}：{names[index]}")
    if showTime:
        # 获得当前时间时间戳
        now = int(time.time())
        # 转换为其他日期格式，如："%Y-%m-%d %H:%M:%S"
        timeArr = time.localtime(now)
        other_StyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArr)
        print(other_StyleTime)
