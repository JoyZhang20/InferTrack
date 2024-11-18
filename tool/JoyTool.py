from math import log2
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os
import numpy as np
from datetime import datetime
import time
from collections import OrderedDict
import cv2
import random


def form(num):
    return round(num, 3)


def copy_list(src, dst):
    for i in range(len(src)):
        dst.append(src[i])
    return dst


def write_list_to_txt(record, filename=""):
    filename += "_data_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    with open("output/" + filename, "w") as f:
        for i in range(len(record)):
            f.write(str(record[i]) + '\n')

def write_multi_list_to_txt(record_list, table_head, filename="data"):
    filename += "_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    with open("output/" + filename, "w") as f:
        for i in range(len(table_head)):
            f.write(table_head[i])
            f.write('\t')
        f.write('\n')
        for i in range(len(record_list[0])):
            for j in range(len(record_list)):
                f.write(str(form(record_list[j][i])))
                f.write('\t')
            f.write('\n')


def read_txt_to_list(txt_path):
    with open(txt_path, 'r') as file:
        res_list = []
        for line in file:
            res_list.append(float(line.strip()))
        return res_list


def read_excel_to_list(dataset_path, sheet_name, head_name, data_len, is_int=True):
    traffic_data = pd.read_excel(dataset_path, header=0, sheet_name=sheet_name)
    res = []
    for i in range(data_len):
        if is_int:
            tmp = int(1.0 * traffic_data.loc[i][head_name])  # 控制流量倍数p7
            if tmp < 1:
                tmp = 1
            res.append(tmp)
        else:
            res.append(float(traffic_data.loc[i][head_name]))
    return res


def transfer_plot_to_gray(file_path, file_name_list):
    for i in range(len(file_name_list)):
        image = cv2.imread(file_path + file_name_list[i] + ".png", cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(file_path + "gray_" + file_name_list[i] + ".png", image)


class JoyPlot:
    def __init__(self, ):
        self.plot_list = []
        self.label_list = []
        self.line_width = 1.5
        self.line_style = '-'
        self.color_list = ["#0072BD", "#D95319", "#77AC30", "#EDB120", "#7E2F8E", "#4DBEEE"]

    def add_plot(self, record, label="label"):
        self.plot_list.append(record)
        self.label_list.append(label)

    def show(self):
        plt.figure()
        x_ran = [i for i in range(len(self.plot_list[0]))]
        for i in range(len(self.plot_list)):
            plt.plot(x_ran, self.plot_list[i], color=self.color_list[i], label=self.label_list[i],
                     linewidth=self.line_width, linestyle=self.line_style)  # 红虚线为损失值
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.legend()
        plt.show()


class JoyTimer:
    def __init__(self, round_num=2):
        self.round_num = round_num
        self.used_time = []
        self.record_time_list = []
        self.start_time = 0
        self.start()

    def start(self):
        self.start_time = time.time()

    def record(self):
        self.record_time_list.append(time.time())

    def show(self):
        self.record_time_list.append(time.time())
        res = []
        for i in range(len(self.record_time_list)):
            if i == 0:
                res.append(round(self.record_time_list[i] - self.start_time, self.round_num))
            else:
                res.append(round(self.record_time_list[i] - self.record_time_list[i - 1], self.round_num))
        # print("time: ", end='')
        # for i in range(len(res)):
        #     print("{}s".format(res[i]), end='\t')
        # print("")
        del res[-1]
        return res


def main_generate_task():
    data_size = [200, 500]
    cycles = [800, 1200]
    priority = [1, 3]
    distance = [0, 1500]
    para = [data_size, cycles, priority, distance]
    task_num = 5000
    res = []
    for j in range(len(para)):
        tmp = []
        for i in range(task_num):
            tmp.append(random.randint(para[j][0], para[j][1]))
        res.append(tmp)
    write_multi_list_to_txt(res)


def Poisson_distribution():
    # 泊松分布的参数 λ
    lam = 1
    # 时间间隔（单位：秒）
    time_interval = 1
    # 模拟 10 个时间间隔的任务请求
    num_intervals = 10
    # 生成符合泊松分布的随机数
    requests = np.random.poisson(lam, num_intervals)
    # 处理生成的任务请求
    for interval in range(num_intervals):
        num_requests = requests[interval]
        if num_requests > 0:
            print(f"At interval {interval}, {num_requests} requests occurred.")
        else:
            print(f"At interval {interval}, no request occurred.")


def joy_print_eval_result(head_name_list, res_list):
    for i in range(len(head_name_list)):
        print(head_name_list[i], end='\t')
        if i == len(head_name_list) - 1:
            print("")
    for i in range(len(res_list)):
        for j in range(len(res_list[i])):
            print(res_list[i][j], end='\t')
            if j == len(res_list[i]) - 1:
                print("")


if __name__ == '__main__':
    # Poisson_distribution()

    list = [[1, 2, 3], [6, 7, 8]]
    write_multi_list_to_txt(list,['A','B'])

    # main_generate_task()

    # dataset_path = r'data/dataset.xlsx'
    # sheet_name = "uav_traffic"
    # head_name = "sag"
    # res = read_excel_to_list(dataset_path, sheet_name, head_name, 144)
    # print(res)

    # timer = JoyTimer()
    # for i in range(10000):
    #     print("hello")
    # timer.record()
    # for i in range(10000):
    #     print("second")
    # timer.output()

    # timer=[1,2,3]
    # joy_printer(timer)

    # print("hello")
    # head = ["SAGOff", "DDQN", "DQN"]
    # res = [[1622.097, 1931.097, 2282.097], [50, 30, 20]]
    # joy_print_eval_result(head, res)
