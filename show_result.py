# from tensorflow.python.ops.rnn_cell import BasicLSTMCell
# import tensorflow as tf
import numpy as np
import math
import re
import matplotlib.pyplot as plt
incorrect_path = 'F:\\HRNN.txt'
file_incorrect = open(incorrect_path, 'r')
line = file_incorrect.readline()
accuracy = list()
epoch = list()
loss = list()

read_list=['Epoch - \s?\d+', 'MSE:  s -   \d+.\d+','f -   \d+.\d+','v -   \d+.\d+', 'q -   \d+.\d+']
number_regex_list=['\d+','\d+.\d+','\d+.\d+','\d+.\d+','\d+.\d+']
show_list=[]
index_str='Epoch'

start_index = 3000
row_index=0
data=list()
while line:
    col_index=0
    data_row = list()
    for read_str in read_list:
        regex_str = read_str
        regex = re.findall(regex_str, line)
        if len(regex)>0:
            data_str = re.findall(number_regex_list[col_index],regex[0])
            data_row.append(float(data_str[0]))

        col_index += 1
    data.append(data_row)
    line=file_incorrect.readline()
    row_index += 1

data=np.array(data)

# Error Rate

min = np.min(data, 0)


pass
