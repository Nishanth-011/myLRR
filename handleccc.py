## this code is used to handle the CCC dataset 
# first, I'd like to test using the vector of CCC
import numpy as np
import scipy.io as sio
filename = 'G:/mycode/myLRR/Datasets/testOrdered.vec'
#filename = 'G:/mycode/myLRR/Datasets/test.txt'
filepath = 'G:/mycode/myLRR/Datasets/CCC'
fea = []
gtlabels = []
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines: 
            break
        gtlabels.append(lines[-2])
        lines = lines[0:-3]#+lines[-1]
        p = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        fea.append(p)  # 添加新读取的数据
        pass
    #fea = np.array(fea) # 将数据从list类型转换为array类型。
    pass
filename = 'G:/mycode/myLRR/Datasets/trainingOrdered.vec'
with open(filename, 'r') as file_to_read:
    while True:
        lines = file_to_read.readline() # 整行读取数据
        if not lines: 
            break
        gtlabels.append(lines[-2])
        lines = lines[0:-3]#+lines[-1]
        p = [float(i) for i in lines.split()] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
        fea.append(p)  # 添加新读取的数据
        pass
    fea = np.array(fea) # 将数据从list类型转换为array类型。
    pass
sio.savemat(filepath+'.mat', mdict={'fea':fea, 'gtlabels':gtlabels})