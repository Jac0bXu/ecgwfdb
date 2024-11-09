import wfdb
import helper as h
# print(wfdb.__version__)

#显示心跳类型
# wfdb.show_ann_labels()

import wfdb
import matplotlib.pyplot as plt

# record = wfdb.rdrecord("C:/Users/jxxzh/PycharmProjects/ecgwfdb/ecgdata/100",  # 文件所在路径
#                        sampfrom=0,  # 读取100这个记录的起点，从第0个点开始读
#                        sampto=1000,  # 读取记录的终点，到1000个点结束
#                        physical=False,  # 若为True则读取原始信号p_signal，如果为False则读取数字信号d_signal，默认为False
#                        channels=[1])  # 读取那个通道，也可以用channel_names指定某个通道;如channel_names=['MLII']
#
# # 转为数字信号
# signal = record.d_signal[0:1000]
# denoised_signal = h.denoise(signal)
#
# # 绘制波形
# # plt.plot(signal,color='blue')
# plt.plot(denoised_signal,color='red')
# plt.title("ECG Signal")
#
# # # 读取annatations
# # signal_ann = wfdb.rdann("C:/Users/jxxzh/PycharmProjects/ecgwfdb/ecgdata/100", "atr", sampfrom=0, sampto=1000)
# #
# # # 将读取到的annatations的心拍绘制到心电图上
# # for index in signal_ann.sample:
# #     plt.scatter(index, signal[index][0], marker="*")
# #
# # # 并打印出改心拍标注的类型
# # print(signal_ann.symbol)
# plt.show()

h.loadData()