"""
University of Sydney
Shuzheng Huang
Creating time 4/10/2022 8:01 pm

========================================
【脚本主要流程图与说明】
========================================
流程概览：

1. 参数设置（刺激参数、网络参数、重复次数等）
   ↓
2. 计算刺激点分布（沿对角线分布，中心对称）
   ↓
3. 获取每个刺激点对应的神经元索引
   ↓
4. 主循环：遍历所有刺激强度和重复次数
   ├─ 计算自发放电率（未刺激状态）
   └─ 计算刺激下的群体放电率（刺激状态）
   ↓
5. 统计并整理每组数据（自发/刺激响应）
   ↓
6. 计算 ΔF/F（响应变化百分比）
   ↓
7. 绘制调谐曲线（tuning curve），展示刺激强度与空间分布的响应关系

【一句话总结】
本脚本用于模拟神经元网络在不同空间位置和刺激强度下的群体响应，自动统计自发放电与刺激响应，并可视化调谐曲线。
========================================
"""
# University of Sydney
# Shuzheng Huang
# Creating time 4/10/2022 8:01 pm

##
stiSiz = [6]           # 刺激区域的大小（如圆形半径，单位为神经元格数）
stiPosi = [0, 0]       # 刺激中心位置（x, y坐标，网络中心为[0, 0]）
stiRateScope = [700, 700]  # 刺激强度范围（最小值, 最大值，单位Hz），用于生成不同强度的刺激
rateStep = [300]       # 刺激强度步长（每次递增的Hz数），决定刺激强度的分辨率

pres_T = [1500]        # 刺激持续时间（单位ms），每次刺激的长度
trans_T = [3000]       # 过渡期/预热期时长（单位ms），用于网络稳定
inter_T = [1000]       # 刺激间隔时间（单位ms），两次刺激之间的休息时间

diag_slope = [1]       # 刺激点分布的对角线斜率，决定刺激点排列方向
num_point = [17]       # 刺激点数量（必须为奇数，保证中心对称）
group_range = [2]      # 每个刺激点对应的神经元群体半径（单位格数）

repeatT = [1]          # 每种刺激强度的重复次数，用于统计平均响应
stimPresTimes = 1      # 每次仿真中刺激呈现的次数

stimulus = ['claSharp']        # 刺激类型（如泊松、方波、正弦等），底部区域
stimulus_top = ['claSharp']    # 顶部区域的刺激类型
# detect_area = ['bottom', 'top']  # 检测区域（未启用）
area_connection = False        # 是否连接上下区域（True则上下区域有突触连接）
netWorkSize = 64               # 网络边长（神经元阵列的宽度/高度，单位格数）

##
import brian2.numpy_ as np
from brian2.only import *
import matplotlib.pyplot as plt
import random
# import copy
import cal_neuron_index as getI   # 计算神经元索引
import calTwoExpand as calSpike   # 计算神经元群体响应

##
maxRate_min = stiRateScope[0]  # 刺激最小强度
maxRate_max = stiRateScope[1]  # 刺激最大强度
numStep = int((maxRate_max - maxRate_min) / rateStep[0]) + 1  # 刺激步数
stiRate = linspace(maxRate_min, maxRate_max, numStep)         # 刺激强度列表
stiRateLen = len(stiRate)                                     # 刺激强度数量

# 计算对角线斜率K和截距b，使刺激点在对角线上
# Y = K * X + b
if len(diag_slope):
    K = diag_slope[0]
else:
    # 随机生成斜率
    K = round(random.uniform(-10, 10), 2)
# 计算截距b
b = stiPosi[1] - K * stiPosi[0]
b = stiPosi[1] - K * stiPosi[0]

##

# 计算刺激对角线与网络边界的交点
# 保证刺激点在中心，分布更美观
num_remain = num_point[0] - 1
# 左边界交点
yL = K * (-31.5) + b
if -31.5 <= yL <= 31.5:
    coord_L = [-31.5, yL]
else:
    if K > 0:
        xL = (-31.5 - b) / K
        coord_L = [xL, -31.5]
    elif K < 0:
        xL = (31.5 - b) / K
        coord_L = [xL, 31.5]
# 右边界交点
yR = K * 31.5 + b
if -31.5 <= yR <= 31.5:
    coord_R = [31.5, yR]
else:
    if K > 0:
        xR = (31.5 - b) / K
        coord_R = [xR, 31.5]
    elif K <0:
        xR = (-31.5 - b) / K
        coord_R = [xR, -31.5]


# 沿对角线分布刺激点
if abs(coord_L[0] - stiPosi[0]) <= abs(coord_R[0] - stiPosi[0]):
    L = abs(coord_L[0] - stiPosi[0])
else:
    L = abs(coord_R[0] - stiPosi[0])
p_step = round(L / (num_remain / 2), 2)  # 点间距
coord_point = []
for i in range(int((num_remain / 2))):
    x = stiPosi[0] - ((num_remain / 2) - i) * p_step
    y = K * x + b
    x = round(x, 2); y = round(y, 2)
    coord_point.append([x, y])
coord_point.append(stiPosi)  # 中心点
for i in range(int((num_remain) / 2)):
    x = stiPosi[0] + (i + 1) * p_step
    y = K * x + b
    x = round(x, 2); y = round(y, 2)
    coord_point.append([x, y])

##

# 计算每个刺激点对应的神经元索引
n_in_G = getI.cal_n_index_multi(group_center=coord_point, group_size=group_range, n_side=64, width=64)

##

# 初始化数据存储变量
stiRateSum = []; gRespSum = []; spontRpSum = []; a = 0
stiRateSumTop = []; gRespSumTop = []; spontRpSumTop = []


# 主循环：遍历所有刺激强度和重复次数，计算自发放电和刺激响应
for ii in range(stiRateLen):
    print('stimulum strength index:', ii+1, 'in total:', stiRateLen)
    stiRate1 = []; gResp1 = []; spontRp1 = []
    stiRate2 = []; gResp2 = []; spontRp2 = []

    for n in range(repeatT[0]):
        a += 1
        print('repeat time:', n+1, 'in total:', repeatT[0])
        S1 = 4088 # 随机种子
        # 计算自发放电
        print('=== calculate spontaneous firing rate ===')
        response = calSpike.calResponse(trans_T=trans_T, pres_T=pres_T, inter_T=inter_T, stimulus_type=['spontaneous'],
                                        stimulus_type_top=['spontaneous'], seed_num=[S1], areaConnection=area_connection,
                                        stimPresTimes=1, netWorkSize=netWorkSize)

        # 统计自发放电数据
        spontPopuR = []
        spontPopuRTop = []
        data = []
        dataTop = []
        tTotal = response['t_spon']
        for i in range(len(n_in_G)):
            spontPopuR.append(response['spk_e_1'].count[n_in_G[i]])
            spontPopuRTop.append(response['spk_e_2'].count[n_in_G[i]])
        for kk in range(len(spontPopuR)):
            data.append(np.mean(spontPopuR[kk]) / tTotal)
            dataTop.append(np.mean(spontPopuRTop[kk]) / tTotal)
        spontRp1.append(data)
        spontRp2.append(dataTop)

        # 计算刺激下的群体放电率
        print('=== calculate population firing rate ===')
        response = calSpike.calResponse(stiSize=stiSiz, stiPosition=stiPosi, stiMaxRate=stiRate,
                                        stiSize_top=stiSiz, stiPosition_top=stiPosi, stiMaxRate_top=stiRate,
                                        pres_T=pres_T, trans_T=trans_T, inter_T=inter_T, seed_num=[S1],
                                        stimulus_type=stimulus, stimulus_type_top=stimulus_top,
                                        stimPresTimes=stimPresTimes,
                                        areaConnection=area_connection, netWorkSize=netWorkSize,)

        # 统计刺激响应数据
        stim_on = response['stim_on']
        Len_stim_on = len(stim_on)
        spike_T = Len_stim_on * (pres_T[0] / 1000 - 0.3)
        data = []
        dataTop = []

        for kk in range(len(n_in_G)):
            spike_num = 0
            neuron_num = len(n_in_G[kk])
            for mm in range(len(n_in_G[kk])):
                timeInd = response['spk_e_1'].t[response['spk_e_1'].i == n_in_G[kk][mm]] / second
                for nn in range(Len_stim_on):
                    spike_num += len(timeInd[(stim_on[nn][0] + 300 <= timeInd * 1000) \
                                             & (timeInd * 1000 <= stim_on[nn][1])])
            data.append((spike_num / neuron_num) / spike_T)

        for kk in range(len(n_in_G)):
            spike_num = 0
            neuron_num = len(n_in_G[kk])
            for mm in range(len(n_in_G[kk])):
                timeInd = response['spk_e_2'].t[response['spk_e_2'].i == n_in_G[kk][mm]] / second
                for nn in range(Len_stim_on):
                    spike_num += len(timeInd[(stim_on[nn][0] + 300 <= timeInd * 1000) \
                                             & (timeInd * 1000 <= stim_on[nn][1])])
            dataTop.append((spike_num / neuron_num) / spike_T)

        perc = round(a / (stiRateLen * repeatT[0]), 2) * 100
        print('-----This loop ends----------{0} % completed -----'.format(perc))
        # 保存本次循环数据
        stiRate1.append(stiRate[ii]); gResp1.append(data)
        stiRate2.append(stiRate[ii]); gResp2.append(dataTop)
    # 保存每个刺激强度下的所有重复数据
    spontRpSum.append(spontRp1); gRespSum.append(gResp1); stiRateSum.append(stiRate1)
    spontRpSumTop.append(spontRp2); gRespSumTop.append(gResp2); stiRateSumTop.append(stiRate2)

## dispackaging data and calculate mean value for each group at each stimulus max rate
g_avg_resp = []
g_avg_resp_top = []
for i in range(stiRateLen):
    resp = gRespSum[i]; data = []
    for n in range(num_point[0]):
        data2 = []
        for k in range(repeatT[0]):
            data2.append(np.mean(gRespSum[i][k][n]))
        data.append(np.mean(data2))
    g_avg_resp.append(data)

for i in range(stiRateLen):
    resp = gRespSumTop[i]; data = []
    for n in range(num_point[0]):
        data2 = []
        for k in range(repeatT[0]):
            data2.append(np.mean(gRespSumTop[i][k][n]))
        data.append(np.mean(data2))
    g_avg_resp_top.append(data)

## dispackaging fire rate at spontaneous status
g_avg_spont = []
g_avg_spont_top = []

for i in range(stiRateLen):
    spont = spontRpSum[i]; data = []
    for n in range(num_point[0]):
        data2 = []
        for k in range(repeatT[0]):
            data2.append(np.mean(spontRpSum[i][k][n]))
        data.append(np.mean(data2))
    g_avg_spont.append(data)

for i in range(stiRateLen):
    spont = spontRpSumTop[i]; data = []
    for n in range(num_point[0]):
        data2 = []
        for k in range(repeatT[0]):
            data2.append(np.mean(spontRpSumTop[i][k][n]))
        data.append(np.mean(data2))
    g_avg_spont_top.append(data)

## calculate ΔF / F
df_f_sum = []
df_f_sum_top = []

for i in range(stiRateLen):
    df_f = []
    for n in range(num_point[0]):
        d = ((g_avg_resp[i][n] - g_avg_spont[i][n]) / g_avg_spont[i][n]) * 100
        df_f.append(d)
    df_f_sum.append(df_f)

for i in range(stiRateLen):
    df_f = []
    for n in range(num_point[0]):
        d = ((g_avg_resp_top[i][n] - g_avg_spont_top[i][n]) / g_avg_spont_top[i][n]) * 100
        df_f.append(d)
    df_f_sum_top.append(df_f)

## plot tuning curves
x = []

for i in range(num_point[0]):
    if i < (num_remain / 2):
        x.append(round(-1 * np.sqrt((coord_point[i][0] - stiPosi[0])**2 + (coord_point[i][1] - stiPosi[1])**2), 2))
    else:
        x.append(round(np.sqrt((coord_point[i][0] - stiPosi[0]) ** 2 + (coord_point[i][1] - stiPosi[1]) ** 2), 2))

fig = plt.figure(figsize=(13, 6))
plt.subplot(1, 2, 1)
for i in range(stiRateLen):
    plt.plot(x, df_f_sum[i], linewidth = 3.0, label='{0} Hz'.format(stiRate[i]))

stand_x = np.arange(-50, 50, 0.1)
stand_y = [0] * len(stand_x)
plt.plot(stand_x, stand_y, c='green', linestyle='--')
#stand_y = np.arange(np.min(df_f_sum) - 5, np.max(df_f_sum) + 5, 0.1)
#stand_x = [0] * len(stand_y)
#plt.plot(stand_x, stand_y, c='green', linestyle='--')
plt.xlim(-50, 50)
plt.xticks([-45, -30, -15, 0, 15, 30, 45], ['-45', '-30', '-15', '0', '15', '30', '45'])
plt.xlabel('distance to stimulus centre')
plt.ylabel('ΔF / F (percentage)')
plt.title('bottom area')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(stiRateLen):
    plt.plot(x, df_f_sum_top[i], linewidth = 3.0, label='{0} Hz'.format(stiRate[i]))

stand_x = np.arange(-50, 50, 0.1)
stand_y = [0] * len(stand_x)
plt.plot(stand_x, stand_y, c='green', linestyle='--')
#stand_y = np.arange(np.min(df_f_sum_top) - 5, np.max(df_f_sum_top) + 5, 0.1)
#stand_x = [0] * len(stand_y)
#plt.plot(stand_x, stand_y, c='green', linestyle='--')
plt.xlim(-50, 50)
plt.xticks([-45, -30, -15, 0, 15, 30, 45], ['-45', '-30', '-15', '0', '15', '30', '45'])
plt.xlabel('distance to stimulus centre')
plt.ylabel('ΔF / F (percentage)')
plt.title('top area')
plt.legend()

plt.suptitle('stimulus:{0}, Centre:{1}, stiSize:{2}, Repeat:{3}'.\
             format(stimulus[0], stiPosi, stiSiz[0], Len_stim_on * repeatT[0]))
plt.show()

