# University of Sydney
# Shuzheng Huang
# Creating time 18/12/2021 2:08 pm

import numpy as np

##
def cal_n_index (group_center = [], group_size = [], n_side=64, width=64):
    ### group_center is a list with a coordinate
    n_location = group_center[0]
    # calculate the coordinate of each neuron
    #A = np.arange(-31.5, 32.5, 1)
    #B = np.arange(31.5, -32.5, -1)

    #data_coordinate = []; L1 = len(A); L2 = len(B)
    #for i in range(L2):
        #b = B[i]
        #for ii in range(L1):
            #a = A[ii]
            #data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    # calculate the night possible distance. BECAUSE this neural network has periodical boundary.
    Len = len(data_coordinate)
    data_dist = []
    # calculate nine coordinates
    for i in range(Len):
        co1 = data_coordinate[i]
        co2 = [co1[0], co1[1] - width]
        co3 = [co1[0] - width, co1[1]]
        co4 = [co1[0] - width, co1[1] - width]
        co5 = [co1[0], co1[1] + width]
        co6 = [co1[0] + width, co1[1]]
        co7 = [co1[0] + width, co1[1] + width]
        co8 = [co1[0] - width, co1[1] + width]
        co9 = [co1[0] + width, co1[1] - width]

        # calculate four distances
        dist1 = np.sqrt((co1[0] - n_location[0])**2 + (co1[1] - n_location[1])**2)
        dist2 = np.sqrt((co2[0] - n_location[0])**2 + (co2[1] - n_location[1])**2)
        dist3 = np.sqrt((co3[0] - n_location[0])**2 + (co3[1] - n_location[1])**2)
        dist4 = np.sqrt((co4[0] - n_location[0])**2 + (co4[1] - n_location[1])**2)
        dist5 = np.sqrt((co5[0] - n_location[0])**2 + (co5[1] - n_location[1])**2)
        dist6 = np.sqrt((co6[0] - n_location[0])**2 + (co6[1] - n_location[1])**2)
        dist7 = np.sqrt((co7[0] - n_location[0])**2 + (co7[1] - n_location[1])**2)
        dist8 = np.sqrt((co8[0] - n_location[0])**2 + (co8[1] - n_location[1])**2)
        dist9 = np.sqrt((co9[0] - n_location[0])**2 + (co9[1] - n_location[1])**2)

        dist_min = np.min([dist1, dist2, dist3, dist4, dist5, dist6, dist7,dist8, dist9])
        data_dist.append(dist_min)

    # select the positions within the scope wants to select
    data = []
    for i in range(Len):
        if data_dist[i] <= group_size[0]:
            data.append(i)

    return data

#%%
def cal_inhNeuron_index (group_center = [], group_size = [], n_side=32, width=64):
    ### group_center is a list with a coordinate
    n_location = group_center[0]
    # calculate the coordinate of each neuron
    #A = np.arange(-31, 32, 2)
    #B = np.arange(31, -32, -2)

    #data_coordinate = []
    #L1 = len(A)
    #L2 = len(B)
    #for i in range(L2):
        #b = B[i]
        #for ii in range(L1):
            #a = A[ii]
            #data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    Len = len(data_coordinate)
    data_dist = []

    for i in range(Len):
        co1 = data_coordinate[i]
        co2 = [co1[0], co1[1] - width]
        co3 = [co1[0] - width, co1[1]]
        co4 = [co1[0] - width, co1[1] - width]
        co5 = [co1[0], co1[1] + width]
        co6 = [co1[0] + width, co1[1]]
        co7 = [co1[0] + width, co1[1] + width]
        co8 = [co1[0] - width, co1[1] + width]
        co9 = [co1[0] + width, co1[1] - width]

        # calculate four distances
        dist1 = np.sqrt((co1[0] - n_location[0])**2 + (co1[1] - n_location[1])**2)
        dist2 = np.sqrt((co2[0] - n_location[0])**2 + (co2[1] - n_location[1])**2)
        dist3 = np.sqrt((co3[0] - n_location[0])**2 + (co3[1] - n_location[1])**2)
        dist4 = np.sqrt((co4[0] - n_location[0])**2 + (co4[1] - n_location[1])**2)
        dist5 = np.sqrt((co5[0] - n_location[0])**2 + (co5[1] - n_location[1])**2)
        dist6 = np.sqrt((co6[0] - n_location[0])**2 + (co6[1] - n_location[1])**2)
        dist7 = np.sqrt((co7[0] - n_location[0])**2 + (co7[1] - n_location[1])**2)
        dist8 = np.sqrt((co8[0] - n_location[0])**2 + (co8[1] - n_location[1])**2)
        dist9 = np.sqrt((co9[0] - n_location[0])**2 + (co9[1] - n_location[1])**2)

        dist_min = np.min([dist1, dist2, dist3, dist4, dist5, dist6, dist7,dist8, dist9])
        data_dist.append(dist_min)

    data = []
    for i in range(Len):
        if data_dist[i] <= group_size[0]:
            data.append(i)

    return data

#%%
def cal_n_index_line(xRange, yRange, n_side=64, width=64):

    if xRange[0] > xRange[1]:
        xRange = [xRange[1], xRange[0]]

    if yRange[0] > yRange[1]:
        yRange = [yRange[1], yRange[0]]

    # calculate the coordinate of each neuron
    #A = np.arange(-31.5, 32.5, 1)
    #B = np.arange(31.5, -32.5, -1)

    #data_coordinate = []
    #L1 = len(A)
    #L2 = len(B)
    #for i in range(L2):
        #b = B[i]
        #for ii in range(L1):
            #a = A[ii]
            #data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    data = []
    for i in range(len(data_coordinate)):
        if ((xRange[0] <= data_coordinate[i][0]) & (data_coordinate[i][0] <= xRange[1])) & \
                ((yRange[0] <= data_coordinate[i][1]) & (data_coordinate[i][1] <= yRange[1])):
            data.append(i)

    return data


#%%
def cal_inhNeuron_index_line(xRange, yRange, n_side=32, width=64):

    if xRange[0] > xRange[1]:
        xRange = [xRange[1], xRange[0]]

    if yRange[0] > yRange[1]:
        yRange = [yRange[1], yRange[0]]

    # calculate the coordinate of each neuron
    #A = np.arange(-31, 32, 2)
    #B = np.arange(31, -32, -2)

    #data_coordinate = []
    #L1 = len(A)
    #L2 = len(B)
    #for i in range(L2):
        #b = B[i]
        #for ii in range(L1):
            #a = A[ii]
            #data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    data = []
    for i in range(len(data_coordinate)):
        if ((xRange[0] <= data_coordinate[i][0]) & (data_coordinate[i][0] <= xRange[1])) & \
                ((yRange[0] <= data_coordinate[i][1]) & (data_coordinate[i][1] <= yRange[1])):
            data.append(i)

    return data

#%%
def calExNeuronCoordinate(nIndex, n_side=64, width=64):
    # calculate the coordinate of each neuron
    #A = np.arange(-31.5, 32.5, 1)
    #B = np.arange(31.5, -32.5, -1)

    #data_coordinate = []
    #L1 = len(A)
    #L2 = len(B)
    #for i in range(L2):
        #b = B[i]
        #for ii in range(L1):
            #a = A[ii]
            #data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    return data_coordinate[nIndex]

#%%
def calInhNeuronCoordinate(nIndex, n_side=32, width=64):
    # calculate the coordinate of each neuron
    #A = np.arange(-31, 32, 2)
    #B = np.arange(31, -32, -2)

    #data_coordinate = []
    #L1 = len(A)
    #L2 = len(B)
    #for i in range(L2):
        #b = B[i]
        #for ii in range(L1):
            #a = A[ii]
            #data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    return data_coordinate[nIndex]

##
def cal_n_index_multi(group_center = [], group_size = [], n_side=32, width=64):
    # setting coordinates
    #A = np.arange(31.5, -32.5, -1)
    #B = np.arange(31.5, -32.5, -1)
    #data_coordinate = []
    #L1 = len(A)
    #L2 = len(B)
    #for i in range(L1):
    #    b = B[i]
    #    for ii in range(L2):
    #        a = A[ii]
    #        data_coordinate.append([a, b])

    hw = width / 2
    step = width / n_side
    x = np.linspace(-hw + step / 2, hw - step / 2, n_side)
    y = np.linspace(hw - step / 2, -hw + step / 2, n_side)

    xx, yy = np.meshgrid(x, y)

    data_coordinate = np.zeros([n_side * n_side, 2])
    data_coordinate[:, 0] = xx.flatten()
    data_coordinate[:, 1] = yy.flatten()

    # calculate distance and select neurons meeting the settings. Because the group centre is not located at the
    # standard coordinates, use 'round' to take approximate distance
    Len = len(group_center)
    Len2 = len(data_coordinate)
    data_sum = []
    for i in range(Len):
        centre = group_center[i]  # coornidate of group centre
        data = []
        for n in range(Len2):
            co_N = data_coordinate[n]  # coordinate of neuron
            # other possible coordinate of this neuron because of the periodical boundary
            co_N2 = [co_N[0] - width, co_N[1]]
            co_N3 = [co_N[0], co_N[1] - width]
            co_N4 = [co_N[0] - width, co_N[1] - width]
            co_N5 = [co_N[0] + width, co_N[1]]
            co_N6 = [co_N[0], co_N[1] + width]
            co_N7 = [co_N[0] + width, co_N[1] + width]
            co_N8 = [co_N[0] - width, co_N[1] + width]
            co_N9 = [co_N[0] + width, co_N[1] - width]
            # four possible distances
            dist = np.sqrt((co_N[0] - centre[0]) ** 2 + (co_N[1] - centre[1]) ** 2)
            dist2 = np.sqrt((co_N2[0] - centre[0]) ** 2 + (co_N2[1] - centre[1]) ** 2)
            dist3 = np.sqrt((co_N3[0] - centre[0]) ** 2 + (co_N3[1] - centre[1]) ** 2)
            dist4 = np.sqrt((co_N4[0] - centre[0]) ** 2 + (co_N4[1] - centre[1]) ** 2)
            dist5 = np.sqrt((co_N5[0] - centre[0]) ** 2 + (co_N5[1] - centre[1]) ** 2)
            dist6 = np.sqrt((co_N6[0] - centre[0]) ** 2 + (co_N6[1] - centre[1]) ** 2)
            dist7 = np.sqrt((co_N7[0] - centre[0]) ** 2 + (co_N7[1] - centre[1]) ** 2)
            dist8 = np.sqrt((co_N8[0] - centre[0]) ** 2 + (co_N8[1] - centre[1]) ** 2)
            dist9 = np.sqrt((co_N9[0] - centre[0]) ** 2 + (co_N9[1] - centre[1]) ** 2)
            dist = min(dist, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9)
            if round(dist) <= group_size[0]:
                data.append(n)
            else:
                pass
        data_sum.append(data)

    return data_sum

##
#a = cal_inhNeuron_index_line([-0.5, 1.5], [-0.5, 1.5])
#a = calExNeuronCoordinate(2015)
#a = calInhNeuronCoordinate(495)


