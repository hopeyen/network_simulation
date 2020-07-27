from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib import cm

############ Helper functions ################

def getHigherMean(list1, list2):
    avg1 = float(sum(map(sum,list1)))/(len(list1)*len(list1[0]))
    avg2 = float(sum(map(sum,list2)))/(len(list2)*len(list2[0]))

    if avg1 > avg2:
        return list1
    else:
        return list2

def getLowerMean(list1, list2):
    avg1 = float(sum(map(sum,list1)))/(len(list1)*len(list1[0]))
    avg2 = float(sum(map(sum,list2)))/(len(list2)*len(list2[0]))

    if avg1 <= avg2:
        return list1
    else:
        return list2

def calculateFee(list1, list2):
    lower = getLowerMean(list1, list2)
    higher = getHigherMean(list1, list2)

    ans = []
    for i in range(len(lower)):
        temp = []
        for j in range(len(lower[i])):
            temp.append(higher[i][j] -lower[i][j])
        ans.append(temp)
    return ans


def chargeFee(bob, fee):
    ans = []
    for i in range(len(fee)):
        temp = []
        for j in range(len(fee[i])):
            temp.append(bob[i][j] - fee[i][j])
        ans.append(temp)
    return ans

def payFee(alice, fee):
    ans = []
    for i in range(len(fee)):
        temp = []
        for j in range(len(fee[i])):
            temp.append(alice[i][j] + fee[i][j])
        ans.append(temp)
    return ans

# def getIntersections(list1, list2, ps):
#     points = []

#     for i in range(1, len(list1)):
#         if list1[i] == list2[i]:
#             points.append((i, list2[i]))
#         elif ((list1[i-1] > list2[i-1]) and (list1[i] < list2[i])): 
#             # or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
#             points.append((ps[i], (list1[i-1]+list1[i])/2))

#     return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)


############## Constants and global variables ################

num_trial = 10

time = 2

givenP = 0.5

paymentMean = 0.1

paymentSigma = 0.1

psInit = 20

fsInit = 25

psLen = 100

fsLen = 100

psIncre = 10

fsIncre = 10
############# Main functions #####################

def runWithPayment(time):
    # 0.2 to 1.2
    ps = [x* 1.0 /100 for x in range(20, 120)]
    # 0.25 to 1.25
    fs = [x* 1.0 /100 for x in range(25, 125)]
    # random.expovariate(givenP)
    # ps = np.arange(0.0, 1.0 + 0.01, 0.01)

    X = np.array(ps)
    Y = np.array(fs)
    X, Y = np.meshgrid(X, Y)

    costs = [[[0 for x in range(psInit, psInit+psLen)] for x in range(fsInit, fsInit+fsLen)] for x in range(0, 6)]

    for i in range(len(fs)):
        print(str(i))

        for j in range(len(ps)):
            
            for k in range(num_trial):

                res = main(p=ps[j], freq=fs[i], timeRun = time)
                for h in range(len(res)):
                    costs[h][i][j] += res[h]

            for h in range(len(costs)):
                costs[h][i][j] /= float(num_trial)

    #         # alice0         costs[0] 
    #         # bob0           costs[1] 
    #         # alice1         costs[2] 
    #         # bob1           costs[3] 
    #         # alice2         costs[4] 
    #         # bob2           costs[5] 
    maxFee = calculateFee(costs[2], costs[4])
    minFee = calculateFee(costs[5], costs[1])

    # Alice's cost increase while Bob's cost decrease by the fee charged
    aliceAfter_Max = payFee(costs[0], maxFee)
    bobAfter_Max = chargeFee(costs[5], maxFee)
    aliceAfter_Min = payFee(costs[0], minFee)
    bobAfter_Min = chargeFee(costs[5], minFee)
    
    # set up the plot
    Z = np.array(bobAfter_Max)
    U = np.array(aliceAfter_Max)
    V = np.array(bobAfter_Min)
    W = np.array(aliceAfter_Min)

    fig = plt.figure(figsize=plt.figaspect(0.5))

    titles = ['Bob after charging the maximum fee', 
            'Alice after charging the maximum fee', 
            'Bob after charging the minimum fee',
            'Alice after charging the minimum fee']
    Zs = [Z, U, V, W]

    for i in range(0, 4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        ax.set_xlabel('Payment size')
        ax.set_ylabel('Frequency')
        ax.set_zlabel("benefit")
        ax.set_title(titles[i])
        surf = ax.plot_surface(X, Y, Zs[i], rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=10)

        # rotate the axes and update
        for angle in range(0, 4):
            for angle2 in range(0, 3):
                ax.view_init(angle2*30, angle*90)
                plt.draw()
                fig.savefig('3D-3-%d%d%d.png' % (i, angle, angle2))

    # fig.savefig('3D-3.png')


################ Call #####################

runWithPayment(time)
# runWithFreq()




