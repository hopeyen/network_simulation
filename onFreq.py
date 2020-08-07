from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms



############## Constants and global variables ################

alice0, bob0, alice1, bob1, alice2, bob2 = [], [], [], [], [], []

num_trial = 200

time = 50

paymentMean = 0.5

paymentSigma = 0.0001



############ Helper functions ################

def getHigherMean(list1, list2):
    avg1 = float(sum(list1))/len(list1)
    avg2 = float(sum(list2))/len(list2)

    if avg1 > avg2:
        return list1
    else:
        return list2

def getLowerMean(list1, list2):
    avg1 = float(sum(list1))/len(list1)
    avg2 = float(sum(list2))/len(list2)

    if avg1 <= avg2:
        return list1
    else:
        return list2

def calculateFee(list1, list2):
    lower = getLowerMean(list1, list2)
    higher = getHigherMean(list1, list2)

    ans = []
    for i in range(len(lower)):
        ans.append(higher[i] -lower[i])
    return ans

def chargeFee(bob, fee):
    ans = []
    for i in range(len(fee)):
        ans.append(bob[i] - fee[i])
    return ans

def payFee(alice, fee):
    ans = []
    for i in range(len(fee)):
        ans.append(alice[i] + fee[i])
    return ans

def getIntersections(list1, list2, ps):
    points = []

    for i in range(1, len(list1)):
        if list1[i] == list2[i]:
            points.append((i, list2[i]))
        elif (((list1[i-1] > list2[i-1]) and (list1[i] < list2[i])) 
            or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
            points.append((ps[i], (list1[i-1]+list1[i])/2))

    return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)


def getIntercepts(list, ps):
    points = []

    for i in range(1, len(list)):
        if list[i] == 0:
            points.append((ps[i], 0))
        elif (((list[i-1] > 0) and (list[i] < 0))
            or (((list[i-1] < 0) and (list[i] > 0)))):
            points.append(((ps[i]+ps[i-1])/2, 0))

    return points



def getMaxFee(a0, a1):
    return a1 - a0

def getMinFee(a0, a2, b0, b2):
    b22 = b2 + a2 - a0
    return b22 - b0


def getFees(ps, f, time):
    # (a0+c0, b0, a1+c1, b1, a2+c2, b2)
    #    a0   b0   a1    b1   a2   b2
    temp = main(p=ps, freq=f, timeRun = time)
    return (getMaxFee(temp[0], temp[2]), getMinFee(temp[0], temp[4], temp[1], temp[5]))


        

############# Main functions #####################

def runWithFreq(time):
    fs = [x* 1.0 /100 for x in range(150, 152)]
    print(fs)

    for i in range(len(fs)):
        # trial
        print(str(i))
        res = [0, 0, 0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            pm = np.random.lognormal(paymentMean, paymentSigma)
            temp = main(p=pm, freq=fs[i], timeRun = time)
            for j in range(len(temp)):
                res[j] += temp[j]
        for j in range(len(res)):
            res[j] = res[j]/float(num_trial)

        alice0.append(res[0])
        bob0.append(res[1])
        alice1.append(res[2])
        bob1.append(res[3])
        alice2.append(res[4])
        bob2.append(res[5])


    # if Bob is taking the highest fee he can, then we look at how his costs changes
    # the highest fee Bob can take is Alice's maximum difference of channel costs
    maxFee = chargeFee(alice1, alice0)
    
    # aliceAfter_max = payFee(alice0, maxFee)

    # bob2'=bob2+(alice2-alice0)
    # minFee = bob2'-bob0 = bob2+(alice2-alice0)-bob0 
    bob22 = payFee(bob2, chargeFee(alice2, alice0))

    minFee = chargeFee(bob22, bob0)
    print("alice OG")
    print(alice0)
    print("alice1")
    print(alice1)
    print("alice2")
    print(alice2)
    print("bob OG")
    print(bob0)
    print("bob1")
    print(bob1)
    print("bob2")
    print(bob2)
    print("\n")

    print("max fee")
    print(maxFee)
    print("\n")

    print("alice escapes")
    print(chargeFee(alice2, alice0))
    print("Bob22")
    print(bob22)
    print("min fee")
    print(minFee)


    print("max min diff" + str(chargeFee(maxFee, minFee)))
    print("param P " + str(paymentMean))
    



    inter = getIntersections(maxFee, minFee, fs)


    titles = ['Maximum fee and minimum fee vs frequency']
    # Zs = [(aliceBenefit_max, bobBenefit_max), (aliceBenefit_min, bobBenefit_min)]
    xlabels = ['frequency (lambda)']

    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i in range(0, 1):
        ax = fig.add_subplot(1, 1, i+1)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel('fee')

        ax.plot(fs, maxFee, "k--")
        ax.plot(fs, minFee, "r-.")

        for pt in range(len(inter)):
            label = '({:.3f}, {:.3f})'.format(inter[pt][0], inter[pt][1])
            ax.annotate(label, (inter[pt][0], inter[pt][1]),
                textcoords="offset points",
                xytext = (2,2),
                rotation=45)


        fig.text(0, 0, 'Trials: %d; Time: %d; Payment mean: %0.2f' % (num_trial, time, paymentMean))
        ax.set_title(titles[i])
        fig.legend(["maximum fee", "minimum fee"])


    fig.savefig('3nodemaxminFreq.png')


    print("checking")
    for pt in range(len(inter)):
        print(inter[pt])
        print("gives")

        print(getFees(paymentMean, inter[pt][0], time))


    # print("mannual")
    # res = [[],[]]
    # for i in range(100):
    #     tmp = getFees(0.3, 0.1, time)
    #     res[0].append(tmp[0])
    #     res[1].append(tmp[1])
    # print(sum(res[0]), sum(res[1]))

################ Call #####################

runWithFreq(time)
# runWithFreq()
