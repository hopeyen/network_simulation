from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import matplotlib.transforms as mtransforms

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
        elif ((list1[i-1] > list2[i-1]) and (list1[i] < list2[i])): 
            # or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
            points.append((ps[i], (list1[i-1]+list1[i])/2))

    return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)


############## Constants and global variables ################

alice0, bob0, alice1, bob1, alice2, bob2 = [], [], [], [], [], []

num_trial = 200

time = 50

givenP = 0.5

paymentMean = 0.65

paymentSigma = 0.1


############# Main functions #####################

def runWithPayment(time):
    ps = np.random.lognormal(paymentMean, paymentSigma)

    fs = [x* 1.0 / 100  for x in range(1,200)]
    # random.expovariate(givenP)
    # ps = np.arange(0.0, 1.0 + 0.01, 0.01)

    for i in range(len(fs)):
        # trial
        print(i)
        res = [0, 0, 0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            temp = main(p = ps, freq=fs[i], timeRun = time)
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

    # if Bob is taking the lowest fee he can, it would be the difference between
    # if he transfer for Alice (bob2) and if he does not (bob0=bob1) 
    minFee = calculateFee(bob2, bob0)

    # Alice's cost increase while Bob's cost decrease by the fee charged
    aliceAfter = payFee(alice0, minFee)
    bobAfter = chargeFee(bob2, minFee)
    

    alicePoints = transform(getIntersections(alice1, aliceAfter, fs))
    bobPoints = transform(getIntersections(bob0, bobAfter, fs))

    bobxs = bobPoints[0]
    bobys = bobPoints[1]
    alicexs = alicePoints[0]
    aliceys = alicePoints[1]

    aliceBenefit = chargeFee(alice1, aliceAfter)
    bobBenefit = chargeFee(bob0, bobAfter)


    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(fs, bob0, "k")
    ax.plot(fs, alice1, "r")
    # ax.plot(ps, bob2, "m")
    # ax.plot(ps, bob1, "k--")
    # ax.plot(ps, alice2, "c--")
    ax.plot(fs, aliceAfter, "b-")
    ax.plot(fs, bobAfter, "g-")
    # ax.plot(bobxs, bobys, "go")
    
    ax.plot(fs, aliceBenefit)
    ax.plot(fs, bobBenefit)
    ax.plot(fs, [0 for x in range(len(fs))], "k--")
    ax.plot(alicexs, aliceys, "bo")

    trans_offset = mtransforms.offset_copy(ax.transData, fig=fig,
                                       x=0.01, y=-0.30, units='inches')

    for x, y in zip(bobxs, bobys):
        plt.plot(x, y, 'go')
        # plt.text(x, y, '%02f, %02f' % (x, y), transform=trans_offset)

    fig.text(0, 0, 'Alice Points %s; Bob Points %s\n Number of trials: %d; Time of each network: %d' % (str(alicePoints), str(bobPoints), num_trial, time))


    ax.set_title('Transferred payment size vs Cost differences after Min Fee')
    ax.set_xlabel('frequency (lambda)')
    ax.set_ylabel('Cost differences')

    # plt.axis([0, 6, 0, 100])
    fig.legend(["bob", "alice", "alice' ", "bob'", "alice benefit", "bob benefit"])
    fig.savefig('frequencyBenefit_Min.png')


################ Call #####################

runWithPayment(time)
# runWithFreq()




