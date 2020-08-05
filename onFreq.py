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


############## Constants and global variables ################

alice0, bob0, alice1, bob1, alice2, bob2 = [], [], [], [], [], []

num_trial = 500

time = 50

givenP = 0.5

paymentMean = 1

paymentSigma = 0.1


############# Main functions #####################

def runWithFreq(time):
    ps = np.random.lognormal(paymentMean, paymentSigma)

    fs = [x* 1.0 / 100 for x in range(1,50)]
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

    # # if Bob is taking the highest fee he can, then we look at how his costs changes
    # # the highest fee Bob can take is Alice's maximum difference of channel costs
    # maxFee = calculateFee(alice1, alice0)
    # aliceAfter_max = payFee(alice0, maxFee)
    # bobAfter_max = payFee(chargeFee(bob2, maxFee), (payFee(alice2, alice0)))

    # # if Bob is taking the lowest fee he can, it would be the difference between
    # # if he transfer for Alice (bob2) and if he does not (bob0=bob1) 
    # minFee = calculateFee(bob2, bob0)
    # aliceAfter_min = payFee(alice0, minFee)
    # bobAfter_min = payFee(chargeFee(bob2, minFee), (payFee(alice2, alice0)))
    

    # alicePoints = transform(getIntersections(alice1, aliceAfter_max, fs))
    # bobPoints = transform(getIntersections(bob0, bobAfter_max, fs))
    
    # bobxs = bobPoints[0]
    # bobys = bobPoints[1]
    # alicexs = alicePoints[0]
    # aliceys = alicePoints[1]

    # # get the benefit by analyzing the costs in different networks
    # aliceBenefit_max = payFee(aliceAfter_max, alice0)
    # bobBenefit_max = payFee(bobAfter_max, bob0)
    
    # aliceBenefit_min = payFee(aliceAfter_min, alice0)
    # bobBenefit_min = payFee(bobAfter_min, bob0)


    # BobInter = [getIntercepts(bobBenefit_max, fs)]
    # print(getIntercepts(aliceBenefit_min, fs))
    # AliceInter = [getIntercepts(aliceBenefit_max, fs), getIntercepts(aliceBenefit_min, fs)]




    # titles = ['benefit after max fee vs payment size', 
    #         'benefit after min fee vs payment size']
    # Zs = [(aliceBenefit_max, bobBenefit_max), (aliceBenefit_min, bobBenefit_min)]
    # xlabels = ['Payment size', 'frequency (lambda)']

    # fig = plt.figure(figsize=plt.figaspect(0.5))

    # for i in range(0, 2):
    #     ax = fig.add_subplot(1, 2, i+1)
    #     ax.set_xlabel(xlabels[i])
    #     ax.set_ylabel('benefit ')
        
    #     ax.plot(fs, Zs[i][0])
    #     ax.plot(fs, Zs[i][1])
    #     ax.plot(fs, [0 for x in range(len(fs))])

    #     for pt in range(len(BobInter[0])):
    #         label = '{:.4f}'.format(BobInter[0][pt][0])
    #         ax.annotate(label, (BobInter[0][pt][0], 0),
    #             textcoords="offset points",
    #             xytext = (1,1),
    #             rotation=45)
    #     for pt in range(len(AliceInter[i])):
    #         label = '{:.4f}'.format(AliceInter[i][pt][0])
    #         # AliceInter[i][pt][0]
    #         ax.annotate(label, (0, 0),
    #             textcoords="offset points",
    #             xytext = (1,1),
    #             rotation=45)

    #     fig.text(0, 0, 'Trials: %d; Time: %d; payment mean: %0.2f' % (num_trial, time, paymentMean))
    #     ax.set_title(titles[i])
    #     fig.legend(["alice ", "bob"])

    # if Bob is taking the highest fee he can, then we look at how his costs changes
    # the highest fee Bob can take is Alice's maximum difference of channel costs
    maxFee = calculateFee(alice0, alice1)
    print(maxFee)
    aliceAfter_max = payFee(alice0, maxFee)

    # bob is also responsible for Alice's share of change in channels
    # bob2'=bob2+(alice2-alice0)
    # minFee = bob2'-bob0 = bob2+(alice2-alice0)-bob0 
    bob22 = payFee(bob2, calculateFee(alice0, alice2))

    minFee = calculateFee(bob22, bob0)
    print(minFee)

    inter = getIntersections(maxFee, minFee, fs)


    titles = ['Maximum fee and minimum fee vs frequency']
    # Zs = [(aliceBenefit_max, bobBenefit_max), (aliceBenefit_min, bobBenefit_min)]
    xlabels = ['frequency (lambda)']

    fig = plt.figure(figsize=plt.figaspect(0.5))

    for i in range(0, 1):
        ax = fig.add_subplot(1, 1, i+1)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel('fee')

        # ax.plot(ps, bob0, "k--")
        # # ax.plot(ps, alice0, "b--")
        # ax.plot(ps, bob1)
        # # ax.plot(ps, alice1)
        # ax.plot(ps, bob2)
        # ax.plot(ps, payFee(bob2, calculateFee(alice0, alice2)))
        ax.plot(fs, maxFee, "k--")
        ax.plot(fs, minFee, "r-.")
        # ax.plot(ps, alice2)
        
        # ax.plot(ps, Zs[i][0])
        # ax.plot(ps, Zs[i][1])
        # ax.plot(ps, [0 for x in range(len(ps))])
        for pt in range(len(inter)):
            label = '{:.3f}'.format(inter[pt][0])
            ax.annotate(label, (inter[pt][0], inter[pt][1]),
                textcoords="offset points",
                xytext = (2,2),
                rotation=45)

        # for pt in range(len(BobInter[0])):
        #     label = '{:.3f}'.format(BobInter[0][pt][0])
        #     ax.annotate(label, (BobInter[0][pt][0], 0),
        #         textcoords="offset points",
        #         xytext = (2,2),
        #         rotation=45)
        # for pt in range(len(AliceInter[i])):
        #     label = '{:.3f}'.format(AliceInter[i][pt][0])
        #     # AliceInter[i][pt][0]
        #     ax.annotate(label, (0, 0),
        #         textcoords="offset points",
        #         xytext = (2,2),
        #         rotation=45)

        fig.text(0, 0, 'Trials: %d; Time: %d; Payment mean: %0.2f' % (num_trial, time, paymentMean))
        ax.set_title(titles[i])
        fig.legend(["maximum fee", "minimum fee"])



    


    


    fig.savefig('3nodemaxminFreq.png')


################ Call #####################

runWithFreq(time)
# runWithFreq()




