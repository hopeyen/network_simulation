import simulation_1
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

num_trial = 1

time = 50

givenP = 0.5

largeFrequency = 1.0

largePayments = 1

littleP = 0.5

littleF = 1.5


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


def getIntercepts(list, ps):
    points = []

    for i in range(1, len(list)):
        if list[i] == 0:
            points.append((ps[i], 0))
        elif ((list[i-1] > 0) and (list[i] < 0)): 
            # or ((list1[i-1] < list2[i-1]) and (list1[i] > list2[i]))):
            points.append(((ps[i]+ps[i-1])/2, 0))

    return points

def transform(points):
    xs = []
    ys = []

    for i in points:
        xs.append(i[0])
        ys.append(i[1])

    return (xs, ys)

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


        
def networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # a star / fork network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayments, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayments, Bob, Charlie)
    paymentAC = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAC1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentAC.setTransfer(paymentAC1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelBC = simulation_1.Channel(Bob, Charlie, network)

    channelAB.addPaymentList([paymentAB, paymentAC])
    channelBC.addPaymentList([paymentBC, paymentAC1])

    
    network.addPaymentList([paymentAB, paymentBC, paymentAC])
    network.addTransferredList([paymentAC1])

    network.runNetwork()
    network.printSummary()

    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()

    return (a+c, b)

def networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # set up the network
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)

    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayments, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayments, Bob, Charlie)
    paymentAC = simulation_1.Payment(freq, p, Alice, Charlie)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelAB.addPayment(paymentAB)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBC.addPayment(paymentBC)

    # Alice creates a direct channel for network 1
    channelAC = simulation_1.Channel(Alice, Charlie, network)
    channelAC.addPayment(paymentAC)
    
    # print("network")
    network.addNodeList([Alice, Bob, Charlie])
    network.addChannelList([channelAB, channelBC, channelAC])
    network.addPaymentList([paymentAB, paymentBC, paymentAC])
    network.runNetwork()
    


    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()

    return (a+c, b)

def networktransferB(p, freq, onlineTX, onlineTXTime, r, timeRun):
    # network 2 
    network = simulation_1.Network(onlineTX, onlineTXTime, r, timeRun)
    
    Alice = simulation_1.Node("Alice", network)
    Bob = simulation_1.Node("Bob", network)
    Charlie = simulation_1.Node("Charlie", network)

    paymentAB = simulation_1.Payment(largeFrequency, largePayments, Alice, Bob)
    paymentBC = simulation_1.Payment(largeFrequency, largePayments, Bob, Charlie)
    paymentAC = simulation_1.Payment(freq, p, Alice, Bob)
    paymentAC1 = simulation_1.Payment(freq, p, Bob, Charlie)
    paymentAC.setTransfer(paymentAC1)

    channelAB = simulation_1.Channel(Alice, Bob, network)
    channelAB.addPayment(paymentAB)
    channelBC = simulation_1.Channel(Bob, Charlie, network)
    channelBC.addPayment(paymentBC)

    # payment goes through Channel AB and BC
    channelAB.addPayment(paymentAC)
    channelBC.addPayment(paymentAC1)


    network.addNodeList([Alice, Bob, Charlie])
    network.addChannelList([channelAB, channelBC])
    network.addPaymentList([paymentAB, paymentBC, paymentAC])
    # print("network2")
    network.runNetwork()
    # print([paymentAC2, paymentAC2.numPaid])


    a = Alice.getChCostTotal()
    b = Bob.getChCostTotal()
    c = Charlie.getChCostTotal()

    return (a+c, b)

############# Main functions #####################

def run(time):
    
        # trial
        
    temp = []
    temp = networkOG(littleP, littleF, onlineTX=5, onlineTXTime=3, r=0.01, timeRun=time)
    alice0.append(temp[0])
    bob0.append(temp[1])
    
    temp = networkDirectAC(littleP, littleF, onlineTX=5, onlineTXTime=3, r=0.01, timeRun=time)
    alice1.append(temp[0])
    bob1.append(temp[1])

    temp = networktransferB(littleP, littleF, onlineTX=5, onlineTXTime=3, r=0.01, timeRun=time)
    alice2.append(temp[0])
    bob2.append(temp[1])

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

    print("maxfee " + str(maxFee) + "; minfee " + str(minFee))
    print("param pay %f ; freq %f" % (littleP, littleF))




################ Call #####################

run(time)
# runWithFreq()

# getFees(0.3, 0.1, time)

