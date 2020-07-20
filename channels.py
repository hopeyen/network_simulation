from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec


upperbound, lowerbound, intersection = [], [], []
estimatedubd, estimatedlbd = [], []

num_trial = 10
time = 20

times = [x* 10.0 for x in range(1, time)]


# transfer payment values

def runWithPayment(time):
    ps = [x* 1.0 /100 for x in range(1, 100)]
    # ps = np.arange(0.0, 1.0 + 0.01, 0.01)

    for i in range(len(ps)):
        # trial
        res = [0, 0, 0, 0]
        temp = []
        for k in range(num_trial):
            temp = main(p=ps[i], timeRun = time)
            for j in range(len(temp)):
                res[j] += temp[j]
        for j in range(len(res)):
            res[j] = res[j]/float(num_trial)

        upperbound.append(res[0])
        lowerbound.append(res[1])
        estimatedlbd.append(res[2])
        estimatedubd.append(res[3])
    # pnw1.append()

    plt.plot(upperbound, 'b-', lowerbound, 'g-', estimatedubd, 'b--', estimatedlbd, 'g--')

    plt.title('payment size vs costs')

    plt.xlabel("payment")
    plt.ylabel("bound")
    # plt.axis([0, 6, 0, 100])
    plt.legend(["upperbound", "lowerbound", "estimatedubd", "estimatedlbd"])
    plt.savefig('output.png')
    plt.show()


    return getIntersectionPoint(upperbound, lowerbound)


# interescts = runWithTime()

# plt.subplot(211)
# plt.plot(times, interescts)
# plt.subplot(212)

runWithPayment(10)
# runWithFreq()




