from simulation_1 import main
import random
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

def Rand(start, end, num): 
    res = [] 
    for j in range(num): 
        res.append(random.randint(start, end)) 
    return res

upperbound, lowerbound = [], []

ps = [x* 1.0 /10 for x in range(50)]

num_trial = 1000
# transfer payment values
for i in range(len(ps)):
    # trial
    res = [0,0]
    temp = []
    for k in range(num_trial):
        temp = main(p=ps[i])
        for j in range(len(temp)):
            res[j] += temp[j]
    for j in range(len(res)):
        res[j] = res[j]/float(num_trial)

    upperbound.append(res[0])
    lowerbound.append(res[1])

plt.plot(upperbound, 'b-', lowerbound, 'g-')

plt.title('payment size vs costs')
plt.xlabel("payment")
plt.ylabel("bound")
# plt.axis([0, 6, 0, 100])
plt.legend(["upperbound", "lowerbound"])
plt.savefig('output.png')


# fs = [x* 1.0 /10 for x in range(1,50)]

# num_trial = 1000
# # transfer payment values
# for i in range(len(fs)):
#     # trial
#     res = [0,0]
#     temp = []
#     for k in range(num_trial):
#         temp = main(freq=fs[i])
#         for j in range(len(temp)):
#             res[j] += temp[j]
#     for j in range(len(res)):
#         res[j] = res[j]/float(num_trial)

#     upperbound.append(res[0])
#     lowerbound.append(res[1])

# plt.plot(upperbound, 'b-', lowerbound, 'g-')

# plt.title('frequency vs costs')
# plt.xlabel("frequnecy \n 1000 trials")
# plt.ylabel("bound")
# # plt.axis([0, 6, 0, 100])
# plt.legend(["upperbound", "lowerbound"])
# plt.savefig('output.png')