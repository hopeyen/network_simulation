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

a1, a2, b1, b2, c1, c2 = [], [], [], [], [], []

ps = [x* 1.0 /10 for x in range(50)]

num_trial = 100
# transfer payment values
for i in range(50):
    # trial
    res = [0,0,0,0,0,0]
    temp = []
    for k in range(num_trial):
        temp = main(ps[i])
        for j in range(len(temp)):
            res[j] += temp[j]
    for j in range(len(res)):
        res[j] = res[j]/float(num_trial)

    a1.append(res[0])
    a2.append(res[1])
    b1.append(res[2])
    b2.append(res[3])
    c1.append(res[4])
    c2.append(res[5])

plt.plot(a1, 'b--', a2, 'b-', b1, 'r--', b2, 'r-', c1, 'g--', c2, 'g-')

plt.title('payment amt vs costs')
plt.xlabel("payment")
plt.ylabel("bound")
# plt.axis([0, 6, 0, 100])
plt.legend(["a1", "a2", "b1", "b2", "c1", "c2"])
plt.savefig('output.png')