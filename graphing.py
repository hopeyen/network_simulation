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

ubd = []
lbd = []

ps = [x* 1.0 /1000 for x in range(50)]

for i in range(50):
    bd = main(ps[i])
    ubd.append(bd[0])
    lbd.append(bd[1])

plt.plot(lbd, 'b--')

plt.title('payment amt vs lbd, upperbd %f' %ubd[0])
plt.xlabel("payment")
plt.ylabel("bound")
plt.savefig('output.png')