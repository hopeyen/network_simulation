import numpy as np 
import pandas as pd 
import random
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Node(object):
	def __init__(self, name):
		self.name = name
		self.channels = []
		self.payments = []
		self.revenue = 0
		self.channelCost = 0

	def addChannel(self, channel, network):
		self.channels.append(channel)
		self.channelCost += (channel.getChannelCost(network)/2)

	def updateChannelCost(self, network):
		ans = 0
		for c in self.channels:
			ans += c.getChannelCost(network)/2
		self.channelCost = ans

	def getChannelCost(self):
		return self.channelCost


	def addPayment(self, payment, network):
		self.payments.append(payment)

		self.updateChannelCost(network)
		# get cost of the payment, calculate tx fee
		# calculate revenue with cost

	def __eq__(self, other):
	    return (isinstance(other, Node) and (self.name == other.name))

	def __repr__(self):
	    return "Node(%s, %f)" % (self.name, self.revenue)

	def costDirectChannel(self, payment, network):
		# if the node creates an unidir channel for the payment
		if payment != None:
			return math.sqrt(2 * network.onlineTX * payment.freq * payment.amt / network.r)
		return 0

class Payment(object):
	"""docstring for Payment"""
	payments = []
	def __init__(self, freq, amt, sender, reciever):
		self.freq = freq
		self.amt = amt
		self.sender = sender
		self.reciever = reciever
		self.txTimes = []

		Payment.payments.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Payment) and (self.freq == other.freq)
	    	and (self.amt == other.amt) and (self.sender == other.sender)
	    	and (self.reciever == other.reciever))

	def __repr__(self):
	    return ("Payment(%s sends amount %f to %s with average frequency %f" 
	    	    	% (self.sender, self.amt, self.reciever, self.freq))

class Channel(object):
	"""docstring for Channel"""
	channels = []
	def __init__(self, nodeA, nodeB, network):
		self.nodeA = nodeA
		self.nodeB = nodeB

		nodeA.addChannel(self, network)
		nodeB.addChannel(self, network)
		Channel.channels.append(self)

class UniChannel(Channel):
	unichannels = []
	def __init__(self, m, sender, reciever, network):
		# super().__init__(sender, reciever, network)
		self.m = m
		self.sender = sender
		self.reciever = reciever
		self.balance = m
		self.payments = []
		# calculate cost and generate time transacted 
		self.cost = 0
		self.txTimes = []
		self.interval = 0
		self.transferPayment = None 

		sender.addChannel(self, network)
		reciever.addChannel(self, network)

		UniChannel.unichannels.append(self)

	def getChannelCost(self, network):
		# a grossly simplied version of calculating the cost
		csum = 0
		for p in self.payments:
			csum += (p.freq * p.amt)
		
		self.cost = math.sqrt(2 * network.onlineTX * csum / network.r)
		return self.cost

	def addPayment(self, payment, network):
		print("add payment in unichannel " + payment.sender.name + " to " + payment.reciever.name)
		self.payments.append(payment)
		self.sender.addPayment(payment, network)
		self.reciever.addPayment(payment, network)


		
		for i in range(int(self.m//payment.amt)):
			self.txTimes.append(random.expovariate(payment.freq))
		# self.txTimes.append(random.expovariate(freq))

	def removePayment(self, payment):
		if payment in self.payments:
			self.payments.remove(payment)
		
			for i in range(int(self.m//payment.amt)):
				self.txTimes.append(random.expovariate(payment.freq))

	def addTransferPayment(self, payment, network):
		self.addPayment(payment, network)
		self.transferPayment = payment

	def rmTransPayment(self, payment):
		self.removePayment(payment)
		self.transferPayment = None

	def getLifetime(self):
		# payments transact independently, channel alive until balance = 0 
		timeline = []
		for p in self.payments:
			for i in range(int(self.m//p.amt)):
				p.txTimes.append(random.expovariate(p.freq))
			times = np.cumsum(np.array(p.txTimes))

			numTX = np.arange(int(self.m//p.amt), dtype=np.double)
			ps = np.full_like(numTX, p.amt)

			event = np.hstack([times, ps])
			print(event)

			
			timeline = np.sort(np.concatenate([timeline, times]))
		
		timeOfLastTX = 0
		for event in timeline:
			if self.balance >= event[1]:
				self.balance -= event[1]
				timeOfLastTX = event[0]

		return timeOfLastTX

	def getSingleTXamt(self):
		c = 0
		for p in self.payments:
			c += p.amt
		return c

	def getTxFee(self, p, network):
		if self.transferPayment != None:
			return p * ((self.transferPayment.amt + self.getSingleTXamt())// self.getSingleTXamt()) * math.exp(network.r * self.getLifetime())
		
		return 0

	def getOppoCost(self, network):
		if self.transferPayment != None:
			return self.transferPayment.amt - self.transferPayment.amt * (self.getLifetime() // (self.getLifetime() + network.r))
		return 0
		

class BiChannel(object):
	bichannels = []
	def __init__(self, A, B, mA, mB, network):
		# super().__init__(A, B, network)
		self.A = A
		self.B = B
		self.mA = mA
		self.balanceA = mA
		self.mB = mB
		self.balanceB = mB
		self.paymentsA = []
		self.paymentsB = []
		self.txTimesA = []
		self.txTimesB = []

		self.cost = 0
		self.transferPayment = None

		A.addChannel(self, network)
		B.addChannel(self, network)		

		BiChannel.bichannels.append(self)

	def getChannelCost(self, network):
		# simplied version of calculating the cost
		freq = 0
		for p in self.paymentsA:
			freq += p.freq
		for p in self.paymentsB:
			freq -= p.freq
		freq = abs(freq)
		self.cost = 3* ((2* network.onlineTX * freq / network.r)**(1.0/3))
		return self.cost

	def addPayment(self, payment, network):
		print("add payment in bichannel " + payment.sender.name + " to " + payment.reciever.name)
		if payment.sender == self.A:
			self.paymentsA.append(payment)

			for i in range(int(self.mA//payment.amt)):
				self.txTimesA.append(random.expovariate(payment.freq))

		elif payment.sender == self.B:
			self.paymentsB.append(payment)
		
			for i in range(int(self.mB//payment.amt)):
				self.txTimesB.append(random.expovariate(payment.freq))
		# self.txTimes.append(random.expovariate(freq))

		self.A.addPayment(payment, network)
		self.B.addPayment(payment, network)

	def removePayment(self, payment):
		if payment in self.paymentsA:
			self.paymentsA.remove(payment)
		
			for i in range(int(self.mA//payment.amt)):
				self.txTimesA.append(random.expovariate(payment.freq))
		
		elif payment in self.paymentsB:
			self.payments.remove(payment)
		
			for i in range(int(self.mB//payment.amt)):
				self.txTimesB.append(random.expovariate(payment.freq))


	def addTransferPayment(self, payment, network):
		self.addPayment(payment, network)
		self.transferPayment = payment

	def rmTransPayment(self, payment):
		self.removePayment(payment)
		self.transferPayment = None

	def getLifetime(self):
		return (max(sum(self.txTimesA)), sum(self.txTimesB))

	def getTxFee(self):
		return 42

	def getOppoCost(self, network):
		return self.transferPayment.amt - self.transferPayment.amt * (self.getLifetime() // (self.getLifetime() + network.r))



class Network(object):
	# keep track of the state of the network, include structure and flow
	def __init__(self, onlineTX, r):
		self.onlineTX = onlineTX
		self.r = r
		self.nodes = []
		self.channels = []

	def addNode(self, node):
		self.nodes.append(node)

	def addNodeList(self, ns):
		self.nodes.extend(ns)

	def addChannel(self, channel):
		self.channels.append(channel)

	def addChannelList(self, chs):
		self.channels.extend(chs)



def main(p=0.1, onlineTX = 5, r = 0.01):
	# set up the network
	Alice = Node("Alice")
	Bob = Node("Bob")
	Charlie = Node("Charlie")

	network = Network(onlineTX, r)
	network.addNodeList([Alice, Bob, Charlie])
	
	paymentAB = Payment(1, 1, Alice, Bob)
	paymentBC = Payment(1, 1, Bob, Charlie)

	channelAB = UniChannel(5, Alice, Bob, network)
	channelAB.addPayment(paymentAB, network)
	channelBC = BiChannel(Bob, Charlie, 10, 10, network)
	channelBC.addPayment(paymentBC, network)
	network.addChannelList([channelAB, channelBC])

	aliceOldCost = Alice.getChannelCost()
	bobOldCost = Bob.getChannelCost()
	charlieOldCost = Charlie.getChannelCost()
	print("old cost: alice " + str(aliceOldCost) + " ; bob " + str(bobOldCost) + " ; charlie " + str(charlieOldCost))

	# add a new payment
	paymentAC = Payment(2, 0.01, Alice, Charlie)
	channelAB.addTransferPayment(paymentAC, network)
	channelBC.addTransferPayment(paymentAC, network)

	bobNewCost = 0
	for c in Bob.channels:
		bobNewCost += c.getChannelCost(network)
	bobNewCost -= (aliceOldCost + charlieOldCost)

	oppo = channelAB.getOppoCost(network)
	txfee = channelAB.getTxFee(p, network)
	
	dirCost = Alice.costDirectChannel(paymentAC, network)
	rev = 0

	print("opportunity cost = %f" % oppo)
	print("tx fee = %f" % txfee)
	print("dirCost (upper bd) = %f" % dirCost)
	print("bob new cost = %f" % bobNewCost)

	if (txfee >= dirCost):
		print("Alice creates a direct channel")
		channelAB.rmTransPayment(paymentAC)
	else:
		print("Alice routes the payment")
		
		# tx >= I + c_n_B - c_o_B
		rev = txfee -(oppo + bobNewCost - bobOldCost)
		print("Bob's revenue (lower bd): %f" % rev)

	return (dirCost, rev)

	# network = Network()
	# network.addNode(Alice)





if __name__ == '__main__':
    main()