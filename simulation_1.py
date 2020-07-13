import numpy as np 
import pandas as pd 
import random
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

class Node(object):
	def __init__(self, name, network):
		self.name = name
		self.channels = []
		self.payments = []
		self.revenue = 0
		self.channelCost = 0

		self.network = network


	def __eq__(self, other):
	    return (isinstance(other, Node) and (self.name == other.name))

	def __repr__(self):
	    return "%s" % (self.name)


	def addChannel(self, channel):
		self.channels.append(channel)

	def addPayment(self, payment):
		self.payments.append(payment)
		

	# def updateChannelCost(self):
	# 	ans = 0
	# 	for c in self.channels:
	# 		ans += c.getChannelCost()/2
	# 	self.channelCost = ans

	# def costDirectChannel(self, payment):
	# 	# if the node creates an unidir channel for the payment
	# 	if payment != None:
	# 		return math.sqrt(2 * self.network.onlineTX * payment.freq * payment.amt / self.network.r)
	# 	return 0

	def getChCostTotal(self):
		self.channelCost = 0
		for c in self.channels:
			self.channelCost += c.cost
		return self.channelCost



class Payment(object):
	"""docstring for Payment"""
	payments = []
	def __init__(self, freq, amt, sender, reciever):
		self.freq = freq
		self.amt = amt
		self.sender = sender
		self.reciever = reciever
		self.numPaid = 0
		self.txTimes = []
		self.nextTime = self.nextPaymentTime()
		self.channel = None
		self.transferTo = None


		Payment.payments.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Payment) and (self.freq == other.freq)
	    	and (self.amt == other.amt))

	def __repr__(self):
	    return ("%s sends %f to %s" 
	    	    	% (self.sender, self.amt, self.reciever))

	def nextPaymentTime(self):
		self.nextTime = random.expovariate(self.freq)
		self.txTimes.append(self.nextTime)
		self.numPaid += 1
		return self.nextTime

	def setChannel(self, channel):
		self.channel = channel

	def setTransfer(self, payment):
		self.transferTo = payment


class Channel(object):
	channels = []
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

		self.network = network

		self.cost = network.onlineTX
		self.transferPayment = None

		A.addChannel(self)
		B.addChannel(self)		

		Channel.channels.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Channel) and (self.A == other.A)
	    	and (self.B == other.B) and (self.network == other.network))

	def __repr__(self):
	    return ("%s has balance %f, %s has balance %f" 
	    	    	% (self.A, self.balanceA, self.B, self.balanceB))

	def getChannelCost(self):
		# # simplied version of calculating the cost
		# freq = 0
		# for p in self.paymentsA:
		# 	freq += p.freq
		# for p in self.paymentsB:
		# 	freq -= p.freq
		# freq = abs(freq)
		# self.cost = 3* ((2* network.onlineTX * freq / network.r)**(1.0/3))
		return self.cost

	def updateCost(self):
		self.cost += self.network.onlineTX

	def addPayment(self, payment):
		# print("add payment in  " + payment.sender.name + " to " + payment.reciever.name)
		if payment.sender == self.A:
			self.paymentsA.append(payment)

			# for i in range(int(self.mA//payment.amt)):
			# 	self.txTimesA.append(random.expovariate(payment.freq))

			# reconstruct the timeline 

		elif payment.sender == self.B:
			self.paymentsB.append(payment)
		

		self.A.addPayment(payment)
		self.B.addPayment(payment)

		payment.setChannel(self)


	# def removePayment(self, payment):
	# 	if payment in self.paymentsA:
	# 		self.paymentsA.remove(payment)
		
	# 	elif payment in self.paymentsB:
	# 		self.payments.remove(payment)


	# def addTransferPayment(self, payment):
	# 	self.addPayment(payment)
	# 	self.transferPayment = payment


	# def rmTransPayment(self, payment):
	# 	self.removePayment(payment)
	# 	self.transferPayment = None


	# def getLifetime(self):
	# 	return (max(sum(self.txTimesA)), sum(self.txTimesB))

	# def getTxFee(self):
	# 	return 42

	# def getOppoCost(self):
	# 	return self.transferPayment.amt - self.transferPayment.amt * (self.getLifetime() // (self.getLifetime() + self.network.r))

	def reopen(self, side):
		self.updateCost()

		payments = []
		if self.A == side:
			self.balanceA = self.mA
			payments = self.paymentsA
		elif self.B == side:
			self.balanceB = self.mB
			payments = self.paymentsB

		# channel suspended for network online transaction time
		for p in payments:
			p.nextTime += self.network.onlineTXTime



	def processPayment(self, payment):
		time = payment.nextTime
		# print([self.A, self.B, payment.sender])

		if self.A == payment.sender:

			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				# able to make the payment, generate the next payment
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.nextPaymentTime()
				return True
	
		elif self.B == payment.sender:

			if self.balanceB < payment.amt:
				# B has to reopen the channel
				self.reopen(self.B)

			else:
				# able to make the payment, generate the next payment
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.nextPaymentTime()
				return True

		# payment is not processed because of reopening 
		return False		

	def processTransfer(self, payment):
		if self.A == payment.sender:
			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.numPaid += 1
				return True
	
		elif self.B == payment.sender:
			if self.balanceB < payment.amt:
				self.reopen(self.B)

			else:
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.numPaid += 1
				return True
		return False	



class Network(object):
	# keep track of the state of the network, include structure and flow
	def __init__(self, onlineTX, onlineTXTime, r, timeLeft):
		self.onlineTX = onlineTX
		self.onlineTXTime = onlineTXTime
		self.r = r
		self.nodes = []
		self.channels = []

		self.timeLeft = timeLeft
		self.payments = []
		self.history = []

	def addNode(self, node):
		self.nodes.append(node)

	def addNodeList(self, ns):
		self.nodes.extend(ns)

	def addChannel(self, channel):
		self.channels.append(channel)

	def addChannelList(self, chs):
		self.channels.extend(chs)

	def addPayment(self, payment):
		self.payments.append(payment)

	def addPaymentList(self, ps):
		self.payments.extend(ps)

	def runNetwork(self):
		# payments can be concurrent on different channels
		# the payment that takes the smallest time should be processed first
		# and when it has been processed, all other payments' interval decrement by the interval of the processed payment
		# and the processed payment has a new interval that gets put into the timeline

		while self.timeLeft >= 0:

			nextPayment = self.payments[0]
			nPTime = self.payments[0].nextTime
			for p in self.payments:
				if p.nextTime < nPTime:
					nextPayment = p
					nPTime = p.nextTime
			if nPTime > self.timeLeft:
				break

			# process the next payment in the channel
			# the channel checks for the channel balance, reopen if balance not enough
			if (nextPayment.channel.processPayment(nextPayment)):
				
				self.timeLeft -= nPTime
				for p in self.payments:
					if p != nextPayment:
						p.nextTime -= nPTime
				if nextPayment.transferTo != None:
					nextPayment.transferTo.channel.processTransfer(nextPayment.transferTo)

				self.history.append((self.timeLeft, nextPayment, nextPayment.channel.balanceA, nextPayment.channel.balanceB))

		# print(self.history)
		# for p in self.payments:
		# 	print([p, p.numPaid])
			




def main(p=0.1, onlineTX = 5, onlineTXTime = 1, r = 0.01, timeRun = 10):
	# set up the network
	network1 = Network(onlineTX, onlineTXTime, r, timeRun)

	Alice1 = Node("Alice", network1)
	Bob1 = Node("Bob", network1)
	Charlie1 = Node("Charlie", network1)

	paymentAB = Payment(2, 1, Alice1, Bob1)
	paymentBC = Payment(2, 1, Bob1, Charlie1)
	paymentAC = Payment(0.5, p, Alice1, Charlie1)

	channelAB1 = Channel(Alice1, Bob1, 5, 0, network1)
	channelAB1.addPayment(paymentAB)
	channelBC1 = Channel(Bob1, Charlie1, 10, 10, network1)
	channelBC1.addPayment(paymentBC)

	# Alice creates a direct channel for network 1
	channelAC = Channel(Alice1, Charlie1, 5, 0, network1)
	channelAC.addPayment(paymentAC)
	
	# print("network1")
	network1.addNodeList([Alice1, Bob1, Charlie1])
	network1.addChannelList([channelAB1, channelBC1, channelAC])
	network1.addPaymentList([paymentAB, paymentBC, paymentAC])
	network1.runNetwork()

	a1 = Alice1.getChCostTotal()
	b1 = Bob1.getChCostTotal()
	c1 = Charlie1.getChCostTotal()
	



	# network 2 
	network2 = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice2 = Node("Alice", network2)
	Bob2 = Node("Bob", network2)
	Charlie2 = Node("Charlie", network2)

	paymentAB1 = Payment(2, 1, Alice2, Bob2)
	paymentBC1 = Payment(2, 1, Bob2, Charlie2)
	paymentAC1 = Payment(0.5, p, Alice2, Bob2)
	paymentAC2 = Payment(0.5, p, Bob2, Charlie2)
	paymentAC1.setTransfer(paymentAC2)

	channelAB2 = Channel(Alice2, Bob2, 5, 0, network2)
	channelAB2.addPayment(paymentAB1)
	channelBC2 = Channel(Bob2, Charlie2, 10, 10, network2)
	channelBC2.addPayment(paymentBC1)

	# payment goes through Channel AB and BC
	channelAB2.addPayment(paymentAC1)
	channelBC2.addPayment(paymentAC2)


	network2.addNodeList([Alice2, Bob2, Charlie2])
	network2.addChannelList([channelAB2, channelBC2, channelAC])
	network2.addPaymentList([paymentAB1, paymentBC1, paymentAC1])
	# print("network2")
	network2.runNetwork()
	# print([paymentAC2, paymentAC2.numPaid])


	a2 = Alice2.getChCostTotal()
	b2 = Bob2.getChCostTotal()
	c2 = Charlie2.getChCostTotal()

	# to get the cost of the channels, we need to simulate a time period 
	# let the network run for an amount of time, with random variables of frequency
	# each time the channel balance depletes, update the channel cost -- the nodes update the channel
	# at the end of the time period, sum up the channel cost




	return (a1, a2, b1, b2, c1, c2)

	# network = Network()
	# network.addNode(Alice)





if __name__ == '__main__':
    main(timeRun = 20)