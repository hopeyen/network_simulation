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
		network.nodes.append(self)


	def __eq__(self, other):
	    return (isinstance(other, Node) and (self.name == other.name))

	def __repr__(self):
	    return "%s with cost of %f" % (self.name, self.getChCostTotal())


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
			self.channelCost += c.cost *1.0 /2
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
		# self.txTimes = []
		self.nextTime = self.nextPaymentTime()
		self.channel = None
		self.transferTo = None
		self.fee = 0
		self.numProcessed = 0
		
		Payment.payments.append(self)

		# self.findRoute()

	def __eq__(self, other):
	    return (isinstance(other, Payment) and (self.freq == other.freq)
	    	and (self.amt == other.amt))

	def __repr__(self):
	    return ("%s sends %f to %s at rate %f, num processed: %d" 
	    	    	% (self.sender, self.amt, self.reciever, self.freq, self.numProcessed))

	def nextPaymentTime(self):
		self.nextTime = random.expovariate(1/self.freq)
		# self.txTimes.append(self.nextTime)
		self.numPaid += 1
		return self.nextTime

	def setChannel(self, channel):
		self.channel = channel

	def setTransfer(self, payment):
		self.transferTo = payment

	def estimateUbd(self):
	    network = self.sender.network
	    return math.sqrt(2 * network.onlineTX * self.freq * self.amt / network.r)

	def estimtedLbd(self, paymentAB):
	    n = self.sender.network
	    # interests = (payment.amt * network.r)/ (lifetime + network.r)
	    cnb1 = math.sqrt(2 * n.onlineTX * (self.freq * self.amt + paymentAB.freq * paymentAB.amt) / n.r)
	    cnb2 = 3 * ((2 * n.onlineTX * (self.freq + paymentAB.freq) / n.r)**(1.0/3))
	    cob1 = math.sqrt(2 * n.onlineTX * paymentAB.freq * paymentAB.amt / n.r)
	    cob2 = 3 * ((2 * n.onlineTX * paymentAB.freq / n.r)**(1.0/3))
	    return cnb1 + cnb2 - cob1 - cob2

	def findRoute(self, start, level = 0):
		if level > 3:
			# create a direct channel
			return 
		for c in self.A.channels:
			if self.reciever == c.B:
				self.route = [c]
				self.setChannel(c)
			else:
				self.route.append(self.findRoute(c.B), level +1)
				c.addTransferPayment(self)
				return

	def setFee(self, fee):
		self.fee = fee

class Channel(object):
	channels = []
	def __init__(self, A, B, network):
		# super().__init__(A, B, network)
		self.A = A
		self.B = B
		self.mA = 0
		self.mB = 0
		self.balanceA = 0
		self.balanceB = 0
		self.paymentsA = []
		self.paymentsB = []
		self.numReopen = 0
		# self.txTimesA = []
		# self.txTimesB = []

		self.network = network

		self.cost = 0
		self.transferPayment = []

		A.addChannel(self)
		B.addChannel(self)		

		Channel.channels.append(self)
		network.channels.append(self)

	def __eq__(self, other):
	    return (isinstance(other, Channel) and (self.A == other.A)
	    	and (self.B == other.B) and (self.network == other.network))

	def __repr__(self):
	    # return ("%s has balance %f, %s has balance %f" 
	    # 	    	% (self.A, self.balanceA, self.B, self.balanceB))
	    return ("%s and %s with cost %f, reopens %d times\n -- average frequencies (%f, %f) \n" 
	    	% (self.A, self.B, self.cost, self.numReopen, self.avergeFreq(self.paymentsA), self.avergeFreq(self.paymentsB))
	    	+ str(self.paymentsA))

	def getChannelCost(self):
		return self.cost

	def updateCost(self):
		# add discounted onlineTx cost to the total cost
		self.cost += self.network.onlineTX * math.exp(-1 * self.network.r * (self.network.totalTime - self.network.timeLeft))

	def addPayment(self, payment):
		# print("add payment in channel " + payment.sender.name + " to " + payment.reciever.name + ", p: %f; f: %f" %(payment.amt, payment.freq))
		if payment.sender == self.A:
			# print("addPaymentA")
			self.paymentsA.append(payment)

		elif payment.sender == self.B:
			# print("addPaymentB")
			self.paymentsB.append(payment)

		

		self.A.addPayment(payment)
		self.B.addPayment(payment)

		payment.setChannel(self)
		self.optimizeSize()


	def addPaymentList(self, payments):
		for p in payments:
			self.addPayment(p)


	# def removePayment(self, payment):
	# 	if payment in self.paymentsA:
	# 		self.paymentsA.remove(payment)
		
	# 	elif payment in self.paymentsB:
	# 		self.payments.remove(payment)


	def addTransferPayment(self, payment):
		self.addPayment(payment)
		self.transferPayment.append(payment)


	# def rmTransPayment(self, payment):
	# 	self.removePayment(payment)
	# 	self.transferPayment = None


	# def getLifetime(self):
	# 	return (max(sum(self.txTimesA)), sum(self.txTimesB))

	# def getTxFee(self):
	# 	return 42

	# def getOppoCost(self):
	# 	return self.transferPayment.amt - self.transferPayment.amt * (self.getLifetime() // (self.getLifetime() + self.network.r))

	def setChannelSize(self, mA, mB):
		self.mA = mA
		self.mB = mB

	def getSlowestFreq(self, payments):
		# at least one payment
		slowest = payments[0]

		for p in payments:
			if p.freq > slowest.freq:
				slowest = p

		return slowest

	def getSlowestFreq(self, payments):
		# at least one payment
		slowest = payments[0]
		sumFreq = 0

		for p in payments:
			if p.freq > slowest.freq:
				slowest = p
				sumFreq += 1/p.freq

		return (slowest, sumFreq)

	def getPortionFreq(self, payments):
		(slowest, sumFreq) = self.getSlowestFreq(payments)
		portion = (1/ slowest.freq) / sumFreq



	def avergeFreq(self, payments):
		# print("calculating averagefreq")
		sumFreq = 0

		for p in payments:
			sumFreq += p.amt / p.freq
			# print("paymnt f: %f; p: %f" %(p.freq, p.amt))

		if len(payments) == 0: return 0
		return sumFreq
		

	def optimizeSize(self):
		# print("---------- optimizeSize")
		# print(self.paymentsA)
		fA = self.avergeFreq(self.paymentsA)
		fB = self.avergeFreq(self.paymentsB)
		oneWay = 0
		bidir = 0

		oneWay = (self.network.onlineTX * abs(fA - fB) / self.network.r) **(1.0/2)
		bidir = (2 * self.network.onlineTX * min(fA, fB) / self.network.r)**(1.0/3)

		if min(fA, fB) == fA:
			self.setChannelSize(bidir, bidir+oneWay)
		else:
			self.setChannelSize(bidir+oneWay, bidir)


		self.balanceA = self.mA
		self.balanceB = self.mB
		self.updateCost()
		# print("2dir: %f; 1dir: %f" %(bidir, oneWay))

		# opens channel


	def reopen(self, side):
		# print("reopening "+ str(self))
		self.updateCost()
		self.numReopen += 1

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
		# print("processing payment\n --")
		# print(payment)
		# print(" -- ")

		if self.A == payment.sender:

			if self.balanceA < payment.amt:
				# A has to reopen the channel
				# print("can't process, reopen")
				self.reopen(self.A)
				
			else:
				# able to make the payment, generate the next payment
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.nextPaymentTime()
				payment.numProcessed += 1
				# print("processed, numProcessed %d" % payment.numProcessed)
				return True
	
		elif self.B == payment.sender:

			if self.balanceB < payment.amt:
				# B has to reopen the channel
				# print("can't process, reopen")
				self.reopen(self.B)


			else:
				# able to make the payment, generate the next payment
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.nextPaymentTime()
				payment.numProcessed += 1
				# print("processed, numProcessed %d" % payment.numProcessed)
				return True

		# payment is not processed because of reopening 
		return False		

	def processTransfer(self, payment):
		# instant transfer
		if self.A == payment.sender:
			if self.balanceA < payment.amt:
				# A has to reopen the channel
				self.reopen(self.A)
				
			else:
				self.balanceA -= payment.amt
				self.balanceB += payment.amt
				payment.numProcessed += 1
				return True
	
		elif self.B == payment.sender:
			if self.balanceB < payment.amt:
				self.reopen(self.B)

			else:
				self.balanceB -= payment.amt
				self.balanceA += payment.amt
				payment.numProcessed += 1
				return True
		return False	

	def expectedTX(self):
		expectedA, expectedB = [], []
		totalTk = 0

		for p in self.paymentsA:
			expectedA.append(self.network.totalTime / p.freq)
			totalTk += (self.network.totalTime / p.freq) * p.amt

		for p in self.paymentsB:
			expectedB.append(self.network.totalTime / p.freq)
			totalTk += (self.network.totalTime / p.freq) * p.amt
		print("expected txs: A:" + str(expectedA) + "; B: " + str(expectedB) + 
			"; total number of transactions: " + str((sum(expectedA)+sum(expectedB))) + 
			"; total token: " + str(totalTk) + "; expected reopens: " + str(totalTk/self.mA))

		if (len(expectedA)> 1):
			print("A with multiple payments: " + str(self.network.totalTime * self.avergeFreq(self.paymentsA)))

		print("while channel size is (%f,%f)" %(self.mA, self.mB))


class Network(object):
	# keep track of the state of the network, include structure and flow
	def __init__(self, onlineTX, onlineTXTime, r, timeLeft):
		self.onlineTX = onlineTX
		self.onlineTXTime = onlineTXTime
		self.r = r
		self.nodes = []
		self.channels = []
		self.totalTime = timeLeft

		self.timeLeft = timeLeft
		self.payments = []
		self.transferredPayments = []
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

	def addTransferred(self, payment):
		self.transferredPayments.append(payment)

	def addTransferredList(self, ps):
		self.transferredPayments.extend(ps)

	def runNetwork(self):
		# payments can be concurrent on different channels
		# the payment that takes the smallest time should be processed first
		# and when it has been processed, all other payments' interval decrement by the interval of the processed payment
		# and the processed payment has a new interval that gets put into the timeline
		
		for c in self.channels:
			c.optimizeSize()

		while self.timeLeft >= 0:
			# print(self.timeLeft)
			nextPayment = self.payments[0]
			nPTime = self.payments[0].nextTime
			for p in self.payments:
				# print(p)
				# print("looping")
				
				if p.nextTime < nPTime:
					nextPayment = p
					nPTime = p.nextTime
				
			if nPTime > self.timeLeft:
				break

			# process the next payment in the channel
			# the channel checks for the channel balance, reopen if balance not enough
			# print(nextPayment)
			if (nextPayment.channel.processPayment(nextPayment)):
				# print("time:" + str(self.timeLeft))
				# print("next payment")
				# print(nextPayment)
				self.timeLeft -= nPTime
				for p in self.payments:
					# print("decrement %s %f " %(nextPayment, self.timeLeft))
					if p != nextPayment:
						p.nextTime -= nPTime
					# print("decreasing time")
				if nextPayment.transferTo != None:
					# print("----- transfer")
					# print(nextPayment.transferTo)
					nextPayment.transferTo.channel.processTransfer(nextPayment.transferTo)

				# self.history.append((self.timeLeft, nextPayment, nextPayment.channel.balanceA, nextPayment.channel.balanceB))
			else:
				# attempt to send failed, payment not processed
				errorTime = 0.001
				self.timeLeft -= errorTime
				for p in self.payments:
					p.nextTime -= errorTime


		# self.printSummary()
		

			
	def printSummary(self):
		print("Summary")
		print("Timeleft: %f" %self.timeLeft)
		print(" - Nodes: ")
		for n in self.nodes:
			print(n)
		print(" - Payments: ")
		for p in self.payments:
			print(p)
		print(" - Transferred Payments: ")
		for p in self.transferredPayments:
			print(p)
		print(" - Channels: ")
		for c in self.channels:
			print(c)
		print(" - testing")
		self.testing()
		print("\n")


	def getTotalCost(self):
		s = 0
		for n in self.nodes:
			s += n.getChCostTotal()
		return s
	
	def testing(self):
		print("--network expected num tx")
		for c in self.channels:
			c.expectedTX()
		




largePayments = 1
largeFrequency = 1



def networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# network 2 
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	network.addNodeList([Alice, Bob, Charlie])
	network.addChannelList([channelAB, channelBC])
	network.addPaymentList([paymentAB, paymentBC])
	# print("network2")
	network.runNetwork()
	# print([paymentAC2, paymentAC2.numPaid])


	a = Alice.getChCostTotal()
	b = Bob.getChCostTotal()
	c = Charlie.getChCostTotal()

	return (a+c, b)


def networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun):
	# set up the network
	network = Network(onlineTX, onlineTXTime, r, timeRun)

	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)
	paymentAC = Payment(freq, p, Alice, Charlie)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
	channelBC.addPayment(paymentBC)

	# Alice creates a direct channel for network 1
	channelAC = Channel(Alice, Charlie, network)
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
	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
	Alice = Node("Alice", network)
	Bob = Node("Bob", network)
	Charlie = Node("Charlie", network)

	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)
	paymentAC = Payment(freq, p, Alice, Bob)
	paymentAC1 = Payment(freq, p, Bob, Charlie)
	paymentAC.setTransfer(paymentAC1)

	channelAB = Channel(Alice, Bob, network)
	channelAB.addPayment(paymentAB)
	channelBC = Channel(Bob, Charlie, network)
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




def main(p=0.1, freq=0.5, onlineTX = 5.0, onlineTXTime = 1.0, r = 0.01, timeRun = 10.0):
	(a0, b0) = networkOG(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a1, b1) = networkDirectAC(p, freq, onlineTX, onlineTXTime, r, timeRun)
	(a2, b2) = networktransferB(p, freq, onlineTX, onlineTXTime, r, timeRun)
	# (a3, b3, c3) = networktransferC(p, freq, onlineTX, onlineTXTime, r, timeRun)
	# (a4, b4, c4) = networktransferA(p, freq, onlineTX, onlineTXTime, r, timeRun)


	# lbd = paymentAC1.estimtedLbd(paymentAB1)
	# ubd = paymentAC1.estimateUbd()

	# graphing 
	# return (a1-a2+c1-c2, b2-b1)

	# channels
	# return (a1+c1, b1, a2+c2, b2, a3+c3, b3, a4+ c4, b4)

	# max_fee based on network 1 and 2
	# return (a0+c0, b0, a1+c1, b1, a2+c2, b2)

	# just bob
	return (a0, b0, a1, b1, a2, b2)





if __name__ == '__main__':
    main()







# def networktransferA(p, freq, onlineTX, onlineTXTime, r, timeRun):
# 	# network 2 
# 	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
# 	Alice = Node("Alice", network)
# 	Bob = Node("Bob", network)
# 	Charlie = Node("Charlie", network)

# 	# BC becomes the transferred payment that go through A
# 	paymentAB = Payment(largeFrequency, largePayments, Alice, Bob)
# 	paymentAC = Payment(freq, p, Alice, Charlie)
# 	paymentBC = Payment(largeFrequency, largePayments, Bob, Alice)	
# 	paymentBC1 = Payment(largeFrequency, largePayments, Alice, Charlie)
# 	paymentBC.setTransfer(paymentBC1)

# 	# existing channels become AC and AB
# 	channelAB = Channel(Alice, Bob, 5, 0, network)
# 	channelAB.addPayment(paymentAB)
# 	channelAC = Channel(Alice, Charlie, 5, 0, network)
# 	channelAC.addPayment(paymentAC)

# 	# payment goes through Channel AC and AC
# 	channelAB.addPayment(paymentBC)
# 	channelAC.addPayment(paymentBC1)


# 	network.addNodeList([Alice, Bob, Charlie])
# 	network.addChannelList([channelAB, channelAC])
# 	network.addPaymentList([paymentAB, paymentBC, paymentAC])
# 	# print("network2")
# 	network.runNetwork()
# 	# print([paymentAC2, paymentAC2.numPaid])


# 	a = Alice.getChCostTotal()
# 	b = Bob.getChCostTotal()
# 	c = Charlie.getChCostTotal()

# 	return (a+c, b)



# def networktransferC(p, freq, onlineTX, onlineTXTime, r, timeRun):
# 	# network 2 
# 	network = Network(onlineTX, onlineTXTime, r, timeRun)
	
# 	Alice = Node("Alice", network)
# 	Bob = Node("Bob", network)
# 	Charlie = Node("Charlie", network)

# 	# AB becomes the transferred payment that go through C
# 	paymentBC = Payment(largeFrequency, largePayments, Bob, Charlie)
# 	paymentAC = Payment(freq, p, Alice, Charlie)
# 	paymentAB = Payment(largeFrequency, largePayments, Alice, Charlie)
# 	paymentAB1 = Payment(largeFrequency, largePayments, Charlie, Bob)
# 	paymentAB.setTransfer(paymentAB1)

# 	# existing channels become AC and BC
# 	channelBC = Channel(Bob, Charlie, 20, 20, network)
# 	channelBC.addPayment(paymentBC)
# 	channelAC = Channel(Alice, Charlie, 5, 0, network)
# 	channelAC.addPayment(paymentAC)

# 	# payment goes through Channel AC and BC
# 	channelAC.addPayment(paymentAB)
# 	channelBC.addPayment(paymentAB1)


# 	network.addNodeList([Alice, Bob, Charlie])
# 	network.addChannelList([channelBC, channelAC])
# 	network.addPaymentList([paymentAB, paymentBC, paymentAC])
# 	# print("network2")
# 	network.runNetwork()
# 	# print([paymentAC2, paymentAC2.numPaid])


# 	a = Alice.getChCostTotal()
# 	b = Bob.getChCostTotal()
# 	c = Charlie.getChCostTotal()

# 	return (a+c, b)