import numpy as np
import torch

#must pass in number of nodes 

class condparam():

	def __init__(self, paramType=2, numNodes=2, maxSeqLen=1, normedVal=0, dtype=torch.FloatTensor):
		self.normedVal=normedVal
		self.paramType=paramType
		self.numNodes=numNodes
		self.maxSeqLen=maxSeqLen

		if paramType==0 :
			assert(numNodes == 1)
		if numNodes==1 :
			assert(paramType == 0)

		self.tensor = torch.zeros(self.maxSeqLen, self.numNodes).type(dtype)


	#set the value and return the coded tensor
	def setTensor(self, normedval, slen=None, verbose=False) :
	#def makeCondTensor(self, normedval, slen=None, verbose=False):
		"""Creates a spatial representation of a floating point value over a tensor.
		seqlen -- the number of components the tensor should have
		normedval -- the floating point value in [0,1)
		Keyword arguments:
		type -- 1-'one-hot' (default), 2- 'two-hot', 4 - 'four hot' (should be 0, 1, or an even number)
		verbose -- use at your own risk

		Returns: a 2-D torch tensor 
		"""

		# default comes from __init__, but a smaller value (eg 1) can be passed in
		if slen==None :
			slen = self.maxSeqLen

		#clear previous tensor to zero
		self.tensor[:slen,:] = 0

		#Single floating point value
		if (self.paramType==0) :
			condindex=0
			if verbose : 
				print('Condtensor type = ' + str(self.paramType) + '. Assign floatval = ' + str(normedval) + " to condindex " + str(condindex) + " in vect with  " + str(self.numNodes) + " nodes")    
			self.tensor[:slen,condindex] = normedval
	    
	    # one hot, out of the numNodes length vector
		elif(self.paramType==1) :
			#rounds down
			floatcondIndex = normedval*(self.numNodes-1) # between 0 and max normedval + 1 
			condindex=int(np.round(floatcondIndex))
			if verbose : 
				print('Condtensor type = ' + str(self.paramType) + '. Assign floatval = ' + str(normedval) + " to condindex " + str(condindex) + " in vect with  " + str(self.numNodes) + " nodes")
			self.tensor[:slen,condindex] = 1

	    
	    # up to paramType values (out of the nomNodes long vector) will be activate, their values summing to 1 except for edge effects
		elif(self.paramType%2==0) :
			halfn=self.paramType/2.   # because the type specifier is also the number to spread the activation over
			#interpolates
			floatcondIndex = normedval*(self.numNodes-1) # between 0 and max normedval
	        
			if verbose : 
				print('Condtensor type = ' + str(self.paramType) + '. Assign floatval = ' + str(normedval) + " to condindex " + str(floatcondIndex) + " in vect with  " + str(self.numNodes) + " nodes")
	        	        
			for i in range(self.numNodes) :
				d=abs(floatcondIndex-i)
				if d < halfn :                          # 0 outside halfn distance from center.
					self.tensor[:slen,i] = (1 - d/halfn)/halfn   #trainglular activate, normed to sum to 1
	                
		return self.tensor[:slen]

