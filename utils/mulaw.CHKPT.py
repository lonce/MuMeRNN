import numpy as np

class mulaw():
    #---  muval stuff
    def __init__(self, nvals):
        if nvals > 256 :
            raise ValueError('mulaw written to accommodate number of descrete value up to 256 only')
        self.nvals=nvals
        self.midpoint=(nvals-1.)/2.
        self.lognvals=np.log(nvals)


        #Brute force method to get the table of muvalues so we can index them with a one-hot vector
        #...just need enough numbers in linspace so we are sure to hit each muval at least once.
        #...12000 is adequate when the mutable length is 256, but goes up with 
        self.mutable=np.unique(self._decimate(self._float2mu(np.linspace(-1,1,num=12000, endpoint=False)))) 
        #print("mutable is of length = " + str(len(self.mutable)))

    #works on floats and nparrays of any size
    def _float2mu(self,x) :
        return np.sign(x)*( (np.log(1+(self.nvals-1)*np.abs(x)))/self.lognvals)
    
    #works on floats and nparrays of any size
    #maps continuous floats in [-1,1] on to nvals equally spaced values in [-1,1]
    def _decimate(self, x) :
        return np.round((self.midpoint-.000000001)*(x+1))/self.midpoint -1.
    
    def _mu2float(self, x) :
        d=1/(self.nvals-1)
        y= np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) 
        return y

    #-----------------------------------------------------------------------
    # -----  one-hot stuff
    # each row is converted from its onhot rep to its mu val
    def _onehot2mu(self, a) :
    	# looks like a can take a tensor, even a cuda tensor, and return a np array!
        argmax = np.argmax(a, 1)
        return   self.mutable[argmax]
    
    # maps mu-values to their mutable indicies
    def _mu2index(self, x) :
        return np.uint8((np.round((self.nvals-1)*(1+ self._decimate(x))/2)))
                        
                        
    def _mu2onehot1(self, m) :
        oh=np.zeros(self.nvals)
        oh[self._mu2index(m)]=1
        return oh
    
    def _mu2onehot(self, a) :
        siqlen = a.size
        oh=np.zeros((siqlen, self.nvals))
        for i in range(siqlen) :
            oh[i, : ] = self._mu2onehot1(a[i])
        return oh 

    def _float2index(self, a) : 
        return self._mu2index(self._decimate(self._float2mu(a)))

                        
    #-----------------------------------------------------------------------
    # *******************  for export only   ****************

    #array of floats to array of one-hot vectors
    def encode(self, a) :
        ilen=len(a)
        ar = np.zeros((ilen, self.nvals), dtype=np.uint8)
        idx=self._float2index(a)
        for i in range(ilen):
            # print(" se1 element " + str(i) + " for " + str(a[i]) + " has sets index " + str(self._float2index1(a[i])))
            ar[i][idx[i]] = 1
        return ar

    # oh is an tensor of one-hot vectors
    # returns: array of floats
    def decode(self, oh) :
        return self._mu2float(self._onehot2mu(oh))

    #input is an array of integers indexing the mutable
    def index2float(self, i) :
    	return self._mu2float(self.mutable[i])
    
    #index a float tensor of any number of dimension 
    # returns a ByteTensor of same shape as input
    # ------- should take an array of floats and return an array of floats
    def float2index(self, a) :
        ilen=len(a)
        #tensor = torch.zeros(ilen, 1, self.nvals).type(dtype)
        #eturn(torch.from_numpy(np.array(self._float2index(a))))
        return(self._float2index(a))
    
    
##===========================================================================
##===========================================================================
# mulaw2 uses a dipole representation (two-element vector representing 
# a floating point value by coding the distance of teh value from each eand-point)
##===========================================================================
class mulaw2() :
    def __init__(self, nvals):
        if nvals > 256 :
            raise ValueError('mulaw written to accommodate number of descrete value up to 256 only')
        self.nvals=nvals
        self.midpoint=(nvals-1.)/2.
        self.lognvals=np.log(nvals)

        self.vlenth=2


        #Brute force method to get the table of muvalues so we can index them with a one-hot vector
        #...just need enough numbers in linspace so we are sure to hit each muval at least once.
        #...12000 is adequate when the mutable length is 256, but goes up with 
        self.mutable=np.unique(self._decimate(self._float2mu(np.linspace(-1,1,num=12000, endpoint=False)))) 
        #print("mutable is of length = " + str(len(self.mutable)))

        #works on floats and nparrays of any size
    def _float2mu(self,x) :
        return np.sign(x)*( (np.log(1+(self.nvals-1)*np.abs(x)))/self.lognvals)

    #works on floats and nparrays of any size
    #maps continuous floats in [-1,1] on to nvals equally spaced values in [-1,1]
    def _decimate(self, x) :
        return np.round((self.midpoint-.000000001)*(x+1))/self.midpoint -1.

    def _mu2float(self, x) :
        d=1/(self.nvals-1)
        y= np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) 
        return y


    # *******************  for export only   ****************

    #array of floats to array of dipole vectors
    def encode(self, a) :
        ilen=len(a)
        ar = np.zeros((ilen, self.vlenth))
        dnmuval=(1.+self._decimate(self._float2mu(a)))/2. #decimated to num quant vales, normalized to [0,1]
        for i in range(ilen):
            ar[i][0]=1-dnmuval[i]
            ar[i][1]=dnmuval[i]
        return ar

    # oh is an array of one-hot vectors
    # returns: array of floats
    def decode(self, a) :
        return self._mu2float(a[:,1]*2.-1.) #the 2nd dipole component is just the decimated and normalized muval
