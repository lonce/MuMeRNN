import numpy as np
#assume data is in [-1,1]
# Note: if you want mu2f(f2mu(0)==0, choose an odd number of descrete values (eg 255)  
class mulaw():
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
        print("mutable is of length = " + str(len(self.mutable)))

    def _float2mu(self, data):
        y=[ np.sign(x)*( (np.log(1+(self.nvals-1)*abs(x)))/self.lognvals)  for x in data] #[-1,1] float
        return y

    def _decimate1 (self, m) : #assume m in [-1,1] - the epsilon is so we can accommodate []
        return np.round((self.midpoint-.000000001)*(m+1))/self.midpoint -1. 

    def _decimate(self, data) :
        return [self._decimate1(m) for m in data]

    # returns [-1,1] on linear scale
    def _mu2float(self, mdata) :
        d=1/(self.nvals-1)
        y=[ np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) for x in mdata ]
        return y

    def _onehot2mu1(self, m) :
        return self.mutable[np.argmax(m)]

    def _onehot2mu(self, a) :
        return [self._onehot2mu1(m) for m in a]

    def _mu2index(self, mval) :
        #This needs to be uint16 if you use ndvals > 256
        return np.uint8(round((self.nvals-1)*(1+ self._decimate1(mval))/2)) 

    def _mu2onehot1(self, m) :
        oh=np.zeros(self.nvals, dtype=np.dtype(int))
        oh[self._mu2index(m)]=1
        return oh

    def _mu2onehot(self, a) :
        return [self._mu2onehot1(x) for x in a]    
    
    # *******************  for export only   ****************
    def encode(self, a) :
        return self._mu2onehot(self._float2mu(a))
        
    def decode(self, oh) :
        return self._mu2float(self._onehot2mu(oh))

    def index2float(i) :
        return self._mu2float(self.mutable[i])
