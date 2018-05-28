
class mulaw():
    #def __init__(self, nvals, dtype):
    def __init__(self, nvals):
        if nvals > 256 :
            raise ValueError('mulaw written to accommodate number of descrete value up to 256 only')
        self.nvals=nvals
        self.midpoint=(nvals-1.)/2.
        self.lognvals=np.log(nvals)
        self.dtype=dtype

        def unique(tensor1d):
            t = np.unique(tensor1d.numpy())
            return torch.from_numpy(t) 

        #Brute force method to get the table of muvalues so we can index them with a one-hot vector
        #...just need enough numbers in linspace so we are sure to hit each muval at least once.
        #...12000 is adequate when the mutable length is 256, but goes up with 
        self.mutable=np.unique(self._decimate(self._float2mu(np.linspace(-1,1,num=12000, endpoint=False)))) 
        #self.mutable=unique(self._decimate(self._float2mu(torch.linspace(-1,1,steps=12000))))
        print("mutable is of length = " + str(len(self.mutable)))

##    def _float2mu1(self, x) :
##        return np.sign(x)*( (np.log(1+(self.nvals-1)*abs(x)))/self.lognvals) 
    
##    def _float2muOLD(self, data):
##        y=[ np.sign(x)*( (np.log(1+(self.nvals-1)*abs(x)))/self.lognvals)  for x in data] #[-1,1] float
##        return y
    
    #x is a tensor of any shape
    def _float2mu_TENSOR(self,x) :
        return torch.sign(x)*( (torch.log(1+(self.nvals-1)*torch.abs(x)))/self.lognvals)

    #works on floats and nparrays of any size
    def _float2mu(self,x) :
        return np.sign(x)*( (np.log(1+(self.nvals-1)*np.abs(x)))/self.lognvals)
    
    
    # -----
##    def _decimate1 (self, m) : #assume m in [-1,1] - the epsilon is so we can accommodate []
##        return np.round((self.midpoint-.000000001)*(m+1))/self.midpoint -1. 
##
##    def _decimateOLD(self, data) :
##        return [self._decimate1(m) for m in data]
    
    #x is a tensor of any shape
    def _decimate_TENSOR(self, x) :
        return torch.round((self.midpoint-.000000001)*(x+1))/self.midpoint -1.
    # -----
    
    #works on floats and nparrays of any size
    def _decimate(self, x) :
        return np.round((self.midpoint-.000000001)*(x+1))/self.midpoint -1.

##    def _mu2float1(self, x) :
##        d=1/(self.nvals-1)
##        return np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1)
##        
##    # returns [-1,1] on linear scale
##    def _mu2floatOLD(self, mdata) :
##        d=1/(self.nvals-1)
##        y=[ np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) for x in mdata ]
##        return y
    
    def _mu2float_TENSOR(self, x) :
        d=1/(self.nvals-1)
        y= torch.sign(x)*d*(torch.pow(self.nvals,torch.abs(x))-1) 
        return y
    
    
    def _mu2float(self, x) :
        d=1/(self.nvals-1)
        y= np.sign(x)*d*(np.power(self.nvals,np.abs(x))-1) 
        return y
    

# -----
##    def _onehot2mu1(self, m) :
##        return self.mutable[np.argmax(m)]
##
##    def _onehot2muOLD(self, a) :
##        return [self._onehot2mu1(m) for m in a]
##    
    # each row is converted from its onhot rep to its mu val
    def _onehot2mu_TENSOR(self, a) :
        _, argmax = torch.max(a, 1)
        return   self.mutable[argmax]
    
    # -----
    
    def _onehot2mu(self, a) :
        _, argmax = np.max(a, 1)
        return   self.mutable[argmax]
    
    
            
##    def _mu2indexOLD(self, mval) :
##        #This needs to be uint16 if you use ndvals > 256
##        return np.uint8(np.round((self.nvals-1)*(1+ self._decimate1(mval))/2)) 

    #x is any shape
    def _mu2index_TENSOR(self, x) :
        return (torch.round((self.nvals-1)*(1+ self._decimate(x))/2)).type(torch.ByteTensor)
    

    def _mu2index(self, x) :
        return np.uint8((np.round((self.nvals-1)*(1+ self._decimate(x))/2)))
                        
    
    # -----
    def _mu2onehot1_TENSOR(self, m) :
        oh=torch.zeros(self.nvals).type(self.dtype)
        oh[self._mu2index(m)]=1
        return oh
                        
    def _mu2onehot1(self, m) :
        oh=np.zeros(self.nvals)
        oh[self._mu2index(m)]=1
        return oh

                        
##     def _mu2onehotOLD(self, a) :
##         return [self._mu2onehot1(x) for x in a]  
    
    #takes a size (siqlen,1) dimensional tensor returns a (seqlen, nvals)
    def _mu2onehot_TENSOR(self, a) :
        siqlen, _ = a.size()
        oh=torch.zeros(siqlen, self.nvals).type(self.dtype)
        for i in range(siqlen) :
            oh[i, : ] = self._mu2onehot1(a[i])
        return oh 
                        
                        
    def _mu2onehot(self, a) :
        siqlen, _ = a.size()
        oh=np.zeros(siqlen, self.nvals)
        for i in range(siqlen) :
            oh[i, : ] = self._mu2onehot1(a[i])
        return oh 
                        
    # -----
    
##     def _float2index1(self, a) :
##         return self._mu2index(self._decimate1(self._float2mu1(a)))
    
##     def _float2indexOLD(self, a) :
##         return [self._float2index1(x) for x in a]
    
    def _float2index_TENSOR(self, a) : 
        return self._mu2index(self._decimate(self._float2mu(a)))

    def _float2index(self, a) : 
        return self._mu2index(self._decimate(self._float2mu(a)))

                        
    # *******************  for export only   ****************
    #takes an arrary of floats and coverts it to 2D tesnor of size(len(a), nvals)
    #---------  Should be called with an array, return a one-hot tensor
    def encode_TENSOR(self, a) :
        ilen=len(a)
        tensor = torch.zeros(ilen, self.nvals).type(dtype)
        for i in range(ilen):
            # print(" se1 element " + str(i) + " for " + str(a[i]) + " has sets index " + str(self._float2index1(a[i])))
            tensor[i][self._float2index1(a[i])] = 1
        return tensor

    def encode(self, a) :
        ilen=len(a)
        ar = np.zeros((ilen, self.nvals), dtype=np.uint8)
        idx=self._float2index(a)
        for i in range(ilen):
            # print(" se1 element " + str(i) + " for " + str(a[i]) + " has sets index " + str(self._float2index1(a[i])))
            ar[i][idx[i]] = 1
        return ar

                        
    #---------  Should be called with a one-hot tensor, return ? - nobody uses this function
    ##def decode(self, oh) :
    ##    return self._mu2float(self._onehot2mu(oh))
    
    # take indexing vector of arbitrary shape (must be a torch.LongTensor), return tabled values in same shape
    #-------------- should take a single float and return a single ingeger
    def index2float_TENSOR(self, i) :
        idx=i.view(1,-1)[0] #for some reason, indexing must be one dimensional!
        sz=i.size()
        return self._mu2float(self.mutable[idx].view(sz))


    def index2float(self, i) :
        return self._mu2float(self.mutable[i])

    
    #inpa a is an arary of floats
    def float2indexOLD(self, a) :
        ilen=len(a)
        #tensor = torch.zeros(ilen, 1, self.nvals).type(dtype)
        return(torch.from_numpy(np.array(self._float2index(a))))
    
    #index a float tensor of any number of dimension 
    # returns a ByteTensor of same shape as input
    # ------- should take an array of floats and return an array of floats
    def float2index(self, a) :
        ilen=len(a)
        #tensor = torch.zeros(ilen, 1, self.nvals).type(dtype)
        #eturn(torch.from_numpy(np.array(self._float2index(a))))
        return(self._float2index(a))
    
    