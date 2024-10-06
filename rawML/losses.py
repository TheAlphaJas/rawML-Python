class MSELoss:
    def __call__(self,x,xt):
        return ((x-xt)**2)/len(x)
    def backprop(self,x,xt):
        x.gd = (2*(x-xt))/len(x) #dLoss/dx