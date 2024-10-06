from . import *
import numpy as np

class MSELoss:
    def __call__(self,x,xt):
        l = ((x-xt)**2)/len(x)
        return (np.mean(l, axis=0)[np.newaxis,...])
    def backprop(self,x,xt):
        x.gd = (2*(x-xt))/len(x) #dLoss/dx
        x.gd = np.mean(x.gd, axis=0)[np.newaxis,...]
        print("X GD SHAPE ",x.gd.shape)