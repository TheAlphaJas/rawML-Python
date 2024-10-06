import numpy as np
from math import sqrt
from .backend import *
    
class linear:
    def __init__(self, n_in, n_out):
        self.w = randn((n_in, n_out))*sqrt(2.0/n_in) #He initialization
        self.b = zeros((1,n_out))
    def __call__(self,x):
        return (np.dot(x,self.w) + self.b)
    def get_wb(self,):
        return (self.w,self.b)
    def backprop(self,x_in,x_out):
        x_in.gd = np.dot(x_out.gd,self.w.T)
        self.w.gd = np.dot(x_in.T,x_out.gd)
        self.b.gd = x_out.gd.sum(0)
    def update_wb(self,tup):
        w_new,b_new = tup
        self.w = w_new
        self.b = b_new
    def show_wb(self,):
        print("Weight Matrix :-")
        print(self.w)
        print("Bias Matrix :-")
        print(self.b)

class relu:
    def __init__(self):
        self.w = None
        self.b = None
    def __call__(self,x):
        return np.clip(x,a_min=0.0, a_max = None)
    def get_wb(self,):
        return (None,None)
    def backprop(self,x_in,x_out):
        temp = x_in
        temp[temp>0]=1.0
        temp[temp<=0]=0.0
        # dL/dx_in = dL/dxout * dxout/xin
        x_in.gd = (x_in > 0).astype(np.float64)*x_out.gd
    def update_wb(self,tup):
        w_new, b_new = None,None
    # def 