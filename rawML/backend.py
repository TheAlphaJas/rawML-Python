import numpy as np

class jTensor(np.ndarray):
    def __new__(cls, ip_arr, gd=None):
        obj = np.asarray(ip_arr).view(cls).astype(np.float64)
        obj.gd = gd
        return obj

    
class createModel:
    def __init__(self, layers_list, optimizer, loss_fn):
        self.layers_list = layers_list
        self.optimizer = optimizer
        self.loss = loss_fn
    def __call__(self,x_in):
        x_out = x_in
        for l in self.layers_list:
            x_out = l(x_out)
        return x_out
    def train(self, xytrain, xyval, epochs=1, verbose_freq = 1):
        x_in, y_true = xytrain
        x_val, y_val = xyval
        # assert (len(x_in) == len(y_true)) "Input vector and Output vector batch size should be equal!"
        for i in range(epochs):
            x_list=[]
            x_list.append(x_in)

            #Train loop
            for l in self.layers_list:
                y = l(x_list[-1])
                x_list.append(y)

            #Val loop
            yv = x_val
            for l in self.layers_list:
               yv = l(yv)

            if (i%verbose_freq == 0):
                print("e:{} | trainLoss: {:.5f}, valLoss: {:.5f}, optLr: {}".format(i+1,mean(self.loss(x_list[-1],y_true)),mean(self.loss(yv,y_val)),self.optimizer.get_lr()))
            
            #Backprop
            self.loss.backprop(x_list[-1],y_true)
            for i in range(len(self.layers_list)):
                self.layers_list[-i-1].backprop(x_list[-i-2],x_list[-i-1])
            #Update
            for l in self.layers_list:
                l.update_wb(self.optimizer.update(l.get_wb()))
            x_list.clear()

def rand(shape):
    return jTensor(np.random.rand(*shape))

def randn(shape):
    return jTensor(np.random.randn(*shape))

def zeros(shape):
    return jTensor(np.zeros(shape))

def zeroes(shape):
    return jTensor(np.zeros(shape))

def about():
    print("rawML/rML is a simple numpy-powered Machine Learning library. It allows one to go into depth of ML fundamentals (by examining the code-base), to understand and appreciate how things are built \"from scratch\". It is currently under active development, and more layers, functions etc will be added in due time. It is made as a hobby project, but needless to say this should not undermine its use. One can easily develop, train and evaluate simple models using rML. Developed with ♥ by jas [Github:TheAlphaJas]")

def mean(*args):
    return jTensor(np.mean(*args))

def max(*args):
    return jTensor(np.mean(*args))

def min(*args):
    return jTensor(np.mean(*args))

def std(*args):
    return jTensor(np.mean(*args))