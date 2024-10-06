class GDOptimizer:
    def __init__(self,lr):
        self.lr = lr
    def update(self,tup):
        w,b = tup
        if (w is None):
            return None,None
        return (w - self.lr*w.gd, b - self.lr*b.gd)
    def get_lr(self,):
        return self.lr
    def update_lr(self,lr):
        self.lr = lr