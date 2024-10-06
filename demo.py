import rawML as rML
from rawML.layers import relu, linear
from rawML.optimizers import GDOptimizer
from rawML.losses import MSELoss
from sklearn.model_selection import train_test_split as tts

# b = 16 #batch_size

LayerList = [
    linear(100,20),
    relu(),
    linear(20,40)
]

opt = GDOptimizer(lr = 1e-2)
loss = MSELoss()
model = rML.createModel(LayerList, opt, loss)

X = rML.rand((16, 100))
Y = rML.rand((16, 40))

##sklearn train-test-split works with jTensors
x_train, x_val, y_train, y_val = tts(X,Y,train_size=0.8)

#Training
model.train((x_train,y_train),(x_val,y_val),epochs=20,verbose_freq=5)

#Predicting
y = model(rML.randn([40,100]))
print(y.shape)


