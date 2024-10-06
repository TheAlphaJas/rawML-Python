# rawML

rawML is a hobby project where neural networks are implemented from scratch using pure Python and Numpy, with a class-based structure to define layers, optimizers, and loss functions. The goal is to implement how neural networks work at a low level, and create a (somewhat) modular custom code framework to implement ML algorithms, without relying on high-level deep learning frameworks like TensorFlow or PyTorch.
rawML is a hobby project where neural networks are implemented from scratch using pure Python and Numpy, with a class-based structure to define layers, optimizers, and loss functions. The goal is to implement how neural networks work at a low level, and create a (somewhat) modular custom code framework to implement ML algorithms, without relying on high-level deep learning frameworks like TensorFlow or PyTorch. In a nutshell, my lite version of PyTorch/TF, powered by NumPy.


### Overview of current implementations:

- **Linear Layers**: Fully connected layers with He-initialization of weights.
- **Activation Functions**: ReLU activation implemented using numpy.
- **Custom Tensor Class**: `jTensor`, an extension of Numpy's ndarray, supports storing gradients in the `.gd` attribute.
- **Optimization**: Gradient Descent Optimizer (`GDOptimizer`) is implemented with learning rate control.
- **Loss Function**: Mean Squared Error (MSE) Loss is implemented to compute the loss during training.
- **Model Class**: The `CreateModel` class stiches all layers together, providing methods for forward passes and training with backpropagation. I plan on making it more customizable.
- **Other basic functionality**: General essential functions, like mean, min, max, std, rand, randn etc are implemented in the rawML library. All are powered by NumPy.

### Requirements
just numpy :)
```
pip install numpy
```

### Usage Instructions
Its available on PyPI!
```
pip install rawML
```


### Code Example
The getting_started.ipynb will show a brief overview on rawML, and how to use it as an high level API.
The following code is an example of the same.
```
import rawML as rML
from rawML.layers import relu, linear
from rawML.optimizers import GDOptimizer
from rawML.losses import MSELoss
from sklearn.model_selection import train_test_split as tts

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
```
- As jTensors inherit(and behave very similar to) from numpy arrays, they support operations like .shape, and they can also be fed into scikit-learn's train test split

### Future Implementations
This project is at a very initial stage, and I aim to expand it further. I will add implementations of more Optimizers, Loss functions along with other layers like MaxPool2D, CNNs. 
There is no implementation of the concept of "batch size" which will be added very soon.
An easier way to add more custom metrics will also be implemented into the model.train() method.
Verbose control will be added
Currently, the CreateModel class is restrictive to a sequential NN, which I plan on changing by implementing a more "functional" NN, to make more complex architectures like skip connections etc. The further aim to implement the famed UNet architecture using RawML.
I also plan to explore GPU acceleration possibilities by migrating to CuPy instead of NumPy
and etc...

(PS, there exists an rML.about())
