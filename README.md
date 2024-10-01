# RawML

RawML is a small hobby project where neural networks are implemented from scratch using pure Python and Numpy, with a simple class-based structure to define layers, optimizers, and loss functions. The goal is to implement how neural networks work at a low level, and create a (somewhat) modular custom code framework to implement ML algorithms, without relying on high-level deep learning frameworks like TensorFlow or PyTorch.

## Project Structure

- **Python Folder**: Contains a Jupyter notebook where neural networks are built from scratch using only Numpy and Python's math library. This folder demonstrates how to implement fundamental components of a neural network, including layers, activation functions, optimizers, and loss functions.
  
- **C++ Folder**: Currently empty but will contain future C++ implementations of neural network components.

### Overview of current implementations:

- **Linear Layers**: Fully connected layers with He-initialization of weights.
- **Activation Functions**: ReLU activation implemented using numpy.
- **Custom Tensor Class**: `jTensor`, an extension of Numpy's ndarray, supports automatic differentiation by storing gradients in a `.gd` attribute.
- **Optimization**: Gradient Descent Optimizer (`GDOptimizer`) is implemented with learning rate control.
- **Loss Function**: Mean Squared Error (MSE) Loss is implemented to compute the loss during training.
- **Model Class**: The `CreateModel` class stiches all layers together, providing methods for forward passes and training with backpropagation. I plan on making it more customizable.


### Code Example
The following is an example on how one could implement a simple NN purely using RawML defined classes.

```python
import numpy as np
from math import sqrt

# Create layers for the model
layers_list = [
    LinearLayer(50, 30, 10),  # Input layer with 50 inputs, 30 outputs. Third argument is batch size. It has to same across all layers
    ReluLayer(),              # Activation layer
    LinearLayer(30, 40, 10)   # Output layer with 30 inputs, 40 outputs
]

# Define optimizer and loss
optim = GDOptimizer(lr=1e-2)
loss = MSELoss()

# Create the model
model = CreateModel(layers_list, optim, loss)

# Generate random input data (X_in) and true labels (Y_true)
X_in = jTensor(np.random.rand(10, 50))  # Batch size 10, input dimension 50
Y_true = jTensor(np.random.rand(10, 40))  # Batch size 10, output dimension 40

# Forward pass and training
y = model(X_in)
print(f"Output shape: {y.shape}")
model.train(X_in, Y_true, epochs=300)
```

### Future Implementations
This project is at a very initial stage, and I aim to expand it further. I will add implementations of more Optimizers, Loss functions along with other layers like MaxPool2D, CNNs. 

Currently, the CreateModel class is restrictive to a sequential NN, which I plan on changing by implementing a more "functional" NN, to make more complex architectures like skip connections etc. The further aim to implement the famed UNet architecture using RawML.

All implementations will be done in Python as well as C++.
