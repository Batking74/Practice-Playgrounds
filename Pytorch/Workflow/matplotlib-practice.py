import matplotlib.pyplot as plt
import numpy as np

labels = ['Nazir', 'Aria', 'Nazoro', 'Batking74']
# Plot a line graph of the values in dev_x (x-axis) and dev_y (y-axis)
for i in range(4):
    dev_x = np.random.rand(4)
    dev_y = np.random.rand(4)
    plt.plot(dev_x, dev_y, label=labels[i])

plt.legend()
plt.xlabel('Ages')
plt.ylabel('Median Salary (USD)')
plt.title('Median Salary (USD) by Age')
plt.show()

# This is the entire process on how generative ai can do what it does:
# Forward Propagation is like making a guess (The activation function is apart of the forward propagation process)
# Loss is like checking how wrong the guess/prediction/inference was
# Optimizer Updates the model parameters to reduce the loss	
# Backpropagation is like figuring out which parts of your guess caused the mistake
# Gradient Descent is like correcting your guess based on that info



# A training loop essentially comprises of all the following steps:
# 1. Forward pass ➝ makes a guess  
# 2. Loss/cost/criterion function ➝ checks the guess
# 3. Optimizer Updates the model parameters to reduce the loss	
# 4. Backpropagation ➝ figures out who’s responsible for the mistake by moving backwards through the network to calculate the gradients of each of the parameters of our model with respect to the loss.
# 5. Gradient descent ➝ fixes the weights (learning step)



# Gradient is change in y and change in x so it is a slope. So if you were a machine learning model and you were on top of a hill, and you wanted to get to the bottom of it. Imagine your loss was the height of the hill (start of with your losses really high), and you want to take your loss down to zero. If you measure the gradient (slope) of the hill, the bottom of the hill is in the opposite direction to where the gradient is steep. So the gradient is an incline, but I want my model to move towards the gradient being nothing which would be at the bottom of the hill. One of the ways an optimization algorithm works is it moves our model parameters so that the gradient equals zero, and then if the gradient of the loss equals zero then the loss equals zero to.