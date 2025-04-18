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

# Forward Propagation is like making a guess (The activation function is apart of the forward propagation process)
# Loss is like checking how wrong the guess/prediction/inference was
# Optimizer Updates the model parameters to reduce the loss	
# Backpropagation is like figuring out which parts of your guess caused the mistake
# Gradient Descent is like correcting your guess based on that info




# 1. Forward pass ➝ makes a guess  
# 2. Loss/cost/criterion function ➝ checks the guess
# 3. Optimizer Updates the model parameters to reduce the loss	
# 4. Backpropagation ➝ figures out who’s responsible for the mistake  
# 5. Gradient descent ➝ fixes the weights (learning step)