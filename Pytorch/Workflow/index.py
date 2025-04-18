import matplotlib.pyplot as plt
import torch


# test = torch.tensor([50, 30])
# device = 'cuda' if torch.cuda.device_count() else 'cpu'

# test = test.to(device)

# print(f"Running on: {device}")
# print(f"Tensor device: {test.device}")

# print(device)
# print(torch.cuda.memory_allocated(0))
# print(torch.cuda.memory_reserved(0))


# PyTorch Workflow
what_were_covering = {
    1: "Getting data ready (prepare and load)",
    2: "build model",
    3: "fitting the model to data (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model",
    6: "putting it all together"
}

# torch.cuda.empty_cache()

# print(torch.cuda.memory_allocated(0))

# Data (Preparing and loading) - Data can be almost anything... in machine learning (E.x Excel spreadsheet, images of any kind, videos, audio like songs or podcasts, DNA, text)

# Machine Learning is a game of two parts:
# 1. Get data into numerical representation
# 2. Build a model to learn patterns in that numerical representation
# A parameter is something that a model learns

# Create known parameters
weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
# weights = random numbers; X = inputs (image, audio, etc)
y = weight * X + bias



# Creating my training data
train_split = int(0.8 * len(X))
# train variables gets all input data up to the index train_split (40)
# the training variables X_train and y_train are my sample data!
# train_split and train_set mean the same thing!
X_train, y_train = X[:train_split], y[:train_split]

# test variables gets the rest of the input data starting from train_split (40) to the end
X_test, y_test = X[train_split:], y[train_split:]

print(len(X), len(y))

print(train_split)







def plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=None):
    plt.figure(figsize=(10, 7))
    
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    
    # Plot Test data in green
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')
    
    
    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
        
    plt.legend(prop={'size': 14})
    # plt.show()
    
    
plot_predictions()


# What my model does
# 1. Start with random values (weight & bias)
# 2. My model will look at the training data (input) and adjust the random values to better represent (or get closer to) the ideal values (input).


# Build Modeling Ai Model: Initializing model with random parameters 
class LinearRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initializing Model Parameters
        self.weights = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
        self.bias = torch.nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: # 'x' is input data such as training data.
        return self.weights * x + self.bias


# Gradient Descent
# Backpropagation




torch.manual_seed(42)

# Instantiating Model
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))

# Make predictions with model. Note: inference_mode disables gradient tracking that pytorch does behind the scenes. The benefit of using inference_mode is that PyTorch behind the scenes will keep track of less data, which will make the models predictions potentially a lot faster because a whole bunch of numbers won't be kept track of/stored in memory. This is a good practice when you want your model to just make predictions/inferences (not train), especially for large dataset. So in essence, it disables all of the useful things that are good for training to make your code faster.
with torch.inference_mode():
    y_preds = model_0(X_test)
    
    print(y_preds)
    print(y_test)
    
    plot_predictions(predictions=y_preds)
    plt.show()

# Setup loss function
loss_fn = torch.nn.L1Loss()

# Setup an optimizer (Stochastic Gradient Descent)
# The optimizer starts by randomly adjusting the weights and once it has found some random values or steps that have minimized the loss value, its going to continue adjusting them in that direction. So say it says, if i increase the weights, it reduces the loss. So its going to keep increasing the weights until the weights no longer reduce the loss. say it continues that for a while and it finds out if it increases the weights anymore, the loss is going to go up. The optimizer will then stop there and the same thing happens with the bias.
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01) # lr = Learning Rate (The higher the learning rate, the more the optimizer will adjust each parameter (weights & bias) in one hit. if i have a weight parameter thats 0.4116 and the learning rate is 0.01 it will subtract 0.01 everytime until the Mean Absolute Error starts to increase. When it realizes its increasing it will stop. Summary: The smaller the learning rate, the smaller the change in the parameter)