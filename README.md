# TrafficFlowInPyTorch

The goal of this project is to provide exposure and serve as an introduction to LSTM.

The dataset is of a highway in Minnesota, the original data set is provided, I split them up and normalized them in train_scaled.csv and test_scaled.csv.


##### PREPARING THE DATA #####

To model time-series data, we need to generate sequences of past values as inputs and predict the next value as the target.
One way to do this is by writing a function and passing in the available data, these need to be converted to pytorch tensors and loaded with dataloader due to their size.

3 main steps:
- Creating sequences with function
- converting to tensors
- loading data

the function will generate (input, output) pairs for the model to train on, where the input is all the data before the last column, and the target is the last column, the traffic volume. 

when loading the data with dataloader, we'll shuffle the training data and keep the test data unshuffled. 


##### Creating The Model ####

main steps: 
- Defining the model
  - Choosing and Activation Function
  - Defining the Loss Function
  - Defining the Optimizer
  - Defining the Forward Pass


 ### Defining the model ###

# Choosing an activation Function #

ReLU is often used in image processing, it passes through positive values, and filters out negative values (weak signals). the non-linearity it introduces is useful in an LSTM model.

# Defining the Loss function #
 
Since this is a regression task there is two options
  - Mean absolute error or L1 Loss:  minimizes outlier impact
  - Mean squared error or L2 Loss: amplifies outlier impact

# Defining the optimizer #

The main three options considered were ADAM, SGD, and AdaGrad

In SGD (stochastic gradient descent) we update all the paramenters for each training example (xi) and the target (yi) individually, instead of computing the gradient of the function with respect to the parameters for the whole training set. It's generally noisier than typical gradient descent as it takes a high number of iterations to reach the minima; even though it requires a higher number of iterations it's still computationally less demanding than typical gradient descent.  

In AdaGrad the objective is to minimize the expected value of a stochastic objective function, with respect to a set of parameters, given a sequence of realizations. It updates the parameters in the opposite direction of the sub-gradients. While standard sub-gradient methods use update rules with step-sizes that ignore the information from teh past observations. Adagrad adapts the learning rate for each parameter individually. 

In ADAM (Adaptive Moment Estimation) the algorithm combines Momentum and RMSP to accelerate tthe gradient descent algorithm by accounting for the 'exponentially weighted average' of the gradients.in RMSP instead of taking the cumulative sum of squared gradients like Adagrad would, it takes the 'exponential moving average' (it's easier to look at the math here)

we go with ADAM.



After creating the model and running it for 2 epochs, at a 0.001 learning rate, we get:

Epoch: 0, Loss: 0.27795905392981113
Epoch: 1, Loss: 0.2779772560343258
average MSE:  0.28222275014017145

which indicates that the model is not learning much after the first epoch, this could be due to small number of epochs or slow learning rate.
Adjusting the learning rate to 0.0001 (reducing it by a factor of 10) for some reason yielded better results:

Epoch: 0, Loss: 0.07314351840028749
Epoch: 1, Loss: 0.02920465974515318
average MSE:  0.020546924569370115



this is still only over 2 epochs, so we'll try 20 epochs and see what we have



