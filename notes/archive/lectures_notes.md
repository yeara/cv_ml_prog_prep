Fei Fei Li Notes

ImageNet
CIFAR10

Training - expensive
Test - should be cheap 

So for example, NN labeling based on pixel diff is bad, because training is cheap and evaluation is expensive
Also suffers from: tint, shift, do not correspond well to perceptual similarities

L1 distance depends on coordinate system 

For testing hyperparamters (for example, K-nearest neighbors, etc):
- train, validate and test data tests. Hyperparamters are optimized on the validation dataset

Cross validation - 
used for small datasets, less for deep learning
data is split into folds
hyperparameters are averaged over the different fold choices
higher confidence on which hyperparameters are used

Data sets should probably be created with the same probabilistic distribution

The curse of dimensionality:
\# of training examples is exponential to the # of pixels for examples, if we use pixel dist as the descriptor

kNN is a parametric model

## Linear Classification

image(x) -> f(x,W) -> 10 numbers of scores per class
W are parameters, so this is what we're learning

deep learning - how to get this function f

linear classifier - f = Wx + b 
b is the bias term, which can give class preferences for one class over another, can be used to unbias an unbalanced dataset

We can visualize the linear classifier to have some idea for template matching

Linear classifier only learn one appearance for each class. It does not deal with variation in class appearance

linear classifier try to create a linear barrier in a high dimension space between classes	

Examples where linear classifiers fail:
multi model data
one class that can appear in different areas of space
for example: 1 < l2 < 2


## Loss Functions

The loss function measures how much error we have in our classification

### Multi Class SVM - generalization of binary SVM 

Loss is 0 if the incorrect label is lower than correct label + safety margin
or wrong_score - correct_score + safety_margin
 = max(0, s_wrong - s_correct + safety)
This is summed only over the wrong categories

Also known as hinge loss

After initialization W is small, initialize by setting all params to more or less similar: 
so the first loss should be approx. C-1 where C is the number of categories

squared hinged loss will create a different classifier because the trade off between good and bad scores is different

hinge loss - we don't care if it's a little bit wrong or very wrong, but squared hinge loss heavily penalizes outliers

The loss function is how we design which errors the algorithm cares about

If we have loss = 0, the classifier is not unique, should be scalable 

Also the loss should not depend only on fitting the training data.

This is solved by regularization, which encourages the model to use a "simple" W
What is simple? that depends on the regularization term which you choose

## Multinomial Linear Regression

Now the scores actually have meanings, they're the unnormalized log probabilities of the classes

softmax function - exp of the score ensures positivity
$ P(Y=k|X=x_i) = \frac{e^{s_k}}{\sum_j e^{s_j}} $

Want to maximize log likelihood, minimize the negative log likelihood of the correct class
So the loss function is:
$ L_i = - log P(Y=y_i|X=x_i) $
minus log of the probability of the true class
$ L_i = - log \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} $

## Multinomial Logistic Regression

To get perfect loss function, we need infinite scores for the correct class and minus infinity scores for the wrong class

min loss is 0 
max loss is infinity

when all classifiers are ~0, the first iteration should be log(c)

cross entropy loss
- log probability of the correct class

softmax wants to push the probability towards 1.0 (or score toward infinity) and other class towards negative inifinity. so even small changes of the scores of the classes will change the loss function

score function vs. loss function

Computing finite differences is terrible idea, because very slow due to high dimensions of the function

Numeric gradients are error prone.
Always validate with finite differences. 


Gradient Descent 

Step size or learning rate is one of the most important hyper parameters

Stochastic gradient descent

Momentum method

ADAM optimizer

Computing the loss might be very expensive if we use all of the examples in our training / validation sets

The analytic gradient of the loss is very slow to compute

Stochastic Gradient Descent - at every step compute a minibatch which is a 2^n examples
Estimate of the true gradient


## Back Propagation

Once we  can express something as a computation graph we can use a technique called back-propagation

The loss is at the bottom

Add gate - gradient distributor 
Max gate - gradient router
Mul gate - gradient switcher

When one node is connected to multiple nodes, the gradients are added at this node.

\frac{\partial f}{\partial x} = \sum_i \frac{\partial f}{\partial q_i}\frac{\partial q_i}{\partial x} 

So now we do this for vectors, Jacobians and Hessians:

The size of the Jacobian is the dimension of the input vector times the dimension of the output vector. for minibatches the jacobian is even larger - 400kx400k easy... 

If the gates are element wise, the Jocobian is a diagonal matrix

Exercise: What's the gradient of the L2 norm? 

Sigmoid function: 
sigma (x) = \frac{1}{1-e^{-x}}

Exercise: break down the sigmoid function into gates
Compute the back propagation

The gradient with respect to a variable should have the same shape as the variable

Activation functions: 
Sigmoid
tanh()
ReLU
Leaky ReLU
Maxout
ELU

How to Build Neural Networks

Fully connected linear layers - all outputs of one layer are connected to all inputs of a second layer
The abstraction of layer allows for matrix - vector operations
 

 ## Lecture 5

 Krizhevsky 2012 - first use of modern nn on imagenet that generated good results


 1980s - Experiments on Vision in Cats: 
 - nearby regions in vis cortex represent nearby region in vis field
 - neurons had heirarchical org 

 Simple cells - response to light and orientation 
 Complex cells - light, orientation and movement 
 Hypercomplex - movement with an end point

 Complex cells sort of do pooling


1998 - gradient based learning applied to document recognition

2012 - modernized version
dense connections 
max connections between layers

Fully Connected Layers:
Image 32x32x3 -> stretch out all pixels to a single vector x = 1x3072 
Layer W 10x3027
Wx = activation per "neuron" -> 1x10

Convolution Layer:
Preserve spatial structure: 
32x32x3 x 
and the filter is going to be 5x5x3 
compute dot product at every spatial location

Filters always extend the full depth of input volume

The dot product is called activation - which is the size of the image - filter + 1

usually: 
CNN
Pool
CNN
...
Fully connected layer to compute max scores

Stride size - the interval of sampling locations

Stride and image size need to match - 
(N - F) / Stride + 1 

\# of parameters per filter - size x x size y x depth

It is common to pad the borders to get the appropriate size 

also known as recepetieve field

### Pooling layers

make representation smaller and more manageable 
pooling is only spatial, not over depth
max pooling is commonly used

three hyperparamters:
spatial extent 
stride 

produced w2 x h2 x d1
w2 = (w1 - f)/s + 1
h2 = (h1 - f)/s + 1
d2 = d1

introduced zero parameters
zero padding isn't usually used

common parameters:
f=2, s=2 
f=3, s=2

### Fully Connected Layer

## Lecture 6 

Activation Functions
Sigmoid - \frac{1}{1+e^-x}
maps all variables to [0,1]
interpetation - saturating firing rate of a neuron 
