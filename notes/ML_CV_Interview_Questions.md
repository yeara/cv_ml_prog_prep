# ML & CV Interview Questions

## Other Resources

https://github.com/andrewekhalel/MLQuestions

# Review Questions

###Coding Math

Implement SQRT(const double & x) without using any special functions, just fundamental arithmetic

Compute the dot product of two large sparse vectors

Imp. a data structure that allows you to iterate through a vector of sorted vectors

How do you make code go fast?

Implement [connected components](http://aishack.in/tutorials/labelling-connected-components-example/) on an image/matrix. I've been asked this twice; neither actually said the words "connected components" at all though. One wanted connected neighbors if the values were identical, the other wanted connected neighbors if the difference was under some threshold.

How would you implement a [sparse matrix](https://en.wikipedia.org/wiki/Sparse_matrix) class in C++? (On-site) Implement a sparse matrix class in C++. Implement a dot-product method on the class.

How do you average two integers without overflow?

### General ML

What's regression?

What's normalized cross validation?

What's normalized cross correlation? 

What's logistic regression?

What is the appropriate use case for generative modeling? 

Supervised vs. unsupervised learning

Regression vs. classification

Explain the tradeoffs between overfitting and regularization, L1 and L2 loss

How does bias affect the model?

### Image Processing

How do you efficiently compute integral images?

Create a function to compute an [integral image](https://en.wikipedia.org/wiki/Summed-area_table), and create another function to get area sums from the integral image.

How do you rotate an image 90 degrees most efficiently if you don't know anything about the cache of the system you're working on?

Implement voronoi clustering

What's plane sweep?

Implement non maximal suppression as efficiently as you can.
Implement a circle drawing function

### CV

How would you [remove outliers](https://en.wikipedia.org/wiki/Random_sample_consensus) when trying to estimate a flat plane from noisy samples? 

Given n correspondences between n images taken from cameras with approximately known poses, find the position of the corresponding 3D feature point.

How do you most precisely and reliably find the pose of an object (of a known class) in a monocular RGB image?

"How does loop closure work?" Followup: In a SLAM context, for do you make loop closure work even with very large baselines? (Say, anti-aligned views of the same scene.)

How  does stereo matching work?

What's the difference between the fundamental and essential matrices?

You're in deep space with a star atlas and a calibrated camera. Find your orientation?

How do you build a model of a table from RGBD images?

### Deep Learning

Draw  sigmoid gate response function, draw the computational graph and compute the backprop.

What's the difference between ReLU/Leaky ReLU?

Given stride and kernel sizes for each layer of a (1-dimensional) CNN, create a function to compute the [receptive field](https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network) of a particular node in the network. This is just finding how many input nodes actually connect through to a neuron in a CNN.

What's max pooling? Soft vs hard max pooling?

### Misc

1. How does [CBIR](https://www.robots.ox.ac.uk/~vgg/publications/2013/arandjelovic13/arandjelovic13.pdf) work?
2. How do you create concurrent programs that operate on the same data without the use of locks?
3. Reverse a bitstring.
4. "Talk to us about rotation."
5. [Project Euler #14.](https://projecteuler.net/problem=14)
6. Reverse a linked list in place.
7. What's a convolution?



