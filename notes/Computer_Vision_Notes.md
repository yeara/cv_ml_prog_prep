# Computer Vision & Machine Learning

Disclaimer: these are my personal notes reviewing material for CV/M interviews. The notes contain text from Quora, as well as screengrabs from various sources including Andrew Ng's Machine Learning Course on Coursera, Wikipedia, The Computer Vision Course on Coursera, and other sources. Reproduced without permission. The notes are incomplete, work in progress, and may contain mistakes.

#### CV Interview Topics

Geometric vision - geometric transforms,projective geometry, homography, stereo vision, epipolar geometry, fundamental and essential matrices, geometric calibration, triangulation

Photometric vision - filtering, convolution, denoising , deblurring etc

Semantic vision - concepts of boosting, neural nets, svm,  image descriptors etc

#Geometric Vision

##Geometric Transformations

#### Scaling 

$\left(  \begin{matrix} s_x & 0 \\ 0 & s_y  \end{matrix} \right)$

#### Rotation

Symmetry group 

SO3 vs . SE3

Reflection

#### Translation

Only possible in homogeneous coordinates

## Projective Geometry

### Homogeneous Coordinates

$x = \left[\begin{matrix} u \\ v\\ 1 \end{matrix}\right]$ pixel space position
$ x_w = \left[ \begin{matrix} x_w \\  y_w \\ w_t\\  1 \end{matrix} \right] $ world space position

Projection matrix - from  world position $x \in \mathbb R^4 \to p \in \mathbb R^3$ pixel homogenous coordinates

A point in an image projected to the world is known up to a scale factor (that's why we have this 1 at the bottom of the vector)

#### Homography

Relates planar images

### Pinhole Camera Model

The camera matrix projects from the world space to image space. Using homogenous coordinates allows for translation


### Camera Intrinsic $K$

$$ K = \left( \begin{matrix} 
\alpha_x & \gamma & u_0 & 0 \\
0 & \alpha_y  & v_0 & 0 \\
0 & 0 &  1 & 0 \\
\end{matrix}
\right) $$

5 intrinsic parameters
 $f$ - focal length
image sensor format - $m_x$ and $m_y$ are the pixel dimensions
$\gamma$ skew coefficient between x-y
$(u_0,v_0)$ *principal point*, usually in the middle of the sensor

$\alpha_x = f m_x $
$\alpha_y = f m_y $

This model cannot represent lens distortion

### Camera Extrinsic Parameters $R,T$

$$ \left[ \begin{matrix} 
R_{3x3} & T_{3x1} \\
0_{1x3} & 1 \end{matrix}
\right] $$

The extrinsic parameters define the position of the camera center and the camera's heading in world coordinates. 

$T$ is the position of the origin of the world coordinate system expressed in coordinates of the camera-centered coordinate system. 

The actual camera position in world coordinates is $C = -R^{-1}T=-R^{T}T $

### Disparity

Definition: difference in image location of the same 3D point between stereo images

<img src="/Users/kozlovy/Documents/2019_JobApplications/Notes/cv/disparity.png" alt="disparity" style="zoom:33%;" />

Baseline - dist. between camera centers

- $f$ - focal length
- $d$ = disparity between the points
- $z$ = dist from object

From triangle similarity we get:

$\frac{B}{z} = \frac{d}{f}$

Looking at this relationship in depth:

${d}=\frac{Bf}{z} $

$\frac{dd}{dz}=-\frac{Bf}{z^2}$

$dd = \frac{f}{B}dz$

$\Delta z = \frac{Z^2}{Bf}dd$

Depth resolution is better when the camera is closer to the objects.

The disparity between two points in a stereo pair is inversely proportional to their distance from the observer.

## Stereo Vision

Two cameras, $C_L$ and $C_R$

Their respective optical centers $O_L$ and $O_R$

The world space point $X$ and its projection in each one of the cameras: $x_L, x_R$

#### Epipolar Point

The epipolar points $e_L, e_R$ are defined as the points where the baseline intersects each one of the camera images, or the center of each camera as projected into the other cameras image.

#### Epipolar Line

A line segment between the epipolar point and the projection of the X on the image.

A second epipolar line segment is the projection of this line onto the first camera image plane

A point in one image generates a line in the other on which its corresponding point must lie

#### Epipolar Plane

A plane that passes through both camera centers.

### Essential Matrix

For two points in two images of a camera stereo pair which correspond to the same $3D$ world position, the following is true: 
$\mathbf{y}^{\prime T}\mathbf{Ey}=0$

This relates the two calibrated cameras.

The essential matrix can be seen as a precursor to the fundamental matrix. 

The essential matrix can only be used in relation to calibrated cameras, it requires known intrinsic camera parameters (for normalization).

The essential matrix can be useful for determining both the relative position and orientation between the cameras and the 3D position of corresponding image points.

$\mathbf{E}=\mathbf{RS}$

where 

$\mathbf{S} = \left| \begin{matrix} 0 & -T_z & T_y \\ T_z & 0 & -T_x \\ -T_y & T_x & 0  \end{matrix} \right|$

weird transformation matrix with 3 DoF and R also has 3 DoFs

rank 2 

Has both left and right nullspaces

Depends only on extrinsic parameters

### Fundamental matrix

$\mathbf F \in \mathbb{R}^{3\times3} = \mathbf{M}_r\mathbf{RSM}_l^{-1}$ 

The Fundamental matrix relates corresponding points in stereo images.
$\mathbf{X}^{\prime T}\mathbf{Fx}=0$ and $\mathbf{F}_{3x3}$

Analogous to essential matrix. The fundamental matrix relates pixels (points) in each image to epipolar lines in the other image.

It is related to the essential matrix $\mathbf{E} = \mathbf{K}^{\prime T} \mathbf{FK}$ where $\mathbf{K}^{\prime}, \mathbf{K}$ are in the intrinsic matrices of both cameras.

$rank(\mathbf{F}) = 2$

### Multiple View Geometry - Single Center of Projection

Three camera views are related via a trifocal tensor 

Having multiple cameras close together results in better depth resolution, less noise, etc.

###Rectification

Pre-warping images such that the corresponding epipolar lines are coincident

For a rectified image pair:

- All epipolar lines are parallel to the horizontal axis of the image plane
- Corresponding points have identical vertical coordinates.

Rectification can be done for image pairs, but may prove impossible for a collection of random cameras, unless they are "parallel" of some sort

How to compute rectification?

1. Rotate both cameras s.t. they're perpendicular to the line connecting both camera centers - using the smallest rotation possible and relying on the freedom of tilt. 
2. To determine the desired twist around the optical axes, make the up vector perpendicular to the camera center line -> the corresponding epipolar lines are horizontal and the disparity for points at inifinity is 0. 
3. Rescale images if necc.

Then the pixel matching can be done for a single dimension on every scanline - reduces the dimensionality of the problem to 1D search

#### How does it look in math?

Assuming one camera is K = [ I 0]

/TODO

### Assumptions for Stereo Matching

- Small baseline
- Most scene points are visible in both images
- image regions are similar in appearance

Left view images will move to the left in the right image - optimization

### Similarity Metrics for Patch/Line Matching

#### Squared Patch Distance

$MSD =  \frac{1}{2xy}\sum_{i,y}\left|P_{x,y}^{(i)} - P^{(j)}_{x,y}\right|^2$

#### Normalized Cross Correlation

Consider two real valued functions  $f,g$  differing only by an unknown shift along the x-axis (i.e. disparity). One can use the cross-correlation to find how much $g$ must be shifted along the x-axis to make it identical to $f$ 

The probability density of the difference $Y-X$ is formally given by the cross-correlation.

The formula essentially slides the $g$ function along the x-axis, calculating the integral of their product at each position.

$NCC = $

##### Zero-normalized cross-correlation

For image-processing applications in which the brightness of the image and template can vary due to lighting and exposure conditions, the images can be first normalized

##### Correlation vs. Convolution

The difference is in how the summarization iterates over the elements.

#### Total Variation

TODO

### Homography

## Geometric Calibration

### Camera Calibration

If we have an image of a known object we can calibrate easily.

### Triangulation

Questions: How do you translate cameras?

### Eight Point Problem

https://en.wikipedia.org/wiki/Eight-point_algorithm

# Stereo Reconstruction Pipeline

### Stereo Photogrammetry

Small vs large baseline:

robust binocular stereo
point matching
adaptive point-based filtering of the
merged point clouds, and efficient, high-quality mesh generation.

#### Bundle Adjustment

Bundle adjustment amounts to jointly refining a set of initial camera and structure parameter estimates for finding the set of parameters that most accurately predict the locations of the observed points in the set of available images. 

Input: $n \; 3D$ points, $m$ views, $x_{ij}$ is the projection of theh $i$th point on image $j$. $v_{ij}$}

![\displaystyle v_{{ij}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/c3d1bb3a51f8bafc07c30a99c3f3f15e008d0259) denote the binary variables that equal 1 if point $i$ is visible in image $j$. Assume also that each camera $j$  is parameterized by a vector $\mathbf {a} _{j}$ and each 3D point $i$ by a vector $\mathbf {b} _{i}$ . 

Bundle adjustment minimizes the total reprojection error with respect to all 3D point and camera parameters, specifically

$$\min _{{{\mathbf  {a}}_{j},\,{\mathbf  {b}}_{i}}}\displaystyle \sum _{{i=1}}^{{n}}\;\displaystyle \sum _{{j=1}}^{{m}}\;v_{{ij}}\,d({\mathbf  {Q}}({\mathbf  {a}}_{j},\,{\mathbf  {b}}_{i}),\;{\mathbf  {x}}_{{ij}})^{2}$$

where $\mathbf {Q} (\mathbf {a} _{j},\,\mathbf {b} _{i})$ is the predicted projection of point $i$ on image $j$ and $d(\mathbf {x} ,\,\mathbf {y} )$  denotes the Euclidean distance between the image points represented by vectors $\mathbf{x,y}$

Bundle adjustment is tolerant to missing image projections 

Minimizes a physically meaningful criterion

This is typically solved using Levenberg–Marquardt Algorithm

When solving the minimization problems arising in the framework of bundle adjustment, the normal equations have a [sparse](https://en.wikipedia.org/wiki/Sparse_matrix) block structure owing to the lack of interaction among parameters for different 3D points and cameras.

#### Surface Reconstruction

- Match points and compute depth field
- Approximate normals
  - By for example, approximating the planarity from the point neighborhood
- Reconstruct surface - for example, fit planes, or Poisson surface reconstruction

## Point Cloud Merge

ICP for point cloud matching

Normal Estimation

Outlier detection / removal 

Surface / mesh fitting / template fitting

### Visual Hull

## Optic Flow

### Lukas-Kaande

### Brox

Gradient consistency assumption + intensity consistency assumption

Iterative multi scale + warping

Uses an analytic formulation derived from Euler-Lagrange Equations

Results in a dense optic flow field.

Works well for small changes.

# Geometry Processing 

## ICP

Algorithm:

## Laplacian Deformation

### Cameras

Sensor RGB Pattern - Bayer pattern

Noise

# Image Processing

### Integral Images

The value at any point (*x*, *y*) in the summed-area table is the sum of all the pixels above and to the left of (*x*, *y*), inclusive where $i(x,y)$  is the value of the pixel at (x,y). The summed-area table can be computed efficiently in a single pass over the image:

$I(x,y) = i(x-1,y-1) + I(x,y-1) + I(x-1,y)-I(x-1,y-1)$

and similarly for any rectangular region:

$ i(A,B,c,D) = I(D) - I(B) - I(C)+I(A)$

[![img](https://upload.wikimedia.org/wikipedia/commons/thumb/5/58/Summed_area_table.png/220px-Summed_area_table.png)](https://en.wikipedia.org/wiki/File:Summed_area_table.png)



## Image Descriptors

#### SIFT

#### HOG

####RGB Historgram

### Edge Detection

Cascade Detectors

###Harris Corner Detection

Patches

## POI / Other Features

## Patch Match

## Filtering

### Convolution

### Denoising

Median/mean filtering

### Deblurring

### Gaussian Pyramids

## Up and Down Sampling

Nearest neighbor
Linear
Bi-linear

## Noise Models

Pepper / black white noise
Gaussian noise
Blue noise

# Semantic Computer Vision

Visual Odometry

Sillouhette segmentation

##Optic Flow

##Classification Problems

## SfM

 SfM approaches often have to work on an unordered set of images often computed in the cloud with little to no time constraints and might employ different cameras

One of the challenges in SfM is to retrieve near-by images, and add images one by one to a growing graph while accounting for potential outlier images so that robust reconstruction maybe performed. (i.e. that MS/GOOG Building  Rome in a Day also talked a lot about system design)

## SLAM

SLAM uses scene matches over the previous N frames to estimate camera pose and 3D keypoint locations. There are different algorithms that can do this task:

- Kalman filtering
- Particle filtering
- Bundle Adjustment.

Visual SLAM is supposed to work in real-time on an ordered sequence of images acquired from a fixed camera set-up. Large scale visual SLAM is typically restricted to trajectories of a few kilometers.

FSM - structure. not necc. coherent map

SLAM - structure + map

SLAM is more complete than BA/SFM since SLAM provides 3D structures, camera localization (the L of SLAM) and mapping.

SLAM there isn’t enough computational budget to run BA on all frames, and the time constraints are tough. SLAM approaches try to cut corners when it comes to feature descriptors and matching, really every stage of the pipeline, to ensure real-time performance in a budget. 

The matching problem is easier, as the  neighboring images are expected to heavily overlap with each other and are known (and sequential).

##### Depth Map Matching

A common approach is to use the iterative closest point algorithm to align the sequential depth maps to the previous one (or to the map), which works when the frame-rate is high enough to expect overlap between every depth-map and for the initialization to be close enough to the correct answer to converge. This way all correspondence are computed at once in 3D space, without difficult search associated with feature matching between RGB images.

##### Loop Detection/Closure

Recognizing features/structures that are already seen. This is used to correct camera's trajectory when it comes back to its starting point and minimize drift.

### Kalman filters

The **Kalman filter** is an efficient recursive **filter** that estimates the internal state of a linear dynamic system from a series of noisy measurements.



## Object Detection

Before deep learning,  was a several step process: 

1. edge detection and feature extraction using techniques like SIFT, HOG 
2. Build multi-scale object representation
3. Descriptor were then compared with existing object templates to detect objects
4. Localize objects present in the image.

For example, for pedestrian detection:

SVM template + image pyramid -> template matching

##### Quality Metrics

**Intersection over Union (IoU) :** Bounding box prediction cannot be expected to be precise on the pixel level, and thus a metric needs to be defined for the extend of overlap between 2 bounding boxes.

**Average Precision and Average Recall :** Precision meditates how accurate are our predictions while recall accounts for whether we are able to detect all objects present in the image or not. Average Precision (AP) and Average Recall (AR) are two common metrics used for object detection.

## Image Segmentation

## Surface Representations 

- Mesh

- Spline surface

- Oriented planes

- Signed distance fields (implicit)



# ML

https://github.com/afshinea/stanford-cs-229-machine-learning

https://ml2.inf.ethz.ch/courses/aml/

https://las.inf.ethz.ch/pai-f19

https://www.coursera.org/learn/machine-learning/lecture/zcAuT/welcome-to-machine-learning

Bias-variance trade-off

Overfitting

Regression: predict real-valued output

Classification: discrete valued outputs

## Supervised Learning

Training set - with $m$ number of training examples, $x$ input variables / features, $y$ outputs/targets

$(x^{(i)},y^{(i)})$ is a training example

### Linear Regression (Uni-variable)

Hypothesis: $h_\theta(x) = \theta_0 + \theta_1x$ 

Cost function: a function that measures the performance of the hypothesis

For linear regression: 

$$\min_{\theta_0,\theta_1} \frac{1}{2m}\sum_i| h_\theta(x^{(i)})-y^{(i)} |^2 $$

Squared Error Cost Function: $J = \frac{1}{2m}\sum_i( h_\theta(x^{(i)})-y^{(i)} )^2 $ 

#### Gradient Descent for Linear Regression

For linear regression - the least squared cost function has no local minimum

GD will converge 

Normal Equations can be used to perform a single step solution for linear models, but GD scales better for large training sets

### Multi Variable Linear Regression

For $n$ features, define $ x\in\mathbb{R}^{n+1}$, $0th$ indexed vector,  the features vector, where $x_0^{(i)} := 1 \forall i$  

And $\theta = (0_0,\dots,\theta_n)$  the model

The hypothesis: $h_{\theta} = \sum \theta_i x_i = \theta^T x$  

Update rule for linear regression:

$\theta_0 = \theta_0 -\alpha\frac{1}{m}\sum_i (h_\theta(x^{(i)})-y^{(i)})\cdot x_0$

and similarly for all other variables

**Feature Scaling**

If features are of very different dimensions, the cost function will have skewed contours in the energy landscape. The gradient descent has this ping-pong behavior.  

It helps to scale the parameters to approx. $-1 \le x_j^{(i)}\le 1$

**Mean Normalization**

Replace $x_i$ with $x_i - \mu_i$ to make the variable approx. 0-mean

$x_i \leftarrow \frac{x_i-\mu_i}{range}$

s = Range will be $\text{max}-\text{min}$

### Debugging GD

Plot the cost function when GD runs

Num of iterations depends on the algorithm / model

Automatic convergence tests:

- change in $J(\theta)$ decreases by less than $10^{-3}$

If the cost function value increases, try smaller $\alpha$

When visualization - either wave behavior or increase in the model

Gradient verification with FD

If $\alpha$ is small enough, GD should decrease for every iteration

![convergence](/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/convergence.png)

A - good convergence

B - slow convergence

C - learning rate too high

Run GD with $\alpha$ with a range of values with 10-scale factor  3x from previous values

Until you find one value which is too small and one value which is too large

#### Momentum

#### Netwon

### Polynomial Regression

Basically, the idea here is to cheat and pre-compute the feature vector. 

For example, $(x_1 := x, x_2 := x^2, x_3 := x^3)$. 

The previous formulation and update rules hold: $\theta^Tx$

In this case it's important to scale the variables!

Other options: sqrt, cubic, squared (which might not fit a lot of models)

### Normal Equation

For a feature vector $n$ features and $m$ data points: 

Construct a matrix $X  \in \mathbb{R}^{m\times (n+1)}$ which contains all of features for all the variables + (n+1) column which contains all 1s.

$\left( \begin{matrix} 1 & x_1^1 & ... & x_1^n \\ \vdots & x_2^1 & ... & x_2^n \\   1 & x_m^1 & ... & x_m^n \\ \end{matrix} \right)$

And collect all of the observations in a vector $y \in \mathbb{R}^m $:

And we solve for a model:

$\theta = (X^T X)^{-1} X^T y$



Now, this is true only if $X^T X$ is invertible

Feature scaling is not necc. when using the normal equation.

### GD vs. Normal Equation

GD

- need to choose learning rate
- need many iterations
- works well when $n$ is large

Normal Equation 

- slow for large $n$  $O(n^3)$, n=10k is where switching over could be beneficial
- no need to choose learning rate
- direct

### When is $X^TX$ non-invertible? 

- linearly dependent features - i.e. size in m^2 and size in feet squared
  - remove features
- too many features $n\ge m$ 
  - delete features 
  - use regularization

## Classification

### Two Class Problems

<img src="/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/log_threshold.png" alt="log_threshold" style="zoom:33%;" />

Using linear regression model + threshold:

Classification is not actually a linear function - using linear models doesn't work well.

Labels are usually {0,1} known as negative and positive classes.

#### Logistic Regression

Want a model that predict a value $0\le h_\theta(x)\le 1$

Model: $h_\theta(x)=g(\theta^T x)$ 

Logistic/sigmoid function: $g(z) = \frac{1}{1+e^{-z}}$

Together: $h_\theta(x)=\frac{1}{1+e^{-\theta^T x}}$

![sigmoid](/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/sigmoid.png)

Has  asymptotes at {0,1}

#### Interpretation of Output

$h_\theta(x)$ is the estimated probability that  $y=1$ on input $x$ 

$h_\theta(x) = P(y=1|x;\theta)$ 

$P(y=0|x;\theta) = 1-P(y=1|x;\theta)$

#### Decision Boundary

$g(z) \ge 0$.5 when $z>0$ 

$g(\theta^T x ) \ge 0.5$ when $\theta^Tx \ge 0$

(basically, here we  can derive this from $1+e^{-\theta^T x}  = 2$

The decision boundary is a function of the hypothesis and its parameters

#### Non Linear Decision Boundaries

Can perform a similar trick as with linear regression -> polynomial regression - build features such as $x_1^2$ etc...

So for example: 

$\theta = \left[ \begin{matrix} -1 & 0 & 0  & 1 & 1 \end{matrix} \right]$ 

$h_\theta(x) = g(\theta^T(1,x_1,x_2,x_1^2,x_2^2 )) $

The decision boundary will lie at $x_1^2 + x_2^2 = 1$

#### Cost Function

Using the linear regression cost function is non convex for the logistic regression.

$Cost(h_\theta(x),y) = \begin{cases} -\log(h_\theta(x))  ;\ \text{if} \;  y=1 \\ -\log(1-h_\theta(x))   ;\ \text{if} \;  y=0  \end{cases}$

This formulation has desirable properties: 

$(h(x)=0, y = 0)$ or$(h(x) = 1, y = 1)$  - cost = 0

Very high penalization if $(h(x)=1, y = 0)$ or$(h(x) = 0, y = 1)$ due to the cost function going asymptotically to $\infty$ :

<img src="/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/cost_function.png" alt="cost_function" style="zoom:25%;" />

#### Simplified Cost Function

A generalized cost function is: 

$Cost(h_\theta (x), y) =-y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))  $

And summarizing over all examples:

$J(\theta)= -\frac{1}{m} \sum_{i=1}^{m} y^{(i)}\log(h_\theta(x^{(i)}))+(1-y^{(i)})\log(1-h_\theta(x^{(i)}))$

Maximum likelihood estimation + convexity

To minimize, solve for parameters:

$\min_{\theta} J(\theta) $ 

Output / new prediction: $h_\theta(x) = \frac{1}{1+e^{-\theta^T x}}$

$\frac{\partial}{\partial_{\theta_j}} J(\theta) = \frac 1 m \sum_i (h_\theta(x^{(i)})-y^{(i)})x_j^{(i)}$

Exactly the same update as linear regression. Here the main difference is that $h_\theta$  went from $\theta^T x $ to $\frac{1}{1+e^{-\theta^Tx}}$

And the update rules are:

$\theta_j := \theta_j - \frac{\alpha}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}$

*And vectorized:

$\theta:=\theta - \frac{\alpha}{m}X^T(g(X\theta)-\vec{y})$

### Optimization Techniques 

There following algorithms are alternatives to GD that do not require choosing a learning rate:

- Conjugate Gradient
- BFGS
- L-BFGS

Advantages:

- No learning rate
- Faster than GD
- Line search

Disadvantages

- More complex
- Prob. don't imp. yourself

#### Multi-Class Classification Problems

#### One vs. All

For example: tagging emails according to multiple classes; weather (rainy, sunny)

For each class, train a logistic regression classifier $h_{\theta}^{(i)}(x)$ that predicts that probability that $y=i$.

For new input choose $\max_ih_\theta^i(x)$

### Overfitting vs. Bias

 ![overfitting](/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/overfitting.png)Underfitting -> high bias. 

Overfitting, high variance

High variance - fitting a high order polynomial can be used to fit almost any function, not enough data to give a good hypothesis

- If we have too many features, the learned hypothesis may fit the training data very well, but fail to generalize

##### Addressing Overfitting

Reduce number of features

- requires deciding which feature to keep and discard
- model selection algorithms

Regularization

- keep features but reduce magnitude / values of $\theta_j$ 
- works well when there are a lot features, each of which contributes less

Modify the cost function by penalizing the parameters:

Penalize higher order parameters: equiv to reducing the model to lower order model - simplfying the model

Penalize all parameters  - trying to keep the hypothesis small, usually corresponds to smoother functions

So now the objective has a data term and a regularization term.

The regularization term: $\lambda\sum_{j=1} \theta_j^2 $ keeps all of them small

If $\lambda $ is very large, in linear reg., all model params will be close to 0 and $h_\theta(x) = \theta_0$  

## SVM

[https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72##targetText=A%20Support%20Vector%20Machine%20(SVM,hyperplane%20which%20categorizes%20new%20examples.](https://medium.com/machine-learning-101/chapter-2-svm-support-vector-machine-theory-f0812effc72##targetText=A Support Vector Machine (SVM,hyperplane which categorizes new examples.)

### Fern

### Decision Trees

Recursive repartition of the data

### Random Forest Regression

An ensemble of decision trees. During learning tree nodes are split using random variable subset of data features.

All trees vote to produce final result.

For best results trees should be as independent as possible. Splitting using a random subset of features achieves this.

Averaging the product of the trees reduces overfitting to noise

5-100 Trees.

### Boosting



### Measuring Model Performance

False positive - Predict an event when there was no event

False negative - Predict no event when in fact there was an event.

#### Precision-Recall

Precision-Recall curves summarize the trade-off between the true positive rate and the positive predictive value for a predictive model using different probability thresholds.

Precision-recall curves are appropriate for imbalanced datasets.

#### ROC -  Receiver Operating Characteristic curve

Summarize the trade-off between the true positive rate and false positive rate for a predictive model using different probability thresholds.

ROC curves are appropriate when the observations are balanced between each class

**Convolution** is a [mathematical operation](https://en.wikipedia.org/wiki/Operation_(mathematics)) on two [functions](https://en.wikipedia.org/wiki/Function_(mathematics)) (*f* and *g*) that produces a third function expressing how the shape of one is modified by the other

$(f\star g)(t) = \int_{-\infty}^{\infty} f(\tau)\cdot g(t-\tau)d\tau = \int_{-\infty}^{\infty} f(t-\tau)\cdot g(\tau)d\tau $

Commutative. 

For functions which only have limited support the integration is only done on the valid domain.

### L1 vs L2 Norm

L2 norm strongly penalizes outliers. For good data with some very far outlier it might not generate the "best" fit as judged by a human observer.

L1 favors sparse coefficients.

## RANSAC

A method for dealing with noisy data. 

Partition the method 

Is not determinant, depends on the subset selection, and is not guaranteed to converge.

1. Select a random subset of the original data. Call this subset the *hypothetical inliers*.
2. A model is fitted to the set of hypothetical inliers.
3. All other data are then tested against the fitted model. Those points that fit the estimated model well, according to some model-specific [loss function](https://en.wikipedia.org/wiki/Loss_function), are considered as part of the *consensus set*.
4. The estimated model is reasonably good if sufficiently many points have been classified as part of the consensus set.
5. Afterwards, the model may be improved by reestimating it using all members of the consensus set.

```
Given:
    data – a set of observations
    model – a model to explain observed data points
    n – minimum number of data points required to estimate model parameters
    k – maximum number of iterations allowed in the algorithm
    t – threshold value to determine data points that are fit well by model 
    d – number of close data points required to assert that a model fits well to data

Return:
    bestFit – model parameters which best fit the data (or nul if no good model is found)

iterations = 0
bestFit = nul
bestErr = something really large
while iterations < k {
    maybeInliers = n randomly selected values from data
    maybeModel = model parameters fitted to maybeInliers
    alsoInliers = empty set
    for every point in data not in maybeInliers {
        if point fits maybeModel with an error smaller than t
             add point to alsoInliers
    }
    if the number of elements in alsoInliers is > d {
        % this implies that we may have found a good model
        % now test how good it is
        betterModel = model parameters fitted to all points in maybeInliers and alsoInliers
        thisErr = a measure of how well betterModel fits these points
        if thisErr < bestErr {
            bestFit = betterModel
            bestErr = thisErr
        }
    }
    increment iterations
}
return bestFit
```



## Bagging/Boosting

Collaborative filtering

## Dimension Reduction

### 



Implement simple models such as decision trees and K-means clustering.

 If you put some models on your resume, make sure you understand it thoroughly and can comment on its pros and cons.

# Unsupervised Learning

Algorithms for finding structure in data.

## Clustering

The clustering problem: given an unlabeled data set, group the data into coherent  subsets or into coherent clusters for us.

### K Means

- $K$ number of clusters + initialization
- Training set ${x^{(1)},x^{(2)}\dots,x^{(m)}}$
- $x\in\mathbb{R}^n$
- By convention, drop $x_0=1$

```
Randomly initialize K cluster centers
While not converged:
1. iterate over data and assign a cluster for each data point based on distance to center
2. re-compute the cluster mean
```

If a cluster becomes empty - remove the cluster

Or randomly re-initialize the cluster

##### K Means for Non Separated Clusters

##### K Means Cost Function

Assuming: 

$c^{(i)}$ index of cluster to which the example $x^{(i)}$ belongs to.

$\mu_k \in \mathbb R ^n $ cluster centroid 

$\mu_{c^{(i)}} \in \mathbb R ^n $ location of the cluster centroid to which example $x^{(i)}$ has been assigned

Example cost for point $Cost(x^{(i)}) = \|x^{(i)} - \mu_{c^{(i)}} \|^2$ 

$J(c^{(1)},\dots, c^{(k)})= \frac{1}{m}\sum_i \|x^{(i)} - \mu_{c^{(i)}} \|^2 $

The objective is to minimize the cost function *distortion* with respect to the clusters (both labelling and centers).

So what k-means algorithm is actually doing is:

1. minimize the cost function with respect to cluster assignments $c^{(i)}$
2. minimize the cost function with respect to cluster centroids $\mu_k$ 

(so basically block coordinate descent?)

##### Random Initialization

- $K < m$
- Randomly pick $K$ training examples and set the cluster means to these examples

K-mean can get stuck in a local optima - to avoid this a good option is to run k-mean multiple times and get as good global optimum

For multiple initializations - run K-means loads of times, pick the clustering which results in the lowest cost function

This works well for small $K < 10$ .

For large $K$s it is not as effective.

##### Number of Clusters - Elbow Method

Choosing the right K 

Plot the cost function with respect to the number of clusters.![elbow](/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/elbow.png)



In practice it is usually a bit harder, and it is not clear that there is such a transition where the distortion stops.

## Dimensionality Reduction

### PCA

PCA is trying to find a lower dimension representation of that data which minimizes the squared distance error of the data from the representation.

Before PCA it is standard practice to perform mean normalization and feature scaling. 

##### PCA vs Linear Regression

<img src="/Users/kozlovy/Documents/2019_JobApplications/Notes/ml_figures/PCA_Linear.png" alt="PCA_Linear" style="zoom:50%;" />

We do not treat $y$ as a special variable

Minimized projected error vs. minimize distance from line

###GMM and EM





Generating good synthetic data: 
realism, 
diverse,


Want to render images which are as different as possible from each other


Parametric model of humans - procedural generation



## Neural Networks

http://karpathy.github.io/neuralnets/

# ML Design

For ML product design, understand the general process of building a ML product. Here’s what I tried to do:

1. Figure out what the objective is: prediction, recommendation, clustering, search, etc.
2. Pick the right algorithm: supervised vs unsupervised, classification vs regression, generalized linear model / decision tree / neural network / etc. Be able to reason about the choice.
3. Pick / engineer relevant features based on available data.
4. Pick metrics for model performance.
5. Optionally, comment on how to optimize the model for production.

#Math

### Algebra

Matrix $A \in\mathbb R^{m\times n}$ is a matrix with $m$ rows and $n$ columns

$AA^{-1}=\mathbb{I}$

A matrix that does not have inverse is _singular_ or _degenerate_

Transpose: $A_{ij} = A_{ji}^T$

Taylor Expansion:

Gradient:

Chain rule:

Finite Difference Approximation:

### Convexity

A function is convex is $f(x) - f(y) \ge $

Examples of convex functions: 



### Exponential and Logarithm Arithmetics 

$exp^{ab} = exp^a*exp^b$

$\log_ab = \frac{log_n a}{log_n b}$

$\log\frac{a}{b} = \log(a) -\log(b)$

$\log(ab) = \log a + \log b$

# Statistics

Expected Value - $E[X] = \sum_i p_i x_i $ 
Variance - $x - E[X]$
Covariance - 
Covariance Matrix -

## Probability

### Bayes Theorem

### Distributions

##### Gaussian Distribution

\frac{1}{2\pi^{d/2}

### PDF

## Alegbra

Matrix rank
Matrix decomposition
Inverse Matrix - when, under what conditions

Prior
Residual 
Approximation
