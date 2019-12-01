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

In 2D, around some axis: 

$\left(  \begin{matrix} \cos  \alpha & \sin \alpha  & 0 \\  -\sin \alpha & \cos \alpha & 0 \\ 0 & 0 & 1  \end{matrix} \right)$

Symmetry group 

SO3 vs . SE3

Reflection - Eigenvalues of the rotation matrix contain (-1)

#### Translation

Only possible in homogeneous coordinates

## Projective Geometry

### Homogeneous Coordinates

$x = \left[\begin{matrix} u \\ v\\ 1 \end{matrix}\right]$ pixel space position, only 2 degrees of freedom.

Inhomgenous coordinates: $x = [\begin{matrix} u & v \end{matrix}]^T $

$ x_w = \left[ \begin{matrix} x_w \\  y_w \\ w_t\\  1 \end{matrix} \right] $ world space position

The scale is unknown: $x \mapsto  w x_w$

##### Homogenous Line/Plane Representation:

$ax+by+c = 0 \to (a,b,c)^T p = 0$ for every $p = (x,y,1)$ on the line.

##### Projection

A projectivity is an invertible mapping $h$ from  to itself such that three points $x_1 , x_2 , x_3$ lie on the same line if and only if $h(x_1), h(x_2 ), h(x_3)$ do

- check by fitting a line to the point and checking the third point is on the same line
- line normal coordinates in 3D: ()

##### Projective Transformation

From  world position $x \in \mathbb R^4 \to p \in \mathbb R^3$ pixel homogenous coordinates. 

8 DoF

Projectivity : collineation :  proj. transformation : **homography**

A point in an image projected to the world is known up to a scale factor (that's why we have this 1 at the bottom of the vector)

### Transformations Hierarchy (2D)

Projective 8 DoF

- colinearity

Affine 6 DoF

- parallelism, ratio of areas, ratio of lengths of parallel lines

Similarity (4DoF)

$\left(  \begin{matrix} s\cos  \alpha &  s\sin \alpha  & t_x \\  -s \sin \alpha & s\cos \alpha & t_y \\ 0 & 0 & 1  \end{matrix} \right)$

- ratios of lengths, angels

Euclidean (3DoF)

$\left(  \begin{matrix} \cos  \alpha & \sin \alpha  & t_x \\  -\sin \alpha & \cos \alpha & t_y \\ 0 & 0 & 1  \end{matrix} \right)$

#### Homography

Relates planar images

Working with these coordinates:

apply homography to homogenous coordinates and then divide by $z$.

Simple Geometric Problems:

Two lines intersection - $l_1,l_2$ - find $x = l_1 \times l_2$  (based on normal equations)

Line through two points: $l = x_1 \times x_2 $

#### Homography Line Transformation

$l^\prime = H^-T l$

##### Ideal Points

Intersection points of parallel lines

#### 3D Homography

Plane normal equation: $ax_1 + bx_2 + c_x3 + d x_4 = 0$ 

If $\pi^T p = 0$ the point lies on the plane / the plane passes through the point

We can fit a plane to three points by solving: 

$\left( \begin{matrix} x_1^T \\ x_2^T \\ x_3^t \end{matrix} \right) \pi = 0 $

### Cameras and Image Formation

The camera model relates  pixels and rays in space

Principal axis - usually  denoted as $z$, faces into the world

Principal point - the point at which the principal axis passes through the image sensor

Sensor RGB Pattern - Bayer pattern

Noise

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

Usually the model is simplified to:

$$ K = \left( \begin{matrix} 
f & 0 & h/2 & 0 \\
0 & f  & w/2 & 0 \\
0 & 0 &  1 & 0 \\
\end{matrix}
\right) $$

### Camera Extrinsic Parameters $R,T$

$$ \left[ \begin{matrix} 
R_{3x3} & T_{3x1} \\
0_{1x3} & 1 \end{matrix}
\right] $$

The extrinsic parameters define the position of the camera center and the camera's heading in world coordinates. 

$T$ is the position of the origin of the world coordinate system expressed in coordinates of the camera-centered coordinate system. 

The actual camera position in world coordinates is $C = -R^{-1}T=-R^{T}T $

Transformation of a point using 

### Camera Intrinsic Estimation

1. For capturing a planar object:

   1. set the planar object as the infinity plane: $x = ( \begin{matrix} x_1 & x_2 & 0 & 1 \end{matrix})^T$

   2. estimate the transformation for each point using homography relations:

      $ x \times H x = 0$ resolves to a series of equations that can be solved for h.

   

2. Normalize the points:

   - Translate points s.t. centroid is at origin
   - Isotropic - mean distance from origin of $\sqrt 2 $

3. minimize the 2 sided reprojection error - from img1 -> img2 and from img2 -> img1

4. This relates to the maximum likelihood estimate

   

 Given n≥4 2D to 2D point correspondences {**x**i↔**x**i’},

Determine the Maximum Likelihood Estimation of **H**

(this also implies computing optimal **x**i’=**Hx**i) Algorithm

**(i)** **Initialization:** compute an initial estimate using normalized DLT or RANSAC

**(ii)** **Geometric minimization of symmetric transfer error:** 

• Minimize using Levenberg-Marquardt over 9 entries of **h or reprojection error:

- compute initial estimate for optimal $x_i
- minimize cost over {**H**,**x**1,**x**2,...,**x**n}
- if many points, use sparse method

#### Other things to consider

- Radial lens distortion
- Rolling shutter effects

# Features

**Local features** - image regions with salient structures that can be identified across multiple images and viewpoints

**Image Descriptors** - compact representation of  image regions which allows us to compare and localize matches as sub pixel accuracy. 

**Feature matching**  - extract features independently and match by comparing descriptors.

**Feature tracking** - extract features from the first frame, find same feature in the next frame.

They also correspond to the three stages of feature correspondence according to Szeliski - feature detection, description and matching.