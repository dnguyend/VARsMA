
<a href="https://colab.research.google.com/github/dnguyend/VARsMA/blob/master/VARsMA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Vector Autoregressive with scalar Moving Average Model
## Introduction.
This notebook explains the underlying idea of the the paper [Vector Autoregressive with scalar Moving Average Model](https://github.com/dnguyend/VARsMA/blob/master/docs/VARsMA.pdf). The main ideas are the following

* When the Autoregressive and the Moving Average polynomials commute with each other, there is a simple formula for the conditional likelihood function.
* The formula is a Generalized Linear Regression type formula, with the inner product given by the Toeplitz matrix of the moving average process.
* There is a simple inversion formula for the Toeplitz matrix using the Woodbury Matrix Identity.  The inversion involves a convolution with the invert of the MA polynomial, and an inversion of a smaller matrix of size $k\times k$ ($k$ is the dimension of vector.) 
* This inversion formula is related to the Borodin-Okounkov formula in operator algebra.
* Since the AR coefficients are given by GLS for a fixed MA polynomial, we have a close form of the likelihood function for each set of MA coefficients. We can optimize over the MA coefficients to estimate the model parameters.

We present the python code in our github directory. We also have R and C++ codes which we plan to make open source in the future.

The AR part of the model is dense. A reduced rank AR model will be presented in future work. This model is a simple extension of the Vector Autogressive Model widely use in Time Series Analysis. This model should be competitive versus the traditional VAR model if the real world data contains a moving average component, even if the MA component is not scalar.

### TL;DR:
Anywhere we use a Vector Autoregressive Model, we should try VARsMA. At the cost of a few parameters it could deal with moving average effect. The package is open source and easy to use. Open the notebook https://github.com/dnguyend/VARsMA/blob/master/VARsMA.ipynb in colab for more details.

