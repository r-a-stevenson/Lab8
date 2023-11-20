### Lab 10 Collab by Andrew, Giorgio, Merdeka and Owen

import numpy as np
import matplotlib.pyplot as plt

## Part 1 : Andrew
def make_tridiagonal(N, b, d, a):
    """
    Function to form a tridiagonal maxtrix from given inputs
    Input:
    \tN = Number of rows and columns for the square matrix (int)\n
    \tb =  "Left" number of the tridiagonal\n
    \td = Number on the true diagonal (i.e. the middle)\n
    \ta = "Right" number of the tridiagonal\n
    """

    myLeftMatrix = b*np.eye(N, N, k = -1)
    
    myMiddleMatrix = d*np.eye(N, N, k = 0)

    myRightMatrix = a*np.eye(N, N, k = 1)

    

    fullTridiag = myLeftMatrix + myMiddleMatrix + myRightMatrix

    return fullTridiag


print(make_tridiagonal(5, 3, 1, 5))

## Part 2: Merdeka & Owen
L = 5 # Setting the adjustable parameters of the function:
Nspace = 300
sigma = 0.2
k = 35
h = L/Nspace
x = np.arange(Nspace)*h - L/2.

def make_initialcond(sigma, k, x_i):
    """ Accepts parameters sigma, k, and xgrid and returns the initial condition, a(x, 0), given the function: 
a(x, 0)=exp[-x**2/(2sigma^2)]cos(kx)\n
    sigma = adjustable parameter (float)\n
    k = adjustable parameter (float)\n
    x_i = a spatial grid, for i = [1, 2, ..., Nspace] (array)"""

    initialcond = np.exp(-x_i**2/(2*sigma**2))*np.cos(k*x_i)
    return initialcond

print(make_initialcond)

# Part 3: Giorgio
def spectral_radius(A):
    """
    Input A: 2-D array
    Returns the eigenvalue with the largest absolute value
    """
    eigenvalues = np.linalg.eig(A)[0]
    return np.max(np.abs(eigenvalues))