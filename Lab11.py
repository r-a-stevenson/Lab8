# Lab 11 collab by Andrew, Giorgio and Merdeka
import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
print('\n\nNEW RUN\n')

# Functions from lab 10
def make_tridiagonal(N, b, d, a):
    """
    Function to form a tridiagonal maxtrix from given inputs
    Input:
    \tN = Number of rows and columns for the square matrix (int)\n
    \tb =  "Left" number of the tridiagonal\n
    \td = Number on the true diagonal (i.e. the middle)\n
    \ta = "Right" number of the tridiagonal\n
    """

    myLeftMatrix = b*np.eye(N, k = -1)
    myMiddleMatrix = d*np.eye(N, k = 0)
    myRightMatrix = a*np.eye(N, k = 1)


    fullTridiag = myLeftMatrix + myMiddleMatrix + myRightMatrix

    return fullTridiag

def make_initialcond(sigma, k, x_i):
    """ Accepts parameters sigma, k, and xgrid and returns the initial condition, a(x, 0), given the function: 
a(x, 0)=exp[-x**2/(2sigma^2)]cos(kx)\n
    sigma = adjustable parameter (float)\n
    k = adjustable parameter (float)\n
    x_i = a spatial grid, for i = [1, 2, ..., Nspace] (array)"""

    initialcond = np.exp(-x_i**2/(2*sigma**2))*np.cos(k*x_i)
    return initialcond

def spectral_radius(A):
    """
    Input A: 2-D array
    Returns the eigenvalue with the largest absolute value
    """
    eigenvalues = np.linalg.eig(A)[0]
    return np.max(np.abs(eigenvalues))

# Part 1: Integrate the 1D advection equation - Andrew and Giorgio

def advection1d(method, nspace, ntime, tau_rel, params):
    """
    Function to integrate the 1D advection equation using the Lax-Friedrichs or FTCS method
    Inputs:
    \tmethod = "lax" or "ftcs" (string)
    \tnspace = number of grid points in space (int)
    \tntime = number of grid points in time (int)
    \ttau_rel = relative time step (float)
    \tparams = list of parameters [L, c] where L is the length of the domain and c is the wave speed
    Outputs:
    \ta = 2D array of the solution to the advection equation (nspace, ntime)
    \tx = 1D array of the spatial grid (nspace)
    \tt = 1D array of the time grid (ntime)
    """
    # Unpack the length L and the wave speed c
    L = params[0]
    c = params[1]

    # Discretize the space and time
    h = L/(nspace-1)

    x = np.linspace(-L/2, L/2, nspace)
    t = np.linspace(0, tau_rel, ntime)

    # Constant in front of matrix B
    k = (c*tau_rel)/(2*h)
    sigma = 0.2

    # Construct the matrices
    if method.lower() =="lax":
        matrixC = make_tridiagonal(nspace, 1, 0, 1)
        matrixB = make_tridiagonal(nspace, -1, 0, 1)
        matrixC[0][-1] = 1
        matrixC[-1][0] = 1
        matrixB[0][-1] = 1
        matrixB[-1][0] = -1

        properMatrixA = (0.5)*matrixC - k*matrixB

        if spectral_radius(properMatrixA)<=1:
            print("Method is expected to be stable.")
        else:
            print("WARNING: method is expected to be unstable.")

    elif method.lower() =="ftcs":
        properMatrixA = make_tridiagonal(nspace, 1, 1, -k)
        properMatrixA[0][-1] = 1
        properMatrixA[-1][0] = -k


        print("WARNING: FTCS method is (always) expected to be unstable.")

    else:
        print("Unknown method.")

    init = make_initialcond(sigma, k, x)
    product = np.dot(properMatrixA, init)

    a = np.zeros([ntime, nspace])
    a[0] = product

    for i in range(1, len(t)):
        if i+1>=len(t):
            break
        a[i] = np.dot(properMatrixA, a[i-1])

        # print(a[i])

    print(a[0], a[10], a[100])

    return (a, x, t, init) 

c = 1
tau_rel = 1
L = 5
nspace = 300
ntime = 501

params = [L, c]

a_lax, x_lax, t_lax, init = advection1d("lax", nspace, ntime, tau_rel, params)
a_FTCS, x_FTCS, t_FTCS, init_FTCS = advection1d("ftcs", nspace, ntime, tau_rel, params)

# print(a_lax)

# fig = plt.figure()
# plt.plot(x_lax, a_lax)
# plt.show()

# print(x_lax)
# print(make_initialcond(0.2, (c*tau_rel)/(2*(L/(nspace-1))), np.linspace(-L/2, L/2, nspace)))

# Part 2: Visualize the propogating wave through time - Merdeka

# print(a_lax)
fig, ax = plt.subplots()# figsize=[10, 5]) # generating figure
camera = Camera(fig)

for i in [0, 1, 2, 3, 4]:
    plt.plot(x_lax, a_lax[i], label="{}".format(i))
    # plt.plot(x_FTCS, a_FTCS[i], c="navy", label="FTCS")
    plt.suptitle("Animation of Wave Solved by Lax and FTCS Methods")
    # plt.annotate("Time (s) = {}".format(np.round(t_lax[i], 2)))
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    plt.legend()

    camera.snap() # saving an image of the plot for the animation at every point in time
plt.show()
# # animation = camera.animate(interval=10, repeat=True) # animating the collected images

# plotskip = 50
# fig, ax = plt.subplots()
# # space out the plots vertically to make the visualization clearer
# yoffset = a_lax[:,0].max() - a_lax[:,0].min()
# # loop in reverse order to make the legend come out right
# for i in [50, 100, 200]:
#     ax.plot(x_lax, a_lax[:,i]+yoffset*i/plotskip,label = 't ={:.3f}'.format(t_lax[i]))
# ax.legend()
# ax.set_xlabel('X position')
# ax.set_ylabel('a(x,t) [offset]')
# plt.show()
