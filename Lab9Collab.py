# Program to solve the generalized Burger  
# equation for the traffic at a stop light problem

# Coded by Andrew, Giorgio, Owen and Merdeka

## Part 1: Andrew and Giorgio

# Set up configuration options and special features
import numpy as np
import matplotlib.pyplot as plt

def traffic_flow(N, nstep):
    """
    Solves the advection equation for the time 
    evolution of ρ(x, t), using the Lax method 
    with ρSL(x) as the initial condition.

    Inputs:
    \tN: number of grid points (int)
    \ttau: time step (float in seconds)
    \tnstep: number of time steps (int)
    Outputs:
    \trplot: density profile (solution to the advection equation)
    \ttplot: time (time space for plotting)
    \txplot: position (positions for plotting)
    """
    L = 1200.      # System size (meters)
    h = L/N       # Grid spacing for periodic boundary conditions
    v_max = 25.   # Maximum car speed (m/s)
    tau = h/v_max
    coeff = tau/(2*h)          # Coefficient used by all schemes
    coefflw = tau**2/(2*h**2)  # Coefficient used by Lax-Wendroff

    #* Set initial and boundary conditions
    rho_max = 1.0                   # Maximum density
    Flow_max = 0.25*rho_max*v_max   # Maximum Flow
    Flow = np.empty(N)
    cp = np.empty(N);  cm = np.empty(N)
    # Initial condition is a square pulse from x = -L/4 to x = 0
    rho = np.zeros(N)
    for i in range(int(N/4),int(N/2)) :
        rho[i] = rho_max     # Max density in the square pulse

    rho[int(N/2)] = rho_max/2   # Try running without this line

    # Use periodic boundary conditions
    ip = np.arange(N) + 1  
    ip[N-1] = 0          # ip = i+1 with periodic b.c.
    im = np.arange(N) - 1  
    im[0] = N-1          # im = i-1 with periodic b.c.

    #* Initialize plotting variables.
    iplot = 1
    xplot = (np.arange(N)-1/2.)*h - L/2.    # Record x scale for plot
    rplot = np.empty((N,nstep+1))
    tplot = np.empty(nstep+1)
    rplot[:,0] = np.copy(rho)   # Record the initial state
    tplot[0] = 0                # Record the initial time (t=0)

    #* Loop over desired number of steps.
    for istep in range(nstep) :

        #* Compute the flow = (Density)*(Velocity)
        Flow[:] = rho[:] * (v_max*(1 - rho[:]/rho_max))
    
        #* Compute new values of density using  
        #  FTCS, Lax or Lax-Wendroff method.
        method = 2
        if method == 1 :      ### FTCS method ###
            rho[:] = rho[:] - coeff*( Flow[ip] - Flow[im] )
        elif method == 2 :    ### Lax method ###
            rho[:] = .5*( rho[ip] + rho[im] ) - coeff*( Flow[ip] - Flow[im] )
        else :                ### Lax-Wendroff method ###
            cp[:] = v_max*(1 - (rho[ip]+rho[:])/rho_max);
            cm[:] = v_max*(1 - (rho[:]+rho[im])/rho_max);
            rho[:] = rho[:] - coeff*( Flow[ip] - Flow[im] ) + coefflw*(
                cp[:]*(Flow[ip]-Flow[:]) - cm[:]*(Flow[:]-Flow[im]) )

        #* Record density for plotting.
        rplot[:,iplot] = np.copy(rho)
        tplot[iplot] = tau*(istep+1)
        iplot += 1
    return rplot, tplot, xplot

rplot, tplot, xplot = traffic_flow(600, 1500)

## Part 2: Owen and Deka (including questions)

#* Graph density versus position and time as wire-mesh plot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
    
#* Graph contours of density versus position and time.
levels = np.linspace(0., 1., num=11)
ct = plt.contour(xplot, tplot, np.flipud(np.rot90(rplot)), levels)
plt.clabel(ct, fmt='%1.2f') 
plt.xlabel('x Values')
plt.ylabel('Time')
plt.title('Density contours')
plt.show()

#* Graph "snapshots" of rho vs. x for selected time points
selectedTimes = np.arange(0, 1201, 300) # picking the time values

fig2 = plt.figure()
for i in selectedTimes:
    plt.plot(xplot, rplot[:,i], label="t={}s".format(i)) # plotting rho vs x for the values of rho in selectedTimes array
plt.xlabel('x Values')
plt.ylabel('rho Values')
plt.title('Density contours')
plt.legend()
plt.show()

# In reference to your contour plot, does a shock 
# front seem to form in the flow? 
# If so, why and at what time does it first appear? 
# In what direction does the front propagate?

""" A shock front does form.

It appears at time t = 0 and at position 
x = -300, because that is the decided location of
the highest density in a 1-d 'x' space.

It propogates directly upward at first because the 
rate of particles entering the shock front is the 
same as the rate of particles leaving the shock front. 
It then travels towards positive X because as particles
'pile up' at the shock front, the shock front moves
in the direction the particles are coming from to
account for the extra space they take up, since these
new particles contribute to the front as well."""