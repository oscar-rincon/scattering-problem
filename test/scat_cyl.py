# -*- coding: utf-8 -*-
"""

Generates the frames for an animation with the scattering of a pressure
wave generated by a rigid cylinder. The values are computed with an
analytical solution, it should be taken into account that the Bessel
series do not satisfy homogeneous convergence and the bigger the domain
of interest, the larger the quantity of terms to use in the sum.

Author: Nicolas Guarin-Zapata

Last modification (dd/mm/yyyy): date 23/02/2018

"""
from scipy.special import jn, hankel2
from matplotlib import pyplot as plt
from numpy import (mgrid, ceil, amin, amax, cos, zeros_like,
                   pi, exp, sqrt, arctan2, log10)
from scipy.fft import ifft
from scipy.fft import fftshift
from matplotlib.colors import LightSource


def u(nmax, k, a, r, theta):
    u = zeros_like(k)
    for n in range(nmax, -1, -1):
        if n==0:
            an = -jn(1, k*a)/hankel2(1, k*a)
            en = 1.0
        else:
            an = -2.0*(-1.0)**n*((jn(n + 1, k*a) - jn(n - 1, k*a))/
                 (hankel2(n + 1, k*a) - hankel2(n - 1, k*a)))
            en = 2.0
        usn = an*1.0j**n*hankel2(n,k*r)*cos(n*theta)
        uin = en*1.0j**n*jn(n,-k*r)*cos(n*theta)
        u = u + usn + uin
    return u


def trans_ricker(b, omega):
    """
    Fourier transform of a Ricker wavelet
    """
    return -pi*b**3*omega**2*exp(-b**2*omega**2/24.0)/(48.0*sqrt(6.0))


def ricker(b, t):
    """
    Ricker wavelet
    """
    return 0.5*sqrt(pi)*(6.0*t**2/b**2 - 0.5)*exp(-6.0*t**2/b**2)



if __name__ == "__main__":
    a = 0.5  #  Radius of the cylinder
    nmax = 50 #  number of terms in the series
    vel = 5.0 #  Wave phase speed
    nfreq = 256  #  Sampling number for frequencies
    b = 0.5     #  Parameter for the Ricker pulse (time between peaks)
    xmin = -10.0 #  Minimum x
    xmax = 10.0  #  Maximum x
    nx = 201     #  Sampling number in x
    ymin = -10.0 #  Minimum y
    ymax = 10.0  #  Maximum y
    ny = 201     #  Sampling number in y
    min_freq = 1e-3
    max_freq = 10.0
    X, Y, F = mgrid[xmin:xmax:1j*nx,
                    ymin:ymax:1j*ny,
                    min_freq:max_freq:1j*nfreq]
    
    
    R = sqrt(X**2 + Y**2)
    Theta = arctan2(Y, X)
    U =  trans_ricker(b, F)*u(nmax, F/vel, a, R, Theta)
    U[R<=a] = 0.0
    Ut = fftshift(ifft(U, axis=2), axes=2)
    
    #%%
    ndigits = int(ceil(log10(nfreq)))
    Umax = max(amin(Ut).real, amax(Ut).real)
    plt.figure(figsize=(4, 4))
    circ = plt.Circle((0,0), 1.2*a, color='k')
    plt.gca().add_artist(circ)
    for t in range(30, nfreq - 30):  ## Ploting each frame
        print("Printing the %ith frame" % t)
        cmap = plt.cm.RdBu
        ls = LightSource(315, 45)
        rgb = ls.shade(Ut[:, :, t].real, cmap, vmin=-Umax, vmax=Umax)
    #    rgb = ls.shade(Ut[:, :, t].real, cmap)
        plt.imshow(rgb, extent=(X.min(), X.max(), Y.min(), Y.max()),
                   interpolation="bilinear")
    #    plt.pcolormesh(X[:, :, 0], Y[:, :, 0], Ut[:, :, t].real, vmin=-Umax,
    #                   vmax=Umax, color=rgb)
        plt.axis("image")
        plt.axis("off")
#        plt.savefig("scat%s.png" %str(t).zfill(ndigits), transparent=True,
#                    dpi=300, bbox_inches="tight")
