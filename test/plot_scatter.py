#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the scattering for a monochromatic plane wave.

@author: Nicolás Guarín-Zapata
@date: May 2020
"""
import numpy as np
from scat_cyl import u
import matplotlib.pyplot as plt
import numpy.ma as ma


def gen_disp(r, theta, k, nmax=50):
    return  u(nmax, k, 1.0, r, theta)


Y, X = np.mgrid[-5:5:501j, -5:5:501j]
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
k = 5.0
disp = gen_disp(R, Theta, k)
savefig = False

#%% Visualization
amp = ma.masked_where(R<1.0, np.real(disp))
phase = ma.masked_where(R<1.0, np.angle(disp))

repo = "https://raw.githubusercontent.com/nicoguaro/matplotlib_styles/master"
style = repo + "/styles/minimalist.mplstyle"
plt.style.use(style)
plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
plt.pcolormesh(X, Y, amp, cmap="RdYlBu", vmin=-1, vmax=1)
cb = plt.colorbar(shrink=0.8, orientation="horizontal")
cb.set_label("Amplitude")
cb.set_ticks([-1, 0, 1])
plt.axis("image")
plt.yticks([])
plt.xticks([])

plt.subplot(1, 2, 2)
plt.pcolormesh(X, Y, phase, cmap="twilight_shifted", vmin=-np.pi,
               vmax=np.pi)
cb = plt.colorbar(shrink=0.8, orientation="horizontal")
cb.set_label("Phase")
cb.set_ticks([-np.pi, 0, np.pi])
cb.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])
plt.axis("image")
plt.yticks([])
plt.xticks([])

if savefig:
    plt.savefig("cyl_scat-ka={:g}.png".format(k), bbox_inches="tight", 
                dpi=300)