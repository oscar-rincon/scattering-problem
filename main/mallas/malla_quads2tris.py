# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 17:46:06 2024

@author: nguarinz
"""
import numpy as np


pts = np.loadtxt("valle_aburra-quads.pts")
quads = np.loadtxt("valle_aburra-quads.quad", dtype=int)

nquads = quads.shape[0]
ntris = 2 * nquads

tris = np.zeros((ntris, 3), dtype=int)
tris[:nquads, :] = quads[:, :3]
tris[nquads:, 0] = quads[:, 2]
tris[nquads:, 1] = quads[:, 3]
tris[nquads:, 2] = quads[:, 0]

np.savetxt("valle_alta.pts", pts)
np.save("valle_alta.tri", tris, dtype=int)