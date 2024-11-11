
from scipy.special import jn, hankel2
from numpy import pi, exp, cos, zeros_like

def u_exact(r, theta, r_i, k):
    u_tot = zeros_like(r, dtype=complex)  # Initialize total displacement field
    u_scn = zeros_like(r, dtype=complex)  # Initialize scattered wave
    nmax = int(30 + (k * r_i)**1.01) # Number of terms in the series
    for n in range(nmax, -1, -1):
        if n == 0:
            # Coefficient for n = 0
            an = -jn(1, k*r_i) / hankel2(1, k*r_i)
            en = 1.0
        else:
            # Coefficients for n > 0
            an = -2.0 * (-1.0)**n * ((jn(n + 1, k*r_i) - jn(n - 1, k*r_i)) /
                                     (hankel2(n + 1, k*r_i) - hankel2(n - 1, k*r_i)))
            en = 2.0
        # Sum terms for both scattered and incident waves
        usn = an * 1.0j**n * hankel2(n, k*r) * cos(n*theta)
        uin = en * 1.0j**n * jn(n, -k*r) * cos(n*theta) * exp(1j*pi)
        u_tot = u_tot + usn + uin
        u_scn = u_scn + usn
    return u_tot, u_scn

 