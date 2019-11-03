import numpy as np

# Unit for length [cm]
U_L = 1e8
# Unit for time [s]
U_T = 1e2
# Unit for mass density [g/cm^3]
U_R = 1e-7
# Unit for speed [cm/s]
U_U = U_L/U_T
# Unit for pressure [dyn/cm^2]
U_P = U_R*(U_L/U_T)*(U_L/U_T)
# Unit for Rosseland opacity [cm^2/g]
U_KR = 1.0/(U_R*U_L)
# Unit for energy per mass [erg/g]
U_EE = U_U*U_U
# Unit for energy per volume [erg/cm^3]
U_E = U_R*U_EE
# Unit for thermal emission [erg/(s ster cm^2)]
U_TE = U_E/U_T*U_L
# Unit for volume [cm^3]
U_L3 = U_L*U_L*U_L
# Unit for magnetic flux density [gauss]
U_B = U_U*np.sqrt(4.0*np.pi*U_R)
