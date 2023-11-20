import numpy as np

# Atomic mass unit [g].
AMU = 1.660_540_2e-24
# Electron charge [esu].
Q_ELECTRON = 4.80325e-10
# Electron mass [g].
M_ELECTRON = 9.109_389_7e-28
# Hydrogen mass [g].
M_H = 1.672_621_9e-24
# Helium mass [g].
M_HE = 6.65e-24
# Ion mass [g].
MION = M_H
# Speed of light in vacuum [cm/s].
CLIGHT = 2.997_924_58e10
# Boltzmann constant [erg/K].
KBOLTZMANN = 1.380_658e-16
# Planck constant [erg s].
HPLANCK = 6.626_075_5e-27
# Stefan-Boltzmann constant [erg/(cm^2 s K^4)].
STEFBOLTZ = 5.670_400e-5
# Bohr radius [cm].
RBOHR = 5.291_773_49e-9
# Ionization potential of hydrogen [erg].
E_RYDBERG = 2.179_874_1e-11
# H2 dissociation energy [eV].
EH2DISS = 4.478
# pi e^2 / m_e c [cm^2 Hz].
PIE2_MEC = 0.02654
# Electron rest energy [erg].
MC2_ELECTRON = M_ELECTRON * CLIGHT * CLIGHT
# Ionization energy of an hydrogen atom [eV].
XI_H = 13.595

# Conversion factor from electron volts to ergs.
EV_TO_ERG = 1.602_177_33e-12
# Conversion factor from kilo electron volts to ergs.
KEV_TO_ERG = EV_TO_ERG * 1e3
# Conversion factor from rydberg to ergs.
RYD_TO_ERG = 2.17987197e-11
# Conversion factor from electron volts to kelvin (1/KBOLTZMANN).
EV_TO_K = 11604.50520
# Conversion factor from electron volts to Joules.
EV_TO_J = 1.602_177_33e-19
# Conversion factor from nanometers to meters.
NM_TO_M = 1e-9
# Conversion factor from centimeters to meters.
CM_TO_M = 1e-2
# Conversion factor from kilometers to meters.
KM_TO_M = 1e3
# Conversion factor from ergs to Joules.
ERG_TO_JOULE = 1e-7
# Conversion factor from grams to kilograms.
G_TO_KG = 1e-3
# Conversion factor from microns to nanometers.
MICRON_TO_NM = 1e3
# Conversion factor from megabarns to square meters.
MEGABARN_TO_M2 = 1e-22
# Conversion factor from atmospheres to Pascal.
ATM_TO_PA = 1.0135e5
# Conversion factor from dynes per square centimeter to Pascal.
DYNE_CM2_TO_PASCAL = 0.1
# Conversion factor from statvolts to volts.
STATV_TO_V = 299.792458

# Fraction of a mass of plasma assumed to be made up of hydrogen.
HYDROGEN_MASS_FRACTION = 0.735
# Conversion factor from mass density [g] to electron density [1/cm^3],
# assuming a fully ionized plasma with no metals and the hard-coded value for
# the hydrogen mass fraction.
MASS_DENSITY_TO_ELECTRON_DENSITY = (1.0 + HYDROGEN_MASS_FRACTION) / (2.0 * AMU)

# Unit for length [cm]
U_L = 1e8
# Unit for time [s]
U_T = 1e2
# Unit for mass density [g/cm^3]
U_R = 1e-7
# Unit for speed [cm/s]
U_U = U_L / U_T
# Unit for pressure [dyn/cm^2]
U_P = U_R * (U_L / U_T) * (U_L / U_T)
# Unit for Rosseland opacity [cm^2/g]
U_KR = 1.0 / (U_R * U_L)
# Unit for energy per mass [erg/g]
U_EE = U_U * U_U
# Unit for energy per volume [erg/cm^3]
U_E = U_R * U_EE
# Unit for thermal emission [erg/(s ster cm^2)]
U_TE = U_E / U_T * U_L
# Unit for volume [cm^3]
U_L3 = U_L * U_L * U_L
# Unit for magnetic flux density [gauss]
U_B = U_U * np.sqrt(4.0 * np.pi * U_R)
# Unit for electric field strength [statV/cm]. (The first factor is to account
# for that E is premultiplied with c in Bifrost)
U_EL =  (1.0 / (CLIGHT / U_U)) * U_B * U_U / CLIGHT
# Unit for power density [erg/s/cm^3]
U_PD = U_E / U_T
# Unit for power [erg/s]
U_PW = U_L3 * U_PD
# Unit for energy [erg]
U_EN = U_PW * U_T
