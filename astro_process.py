#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 15:57:04 2022

@author: skusmic

This reads in the FITS catalogs, outputs processed data table.
"""

from astropy.table import Table
import numpy as np

photom_tab = Table.read("hlsp_candels_hst_wfc3_cos-tot-multiband_f160w_v1-1photom_cat.fits")
z_tab = Table.read("hlsp_candels_hst_wfc3_cos_v1_photoz_cat.fits")

# First all the photometry, both linear and log, reading them in
F606W_flux = photom_tab["ACS_F606W_FLUX"].value
F814W_flux = photom_tab["ACS_F814W_FLUX"].value
F125W_flux = photom_tab["WFC3_F125W_FLUX"].value
F160W_flux = photom_tab["WFC3_F160W_FLUX"].value

F606W_fluxerr = photom_tab["ACS_F606W_FLUXERR"].value
F814W_fluxerr = photom_tab["ACS_F814W_FLUXERR"].value
F125W_fluxerr = photom_tab["WFC3_F125W_FLUXERR"].value
F160W_fluxerr = photom_tab["WFC3_F160W_FLUXERR"].value

for arr in [F606W_flux, F814W_flux, F125W_flux, F160W_flux]:
    for i in range(len(arr)):
        if arr[i] < 0.:
            arr[i] = np.nan
            
for arr in [F606W_fluxerr, F814W_fluxerr, F125W_fluxerr, F160W_fluxerr]:
    for i in range(len(arr)):
        if arr[i] < 0.:
            arr[i] = np.nan

# Getting color index (ratios in linear)

F606W_F814W_flux = F606W_flux / F814W_flux
F814W_F125W_flux = F814W_flux / F125W_flux
F125W_F160W_flux = F125W_flux / F160W_flux


# and their errors
F606W_F814W_fluxerr = np.sqrt( (F606W_F814W_flux/F606W_flux)**2 * F606W_fluxerr**2
                              + (F606W_F814W_flux/F814W_flux)**2 * F814W_fluxerr**2)
F814W_F125W_fluxerr = np.sqrt( (F814W_F125W_flux/F814W_flux)**2 * F814W_fluxerr**2
                              + (F814W_F125W_flux/F125W_flux)**2 * F125W_fluxerr**2)
F125W_F160W_fluxerr = np.sqrt( (F125W_F160W_flux/F125W_flux)**2 * F125W_fluxerr**2
                              + (F125W_F160W_flux/F160W_flux)**2 * F160W_fluxerr**2)



F606W_mag = -2.5 * np.log10(F606W_flux)
F814W_mag = -2.5 * np.log10(F814W_flux)
F125W_mag = -2.5 * np.log10(F125W_flux)
F160W_mag = -2.5 * np.log10(F160W_flux)

F606W_magerr = 2.5 / (F606W_flux * np.log(10)) * F606W_fluxerr
F814W_magerr = 2.5 / (F814W_flux * np.log(10)) * F814W_fluxerr
F125W_magerr = 2.5 / (F125W_flux * np.log(10)) * F125W_fluxerr
F160W_magerr = 2.5 / (F160W_flux * np.log(10)) * F160W_fluxerr

# color index (subtraction in magnitudes)

F606W_F814W_mag = F606W_mag - F814W_mag
F814W_F125W_mag = F814W_mag - F125W_mag
F125W_F160W_mag = F125W_mag - F160W_mag

# ADDME errors in color index for mag
F606W_F814W_magerr = np.sqrt( (1. - F814W_mag)**2 * F606W_magerr**2
                             +(F606W_mag - 1.)**2 * F814W_magerr**2)
F814W_F125W_magerr = np.sqrt( (1. - F125W_mag)**2 * F814W_magerr**2
                             +(F814W_mag - 1.)**2 * F125W_magerr**2)
F125W_F160W_magerr = np.sqrt( (1. - F160W_mag)**2 * F125W_magerr**2
                             +(F125W_mag - 1.)**2 * F160W_magerr**2)


# Now for redshift
z_techs = ["Photo_z_Wuyts","Photo_z_Pforr","Photo_z_Wiklind","Photo_z_Finkelstein","Photo_z_Gruetzbauch","Photo_z_Salvato"]
z = np.zeros(z_tab["Spec_z"].value.size)
N = 0
for colname in z_techs:
    z += z_tab[colname].value
    N += 1
    
z /= N # averaging the redshifts

# ADDME split data files further into two, so 4 total file outputs: 
# lin estimates, lin errors, mag estimates, and mag errors

lin_est_names = ["F606W flux (muJy)","F814W flux (muJy)","F125W flux (muJy)",
                 "F160W flux (muJy)","F606W/F814W","F814W/F125W","F125W/F160W",
                 "redshift"]

lin_err_names = ["F606W err (muJy)","F814W err (muJy)",
                 "F125W err (muJy)","F160W err (muJy)","F606W/F814W err",
                 "F814W/F125W err","F125W/F160W err"]

log_est_names = ["F606W","F814W","F125W","F160W","F606W-F814W","F814W-F125W",
                 "F125W-F160W","redshift"]

log_err_names = ["F606W (magerr)","F814W (magerr)","F125W (magerr)","F160W (magerr)",
                 "F606W-F814W err","F814W-F125W err","F125W-F160W err"]

# Making Table objects for flux and magnitudes
lin_est_tab = Table([F606W_flux,F814W_flux,F125W_flux,F160W_flux,
                     F606W_F814W_flux,F814W_F125W_flux,F125W_F160W_flux,z],
                    names=lin_est_names)

lin_err_tab = Table([F606W_fluxerr,F814W_fluxerr,F125W_fluxerr,F160W_fluxerr,
                     F606W_F814W_fluxerr,F814W_F125W_fluxerr,F125W_F160W_fluxerr],
                    names=lin_err_names)

# Removing rows with nan
data_lin_est = np.lib.recfunctions.structured_to_unstructured(np.array(lin_est_tab))
hasNan_lin_est = np.any(np.isnan(data_lin_est), axis=1)
lin_tab_noNan_est = lin_est_tab[~hasNan_lin_est]

data_lin_err = np.lib.recfunctions.structured_to_unstructured(np.array(lin_err_tab))
hasNan_lin_err = np.any(np.isnan(data_lin_err), axis=1)
lin_tab_noNan_err = lin_err_tab[~hasNan_lin_err]

log_est_tab = Table([F606W_mag,F814W_mag,F125W_mag,F160W_mag,F606W_F814W_mag,
                     F814W_F125W_mag,F125W_F160W_mag,z],
                    names=log_est_names)

log_err_tab = Table([F606W_magerr,F814W_magerr,F125W_magerr,F160W_magerr,
                     F606W_F814W_magerr,F814W_F125W_magerr,F125W_F160W_magerr],
                    names=log_err_names)

# Removing rows with nan
data_log_est = np.lib.recfunctions.structured_to_unstructured(np.array(log_est_tab))
hasNan_log_est = np.any(np.isnan(data_log_est), axis=1)
log_tab_noNan_est = log_est_tab[~hasNan_log_est]

data_log_err = np.lib.recfunctions.structured_to_unstructured(np.array(log_err_tab))
hasNan_log_err = np.any(np.isnan(data_log_err), axis=1)
log_tab_noNan_err = log_est_tab[~hasNan_log_err]


# Splitting data estimates and uncertainties


linfname_est = "mldata_lin_est.csv"
linfname_err = "mldata_lin_err.csv"

logfname_est = "mldata_log_est.csv"
logfname_err = "mldata_log_err.csv"

lin_tab_noNan_est.write(linfname_est, format="csv", overwrite=True)
lin_tab_noNan_err.write(linfname_err, format="csv", overwrite=True)

log_tab_noNan_est.write(logfname_est, format="csv", overwrite=True)
log_tab_noNan_err.write(logfname_err, format="csv", overwrite=True)

print(len(log_tab_noNan_est), len(lin_tab_noNan_est))
