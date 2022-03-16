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

# First all the photometry, both linear and log
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
        if arr[i] == -9.90e+01:
            arr[i] = np.nan
            
for arr in [F606W_fluxerr, F814W_fluxerr, F125W_fluxerr, F160W_fluxerr]:
    for i in range(len(arr)):
        if arr[i] == -9.90e+01:
            arr[i] = np.nan

F606W_mag = -2.5 * np.log10(F606W_flux)
F814W_mag = -2.5 * np.log10(F814W_flux)
F125W_mag = -2.5 * np.log10(F125W_flux)
F160W_mag = -2.5 * np.log10(F160W_flux)

F606W_magerr = 2.5 / (F606W_flux * np.log(10)) * F606W_fluxerr
F814W_magerr = 2.5 / (F814W_flux * np.log(10)) * F814W_fluxerr
F125W_magerr = 2.5 / (F125W_flux * np.log(10)) * F125W_fluxerr
F160W_magerr = 2.5 / (F160W_flux * np.log(10)) * F160W_fluxerr

# Now for redshift
z_techs = ["Photo_z_Wuyts","Photo_z_Pforr","Photo_z_Wiklind","Photo_z_Finkelstein","Photo_z_Gruetzbauch","Photo_z_Salvato"]
z = np.zeros(z_tab["Spec_z"].value.size)
N = 0
for colname in z_techs:
    z += z_tab[colname].value
    N += 1
    
z /= N

lin_names = ["F606W flux (muJy)","F814W flux (muJy)","F125W flux (muJy)",
             "F160W flux (muJy)","F606W err (muJy)","F814W err (muJy)",
             "F125W err (muJy)","F160W err (muJy)","redshift"]

log_names = ["F606W (mag)","F814W (mag)","F125W (mag)","F160W (mag)",
             "F606W (magerr)","F814W (magerr)","F125W (magerr)","F160W (magerr)",
             "redshift"]

lin_tab = Table([F606W_flux,F814W_flux,F125W_flux,F160W_flux,
                 F606W_fluxerr,F814W_fluxerr,F125W_fluxerr,F160W_fluxerr,z],
                names=lin_names)

log_tab = Table([F606W_mag,F814W_mag,F125W_mag,F160W_mag,
                 F606W_magerr,F814W_magerr,F125W_magerr,F160W_magerr,z],
                names=log_names)

linfname = "mldata_lin.csv"
logfname = "mldata_log.csv"

lin_tab.write(linfname, format="csv", overwrite=True)
log_tab.write(logfname, format="csv", overwrite=True)