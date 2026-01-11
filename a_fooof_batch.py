# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:28:41 2025

@author: 86132
"""

from neurodsp import spectral
from fooof.utils import interpolate_spectrum
import numpy as np
from fooof import FOOOFGroup
import h5py
import os
import re
import pickle

def convert_knee_val(knee, exponent=2.):
    knee_freq = knee**(1./exponent)
    knee_tau = 1./(2*np.pi*knee_freq)
    return knee_freq, knee_tau

folder = r'F:\simulate_fooof\4_awake_preproc_data'
save_folder = r'F:\simulate_fooof\10_fooof_exp'

# correct order
def extract_number(filename):
    match = re.search(r"sub-(\d+)", filename)
    return int(match.group(1)) if match else float('inf')
file_list = [f for f in os.listdir(folder) if f.endswith('.mat')]
file_list_sorted = sorted(file_list, key=extract_number)

for filename in os.listdir(folder):
    if filename.endswith('.mat'):
        filepath = os.path.join(folder, filename)
        with h5py.File(filepath, 'r') as f:
            data_sub = f['data'][:].T
            print(filename)
    
            # calculate psd
            fs = 1000
            PSD_all = []
            for i in range(data_sub.shape[1]):
                f_axis, PSD = spectral.compute_spectrum(data_sub[:, i], fs, f_range=[0.1, 195],
                                                        nperseg=5000, noverlap=2500)
                PSD_all.append(PSD)
                # plt.figure()
                # plt.loglog(f_axis, PSD)
                
            # deal with line noise
            spec_int_all = []
            interp_ranges = [[47, 53], [97, 103], [147, 153]]
            for j in range(data_sub.shape[1]):
                freqs_int, spec_int = interpolate_spectrum(f_axis, PSD_all[j], interp_ranges)
                spec_int_all.append(spec_int)
                # plt.figure()
                # plt.loglog(freqs_int, spec_int)
            
            # fit with fooof
            spec_int_array = np.array(spec_int_all)
            
            fg = FOOOFGroup(aperiodic_mode='knee', max_n_peaks=3, peak_width_limits=[1,12],
                            min_peak_height=0.2, peak_threshold=2)
            fg.fit(freqs=freqs_int, power_spectra=spec_int_array, freq_range=(0.1,195))
            
            fit_exp_all = []
            knee_freq_all = []
            taus_all = []
            for h in range(data_sub.shape[1]):
                fm = fg.get_fooof(ind=h, regenerate=True)
                fit_exp = fm.get_params('aperiodic_params', 'exponent')
                fit_knee = fm.get_params('aperiodic_params', 'knee')
                
                knee_freq, taus = convert_knee_val(fit_knee, fit_exp)
                fit_exp_all.append(fit_exp)
                knee_freq_all.append(knee_freq)
                taus_all.append(taus)
                
            # save pkl
            out_name = filename.replace('.mat', '.pkl')
            out_path = os.path.join(save_folder, out_name)
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'fit_exp_all': fit_exp_all,
                    'knee_freq_all': knee_freq_all,
                    'taus_all': taus_all
                }, f)