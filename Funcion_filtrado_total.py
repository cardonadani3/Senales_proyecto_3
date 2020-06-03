# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 16:44:27 2020

@author: user
"""

import pywt
import numpy as np
import matplotlib.pyplot as plt;
import librosa
import scipy.signal as signal
from linearFIR import filter_design


def filtrado_total(filename):
    
    def recibir_filter(filename): 
        
        y, sr = librosa.load(filename) #signal and sampling rate
        fs = sr;
        order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0);
        order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);
        y_hp = signal.filtfilt(highpass, 1, y);
        y_bp = signal.filtfilt(lowpass, 1, y_hp);
        data = np.asfortranarray(y_bp)
        return(data,sr)
    
    data, sr = recibir_filter(filename)
    
    def wavelet(data):
        
        def wthresh(coeff,thr):
            y   = list();
            s = wnoisest(coeff);
            for i in range(0,len(coeff)):
                y.append(np.multiply(coeff[i],np.abs(coeff[i])>(thr*s[i])));
            return y; 
      
        def thselect(y_bp):
            Num_samples = 0;
            for i in range(0,len(y_bp)):
                Num_samples = Num_samples + y_bp[i].shape[0];
            thr = np.sqrt(2*(np.log(Num_samples)))
            return thr  
    
        def wnoisest(coeff):
            stdc = np.zeros((len(coeff),1));
            for i in range(1,len(coeff)):
                stdc[i] = (np.median(np.absolute(coeff[i])))/0.6745;
            return stdc;

        LL = int(np.floor(np.log2(data.shape[0])));

        coeff = pywt.wavedec( data, 'db6', level=LL );

        thr = thselect(coeff);
        coeff_t = wthresh(coeff,thr);
    
        x_rec = pywt.waverec( coeff_t, 'db6');

        x_rec = x_rec[0:data.shape[0]];
    
        x_filt = np.squeeze(data - x_rec);    
        
        return(x_filt,x_rec)
        
    x_filt, x_rec = wavelet(data)
    
    return( x_filt,x_rec,data)
        
    
        
        
origi, sr = librosa.load("101_1b1_Al_sc_Meditron.wav")

x_filt, x_rec, data= filtrado_total("101_1b1_Al_sc_Meditron.wav") 

plt.plot(origi[0:800],label = 'Senal Original') 
plt.plot(data[0:800],label='Filtro lineal')
plt.plot(x_filt[0:800],label='Senal con Filtro Lineal - Umbralizada')
plt.plot(x_rec[0:800],label='Umbralizada por Wavelet')
plt.legend()
plt.grid()

#filter_signal = wavelet(uno)
#plt.plot(filter_signal)   