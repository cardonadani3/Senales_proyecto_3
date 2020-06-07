# -*- coding: utf-8 -*-

import numpy as np
import scipy.signal as signal

def extraccion_ciclos(rutatxt, senal_filtrada,sr): #funcion que recibe un archivo txt, una senal y la frecuencia de muestreo
    ciclos=[]
    data=np.loadtxt(rutatxt) #se carga el archivo txt
    numciclos= len(data) 
    
    for i in range(numciclos): # for para recorrer los datos de los ciclos 
        senal=(senal_filtrada[int(round((data[i][0])*sr)):int(round((data[i][1])*sr))]) #extrae los ciclos de la senal
        ciclos.append(senal) #agrega los ciclos a una lista 
        
    return ciclos,data  # retorna la lista de los ciclos y la informacion del archivo txt en un array


def indices(ciclo1,sr): #funcion que recibe un ciclo, la frecuencia de muestreo y devueleve los inidices 
    sma_coarse=0
    ciclo1=np.array(ciclo1) # see convierte en array
    
    varianza= np.var(ciclo1) #se obtiene el primer indice:varianza
    
    ind_range= np.amax(ciclo1)-np.amin(ciclo1)  # se obtiene el segundo indice: rango
    

    f, Pxx = signal.welch(ciclo1,sr,'hamming', 1024 , scaling='density');
    ind_Spect_mean=np.mean(Pxx) # se obtiene el tercer indice:media del espectro
    
    for i in range(len(ciclo1)-1): # se obtiene el cuarto indice: SMA fine (Suma de media móvil simple)
    	resta=np.abs(ciclo1[i]-ciclo1[i+1])  
    	sma_coarse= sma_coarse+resta
        
    # se obtiene el quinto indice: SMA Coarse     
    maximo=[]
    sumatoria=0
    for i in range(0,len(ciclo1),100):
        w1=ciclo1[i:i+800]
        
        for j in range(len(w1)-1):
            restafine=np.abs(w1[j]-w1[j+1])
            sumatoria= sumatoria+restafine
        
        maximo.append(sumatoria)
        sumatoria=0
            
    sma_fine=max(maximo)
   
    return(varianza,ind_range,sma_coarse, ind_Spect_mean,sma_fine)