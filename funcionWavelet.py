# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:42:27 2020

@author: LENOVO
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from linearFIR import filter_design, mfreqz

#data = sio.loadmat('C001R_EP_reposo.mat')
#data = data["data"]

mat_contents = sio.loadmat('senal_prueba_wavelet.mat')
data = np.squeeze(mat_contents['senal']);
#sensores,puntos,ensayos=data.shape
#senal_continua=np.reshape(data,(sensores,puntos*ensayos),order="F")
#signal=(senal_continua[2,0:2000])
#signal=(data[0:1400])


#plt.figure(1)
#plt.plot(signal)



def recibirsenal(filename):
    import scipy.signal as signal
    y, sr = librosa.load(filename) #signal and sampling rate
    fs = sr;
    order, lowpass = filter_design(fs, locutoff = 0, hicutoff = 1000, revfilt = 0);
    order, highpass = filter_design(fs, locutoff = 100, hicutoff = 0, revfilt = 1);
    y_hp = signal.filtfilt(highpass, 1, y);
    y_bp = signal.filtfilt(lowpass, 1, y_hp);
    y_bp = np.asfortranarray(y_bp)
    return(y_bp,sr)


def transf_wavelet(y_bp,eleccion_thr,pond,umb):#analisis usando wavelet 
    import math
    import numpy as np
    
    def descomponer (senal_descomponer): 
        # funcion que recibe una senal y la descompone en detalles y la ultima aproximacion
        # retorna una lista con todos los detalles y otra con la ultima aproximacion
        wavelet = [-1/np.sqrt(2) , 1/np.sqrt(2)];
        scale = [1/np.sqrt(2) , 1/np.sqrt(2)];
    
        jmax= np.floor(math.log2(senal_descomponer.shape[0]/2)-1) # se halla cual es el numero de detalles apropiados
        aprox=[] # se crea la lista donde se va a guardar la aproximacion
        detalles=[] # se crea la lista donde se va a guardar los detalles 
        
        for i in range(int(jmax)): 
            # itera en numero de veces de los detalles que se necesitam
            # los va calculando y guardando 
            if (senal_descomponer.shape[0] % 2) != 0:
                print("Anadiendo ceros");
                senal_descomponer = np.append(senal_descomponer, 0);
            
            Aprox = np.convolve(senal_descomponer,scale,'full');
            #a partir del primero toma cada dos
            Aprox = Aprox[1::2];
            
            Detail = np.convolve(senal_descomponer,wavelet,'full');
            #a partir del primero toma cada dos
            Detail = Detail[1::2];
            
            detalles.append(Detail)
            aprox= Aprox
            #en los niveles siguientes descompongo las ultimas aproximaciones
            senal_descomponer= Aprox
            
        detalles_cont= detalles[::-1] #invierte la lista que se crea de detalles para
                                            #que quede organizado del ultimo detalle al primero

        return(aprox,detalles_cont)   
        
    aproxi,detailles= descomponer(y_bp)
    
    def seleccion_umbral(aproxi,detailles,eleccion_thr):  
            # funcion que calcula y devuelve el umbral que se elija: universal,minimax
            # Recibe la ultima aproximacion, los detalles y la eleccion de que umbral se quiere
    
            Num_samples= aproxi.shape[0]
            
            for dn in detailles:   
                Num_samples=Num_samples+dn.shape[0]
                
            if eleccion_thr == 1: # universal
                thr= np.sqrt(2*(np.log(Num_samples)))
                
            elif eleccion_thr == 2 : # minimax
                thr= 0.3936 + 0.1829*(np.log(Num_samples)/np.log(2))
                
#            elif eleccion_thr == 3: # sure
#                x= reconstruccion(aprox,detail,N) #se necesita la señal recostruida para proceder con cada muestra de estas
#                n = np.size(x) #se calcula el tamaño de esta señal
#                sx2 =np.sort(abs(x))**2 #de acuerdo a formula
#                c = np.linspace(n-1,0,n) #vector n-1:-1:0
#                s = np.cumsum(sx2)+c*sx2 #se calcula esta operacion de acuerdo a la formula
#                risks = (n -(2*np.arange(n))+s)/n #se define el vector de riesgo
#                best = np.argmin(risks) # el mejor es el minimo valor
#                sure = np.sqrt(sx2[best]) # se calcula de acuerdo a la formula
                
                       
            return(thr) 
            
    thr= seleccion_umbral(aproxi,detailles,eleccion_thr)   
    
    def ponderacion(details,pond):
        # funcion que calcula y devuelve el sigma que se elija para la ponderacion
        # recibe una lista de detalles y la eleccion del sigma que se quiere 
    
        if pond==1: # One
            stdc=np.ones((len(details),1))
            
        elif pond==2: # Single Level Noise (sln)
            stdc= np.zeros((len(details),1))
            for i in range(len(details)):    
              stdc[i] = np.median(np.absolute(details[len(details)-1]))/0.6745;
            
        elif pond==3: # Multiple Level Noise (mln):
            stdc=np.zeros((len(details),1))
            for i in range(len(details)):    
                stdc[i] = np.median(np.absolute(details[i]))/0.6745;
                
        return(stdc)    
    stdc= ponderacion(detailles,pond)
    
    def umbral(detalles,thr,stdc,umb): 
        # funcion que calcula y devuelve una lista de detalles al aplicarles los diferentes umrales
        # recibe una lista de detalles, un lambda, un sigma y la eleccion del umbral a plicar 
        
        umbrales=stdc*thr
        
        if umb==1: # umbral duro 
            i=0
            for dn in detalles:
                umbral=umbrales[i]  
                for x in range(len(dn)):
                    if np.absolute(dn[x])<umbral:
                        dn[x]=0
                    else:
                        dn[x]=dn[x]
                i=i+1
        
        elif umb==2: # umbral suave
            i=0
            for dn in detalles:
                umbral=umbrales[i]  
                for x in range(len(dn)):
                    if np.absolute(dn[x])<umbral:
                        dn[x]=0
                    else:
                        dn[x]=np.sign(dn[x])*(np.absolute(dn[x])-umbral)
                i=i+1
              
        detalles=detalles
        
        return(detalles)
        
    detallesumbral=umbral(detailles,thr,stdc,umb)
    
    def reconstruir(detail,aprox,signal):
        # funcion que reconstruye y devuelve una senal
        # recibe una lista de detalles( que debe ir desde el ultimo detalle hasta el primero)
        # recibe un array con la ultima aproximacion 
        
        wavelet_inv = [1/np.sqrt(2) , -1/np.sqrt(2)];
        scale_inv = [1/np.sqrt(2) , 1/np.sqrt(2)];
        longitud_original = signal.shape[0];
        
        for i in range(len(detail)):    
            
            npoints_aprox =aprox.shape[0];
            Aprox_inv3 = np.zeros((2*npoints_aprox));
            Aprox_inv3[0::2] = aprox;
            Aprox_inv3[1::2] = 0;
            
            APROX = np.convolve(Aprox_inv3,scale_inv,'full');    
            
            npoints_aprox = detail[i].shape[0];
            Detail_inv3 = np.zeros((2*npoints_aprox)); 
            Detail_inv3[0::2] = detail[i];
            Detail_inv3[1::2] = 0;
            
            DETAIL = np.convolve(Detail_inv3,wavelet_inv,'full');
            
            X3 = APROX + DETAIL;
            if i < len(detail)-1: 
                
             #por la expansión de ceros se pueden aumentar las muestres
                if X3.shape[0] > detail[i+1].shape[0]:
                    print("Quitando ceros");
                    X3 = X3[0:detail[i+1].shape[0]];
                aprox=X3 
                
            else:         
                senal_reconstruida = APROX + DETAIL;
                senal_reconstruida = senal_reconstruida[0:longitud_original];
                
        return(senal_reconstruida)
        
    senal_filtrada= reconstruir(detallesumbral,aproxi,signal)     
        
    return(senal_filtrada)
    
#senalfiltrada=transf_wavelet(signal,1,3,2)
#plt.figure(2)
#plt.plot(senalfiltrada)
filename = './example_data/101_1b1_Pr_sc_Meditron.wav'
filtro1 = recibirsenal(filename)
senalfiltrada=transf_wavelet(filtro1,1,3,2)
plt.plot(senalfiltrada)
    