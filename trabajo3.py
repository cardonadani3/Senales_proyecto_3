# -*- coding: utf-8 -*-


#importemos las librerias (omito tildes en el codigo)
import pandas as pd #Manejo de los dataframes de datos


import glob
from funcion_filtrado import filtrado_total
from ciclos import extraccion_ciclos, indices



ruta_archivos = 'D:/Users/ISABEL/OneDrive - Universidad de Antioquia/20192/senales/Trabajo3/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'
#listando los archivos en el directorio
archivostxt= glob.glob(ruta_archivos + '/*.txt') # Genera una lista con los archivos txt 
archivoswav= glob.glob(ruta_archivos + '/*.wav') # Genera una lista con los archivos wav 

# se crean diferentes listas para guardar los infice y estados 
estados=[]
varianza=[]
rango=[]
smafine=[]
smacoarse=[]
spect_mean=[]
list_normal=[]
list_sibilancias=[]
list_crepitancias=[]
list_sibycrep=[]

for j in range (len(archivostxt)):
    
    rutawav = archivoswav[j]
    rutatxt= archivostxt[j]
    
       
    x_filt, x_rec, filtro1,sr= filtrado_total(rutawav) # Llama las funciones que realizan el preprocesamiento
    
    lista_ciclos,infociclos= extraccion_ciclos(rutatxt,x_filt,sr) # LLama las funciones que extrae los ciclos y la informacion de este 
    
    for i in range(len(infociclos)):
        if infociclos[i][2]==0 and infociclos[i][3]==0:
            estado=0
            normal=1
            sibilancias=0
            crepitancias=0
            sibycrep=0
            
        elif infociclos[i][2]==1 and infociclos[i][3]==0:
            estado=1
            normal=0
            sibilancias=0
            crepitancias=1
            sibycrep=0
            
        elif infociclos[i][2]==0 and infociclos[i][3]==1:
            estado=2
            normal=0
            sibilancias=1
            crepitancias=0
            sibycrep=0            
            
        elif infociclos[i][2]==1 and infociclos[i][3]==1:
            estado=3
            normal=0
            sibilancias=0
            crepitancias=0
            sibycrep=1            
    
        estados.append(estado) 
        list_normal.append(normal)
        list_sibilancias.append(sibilancias)
        list_crepitancias.append(crepitancias)
        list_sibycrep.append(sibycrep)
                
            
    
    for k in range(len(lista_ciclos)): # Recorre los ciclos de cada senal 
        ciclo= lista_ciclos[k]
        ind_varianza, ind_range,ind_sma_coarse,ind_Spect_mean,ind_sma_fine=indices(ciclo,sr) # llama la funcion que obtiene los indices de cada ciclo 
        # Se agregan los indices a listas
        varianza.append(ind_varianza)
        rango.append(ind_range)
        smafine.append(ind_sma_fine)
        spect_mean.append(ind_Spect_mean)
        smacoarse.append(ind_sma_coarse)
        
# se crea un dataframe con los datos obtenidos     
datosciclos = pd.DataFrame({'Varianza' : varianza, 'Rango' : rango, 'SMA Fine': smafine,'SMA Coarse': smacoarse, 'Media del espectro': spect_mean,'Crepitancias': list_crepitancias,'Sibilancias':list_sibilancias,'Normal':list_normal,'Sibilancias y Crepitancias':list_sibycrep, 'Estado':estados})

#Se guarda el dataframe 
datosciclos.to_csv('datosciclos3.csv', index=False)


