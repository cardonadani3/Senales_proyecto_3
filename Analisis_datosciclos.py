# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 08:28:56 2020

@author: user
"""

#importemos las librerias (omito tildes en el codigo)
import pandas as pd #Manejo de los dataframes de datos
import matplotlib.pyplot as plt #Graficacion
import seaborn as sns #Graficacion
import numpy as np #Manipulacion de matrices
from scipy import stats
import statsmodels.api as sm

datos = pd.read_csv('datosciclos3.csv');
estadist = datos.describe()


sano = datos.loc[:,'Estado']==0
datos_sano = datos.loc[sano]

crepitancia = datos.loc[:,'Estado']==1
datos_crepitancia = datos.loc[crepitancia]

sibilancia = datos.loc[:,'Estado']==2
datos_sibilancia = datos.loc[sibilancia]

ambos = datos.loc[:,'Estado']==3
datos_ambos = datos.loc[ambos]
 
#%%  Analisis de distribucion de cada variable en el data Frame (qqPlot) 

left = -1.8   #x coordinate for text insert
fig = plt.figure(1,[10,10])

ax = fig.add_subplot(3, 2, 1)
sm.graphics.qqplot(datos_crepitancia['Varianza'], line = 's',ax=ax)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, 'Varianza', verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1,))

ax = fig.add_subplot(3, 2, 2)
sm.graphics.qqplot(datos_crepitancia['Rango'], line='s', ax=ax)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "Rango", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(3, 2, 3)
sm.graphics.qqplot(datos_crepitancia['SMA Fine'], line='s', ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "SMA fine", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(3, 2, 4)
sm.graphics.qqplot(datos_crepitancia['SMA Coarse'], line='s', ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "SMA coarse", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(3, 2, 5)
sm.graphics.qqplot(datos_crepitancia['Media del espectro'], line='s', ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "Media del espectro",verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

fig.tight_layout()
fig.suptitle('Quantile-Quantile Plot para cada índice de los sujetos con crepitancias',fontsize=14,y =1.05)
plt.gcf()
plt.savefig('qqplot_crepitancia.png')


#%%  Analisis de distribucion de cada variable en el data Frame (qqPlot) 

left = -1.8   #x coordinate for text insert
fig = plt.figure(12,[10,8])

ax = fig.add_subplot(3, 2, 1)
sm.graphics.qqplot(datos_sibilancia['Varianza'], line = 's',ax=ax)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, 'Varianza', verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1,))

ax = fig.add_subplot(3, 2, 2)
sm.graphics.qqplot(datos_sibilancia['Rango'], line='s', ax=ax)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "Rango", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(3, 2, 3)
sm.graphics.qqplot(datos_sibilancia['SMA Fine'], line='s', ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "SMA fine", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(3, 2, 4)
sm.graphics.qqplot(datos_sibilancia['SMA Coarse'], line='s', ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "SMA coarse", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

ax = fig.add_subplot(3, 2, 5)
sm.graphics.qqplot(datos_sibilancia['Media del espectro'], line='s', ax=ax)
ax.set_xlim(-2, 2)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "Media del espectro",verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))

fig.tight_layout()
fig.suptitle('Quantile-Quantile Plot para cada índice de los sujetos con sibilancias',fontsize=14,y =1.05)
plt.gcf()
plt.savefig('qqplot_sibilancias.png')

#%% Analisis de distribucion de cada variable en el data Frame (Histograma)

fig = plt.figure(13, [11,12])

plt.subplot(3,2,1);plt.hist(datos['Varianza'])
plt.ylabel('Frecuencia');plt.xlabel('Varianza')

plt.subplot(3,2,2);plt.hist(datos['Rango'])
plt.ylabel('Frecuencia');plt.xlabel('Rango ')

plt.subplot(3,2,3);plt.hist(datos['SMA Fine'])
plt.ylabel('Frecuencia');plt.xlabel('Probabilidad movil delgado')

plt.subplot(3,2,4);plt.hist(datos['SMA Coarse'])
plt.ylabel('Frecuencia');plt.xlabel('Probabilidad movil gruesa')

plt.subplot(3,2,5);plt.hist(datos['Media del espectro'])
plt.ylabel('Frecuencia');plt.xlabel('Promedio del espectro')

fig.suptitle('Histograma para cada índice del data Frame',fontsize=16)

plt.savefig('histogram.png')

#%% Analisis de caja de bigotes 
figs = plt.figure(14,[11,11])

plt.subplot(3,2,1); sns.boxplot(x = 'Estado', y = 'Varianza',data = datos)
plt.subplot(3,2,2); sns.boxplot(x = 'Estado', y = 'Rango',data = datos) 
plt.subplot(3,2,3); sns.boxplot(x = 'Estado', y = 'SMA Fine',data = datos)
plt.subplot(3,2,4); sns.boxplot(x = 'Estado', y = 'SMA Coarse',data = datos)
plt.subplot(3,2,5); sns.boxplot(x = 'Estado', y = 'Media del espectro',data = datos)

figs.suptitle('Caja de bigotes de los indices vs los estados ',fontsize=16)
plt.savefig('bigotes.png')

#%% Correlacion de Person 


#datos.drop(['Crepitancias','Sibilancias','Normal','Sibilancias y Crepitancias'],axis='columns', inplace=True)
correlation_matrix = datos.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.figure(15,[10,8])
plt.title('Grafico de correlacion')
plt.savefig('Corelacion de variables',fontsize=16, y = 1.05)


#%% Histograma de varianza 

fig = plt.figure(16,[12,5])

ax = fig.add_subplot(1, 2, 1)
sm.graphics.qqplot(datos_sano['Varianza'], line = 's',ax=ax)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, 'Varianza de los sanos', verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1,))

ax = fig.add_subplot(1, 2, 2)
sm.graphics.qqplot(datos_crepitancia['Varianza'], line='s', ax=ax)
top = ax.get_ylim()[1] * 0.5
txt = ax.text(left, top, "Varianza de crepitancia", verticalalignment='top')
txt.set_bbox(dict(facecolor='k', alpha=0.1))


#%% Aplicacion del test de Mann - Whitney 
from scipy.stats import mannwhitneyu


    
def analisis_mwhitney(data1,data2):
    Estadistica = []
    probabilidad = []
    sentencia = []
    for i in np.arange(0,5):
        
        dat = data1.iloc[:,i]
        data = data2.iloc[:,i]
        stat, p = mannwhitneyu(dat, data)
        Estadistica = np.append( Estadistica, stat)
        probabilidad = np.append(probabilidad, p)
        alpha = (0.05/4)
        if p > alpha:
            sentencia = np.append(sentencia, 'No rechazar la hipotesis nula H0')
        else:
            sentencia = np.append(sentencia,'Rechazar hipotesis nula H0')
            
        matriz = np.vstack((Estadistica,probabilidad,sentencia)).T
    
    return matriz

matriz_sano_crepitancia = analisis_mwhitney(datos_sano,datos_crepitancia)
matriz_sano_sibilancia = analisis_mwhitney(datos_sano,datos_sibilancia)
matriz_crepitancia_sibilancia = analisis_mwhitney(datos_crepitancia,datos_sibilancia)
matriz_ambos = analisis_mwhitney(datos_sano,datos_ambos)
    

    
    