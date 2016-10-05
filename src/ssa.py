#  main
import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt




def calculateZ(  pcs , indice , M ): # componentes principales
    Z1s = pcs.T[int(indice)][:]
    Z2s = pcs.T[int(indice)][:]
    size = Z1s.size
    
    for lag in range(1, M):
        Zx = np.roll( Z2s , lag)
        Zx[ 0 : lag ] = 0
        #print ( Z2 )
        Z1s = np.concatenate( ( Z1s , Zx )   )

    Z1s =  Z1s.reshape(  M , size ).T

    return Z1s

    
dataFolder = "../data/"
dataFile =  dataFolder + "series1.xlsx"

sheetName = "AgronetExport"
## abrir archivo

xFile = pd.ExcelFile(dataFile)
dataFrame = xFile.parse(sheetName)

M = 4  # lags

correlationList = []

datosNumpy = dataFrame["Datos"].as_matrix()
datosNumpy = datosNumpy # columna

datosOriginal = datosNumpy
size = datosNumpy.size




for lag in range(0, M):
    correlationList.append( float( dataFrame["Datos"].autocorr(lag) ) )
    if lag >= 1:
        datosNumpy2 = dataFrame["Datos"].as_matrix()
        datosNumpy2 = np.roll( datosNumpy2 , -lag )
        #datosNumpy2[0:lag] = 0
        datosNumpy2[ (size-lag) :(size) ] = 0
        print(datosNumpy2)
        datosNumpy = np.concatenate( (datosNumpy ,  datosNumpy2)   ) 
    
datosNumpy = datosNumpy.reshape( M , size ).T
print( "Matriz Y" )
print (datosNumpy) # matriz Y del paper 

correlationVector = np.array(correlationList)    
#cov = dataFrame["Datos"].autocorr(1) #.corr( dataFrame["Datos"])

## correlation matrix 
CovarianceMatrix = toeplitz (correlationVector )


w,v = np.linalg.eig( CovarianceMatrix )

print ( v[:][0] ) 

#plt.show()



PC = np.matmul( datosNumpy  , v.T ) # 
print("Componentes principales")
print( PC )

#  Construir
PCIndex = 0 ; ## este numero cambiara con

Z1 = PC.T[PCIndex][:]
Z2 = PC.T[PCIndex][:]
size = Z1.size

for lag in range(1, M):
    Zx = np.roll( Z2 , lag)
    Zx[ 0 : lag ] = 0
    #print ( Z2 )
    Z1 = np.concatenate( ( Z1 , Zx )   )

print("MATRIZ Z")
Z1 = Z1.reshape(  M , size ).T
print (  Z1  )


#plt.plot( PC )

Z1 = calculateZ( PC , PCIndex , M);
print(Z1)

## componentes reconstruidas 

#RC1 = np.matmul(  Z1 ,  v[:][0].T )/M
#RC2 =  np.matmul(  Z1 ,  v[:][1].T )/M
#RC3 = np.matmul(  Z1 ,  v[:][2].T )/M
#RC4 =  np.matmul(  Z1 ,  v[:][3].T )/M

RC = np.matmul( Z1 , v[:][PCIndex].T)/M

print("Reconstructed")
print(RC)


legends = [ ["Reco 1 "] , ["Reco 2"] , ["Reco 3"] , ["Reco 4"] ]
plt.legend( legends )
plt.plot( RC )
plt.plot( datosOriginal )

#plt.show()
