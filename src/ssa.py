

#  main



import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt 
dataFolder = "../data/"
dataFile =  dataFolder + "series1.xlsx"

sheetName = "Corabastos2"
## abrir archivo

xFile = pd.ExcelFile(dataFile)
dataFrame = xFile.parse(sheetName)

M = 4  # lags

correlationList = []

datosNumpy = dataFrame["Datos"].as_matrix()

datosOriginal = datosNumpy
size = datosNumpy.size




for lag in range(0, M):
    correlationList.append( float( dataFrame["Datos"].autocorr(lag) ) )
    if lag >= 1:
        datosNumpy2 = dataFrame["Datos"].as_matrix()
        datosNumpy2 = np.roll( datosNumpy2 , lag )
        datosNumpy = np.concatenate( (datosNumpy ,  datosNumpy2) , axis=0  ) 
    


correlationVector = np.array(correlationList)    
#cov = dataFrame["Datos"].autocorr(1) #.corr( dataFrame["Datos"])

## correlation matrix 
CovarianceMatrix = toeplitz (correlationVector )

w,v = np.linalg.eig( CovarianceMatrix )


#plt.show()

datosNumpy = np.reshape( datosNumpy , (  M , size ) )

PC = np.matmul( datosNumpy.T  , v ) # 

#  Construir
PCIndex = 0 ; ## este numero cambiara con

Z1 = PC.T[PCIndex][:]
Z2 = PC.T[PCIndex][:]
size = Z1.size

for lag in range(1, M):
    Zx = np.roll( Z2 , -1*lag)
    Zx[ (size-lag) :(size) ] = 0
    #print ( Z2 )
    Z1 = np.concatenate( ( Z1 , Zx )   )

Z1 = Z1.reshape(size , M )
print (  Z1.shape  )
print ( v[:][0] )

#plt.plot( PC )
#plt.show()


## componentes reconstruidas 

RC1 = np.matmul(  Z1 ,  v[:][0].T )/M
RC2 =  np.matmul(  Z1 ,  v[:][1].T )/M
RC3 = np.matmul(  Z1 ,  v[:][2].T )/M
RC4 =  np.matmul(  Z1 ,  v[:][3].T )/M

legends = [ ["Reco 1 "] , ["Reco 2"] , ["Reco 3"] , ["Reco 4"] ]
plt.legend( legends )
plt.plot( (RC1 + RC2 + RC3 + RC4) )
plt.plot( datosOriginal )

plt.show()
