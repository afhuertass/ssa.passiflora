

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

print (  datosNumpy )
print ( v.shape )

plt.plot( PC )
plt.show()
