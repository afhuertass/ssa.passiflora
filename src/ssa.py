

#  main



import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt 
dataFolder = "../data/"
dataFile =  dataFolder + "series1.xlsx"

sheetName = "Corabastos1"
## abrir archivo

xFile = pd.ExcelFile(dataFile)
dataFrame = xFile.parse(sheetName)

M = 10  # lags

correlationList = []

for lag in range(0, M):
    correlationList.append( float( dataFrame["Datos"].autocorr(lag) ) )

correlationVector = np.array(correlationList)    
#cov = dataFrame["Datos"].autocorr(1) #.corr( dataFrame["Datos"])

## correlation matrix 
CovarianceMatrix = toeplitz (correlationVector )

w,v = np.linalg.eig( CovarianceMatrix )
print (  v  )
