#  main
import pandas as pd
import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

from sys  import argv 

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
def CovarianceMatrix( dataSerie , M ):
    correlationList = []
    for lag in xrange( 0, M):
        correlationList.append( float( dataFrame["Datos"].autocorr(lag) )  )

    correlationNp = np.array(correlationList)

    return toeplitz (correlationNp ) # return covariance matrix 
    

def YMatrix ( dataSerie , M ) :
    # calculate the "Y" matriz, aka shifted components of the original data series. 
    # tiene que ser un arreglo de numpy
    
    Y = dataSerie.copy()
    size = Y.size
    for lag in xrange(1,M):
        Yparcial = np.roll( dataSerie , -lag)
        Yparcial[ (size-lag) :(size) ] = 0

        Y = np.concatenate( ( Y , Yparcial)  )

    Y = Y.reshape( M , size ).T

    return Y 


def principalComponents( serieDatos , M  ):

    CovMatrix = CovarianceMatrix(serieDatos , M)
    Ydata = YMatrix( serieDatos , M)

    EigenVals , EigenVecs = np.linalg.eig( CovMatrix )

    PCs = np.matmul( Ydata , EigenVecs.T )

    return PCs , EigenVecs


def reconstructedComponents(  Pcs , index , M , EigenVecs ):

    Z = calculateZ ( Pcs , index , M)
    RC1 = np.matmul( Z  , EigenVecs[:][index].T   )/M

    return RC1
    

def mainProgram( serieDatos , M ):

    CovMatrix = CovarianceMatrix( serieDatos , M)

    PCS , EigenVs = principalComponents( serieDatos , M)

    RC1 = reconstructedComponents (  PCS , 0 , M , EigenVs )

    
    return PCS , RC1
    
    
    
    
dataFolder = "../data/"
dataFile =  dataFolder + "series1.xlsx"

sheetName = "Corabastos2"
## abrir archivo

xFile = pd.ExcelFile(dataFile)
dataFrame = xFile.parse(sheetName)

M = 4  # lags

correlationList = []

datosNumpy = dataFrame["Datos"].as_matrix()


PC , Rc1 = mainProgram( datosNumpy , M)
  

plt.plot( Rc1  )
plt.plot( datosNumpy )

plt.draw()
