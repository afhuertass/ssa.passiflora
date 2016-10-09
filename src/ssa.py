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
def CovarianceMatrix( dataSerie , M , dataPandas  ):
    correlationList = []
    for lag in xrange( 0, M):
        correlationList.append( float( dataPandas.autocorr(lag) )  )

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


def principalComponents( serieDatos , M , dataPandas  ):

    CovMatrix = CovarianceMatrix(serieDatos , M , dataPandas )
    Ydata = YMatrix( serieDatos , M)

    EigenVals , EigenVecs = np.linalg.eig( CovMatrix )

    PCs = np.matmul( Ydata , EigenVecs.T )

    return PCs , EigenVecs


def reconstructedComponents(  Pcs , M , EigenVecs ):

    #Z = calculateZ ( Pcs , index , M)
    RCS = []
    for r in range(0,M):
        Z = calculateZ ( Pcs , r , M)
        RC = np.matmul( Z  , EigenVecs[:][r].T   )/M
        RCS.append(  RC )
    return RCS
    

def mainProgram( serieDatos , M , dataPandas ):

    # CovMatrix = CovarianceMatrix( serieDatos , M ,dataFrame )

    PCS , EigenVs = principalComponents( serieDatos , M , dataPandas )
    RCS = reconstructedComponents( PCS , M , EigenVs )
    return PCS , ( RCS  )
    
    
def getData( dataPath , sheetName, col ):

    xFile = pd.ExcelFile(dataPath)
    dataFrame = xFile.parse( sheetName )

    return dataFrame[col].as_matrix() , dataFrame[col]


    
dataPath = "../data/series1.xlsx"
sheetName = "AgronetExport2"
col = "Datos"
dataSeries , dataPandas  = getData( dataPath , sheetName , col)
## abrir archivo
M = 6  # lags

PC , Rc1 = mainProgram( dataSeries , M , dataPandas)
  

#plt.plot( Rc1  )
plt.plot( dataSeries  )
for i in range(0, M):
    
    plt.plot( Rc1[i] )
    
plt.show()
