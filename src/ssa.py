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

    return toeplitz (correlationVector ) # return covariance matrix 
    

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
    Ydata = YMatrix( dataSerie , M)

    EigenVals , EigenVecs = np.linalg.eig( CovMatrix )

    PCs = np.matmul( Ydata , EigenVecs.T )

    return PCs , EigenVecs


def reconstructedComponents(  Pcs , index , M , EigenVecs ):

    Z = calculateZ ( Pcs , index , M)
    RC1 = np.matmul( Z  , Eigenvecs[:][index].T   )/M

    return RC1
    

def mainProgram( serieDatos , M ):

    CovMatrix = CovarianceMatrix( serieDatos , M)

    PCS , EigenVs = principalComponents( serieDatos , M)

    RC1 = reconstructedComponents (  PCS , 0 , M , EigenVs )

    
    
    
    
    
dataFolder = "../data/"
dataFile =  dataFolder + "series1.xlsx"

sheetName = "Corabastos2"
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



#plt.show()



PC = np.matmul( datosNumpy  , v.T ) # 
print("Componentes principales")
print( PC)
print( "Primera componente principal")
print( PC.T[0][:] )

#  Construir
PCIndex = 0 ; ## este numero cambiara con


Z1 = calculateZ( PC , 0 , M)
Z2 = calculateZ( PC , 1 , M)
Z3 = calculateZ(PC , 2 , M)
Z4 = calculateZ(PC , 3 , M)

print("Z4")
print( Z4 ) 

RC1 = np.matmul( Z1 , v[:][0].T)/M
RC2 = np.matmul(Z2 , v[:][1].T)/M
RC3 = np.matmul(Z3 , v[:][2].T)/M
RC4 = np.matmul(Z4 , v[:][3].T)/M

RC = RC1 + RC2 + RC3 + RC4
print("Reconstructed")
print(RC1)



plt.plot( (RC1 ) )
plt.plot( (RC2 ) )
#plt.plot( (RC3 ) )
#plt.plot( (RC ) )
plt.plot( datosOriginal )

plt.show()
