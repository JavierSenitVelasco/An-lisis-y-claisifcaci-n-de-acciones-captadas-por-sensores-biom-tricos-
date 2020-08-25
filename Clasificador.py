    
from abc import ABCMeta,abstractmethod
import numpy as np
import statistics as stats
import scipy.stats
import matplotlib.pyplot as plt
import math
from math import sqrt

from operator import itemgetter
import random

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion
  # de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
 
  def error(self,datos,pred):
    totalDatos = len(datos)
    error = 0
    for i in range (totalDatos):
      if (datos[i][-1]!=pred[i]):
        error+=1
    return error/totalDatos
    
  
  def calcularMediasDesv(self,datostrain,numAtributos):
      self.medias = []
      self.desviaciones = []
      
      #por cada columna se calcula la media y la desviacion de todos sus valores
      for i in range (numAtributos): 
          atributo = datostrain[:,i]
          self.medias.append(np.mean (atributo))
          self.desviaciones.append (np.std (atributo))

  def normalizarDatos (self, datos, extraeMediasDesv,numAtributos):
    if (extraeMediasDesv):
        self.calcularMediasDesv(datos,numAtributos)
        
    #cada valor de la matriz sera normalizado restandole la media y dividiendo entre la desviacion    
    for i in range (len(datos)):
        for j in range (numAtributos): 
            datos[i][j] = (datos[i][j]-self.medias[j])/self.desviaciones[j]
    return datos

  
  def validacion(self,particionado,dataset,clasificador,seed=None):
    
    random.seed(seed)
    errores=[]
    
    self.posterioris=[]
    self.predicciones=[]
    
    
    particionado.creaParticiones(dataset.datos)
   
    
    #este bucle es el que se encarga de ejecutar todo, hace particiones, extrae datos, entrena y clasifica
    for i in range (len(particionado.particiones)):
      datosTrain = dataset.extraeDatos (particionado.particiones[i].indicesTrain)
      datosTest = dataset.extraeDatos (particionado.particiones[i].indicesTest)
      self.entrenamiento(datosTrain,dataset.nominalAtributos, dataset.diccionarios)
      self.clasifica (datosTest,dataset.nominalAtributos,dataset.diccionarios)
      errores.append(self.error(datosTest,self.predicciones[i]))
      
    self.media_error=np.mean(errores)
    
    return self.media_error
    
  
  #Codigo para generar las curvas ROC    
  def curvaROC (self,dataset,estrategia):
    
    probsPosterioris = self.posterioris
    
    for n in range (0, len(estrategia.particiones)):
     
      totalPositivos = 0
      totalNegativos = 0
      datosTest = dataset.extraeDatos(estrategia.particiones[n].indicesTest)
      for i in range (len(datosTest)):
        if (datosTest[i][-1]==0):
          totalPositivos +=1
        else:
          totalNegativos +=1
      
      curva = []
      
      for i in range (len(datosTest)):
        prob_clase = []
        prob_clase.append(probsPosterioris[n][i][0])
        prob_clase.append(datosTest[i][-1])
        curva.append(prob_clase)
      
      curva = sorted(curva, key=lambda curva_entry: curva_entry[0], reverse=True) 
      
      listaVerdaderosPositivos = [0.0]
      listaFalsosPositivos= [0.0]
      verdaderosPositivos = 0
      falsosPositivos = 0
      
      for i in range (len(datosTest)):
        if (curva[i][1]==0):
          verdaderosPositivos+=1
          listaVerdaderosPositivos.append(verdaderosPositivos/float(totalPositivos))
          listaFalsosPositivos.append(falsosPositivos/float(totalNegativos))
        else:
          falsosPositivos+=1
          listaVerdaderosPositivos.append(verdaderosPositivos/float(totalPositivos))
          listaFalsosPositivos.append(falsosPositivos/float(totalNegativos))
      lw = 2
      if (n==0):
        plt.clf()
        plt.figure()
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Ratios de falsos positivos')
        plt.ylabel('Ratio de verdaderos positivos')
        plt.title('Curva ROC')
        

      cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
      
      plt.plot(listaFalsosPositivos, listaVerdaderosPositivos, color=cycle[n],label='Particion {0} '
              ''.format(n),
              lw=lw)
      if(n==(len(estrategia.particiones)-1)):
        plt.legend(loc="lower right")
        plt.show()   

 
##############################################################################

class ClasificadorNaiveBayes(Clasificador):

  def __init__ (self,laplace=True,normalizar=False):
    self.laplace=laplace
    self.normalizar=normalizar

  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
    self.prioris = []
    
    
    aparicionesClase = np.zeros(len (diccionario[-1]))
    for i in range ((len(datostrain))):
      for j in range (len (diccionario[-1])):
        if (datostrain[i][-1]==j):
          aparicionesClase[j]+=1
    for i in range (len (diccionario[-1])):
      self.prioris.append(aparicionesClase[i]/(len(datostrain)))
    

    #print("Las probs a priori son: " + str(self.prioris))
    #print(datostrain)
    valoresAtributos = []  #Aqui se guardan las verosimilitudes          
    for i in range (len(diccionario)-1):  
      if (atributosDiscretos[i]):
        numValoresAtributo = len (diccionario[i])
        tablasDeOcurrencia = np.zeros ((len (diccionario[i]),len (diccionario[-1])))

        #<-----At0------>  <-----At1----->   <-----At2----->
        #[[2.0, 1.0, 2.0], [1.0, 2.0, 2.0], [1.0, 0.0, 4.0]]          
        #  c0    c1   c2     c0   c1  c2     c0    c1   c2
        for j in range(len(datostrain)):
          for k in range (numValoresAtributo):
            if (datostrain[j][i]==k):
              for v in range (len (diccionario[-1])):
                if (v==datostrain[j][-1]):
                  tablasDeOcurrencia[k][v]+=1
        
        #print(tablasDeOcurrencia)
        #Se calculan las verosimilitudes diviendo entre el numero de apariciones de cada clase
        for j in range (numValoresAtributo):
          for k in range (len (diccionario[-1])):
            if (tablasDeOcurrencia[j][k]==0):
              tablasDeOcurrencia[j][k]= 1 #laplace
            tablasDeOcurrencia [j][k] = tablasDeOcurrencia[j][k]/aparicionesClase[k]
        
      else:
        tablasDeOcurrencia = np.zeros ((len (diccionario[-1]),2))
        for k in range (len (diccionario[-1])):
            
          
          valoresDeCadaClase=[]
          for j in range ((len(datostrain))):
            if (datostrain[j][-1]==k):
              valoresDeCadaClase.append(datostrain[j][i])
        
          tablasDeOcurrencia[k][0]=np.mean(valoresDeCadaClase)
          tablasDeOcurrencia[k][1]=np.std(valoresDeCadaClase)
      valoresAtributos.append (tablasDeOcurrencia)
    
    self.valoresTrain=valoresAtributos
    

  def clasifica(self,datostest,atributosDiscretos,diccionario):
    
    pred=[]
    
    posteriori=[]
    for i in range (len(datostest)):
      
      probabilidades = []
      norm = 0
      for j in range (len (diccionario[-1])):
        
        probabilidadesClase =[]
        for k in range (len(diccionario)-1):     
          if atributosDiscretos[k]:
            valorMatrizTest = datostest[i][k]
            probabilidadesClase.append (self.valoresTrain[k][int(valorMatrizTest)][j])  
          else:
            valorMatrizTest = datostest[i][k]
            media = self.valoresTrain[k][j][0]
            desviacion= self.valoresTrain[k][j][1]
            probabilidad= scipy.stats.norm.cdf(valorMatrizTest,media,desviacion)
            probabilidadesClase.append(probabilidad)
        prod = np.prod(probabilidadesClase)*self.prioris[j]
        norm += prod 
        probabilidades.append(prod)
      for j in range (len(diccionario[-1])):
          probabilidades[j] = probabilidades[j]/norm 
      
      clase = np.argpartition(probabilidades, -1)[-1:][0]
      
      posteriori.append(probabilidades)
      pred.append(clase)
      
    self.posterioris.append (posteriori)
    
    self.predicciones.append(pred)

##############################################################################

class ClasificadorVecinosProximos (Clasificador):
  
  def __init__ (self, K,norm=True):
    self.K = K #la k indica cuantos vecinos se tendran en cuanta para la clasificacion
    self.norm = norm #true o false se se quiere con normalizacion o no

  def entrenamiento (self,datostrain,atributosDiscretos,diccionario):
    if (self.norm):#si se usa normalización simplemente calculamos la nueva matriz con los datosTrain normalizados
        self.datosNormalizados =self.normalizarDatos(datostrain,True,len(diccionario)-1)
    else:#si es sin normalizacion la matriz se queda como estaba
      self.datosNormalizados = datostrain

  #distEucl = raiz[dato-datonorm al cuadrado (se hace con cada atributo y se suman)]
  def distanciaEuclidea (self,datos,numAtributos):
    distancias = []
    for k in range (len(datos)):
      distancia_clase =[]
      for i in range (len(self.datosNormalizados)):
        total = 0
        for j in range (numAtributos):
          total = total + (datos[k][j]-self.datosNormalizados[i][j])**2
        distancia = sqrt(total)
        distancia_clase.append ([distancia,int(self.datosNormalizados[i][-1])])
      distancias.append(distancia_clase)
    #print(distancias)
    return distancias
      
  
  def clasifica (self,datostest,atributosDiscretos,diccionario):
    
    if (self.norm):#si se usa normalización simplemente calculamos la nueva matriz con los datosTest normalizados
      datos = self.normalizarDatos(datostest,False,len (diccionario)-1)
    else:#si es sin normalizacion la matriz se queda como estaba
      datos = datostest
      
    #primero se calculan las distancias euclideanas
    distancias = self.distanciaEuclidea(datos,len (diccionario)-1)
    pred = []
    prob= []
    
    #este bucle calcula las probabilidades y sus predicciones
    for n in range (len (datos)):
      distancia_clase = distancias[n]
      distanciasEnOrden = sorted(distancia_clase, key=itemgetter(0), reverse=False) 
      
      sumatorioClases = np.zeros(len(diccionario[-1]))
      
      #una iteracion por cada vecino
      for i in range (self.K):
        clase = distanciasEnOrden[i][1]
        for j in range (len(diccionario[-1])):
          if (j==clase):
            sumatorioClases[j] +=1
      
      for k in range (len(diccionario[-1])):
        sumatorioClases[k] = sumatorioClases[k]/self.K
      prob.append (sumatorioClases)
      pred.append (np.argpartition(sumatorioClases, -1)[-1:][0])
    self.posterioris.append(prob)
    self.predicciones.append(pred)
    return np.array(pred).astype('float')
    

##############################################################################

class ClasificadorRegresionLogistica (Clasificador):
  def __init__ (self,numEpocas,cAprendizaje,norm=False):
    self.norm = norm
    self.numEpocas = numEpocas
    self.cAprendizaje = cAprendizaje
  
  def entrenamiento (self,datostrain,atributosDiscretos,diccionario):
   
    w = np.zeros (len(diccionario)-1)
    random.seed()
    for d in range (len(diccionario)-1):
      w[d]=random.uniform (-0.5,0.5)
    
    for ie in range (self.numEpocas):
      for i in range (len(datostrain)):
        x = np.zeros (len(diccionario)-1)
        for d in range (len(diccionario)-1):
          x[d] = datostrain[i][d]
        
        resultado = 0.0
        for i in range(len(w)):
            resultado += w[i]*x[i]
        productoEscalar=resultado
        
        sigmoidal= (1/(1+(math.e**(-1*productoEscalar))))
        
        clase = int(datostrain[i][-1])
        for d in range (len(diccionario)-1):
          x[d]= x[d]*(sigmoidal-clase)*self.cAprendizaje
          w[d] = w[d]-x[d]
    self.w=w
        


  def clasifica (self,datostest,atributosDiscretos,diccionario):
    pred = []
    prob = []
    numAtributos = len (diccionario)-1
    for n in range(len(datostest)):
      x = datostest[n][range (numAtributos)]
      resultado = 0.0
      for i in range(len(self.w)):
          resultado += self.w[i]*x[i]
      productoEscalar=resultado
      sigmoidal = (1/(1+(math.e**(-1*productoEscalar))))
      if (sigmoidal<0.5):
        pred.append (0)
      else:
        pred.append (1)

      prob.append([1-sigmoidal,sigmoidal])
      
    self.posterioris.append(prob)
    self.predicciones.append(pred)
    return np.array(pred)
    
      


    
    
    

    
    





  