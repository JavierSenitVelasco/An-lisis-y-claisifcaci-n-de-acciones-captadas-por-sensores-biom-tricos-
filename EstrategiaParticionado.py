
from abc import ABCMeta,abstractmethod
import math
import random


class Particion():

  # Esta clase mantiene la lista de índices de Train y Test para cada partición del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]
  
    


#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta: nombreEstrategia, numeroParticiones, listaParticiones. Se pasan en el constructor 
  
  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  
  def __init__(self, pTrain=0.6):
        
        self.numParticiones = pTrain
        self.particiones = []
  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado.
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):
      
      random.seed(seed)
      
      numFilas = len(datos[:,0])
    
      particion = Particion()
    
      rango = range(numFilas)
      
      listaIndices=list(rango)
      random.shuffle(listaIndices)
      
      numTrain = int(math.ceil(numFilas * self.numParticiones))
        
      for i in range (0, numTrain):
         particion.indicesTrain.append(listaIndices[i])
          
      for i in range (numTrain, len(listaIndices)):
          particion.indicesTest.append(listaIndices[i])
           
      self.particiones.append(particion)
      
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  def __init__(self, k=2):
        
        self.numParticiones = k
        self.particiones = []
        
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self, datos, seed=None):
        
        numFilas = len(datos[:,0])
        
        particion = Particion()
        
        rango = range(numFilas)
        
        listaIndices=list(rango)
        
        random.shuffle(listaIndices)
        
        numElemGrupo = numFilas/self.numParticiones
        numElemGrupo = math.floor(numElemGrupo)
        
        listaRepartida=[]
        
        for i in range(0, len(listaIndices), numElemGrupo):
             listaRepartida.append(listaIndices[i:i + numElemGrupo])
       
        for i in range(0, self.numParticiones):
            particion = Particion()
            particion.indicesTest = listaRepartida[i]
            listaKiteraciones=[]
            
            for j, k in enumerate(listaRepartida):
                 if j != i:
                     listaKiteraciones.append(k)
           
            for iteradorlistaKiteraciones in listaKiteraciones:
                for k in iteradorlistaKiteraciones:
                    particion.indicesTrain.append(k)
          
            self.particiones.append(particion)
           
        