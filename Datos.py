
"""

@author: Javier Senit Velasco y Alberto Pérez Garrido
"""

import numpy as np
import pandas as pd

class Datos:
  
  TiposDeAtributos=('Continuo','Nominal')
 
  # TODO: procesar el fichero para asignar correctamente las variables tipoAtributos, nombreAtributos, nominalAtributos, datos y diccionarios
  # NOTA: No confundir TiposDeAtributos con tipoAtributos
  def __init__(self, nombreFichero):
      self.nombreFichero = nombreFichero
      #print(self.nombreFichero)
      df= pd.read_csv(nombreFichero, names=['index','Arrival_Time', 'Creation_Time', 'x', 'y', 'z', 'User', 'Model', 'device', 'gt'])
      print(df.describe())
      filas= df.shape[0]
      print(df.head(10))
      df[[]] = df[[]].replace(0, np.NaN)
      print(df.isnull().sum())
      df.dropna(inplace=True)
      print("Filas tras eliminar: "+str(df.shape[0]) + " en comparacion con las filas anteriores: "+str(filas))
      print(df.isnull().sum())
      
      grupos = df.groupby(['User', 'Model', 'device', 'gt'])
    
      grupos.head(5).drop(grupos.head(5).columns[0],axis=1).to_csv('nuevo.csv')

      self.nombreFichero = 'nuevo.csv'
      
      
      #CALCULO DE NUMERO DE DATOS
      contador=0
      with open(self.nombreFichero, 'r') as f:
          while True:

              line = f.readline()
          
                     
              if not line:
                  break
            
              contador+=1
              
      self.numeroDeDatos= contador-1
     # print(self.numeroDeDatos)
      
      self.tipoAtributos = [True,True,True,True,True,True,True,True,True,True]
      #print(self.tipoAtributos)
      
      
      
      archivo = open(self.nombreFichero, "r")
      archivo.seek(0) #situamos el cursor de lectura al principio
      
      self.nombreAtributos = archivo.readline() #lectura de la primera linea  
      self.nombreAtributos=self.nombreAtributos.strip("\n") #eliminamos el \n del final
      self.nombreAtributos = self.nombreAtributos.split(",")
      #print(self.nombreAtributos)
      
      
     
              
      self.nominalAtributos=self.tipoAtributos
      #print(self.nominalAtributos)
      archivo.seek(0)
      
      
      self.datos = np.empty(((int(self.numeroDeDatos)+1),len(self.tipoAtributos)), dtype=object)
     
      
      contador = 0
      with open(self.nombreFichero, 'r') as f:
          while True:
              milista=[]
            
              line = f.readline()
              
              line= line.strip("\n")
              caracteres = line.split(",")
            
              for x in caracteres:
                  milista.append(x)
              if not line:
                  break

              self.datos[contador] = milista  
              contador+=1
              
            
      self.datos = np.delete(self.datos, [0], axis=0)
     
    
     
      self.diccionarios=[]
      for i in range (0, len(self.nombreAtributos)):
          listaAtributosDiferentes=[]
          diccionario={} 
          contadorAtributos=0
          if self.nominalAtributos[i] == True:
              
              columna = self.datos[:,i]
              
              for j in columna:
                  j=j.strip("\n")
                  if j in listaAtributosDiferentes:
                     continue 
                  else:
                      listaAtributosDiferentes.append(j)     
              listaAtributosDiferentes.sort()

              for i in listaAtributosDiferentes:
                  diccionario[i]=contadorAtributos
                  contadorAtributos += 1
              self.diccionarios.append(diccionario) 
              
          else:
              self.diccionarios.append({})
                
          
           
    #  print(self.diccionarios)     
      
      for i in range (0, len(self.nombreAtributos)):
          
          columna = self.datos[:,i]
          
          contador = 0
          
          if self.nominalAtributos[i] == True:
              for j in columna:
                  valor = self.diccionarios[i][j]
                  self.datos[:,i][contador] = float(valor)
                  contador += 1
          else:
            for k in columna:
                self.datos[:,i][contador]=float(k)
                contador+=1
              
      #print(self.datos) 
      
      archivo.close()
         
    
  # TODO: implementar en la práctica 1
  def extraeDatos(self, idx):
    return self.datos[idx, :]



  
