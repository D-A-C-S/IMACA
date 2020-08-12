from datetime import timedelta
import pandas as pd
from maderas.Utilidades import EMFechas
from ._Estimacion import TiempoCaptura
def Agrupar(df,tiempo_corte=None):
  """
  Añade al dataframe una columna que aproximadamente identifica a cada foto por
  su tronco de origen
  """
  df = df.copy()
  #Fecha y origen de acuerdo al nombre del archivo o fecha de modificacion
  _fyo = df.apply(lambda p:EMFechas.extraer_fecha(p.filename,p.path),axis=1)
  df["fecha"] =  _fyo.apply(lambda p:p[0])
  df["origen"] = _fyo.apply(lambda p:p[1])
  
  #Conversion a formato que permite realizar operaciones
  df["fecha"] = pd.to_datetime(df.fecha)
  
  df = df.sort_values(by="fecha")
  df["fecha_diffs"] = df.fecha.diff()
  df["fecha_diffs"] = df.fecha_diffs.apply(lambda p:p.total_seconds()).fillna(0.0)
  
  if tiempo_corte:
    max_tA = tiempo_corte
    max_tL = tiempo_corte
  else:
    tiempos_estimados = TiempoCaptura(df)
    max_tA = tiempos_estimados["max_tA"]
    max_tL = tiempos_estimados["max_tL"]

  #Si la diferencia entre tiempos de captura es mayor a cierto numero, se
  #considera que la foto proviene de un tronco diferente y se asigna a un nuevo
  #grupo
  grupos_origen1 = (df[(df["origen"]==0)]["fecha_diffs"]>max_tA).cumsum()
  grupos_origen2 = (df[(df["origen"]==1)]["fecha_diffs"]>max_tL).cumsum()
  tronco = pd.concat([grupos_origen1,grupos_origen2],sort=False)
  df["tronco"] = tronco.apply(str)+df["origen"].apply(str)+df["especie"]
  return df

def Partir(df,ratios):
  """
  Separacion del dataframe en conjuntos que provienen de diferentes troncos.
  Se intenta mantener las relaciones entre conjuntos constantes en cada especie.
  args:
      df:dataframe con columna 'tronco' y 'especie'
      ratios: lista con elementos entre [0-1] que suman 1.
  return:
        tupla con los indices del dataframe para cada conjunto.
  """
  df = df.copy()
  df = df.sample(frac=1,random_state=0)
  #df:dataframe
  #ratios: lista:fraccion de muestras por conjunto
  def PartirTroncos(dc,especies_target):

    especies = dc.especie.value_counts()
    train_target = especies_target

    group_count = dc.tronco.value_counts()
    train_esp = []
    train_count = {especie:0 for especie in dc.especie.unique()}
    for especie in list(especies.index):
      Troncos = (dc[dc.especie==especie]).tronco.unique()

      for Tronco in Troncos:
        train_count[especie]+= group_count[Tronco]
        train_esp.append(Tronco)
        
        if train_count[especie]>train_target[especie]:
          break
    return train_esp

  out = []
  total_especies = df.especie.value_counts()
  reduced_df = df
  for ratio in ratios:
    especies_target = total_especies*ratio
    Partidos = PartirTroncos(reduced_df,especies_target)
    out.append(reduced_df[reduced_df.tronco.isin(Partidos)])
    reduced_df = reduced_df[~reduced_df.tronco.isin(Partidos)]
  return out
 