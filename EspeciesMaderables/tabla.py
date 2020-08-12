from maderas.Utilidades import get_df_from_path
from . import _Troncos
import pandas as pd

def _codificar_etiquetas(df):
  ToCod = {}
  ToEspecie = []
  Especies = sorted(df.especie.unique())
  for i,especie in enumerate(Especies):
    ToCod.update({especie:i})
    ToEspecie.append(especie)
  df['label']= df.especie.apply(lambda p:ToCod[p])
  return df

def build(anotaciones_path,fotos_path,tiempo_corte=None):
  """
  Union de anotaciones manuales y dataframe generado de los nombres de las carpetas
  """
  anotaciones = (pd.read_csv(anotaciones_path).
                set_index("path").
                drop(columns=["especie","aumento","filename"]))
  maderas_df = get_df_from_path(fotos_path,nombre_cols=["especie","aumento"])
  maderas_df["index"] = maderas_df.especie+"/"+maderas_df.aumento+"/"+maderas_df.filename
  maderas_df.set_index("index",inplace=True)
  df = pd.concat([anotaciones,maderas_df],sort= False,axis=1)
  df = _Troncos.Agrupar(df,tiempo_corte=tiempo_corte)
  return df

def filter(df,aumento="160X",calidad=2,min_n=500):
  """Selecciona un subconjunto del dataframe de acuerdo a los argumentos:
     aumento:160X o 30X
     calidad:0-5,califica enfoque de la foto y corte de la muestra
     min_n : numero minimo de fotos necesarias para incluir una especie
  """
  df = df[(df.Clase==1) & (df["Caracteristicas visibles"]>=calidad)]
  df = df[(df["aumento"]==aumento)]
  conteo = df.especie.value_counts()
  especies_seleccionadas = list(conteo[conteo>min_n].index)
  nuevo_df = df[df.especie.isin(especies_seleccionadas)].copy()
  nuevo_df = _codificar_etiquetas(nuevo_df)
  print("Especies seleccionadas:",especies_seleccionadas)
  return nuevo_df


def split(df,ratios,Troncos=True):
  """Divide la tabla, seleccionando aleatoriamente por foto o por tronco
  df:tabla de pandas
  ratios: lista que especifica la proporcion de cada subconjunto.0-1
  Troncos : seleccion por especimen(aproximacion), la alternativa es 
            seleccion completamente aleatoria
  """
  assert abs(sum(ratios)-1.0)<=0.01
  def uniform_split(df,ratios):
    df = df.sample(frac=1,random_state=0)
    splits = [int(len(df)*ratio) for ratio in ratios]
    partes = []
    for split in splits:
      partes.append(df.iloc[:split])
      df = df.iloc[split:]
    return partes
    

  if Troncos:
    return _Troncos.Partir(df,ratios)
  else : 
    return uniform_split(df,ratios)
  

