def TiempoCaptura(df,max_tL=None,max_tA=None,verbose=False):
  """
  df: dataframe con columna 'fecha_diffs',que corresponde 
  a la diferencia del tiempo(en segundos) entre la 
  captura de una foto y la siguiente, y columna origen con
  un id del dispositivo de captura"
  """
  
  ##Estimacion tiempo promedio entre capturas
  Ltimes = df.fecha_diffs[(df.fecha_diffs<60)&(df.origen==1)]
  Atimes = df.fecha_diffs[(df.fecha_diffs<60)&(df.origen==0)]
  
  if verbose:
    print("Tiempo promedio de captura:")
    print(f"origenL:{sum(Ltimes)/len(Ltimes):.1f}s")
    print(f"origenA:{sum(Atimes)/len(Atimes):.1f}s")

  Ltimes = df.fecha_diffs[df.origen==1]
  Atimes = df.fecha_diffs[df.origen==0]
  #Estimacion de "punto de quiebre",cambio de muestra durante la captura
  max_tL = max_tL or 50#Estimado del maximo tiempo que pasa sin tomar fotos sobre un mismo tronco
  max_tA = max_tA or 35#Estos valores han sido ajustados iterativamente
  ltimes = df.fecha_diffs[(df.fecha_diffs<max_tL)&(df.origen==1)]
  atimes = df.fecha_diffs[(df.fecha_diffs<max_tA)&(df.origen==0)]
  #Suponiendo que si el tiempo es menor a max_tX la captura se realiza sobre el mismo tronco
  if verbose:
    print("Numero de capturas realizadas sobre cada tronco:")
    print(f"origenL:{1/(1-len(ltimes)/len(Ltimes)):.2f}")
    print(f"origenA:{1/(1-len(atimes)/len(Atimes)):.2f}")
  return {"max_tL":max_tL,"max_tA":max_tA}