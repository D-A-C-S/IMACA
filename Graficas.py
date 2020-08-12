import pandas as pd
from scipy import signal
import numpy as np
import seaborn as sns; sns.set()

import pandas as pd
from scipy import signal
import numpy as np
import seaborn as sns; sns.set()

def low_pass_filter(sig,frecuency):
  b,a = signal.butter(2,frecuency)#orden 3
  filtered = signal.filtfilt(b, a, sig, method="gust")
  return filtered

def banded_smooth_plot(sig,n_ticks=10,xname='x',yname='y',
                       bounds=(0,1),freq=1/20):
  A = pd.DataFrame()
  min_bound,max_bound = bounds
  #Filtra la señal
  filtered = low_pass_filter(sig,freq)
  #Calcula el ancho de la banda
  point_var = (filtered-sig)**2
  std = np.sqrt(np.abs(low_pass_filter(point_var,freq/3)))
  #Construye un dataframe
  x = np.linspace(0,n_ticks,len(filtered))
  A[yname] = filtered
  y1 = np.clip(filtered+std,None,max_bound)
  y2 = np.clip(filtered-std,min_bound,None)
  A[xname] = x
  #Dibuja
  ax = sns.lineplot(x=xname,y=yname,data=A)
  ax.fill_between(x,y1,y2,alpha=0.2)

def SaveReport(train_accs,train_losses,eval_accs,eval_losses):

  eval_report = pd.DataFrame()
  train_report = pd.DataFrame()

  train_exact= [ele.item() for epoca in train_accs for ele in epoca]
  train_perdi = [ele for epoca in train_losses for ele in epoca]

  eval_report["acc"] = eval_accs
  eval_report["loss"] = eval_losses

  train_report["acc"] = train_exact
  train_report["loss"] = train_perdi

  train_report.to_csv("train_report.csv",index=False)
  eval_report.to_csv("eval_report.csv",index=False)  
  return train_report["acc"],train_report["loss"],eval_report["acc"],eval_report["loss"]