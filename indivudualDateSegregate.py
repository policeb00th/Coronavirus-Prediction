import pandas as pd 
from datetime import datetime
df = pd.read_csv('datewise.csv')

for i in range (len(df)-1,0,-1):
  df['Confirmed'][i]-=df['Confirmed'][i-1]

df.to_csv('CasesPerDay.csv',index=False)