import pandas as pd 
from datetime import datetime
df = pd.read_csv('dataset/covid_19_india.csv')
df=df.drop(['Sno', 'Time', 'State/UnionTerritory','ConfirmedIndianNational', 'ConfirmedForeignNational'], axis=1)
#print(df)
#df['Confirmed']-=(df['Cured']+df['Deaths'])
for i in range(0,len(df['Date'])):
    df['Date'][i]=datetime.strptime(df['Date'][i],'%d/%m/%y')
df=df.groupby(['Date'])['Confirmed'].sum()
df.to_csv('datewise.csv')
print(df.head())