import numpy as np
from datetime import datetime
import pandas as pd

df = pd.read_csv('dataset/covid_19_india.csv')
#print(df)
#df['Confirmed']-=df['Cured']
for i in range(0,len(df['Date'])):
    df['Date'][i]=datetime.strptime(df['Date'][i],'%d/%m/%y')
#print(df)
states = []
for i in df['State/UnionTerritory']:
    if i not in states:
        states.append(i)
for i in states:
    stateCsv=df.loc[df['State/UnionTerritory']==i]
    stateCsv=stateCsv.drop(['Sno', 'Time', 'State/UnionTerritory','ConfirmedIndianNational', 'ConfirmedForeignNational'], axis=1)
    #print(stateCsv.head())
    stateCsv.to_csv('Statewise/'+i+'.csv',index=False)




def convertDate(df):
    for i in range(len(df['Date'])):
        df['Date'][i]=i
    return df

'''for i in states:
    stateCsv = df.loc[df['State/UnionTerritory'] == i]
    stateCsv = stateCsv.drop(['Sno', 'Time', 'State/UnionTerritory','ConfirmedIndianNational', 'ConfirmedForeignNational'], axis=1)
    for j in range(len(stateCsv['Date'])):
        stateCsv['Date'][j] = j
    print(stateCsv.head())
    print(i)
    for j in range(0, len(stateCsv['Confirmed'])):
        print(stateCsv['Confirmed'][j])
        #stateCsv['Confirmed'][j] -= stateCsv['Cured'][j]
    stateCsv = stateCsv.drop(['Cured', 'Deaths'], axis=1)
    print(stateCsv.head())
    print('State saved: ',i)
    stateCsv.to_csv('Statewise/'+i+'.csv',index=False)'''
'''
for j in states:
    df2 = pd.read_csv('Statewise/'+j+'.csv')
    for i in range(len(df2['Date'])):
        df2['Date'][i] = i
    for i in range(0,len(df2['Confirmed'])):
        df2['Confirmed'][i]-=df2['Cured'][i]
    try:
        df2=df2.drop(['Cured','Deaths'],axis=1)
    except:
        print('State already done: ',j)
    finally:
        print(df2.head())
        df2.to_csv('Statewise/'+j+'.csv',index=False)'''

