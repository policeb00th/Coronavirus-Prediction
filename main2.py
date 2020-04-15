import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import fbprophet
from WindowSlider import WindowSlider
    

df = pd.read_csv('dataset/covid_19_india.csv')
#print(df)
states = []
for i in df['State/UnionTerritory']:
    if i not in states:
        states.append(i)
for j in states:
    print(j)
    df2 = pd.read_csv('Statewise/'+j+'.csv')
    # Prophet requires columns ds (Date) and y (value)
    df2 = df2.rename(columns={'Date': 'ds', 'Confirmed': 'y'})
    df_train = df2.loc[df2["ds"]<"2020-04-09"]
    df_test  = df2.loc[df2["ds"]>="2020-04-09"]
    #gm['y'] = gm['y'] / 1e9
    # Make the prophet model and fit on the data
    df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.99)
    df_prophet.add_regressor('Cured')
    df_prophet.add_regressor('Deaths')
    df_prophet.fit(df_train)
    #df_prophet.fit(df2)
    # Python
    #future = df_prophet.make_future_dataframe(periods=10,freq='D')
    #forecast=df_prophet.predict(future)
    forecast = df_prophet.predict(df_test.drop(columns='y'))
    df_prophet.plot(forecast)
    df_prophet.plot_components(forecast)
    #df_prophet.plot_components(forecast)
    plt.show()
