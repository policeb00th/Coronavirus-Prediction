import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import fbprophet
from fbprophet.diagnostics import cross_validation
from WindowSlider import WindowSlider

def MaP(y,yh):
    y,yh=np.array(y),np.array(yh)
    return np.mean(np.abs((y-yh)/y)) *100    

df = pd.read_csv('dataset/covid_19_india.csv')
#print(df)
states = []
for i in df['State/UnionTerritory']:
    if i not in states:
        states.append(i)
holidays=playoffs = pd.DataFrame({
  'holiday': 'lockdown',
  'ds': pd.to_datetime(['2020-03-22','2020-03-24','2020-03-25','2020-03-26','2020-03-27','2020-03-28','2020-03-29','2020-03-30','2020-03-31','2020-04-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05','2020-04-06','2020-04-07','2020-04-08','2020-04-09','2020-04-10','2020-04-11','2020-04-12','2020-04-13','2020-04-15','2020-04-16','2020-04-17','2020-04-18','2020-04-19','2020-04-20','2020-04-21','2020-04-22','2020-04-23','2020-04-24','2020-04-25','2020-04-26','2020-04-27','2020-04-28','2020-04-29','2020-04-30','2020-05-01','2020-05-02','2020-05-03']),
  'lower_window': 0,
  'upper_window': 1,
})
for j in states:
    df2 = pd.read_csv('Statewise/'+j+'.csv')
    # Prophet requires columns ds (Date) and y (value)
    df2 = df2.rename(columns={'Date': 'ds', 'Confirmed': 'y'})
    # Put market cap in billions
    #gm['y'] = gm['y'] / 1e9
    # Make the prophet model and fit on the data
    df_prophet = fbprophet.Prophet(changepoint_prior_scale=0.6,holidays=holidays,holidays_prior_scale=40,seasonality_mode='multiplicative',seasonality_prior_scale=10,daily_seasonality=False,yearly_seasonality=False,weekly_seasonality=False).add_seasonality(
        name='daily',
        period=1,
        fourier_order=15
    ).add_seasonality(
        name='incubation_period',
        period=5,
        fourier_order=20
    )

    df_prophet.fit(df2)
    # Python
    future = df_prophet.make_future_dataframe(periods=10,freq='D')
    forecast = df_prophet.predict(future)
    df_prophet.plot(forecast)
    cv=cross_validation(df_prophet,initial=str(len(df2)-5)+' days',horizon='2 days' )
    mAp=MaP(cv.y,cv.yhat)
    print(mAp)
    #df_prophet.plot_components(forecast)
    #df_prophet.plot_components(forecast)
    plt.show()
    break
