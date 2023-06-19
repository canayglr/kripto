import pandas as pd #Veri işleme
import yfinance as yf # Kripto para çekme
import datetime 
from datetime import date, timedelta
import plotly.graph_objects as go # Görselleştirme
from autots import AutoTS #Makine öğrenmesi

today=date.today()
d1=today.strftime("%Y-%m-%d") # yıl ay güne çevirdi
end_date=d1
d2=date.today()-timedelta(days=730) # bugünü 730 günden çıkarma
start_date=d2
data=yf.download("BTC-USD", 
                 start=start_date, 
                 end= end_date, 
                 progress=False) # kripto para birimi verilerini indirme
data["Date"]=data.index
data=data[["Date","Open","High","Low","Close","Adj Close","Volume"]]
data.reset_index(drop=True,inplace=True)
figure=go.Figure(data=[go.Candlestick(x=data["Date"], #grafik türü - x teki tarih
                                    open=data["Open"],
                                    high=data["High"],
                                    low=data["Low"],
                                    close=data["Close"]
                                    )])
figure.update_layout(title="Son 730 Günün Bit Coin Grafiği",
                     xaxis_rangeslider_visible=True) #yakınlaştırma
figure.show() #figürü yazdır

model= AutoTS(forecast_length=30,frequency="infer", ensemble="simple")
model=model.fit(data,date_col="Date",value_col="Close",id_col=None)
tahmin=model.predict()
tahmin2=tahmin.forecast
print(tahmin2)