import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error



class ProphetForQlik:
    return_param=[]
    prophet_param=[]
    future_param=[]
    regresores_param=[]
    scale_param=[]
    
    def MAPE(self,Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    
    def getMetrics(self,df_1,df2):
        df_joined=pd.merge(df_1,df2,on='ds')
        y_true = df_joined['y'].values
        y_pred = df_joined['yhat'].values
        mae = mean_absolute_error(y_true, y_pred)
        mape=MAPE(y_true,y_pred)
        print('MAE: %.3f' % mae)
        print('MAPE: %.3f' % mape)
    
    # tratar todo como str por implementacion de Qlik
    def parseStrFloat(self,num):
        if(num!=0):
            temp=float(num.replace(',','.'))
            return round(temp,2)
        else:
            return 0
        
    def changeScale(self,data_out):
        type_importe=data_out.iloc[:,1].dtype
        
        if(type_importe=='O'):
            data_out['y'] = data_out.iloc[:,1].apply(lambda i: self.parseStrFloat(i))
        else:
            data_out['y'] = data_out.iloc[:,1]
        
        if self.scale_param!=[]:
            data_out['y']=data_out['y'].apply(lambda x: x/int(self.scale_param[0]))
            
        return data_out
    
    def changePeriodToDate(self,data_in):
        data_out=pd.DataFrame()
        data_in['Year']  = data_in.iloc[:,0].apply(lambda x: str(x)[-6:-2])
        data_in['Month'] = data_in.iloc[:,0].apply(lambda x: str(x)[-2:])
        data_in['Day'] = '01'
        
        data_out['ds'] = pd.DatetimeIndex(data_in['Year']+'-'+data_in['Month']+'-'+data_in['Day'])
        data_out['y']  = data_in.iloc[:,1]
            
        return data_out
    
    def isPeriod(self,data_in):
        if(len(str(data_in.iloc[0,0]))==6):
            return True
        else:
            return False
        
    def _getNumericValue(self,s):
        try:
            if(s.index('.')>0):
                return float(s)
        except:
            if(not s.isdigit()):
                return s
            else:
                return int(s)    
        
    def _init_regresores(self,model):
        
        print('Init Regresores')
        self.data_in['SMA3']=self.data_in['y'].rolling(min_periods=1,window = 3).mean()
        self.data_in['SMA6']=self.data_in['y'].rolling(min_periods=1,window = 6).mean()
        self.data_in['SMA9']=self.data_in['y'].rolling(min_periods=1,window = 9).mean()
        
        exogenous_features = self.regresores_param
        
        for feature in exogenous_features:
            model.add_regressor(feature)
      
        return model
        
    def _init_params(self,str_params):
        
        self.prophet_kwargs={}
        self.make_kwargs={}
        
        params=[]
        params=str_params.split(';')
        
        for i in range(0,len(params)):
            if(i==0):
                self.return_param=params[i].replace('return=','').split(',')
            if(i==1 and len(params[i])>0):  
                self.prophet_param=params[i]
                #print('prophet_param:'+self.prophet_param)
                self.prophet_kwargs = self._vectorToDict(self.prophet_param)
            if(i==2 and len(params[i])>0):
                #print('future_param:'+str(params[i]))
                self.future_param=params[i]
                self.make_kwargs = self._vectorToDict(self.future_param)
            if(i==3):
                #print('Regresores:'+str(params[i]))
                self.regresores_param=params[i].split(',')    
            if(i==4):
                #print('Escala:'+str(params[i]))
                self.scale_param=params[i].split(',') 
                
        print(' prophet_param: {} \n future_param: {} \n Regresores: {} \n Escala: {}'.format(self.prophet_param,self.future_param,self.regresores_param,self.scale_param))    
        
    
    def __init__(self,data_in,str_params):
        
        
        self._init_params(str_params)
        
        if(self.isPeriod(data_in)):
            data_in=self.changePeriodToDate(data_in)
            
        self.data_in=self.changeScale(data_in)
        
    def _vectorToDict(self,params):
        temp=dict(e.split('=') for e in params.split(','))
        return {key: self._getNumericValue(value) for key, value in temp.items()}
    
    def plot(self,forecast):
        self.model.plot(forecast)
    
    def getRegresorValue(self):
        print('getRegresorValue')
        # Futuro aca hay que agregar 
        # 1 opcion forecastear con el makefuture y despues pasar por el rolling 
        # 2 opcion intentar con la curva de rolling antes de forecasteo con una regresion polinomica dibujar la curva futura
        
    
    
    def predict(self):
          
        if len(self.prophet_param) > 0:
            self.model = Prophet(**self.prophet_kwargs)
        else:
            self.model = Prophet()
        
        #if (len(self.regresores_param)>1):
        #    self.model=self._init_regresores(self.model)
        #    print(self.data_in)
        
        
        self.model.fit(self.data_in)
        self.future_df = self.model.make_future_dataframe(**self.make_kwargs)
        
        
        forecast=self.model.predict(self.future_df)
        
        #if (len(self.regresores_param)>1):
        #    self.future_df=pd.merge(forecast,self.data_in,on='ds',how='left')
        #    self.future_df.fillna(0,inplace=True)
        #    print(self.future_df)
    
    
    
        # si tiene mas de dos parametros ds e y
        if (len(self.return_param)>1):
            return forecast[self.return_param]
        else:
            return forecast