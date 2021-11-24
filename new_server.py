from pandas import DataFrame
from pandas import concat
from pandas import to_datetime
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import joblib 
from fastapi import FastAPI
from pydantic import BaseModel
import socket

def ensure_float(x):
    if isinstance(x,np.float):
        return x
    else :
        return np.nan

data_name_list=['h2','c2h2','totalGasppm']
train_group={'day':[3,1],'week':[21,7],'half':[30,14],'month':[60,30]}
sensor_name='sensorId'

class gis_msg(BaseModel):
    msg:str
    code:int
    data:list
    size:int

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        df = DataFrame(data)
        cols = list()
        # 输入序列(t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        # 预测序列(t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        # 把所有放在一起
        agg = concat(cols, axis=1)
        # 删除空值行
        if dropnan:
            agg.dropna(inplace=True)
        return agg
    
#format data and get mean
def process_raw_data(raw_data):
    #valid_data=raw_data['data']
    cols=list()
    sensor_id=0
    frame = DataFrame(raw_data,columns=['creatTime','h2','c2h2','totalGasppm','sensorId']).dropna()
    sensor_id = frame[sensor_name].drop_duplicates().values
    if len(sensor_id)>0:
        sensor_id = str(sensor_id[0])
    else:
        sensor_id =-1
    frame['creatTime'] = to_datetime(frame['creatTime'], format='%Y-%m-%d %H:%M:%S')
    frame['creatTime'] =frame['creatTime'].apply(lambda x:datetime.strftime(x,'%Y-%m-%d'))
    for g_name in data_name_list:
        frame[g_name] = frame[g_name].apply(lambda x: ensure_float(x))
        frame[g_name] = frame[g_name].dropna()
        cols.append(frame.groupby('creatTime')[g_name].mean())
    agg = concat(cols, axis=1)
    return agg,sensor_id

def standard_data(data):
        scaler = StandardScaler()
        output_data=np.array(data).reshape(-1, 1)
        output_data = scaler.fit_transform(output_data)
        return output_data,scaler

def split_data(data,history_val,predict_val):
    train_data = series_to_supervised(data,history_val,predict_val)
    datasets = train_data.values
    trainx,trainy = datasets[:,:history_val],datasets[:,history_val:]
    trainx = trainx.reshape(trainx.shape[0],trainx.shape[1],1)
    return trainx,trainy


def calculate_prediction(raw_data,model_name,scaler_path,history_days,predict_days):
        #wrong json message
        if os.path.exists(model_name) == False or os.path.exists(scaler_path) == False:
            return 'No find model'
        if(len(raw_data) < history_days):
            return 'Data too short'
        #load the latest model
        my_scaler = joblib.load(scaler_path)
        std_data=np.array(raw_data).reshape(-1, 1)
        std_data = my_scaler.fit_transform(std_data)
        model = load_model(model_name)
        trainx,_ = split_data(std_data,history_days,predict_days)
        result = model.predict(trainx)
        result  = my_scaler.inverse_transform(result)
        #plt.plot(raw_data,'b')
        ##plt.plot(result,'r')
        ##plt.show()
        #exit(0)
        return result

def gas_data_process(raw_data,param):
    mean_data,sensor_id = process_raw_data(raw_data)
    predict_result={}
    if(sensor_id == -1):
        return {"Result":"No sensor ID"}
    for gas_name in data_name_list:
        gas_val = mean_data[gas_name].values
        std_val,scaler= standard_data(gas_val)
        days_map={}
        for date_key in train_group.keys():
            time_gap=train_group[date_key]
            history_val = time_gap[0]
            predict_val = time_gap[1]
            trainx,trainy=split_data(std_val,history_val,predict_val)
            model_name = 'models/{}/{}/{}_{}.h5'.format(sensor_id,gas_name,sensor_id,date_key)
            scaler_path='models/{}/{}/{}.save'.format(sensor_id,gas_name,sensor_id)
            if(param == 't'):
                model = Sequential()
                model.add(LSTM(128, input_shape=(trainx.shape[1], trainx.shape[2])))
                model.add(Dense(predict_val))
                model.compile(loss='mae', optimizer='adam')
                model.fit(trainx,trainy,epochs=50,batch_size=50,verbose=2 ,shuffle=False)
                model.save(model_name)
                #model.save(back_up_name)
                joblib.dump(scaler, scaler_path)
            elif (param == 'p'):
                results = calculate_prediction(gas_val,model_name,scaler_path,history_val,predict_val)
                days_map[date_key]=results[-1,:].tolist()
        if (param == 'p'):
            predict_result[gas_name] = days_map
    
    if param == 'p':
        return {'Prediction Result':predict_result}
    elif param == 't':
        return {'Training Result':'Done'}

app = FastAPI()

@app.post('/train')
async def train_user_data(train_data:gis_msg):
    if(len(train_data.data) != train_data.size  ):
        return {'Training Result':'Data length incorrect'}
    if(len(train_data.data)==0 or train_data.size == 0):
        return {'Training Result':'Data empty'}
    return gas_data_process(train_data.data,'t')

@app.post('/predict')
async def predict_oil_data(train_data:gis_msg):
    if(len(train_data.data) != train_data.size  ):
        return {'Prediction Result':'Data length incorrect'}
    if(len(train_data.data)==0 or train_data.size == 0):
        return {'Prediction Result':'Data empty'}
    #print(train_data.data)
    return gas_data_process(train_data.data,'p')
     

@app.get("/")
def read_root():
    return {"GIS":"AI System"}

def get_host_ip(): 
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    if ip == None:
        return "127.0.0.1"
    return ip

if __name__=='__main__':
    import uvicorn
    ip_addr = get_host_ip()
    uvicorn.run(app, host=ip_addr, port=8000)
    #import json
    #with open("msg.json", 'r') as f:
    #    json_data = json.load(f)
    
    #print(gas_data_process(json_data['data'],'p'))
