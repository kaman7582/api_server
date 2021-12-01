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
from fastapi import FastAPI,BackgroundTasks
from pydantic import BaseModel
import os
import numpy as np
import joblib 
import socket
import json
import sys

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def ensure_float(x):
    if isinstance(x,np.float):
        return x
    elif is_number(x):
        return float(x)
    else:   
        return np.nan

data_name_list=['h2','c2h2','tothyd']
train_group={'day':[3,1],'week':[21,7],'2weeks':[30,14],'month':[60,30]}
sensor_name='sensorId'

class gis_msg(BaseModel):
    oilData:list

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
    frame = DataFrame(raw_data,columns=['creatTime','h2','c2h2','tothyd','sensorId']).dropna()
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
        #print(frame[g_name])
        cols.append(frame.groupby('creatTime')[g_name].mean())
    agg = concat(cols, axis=1)
    #agg.ix[~(agg==0).all(axis=1), :]
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
        my_scaler = joblib.load(scaler_path)
        std_data=np.array(raw_data).reshape(-1, 1)
        std_data = my_scaler.fit_transform(std_data)
        model = load_model(model_name)
        #trainx,_ = split_data(std_data,history_days,predict_days)
        #print(trainx.shape)
        trainx = std_data[-history_days:,:].reshape(1,history_days,1)
        result = model.predict(trainx)
        result  = my_scaler.inverse_transform(result)
        return result
    
def gas_data_predict(raw_data,date_key):
    mean_data,sensor_id = process_raw_data(raw_data)
    predict_result={}
    for gas_name in data_name_list:
        gas_val = mean_data[gas_name].values
        model_name = 'models/{}/{}/{}_{}.h5'.format(sensor_id,gas_name,sensor_id,date_key)
        scaler_path='models/{}/{}/{}.save'.format(sensor_id,gas_name,sensor_id)
        history_val,predict_val = train_group[date_key]
        #if gas model does not find
        if os.path.exists(model_name) == False or os.path.exists(scaler_path) == False:
                #return {'Prediction Result':'No find AI model'}
                predict_result[gas_name]='No find AI model'
        else:
            results = calculate_prediction(gas_val,model_name,scaler_path,history_val,predict_val)
            rlist = results[-1,:].tolist()
            rlist = [("%.2f" % abs(i)) for i in rlist]
            predict_result[gas_name]=rlist
    return {date_key:predict_result}

def gas_data_train(raw_data):
    mean_data,sensor_id = process_raw_data(raw_data)
    if(sensor_id == -1):
        print("Sensor id not find")
        return {"Result":"No sensor ID"}
    for gas_name in data_name_list:
        gas_val = mean_data[gas_name].values
        #if gas is almost zero
        #np.count_nonzero(gas_val)
        std_val,scaler= standard_data(gas_val)
        for date_key in train_group.keys():
            history_val,predict_val = train_group[date_key]
            trainx,trainy=split_data(std_val,history_val,predict_val)
            model_name = 'models/{}/{}/{}_{}.h5'.format(sensor_id,gas_name,sensor_id,date_key)
            scaler_path='models/{}/{}/{}.save'.format(sensor_id,gas_name,sensor_id)
            model = Sequential()
            model.add(LSTM(128, input_shape=(trainx.shape[1], trainx.shape[2])))
            model.add(Dense(predict_val))
            model.compile(loss='mae', optimizer='adam')
            model.fit(trainx,trainy,epochs=50,batch_size=50,verbose=2 ,shuffle=False)
            model.save(model_name)
                #model.save(back_up_name)
            joblib.dump(scaler, scaler_path)
    print("Training done")
    return {'Training Result':'Done'}

app = FastAPI()

@app.post('/train')
async def train_user_data(train_data:gis_msg,background_tasks: BackgroundTasks):
    if(len(train_data.oilData) < train_group['month'][0] ):
        return {'Training Result':'Data too short'}
    #background_tasks.add_task(gas_data_process, train_data.oilData,'t')
    #return {"Training Result":"Done"}
    return gas_data_train(train_data.oilData)
    


@app.post('/predict_day')
async def predict_oil_data(train_data:gis_msg):
    if(len(train_data.oilData) < train_group['day'][0]):
        return {'Prediction Result':'Data too short'}
    return gas_data_predict(train_data.oilData,'day')

@app.post('/predict_week')
async def predict_oil_data(train_data:gis_msg):
    if(len(train_data.oilData) < train_group['week'][0]):
        return {'Prediction Result':'Data too short'}
    return gas_data_predict(train_data.oilData,'week')

@app.post('/predict_2weeks')
async def predict_oil_data(train_data:gis_msg):
    if(len(train_data.oilData) < train_group['2weeks'][0]):
        return {'Prediction Result':'Data too short'}
    return gas_data_predict(train_data.oilData,'2weeks')

@app.post('/predict_month')
async def predict_oil_data(train_data:gis_msg):
    if(len(train_data.oilData) < train_group['month'][0]):
        return {'Prediction Result':'Data too short'}
    return gas_data_predict(train_data.oilData,'month')


@app.get("/")
def read_root():
    return {"GIS":"AI System Server Start"}
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
    try:
        f= open("config.json", 'r')
        j_msg = json.load(f)
        serv_info = j_msg['server_info']
        ip_addr = serv_info['ip']
        bind_port = int(serv_info['port'])
    except:
        if( len(sys.argv)>2):
            ip_addr = sys.argv[1]
            bind_port = int(sys.argv[2])
        else:
            ip_addr = get_host_ip()
            bind_port = 8000
    uvicorn.run(app, host=ip_addr, port=bind_port)
    '''
    import json
    with open("new.txt", 'r') as f:
        json_data = json.load(f)
    print( gas_data_process(json_data['oilData'],'p'))
    '''

#config in json ,server ip , port
#config in json, receive json message format