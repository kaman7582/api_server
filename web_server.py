from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

'''
class lstm model
'''

import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
import numpy as np
import os
from pandas import DataFrame
from pandas import concat
import joblib 

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
    
#train the feature data
class gasTrain():
    def __init__(self,name='unknown',epoch=50,step=5,future_day=3) -> None:
        self.epoch = epoch
        self.history_days = step
        self.model_name = name
        self.predict_days = future_day

    def data_set_process(self,raw_data):
        datasets = series_to_supervised(raw_data,self.history_days,self.predict_days)
        datasets = datasets.values
        trainx,trainy = datasets[:,:self.history_days],datasets[:,self.history_days:]
        trainx = trainx.reshape(trainx.shape[0],trainx.shape[1],1)
        return trainx,trainy
    '''
    New lstm model
    7 days -> predict 1 day.
    42 data points predict 6 points
    '''
    def train_start(self,raw_data,gas_name,senser_id):
        cur_date=self.get_current_date()
        self.model_name = 'models/{}/{}/{}.h5'.format(senser_id,gas_name,senser_id)
        back_up_name='models/{}/{}/{}_{}.h5'.format(senser_id,gas_name,senser_id,cur_date)
        scaler_path='models/{}/{}/{}.save'.format(senser_id,gas_name,senser_id)
        #try not normalize the data
        raw_data,scaler= self.standard_data(raw_data)
        #convert 1-d array to [history-days,predict-days]
        trainx,trainy = self.data_set_process(raw_data)
        model = Sequential()
        model.add(LSTM(128, input_shape=(trainx.shape[1], trainx.shape[2])))
        model.add(Dense(self.predict_days))
        model.compile(loss='mae', optimizer='adam')
        model.fit(trainx,trainy,epochs=self.epoch,batch_size=50,verbose=2 ,shuffle=False)
        model.save(self.model_name)
        model.save(back_up_name)
        joblib.dump(scaler, scaler_path)
        return True

    def standard_data(self,data):
        scaler = MinMaxScaler()
        output_data=np.array(data).reshape(-1, 1)
        output_data = scaler.fit_transform(output_data)
        return output_data,scaler
    
    def get_current_date(self):
        return time.strftime("%Y%m%d", time.localtime()) 

    #calc the predict data
    #rightnow, using 42 data points to predict futre 6 data points
    def predict_get(self,raw_data,gas_type,sensor_id):
        #format the name sensor_id,gas type,train_result
        self.model_name = 'models/{}/{}/{}.h5'.format(sensor_id,gas_type,sensor_id)
        scaler_path='models/{}/{}/{}.save'.format(sensor_id,gas_type,sensor_id)
        #wrong json message
        if os.path.exists(self.model_name) == False or os.path.exists(scaler_path) == False:
            return 'No find model'
        if(len(raw_data)<self.history_days):
            return 'Data too short'
        #load the latest model
        my_scaler = joblib.load(scaler_path)
        std_data=np.array(raw_data).reshape(-1, 1)
        std_data = my_scaler.fit_transform(std_data)
        model = load_model(self.model_name)
        trainx,_ = self.data_set_process(std_data)
        #calculate the predict results
        result = model.predict(trainx)
        #reverse
        result  = my_scaler.inverse_transform(result)
        #self.predict_display(raw_data,result)
        return result[-1]

'''
end lstm model
'''

app = FastAPI()
gasModel = gasTrain(step=42,future_day=6)
#json message structure
class train_msg(BaseModel):
    msg:str
    code:int
    data:list
    size:int

sensor_name='sensorId'
scan_list=['c2h2','h2','totalGasppm']

def parser_msg_data(oil_data):
    c2h2_total=[]
    h2_total=[]
    gas_total=[]
    sensorid=set()
    all_data = {'c2h2':c2h2_total,'h2':h2_total,'totalGasppm':gas_total}
    for val in oil_data:
        #extract only h2, c2h2, totalGasppm gas data
        if sensor_name in val:
            sensorid.add(val[sensor_name])
        for gas_n in scan_list:
            #read out the data in the list
            if gas_n in val:
                if(val[gas_n] !=''):
                    all_data[gas_n].append(float(val[gas_n]))
    sensor_id = list(sensorid)[0]
    return all_data,sensor_id
#format the input json data
#right now ,we just train c2h2,h2,totalgas, this three parameters
def train_oil_gas_data(oil_data):
    output_data,sensor_id=parser_msg_data(oil_data)
    for gname in (scan_list):
        trainx = output_data[gname]
        if(len(trainx) == 0):
            continue
        gasModel.train_start(trainx,gname,sensor_id)
    return {'success':False,'msg':'Training done'}

#predict all types of the gas
def predict_oil_gas_data(raw_data):
    output_data,sensor_id=parser_msg_data(raw_data)
    all_result={}
    for gas_name in (scan_list):
        trainx = output_data[gas_name]
        if(len(trainx) == 0):
            continue
        all_result[gas_name]=gasModel.predict_get(trainx,gas_name,sensor_id).tolist()
    #map as json format
    return all_result

@app.post('/train')
async def train_user_data(train_data:train_msg):
    #print_row(train_data.data)
    if(len(train_data.data) != train_data.size  ):
        return {'success':False,'msg':'Data corrupt'}
    if(len(train_data.data)==0 or train_data.size == 0):
        return {'success':True,'msg':'Data empty'}
    return train_oil_gas_data(train_data.data)

@app.post('/predict')
async def predict_oil_data(train_data:train_msg):
    if(len(train_data.data) != train_data.size  ):
        return {'success':False,'msg':'Data corrupt'}
    if(len(train_data.data)==0 or train_data.size == 0):
        return {'success':False,'msg':'Data empty'}
    result= predict_oil_gas_data(train_data.data)
    return {'success':True,'predict':result}

@app.get("/")
def read_root():
    return {"Gas": "Prediction"}

#from fastapi import FastAPI

'''
@app.get("/")
async def read_root():
    return {"Gas": "Prediction"}

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
async def read_predict():
    import json
    with open("msg.json", 'r') as f:
        json_data = json.load(f)
    result=predict_oil_gas_data(json_data['data'])
    return result

'''

import socket

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



'''
if __name__ == "__main__":
    #uvicorn.run("ai_server:app", host="127.0.0.1", port=8000, reload=True,debug=True,log_level="trace")
    import json
    with open("msg.json", 'r') as f:
        json_data = json.load(f)
    result=predict_oil_gas_data(json_data['data'])
    print(result)



from flask import Flask
fapp = Flask(__name__)
@fapp.route('/')
def index():
    import json
    with open("msg.json", 'r') as f:
        json_data = json.load(f)
    result=predict_oil_gas_data(json_data['data'])
    return result

if __name__ == '__main__':
    fapp.debug = True # 设置调试模式，生产模式的时候要关掉debug
    fapp.run()

'''
    
#uvicorn ai_server:app --reload