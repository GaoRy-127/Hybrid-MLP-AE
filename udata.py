#read file
import pandas as pd
import numpy as np
import os



def outdata():
    
    os.chdir('./data') # py에서 쓸 때 os.chdir('code\\cfdeep\\data')  # 주피터 랩에서 쓸때os.chdir('./data')
    file_names = os.listdir()

    udata = []

    for file_name in file_names:
        if os.path.splitext(file_name)[1] == '.txt': # txt 파일만 읽어옴
            # print(file_name)

            data = pd.read_csv(file_name, sep=" ", header=None)
            data = data.dropna()
            # data = data.values.flatten().tolist()
            # data = np.array(data)
            # data = data[0:1000,:]
            udata = udata + [data]
            

    udata = np.array(udata)
    # print(udata)
    # print(np.shape(udata))
            
    return udata


def indata():
    
    
    file_names = os.listdir()
    xdata = []
    for file_name in file_names:
        if os.path.splitext(file_name)[1] == '.txt': # txt 파일만 읽어옴
            # print(file_name)
            indata = os.path.splitext(file_name)[0]
            infront = float(indata.split('_')[1])
            inback = float(indata.split('_')[2])
            indata = [infront, inback]

            xdata = xdata + [indata]

    xdata = np.array(xdata)

    return xdata
