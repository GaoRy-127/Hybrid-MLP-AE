#read file
import pandas as pd
import numpy as np
import os

def indata():
    os.chdir("code\\cfdeep\\result")
    data = pd.read_csv('invisible.txt',sep=",",header=None)

    indata=[]

    for i,rows in data.iterrows():
        indata.append([rows[0],rows[1]])

    indata = np.array(indata)

    # print(indata)
    
    return indata

def outdata():
            
    data = pd.read_csv('invisible.txt',sep=",",header=None)

    encoded=[]

    for i,rows in data.iterrows():
        encoded.append([rows[2],rows[3],rows[4],rows[5],rows[6],rows[7],rows[8],rows[9]])

    encoded = np.array(encoded)
    # print(encoded)
    return encoded
