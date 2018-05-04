#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants as const
"""
Proceding of ARPES data sets from OMICON SES Software.
"""

__author__ = "Alexander Kononov"
__copyright__ = "Royalty-free"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Alexander Kononov"
__email__ = "alexander.kononov@tu-dortmund.de"
__status__ = "Production"


def readIgorTxt(igor_data_path):
    """
    Convert data points from Igor Pro generated .txt file.
    Return: 
        Info - list
        dimOne - 1d numpy array: Kinetic Energy
        dimTwo - 1d numpy array: Coordinates (Transmition Mode) or angles (ARPES) 
        data - 2d numpy array: Each element of array represents a slice for certain Energy along dimTwo (Coordinates or angles). Attention! First element is the Energy of a slice.   
    """
    info = []
    dimOne = []
    dimTwo = []
    data = []                       # Each line is 'vertical' slice by one certan Enery 
    data_field_trigger = False
    with open(igor_data_path) as igor_data:
        for line in igor_data:
            if not data_field_trigger and 'Dimension' and 'scale' in line: 
                dim_dummy = line.split('=')
                if [int(s) for s in dim_dummy[0].split() if s.isdigit()][0] == 1:
                    str_list = dim_dummy[1].strip().split(' ')
                    str_list = list(filter(None, str_list)) # erase empty strings from list
                    dimOne.extend(str_list)
                else:
                    str_list = dim_dummy[1].strip().split(' ')
                    str_list = list(filter(None, str_list)) # erase empty strings from list
                    dimTwo.extend(str_list)
            if not data_field_trigger and not 'scale' in line:
                info.append(line.strip())
            if 'Data' in line:
                data_field_trigger = True
            if data_field_trigger:
                str_list = line.strip().split(' ')
                str_list = list(filter(None, str_list)) # erase empty strings from list
                data.append(str_list)
                data = list(filter(None, data)) # some how one of the elements is empty, erase them!
        del data[0]             # remove first line because it is a string e.g.'[Data 1]' 
        dimOne = np.asfarray(dimOne)
        dimTwo = np.asfarray(dimTwo)
        for i in range(0,len(data)):
            #data[i] = np.asfarray(data[i][1:])
            data[i] = np.asfarray(data[i])
        data = np.asfarray(data)
        data = pd.DataFrame(data=data)
        data = data.set_index([0])
        data.columns = dimTwo
        #data = data.to_panel()
        for i in info:
            if 'Dimension 1 name' in i:
                data.index.name = i.split('=')[1]
            if 'Dimension 2 name' in i:
                data.columns.name = i.split('=')[1]
        return(info, data)
    
def loadObj(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj

    
def sliceData(data, xlim = None, ylim = None):
    if xlim:
        x1 = data.index.values.argmin() if xlim[0] < data.index.values.min() else np.where(data.index.values>=xlim[0])[0][0]
        x2 = data.index.values.argmax() if xlim[1] > data.index.values.max() else np.where(data.index.values>=xlim[1])[0][0] 
    if ylim:
        #hier funkt was nicht. Die Columns werden nicht richtig wiedergegeben.
        y1 = data.columns.values.argmin() if ylim[0] < data.columns.values.min() else np.where(data.columns.values>=ylim[0])[0][0]
        y2 = data.columns.values.argmax() if ylim[1] > data.columns.values.max() else np.where(data.columns.values>=ylim[1])[0][0]
    if xlim and ylim:
        data = data.iloc[x1:x2,y1:y2] 
    elif xlim:
        data = data.iloc[x1:x2,:]
    elif ylim:
        data = data.iloc[:,y1:y2]
    return data


def reduceByX(data):
    '''Integriere Daten entlang einzelnen Energiewerten '''
    #return np.add.reduce(data.T)
    return data.apply(np.sum, axis = 1)
def reduceByY(data):
    '''Integriere Entlang Y.'''
    #return np.add.reduce(data)
    return data.apply(np.sum, axis = 0)


def plotData(data,title = None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title:
        fig.canvas.set_window_title(title)
    else:
        fig.canvas.set_window_title('Data_Set')
    x,y = data.index.values, data.columns.values
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    im = plt.imshow(data.T,extent=extent, origin = 'lower', cmap='hot',  aspect = 'auto')
    plt.xlabel(data.index.name)
    plt.ylabel(data.columns.name)
    plt.colorbar()
    plt.tight_layout()
    return im

def fermiFct(x,y,E_f,T):
    k_b = const.value(u'Boltzmann constant in eV/K')
    return 1./(np.exp((x-E_f)/k_b*T)+1.)



path = 'SS_77K.txt'
# path = 'SS_and_Dband_77K.txt'
# path =  'Ni3Al_2.txt'
#path =  'Ni3Al_1.txt'
#path =  'Ni3Al_2.txt'
info,data = readIgorTxt(path)
# print data.info()
# xlim=[16.53,16.83]
# print [np.where(x>=xlim[0])[0][0], np.where(x>=xlim[1])[0][0]]
# print [x[np.where(x>=xlim[0])[0][0]], x[np.where(x>=xlim[1])[0][0]]]
# exit()
im = plotData(data, title = path[:-4])
# plt.scatter(x=data.index.values,y=data.columns.values)
data2 = sliceData(data, xlim = [16.8,17.0], ylim = [-2,1])
# print data.info()
# print "Energies="+str(len(data.index.values))
# print "mm="+str(len(data.columns.values))
# print data2.info()
im = plotData(data2, title = path[:-4])
plt.figure()
asd = reduceByX(data2)
asd.plot(yerr = asd.apply(np.sqrt))
# xred,yred = reduceByX(data2)
# plt.errorbar(xred,yred)
# plt.figure()
# plt.errorbar(reduceByY(data2))

# ylim = [-1,1]
# x,y,data = sliceData(x,y,sliceY, ylim = ylim)
# data = reduceByX(data)
# xlim = [15.9,16.1]
# plt.plot(x,data)
plt.show()
