#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import constants as const
import workingFunctions as wf
"""
Proceding of ARPES data sets from OMICON SES Software.
"""

__author__ = "Alexander Kononov"
__copyright__ = "Royalty-free"
__credits__ = ""
__license__ = "GPL"
__version__ = "0.2"
__maintainer__ = "Alexander Kononov"
__email__ = "alexander.kononov@tu-dortmund.de"
__status__ = "Production"


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



# path = '/run/media/hexander/main_drive/hexander/Documents/Uni/Promotion/UPS/Data_pkl/180427/'
# a = wf.loadObj(path +'10001.pkl')
# aRedRef = reduceByX(sliceData(a[1], ylim =[-5,5])) 
# #aRed.plot(yerr = aRed.apply(np.sqrt))
# counter = 0
# for i in wf.fileList(path):
#     counter += 1
#     fpath = path+i
#     a = wf.loadObj(fpath)
#     aRed = reduceByX(sliceData(a[1], ylim =[-5,5]))
#     #(aRedRef-aRed).plot()                 # yerr = aRed.apply(np.sqrt)
#     aRed.plot(label=i)
#     if counter == 2:
#         break
# plt.ylabel('Counts')
# plt.legend()
# plt.show()


# info,data = readIgorTxt(path)
# print data.info()
# xlim=[16.53,16.83]
# print [np.where(x>=xlim[0])[0][0], np.where(x>=xlim[1])[0][0]]
# print [x[np.where(x>=xlim[0])[0][0]], x[np.where(x>=xlim[1])[0][0]]]
# exit()
# im = plotData(data, title = path[:-4])
# plt.scatter(x=data.index.values,y=data.columns.values)
# data2 = sliceData(data, xlim = [16.8,17.0], ylim = [-2,1])
# print data.info()
# print "Energies="+str(len(data.index.values))
# print "mm="+str(len(data.columns.values))
# print data2.info()

# im = plotData(data2, title = path[:-4])
# plt.figure()
# asd = reduceByX(data2)
# asd.plot(yerr = asd.apply(np.sqrt))

# xred,yred = reduceByX(data2)
# plt.errorbar(xred,yred)
# plt.figure()
# plt.errorbar(reduceByY(data2))

# ylim = [-1,1]
# x,y,data = sliceData(x,y,sliceY, ylim = ylim)
# data = reduceByX(data)
# xlim = [15.9,16.1]
# plt.plot(x,data)

