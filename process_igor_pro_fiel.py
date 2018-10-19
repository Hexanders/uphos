#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd
from scipy import constants as const
import workingFunctions as wf
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import HoverTool
from collections import OrderedDict

import PySimpleGUI as sg

from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtCore import Qt
plt.switch_backend('Qt4Agg') #### macht segfault hmmmmmmm

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

def plotData_old(data,title = None):
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

def plotRed(dataSet,info, currentPlot = False):
    if currentPlot:
        p = currentPlot
    else:
        p = figure(plot_width=1000, plot_height=600,
                   tools="pan,box_zoom,reset,save,crosshair,hover,wheel_zoom", 
                   title="",
                   x_axis_label=dataSet.index.name, 
                   y_axis_label='Counts',
                   toolbar_location="left"
                   )

    df = dataSet.reset_index()
    df.columns = [dataSet.index.name,'Counts']
    source = ColumnDataSource.from_df(df)
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = OrderedDict(info['[Info 1]'].items())
    p.line(x='index', y='Counts', source=source, legend=info['[Info 1]']['Spectrum Name'])
    return p

def fermiFct(x,y,E_f,T):
    k_b = const.value(u'Boltzmann constant in eV/K')
    return 1./(np.exp((x-E_f)/k_b*T)+1.)


buttoncolor = 'lightskyblue'#'lightgoldenrodyellow'

def plotData(data,title = None):
    fig = plt.figure()
    # cid = fig.canvas.mpl_connect('resize_event', onresize)
    global ax
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
    button1pos= plt.axes([0.79, 0.0, 0.1, 0.075])
    button2pos = plt.axes([0.9, 0.0, 0.1, 0.075])

    bcut1 = Button(button1pos, 'Int. X', color=buttoncolor)
    bcut2 = Button(button2pos, 'Int. Y', color=buttoncolor)
    bcut1.on_clicked(lambda event: on_clickX(event, data))
    bcut2.on_clicked(lambda event: on_clickY(event, data))
    button1pos._button = bcut1 #otherwise the butten will be killed by carbagcollector
    button2pos._button = bcut2
    return im, data

def on_clickX(event,data):
        print('Start to Integrate X')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()      
        slicedData = sliceData(data, xlim = x_lim, ylim = y_lim)
        reducedData = reduceByX(slicedData)
        ax2.plot(reducedData)
        button3pos = plt.axes([0.9, 0.0, 0.1, 0.075])
        bcut3 = Button(button3pos, 'Save', color=buttoncolor)
        plt.show()
        bcut3.on_clicked(lambda event:saveReduceData(event, reducedData))
        button3pos._button = bcut3
        

def on_clickY(event, data):
    print('Start to Integrate X')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    button4pos = plt.axes([0.9, 0.0, 0.1, 0.075])
    bcut4 = Button(button4pos, 'Save', color=buttoncolor)
    slicedData = sliceData(data, xlim = x_lim, ylim = y_lim)
    reducedData = reduceByY(slicedData)
    ax2.plot(reducedData)
    plt.show()
    bcut4.on_clicked(lambda event:saveReduceData(event,reducedData))
    button3pos._button = bcut4

def saveReduceData(event, reddata):
    event, (filename,) = sg.Window('Save data'). Layout([[sg.Text('Filename')], [sg.Input(), sg.SaveAs()], [sg.OK(), sg.Cancel()] ]).Read()
    reddata.to_pickle(filename)
    return event

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

