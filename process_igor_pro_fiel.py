#!/usr/bin/python3
import numpy as np
import sys
from matplotlib.widgets import Button
import pandas as pd
from scipy import constants as const
import workingFunctions as wf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import tkinter as Tk
import matplotlib.backends.tkagg as tkagg

from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.io import output_notebook
from bokeh.models import HoverTool
from collections import OrderedDict

from scipy.optimize import curve_fit


import PySimpleGUI as sg

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# from PyQt5 import QtGui
# from PyQt5 import QtCore
# from PyQt5.QtCore import Qt
# plt.switch_backend('Qt5Agg') #### macht segfault hmmmmmmm

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
    x,y = data[1].index.values, data[1].columns.values
    extent = np.min(x), np.max(x), np.min(y), np.max(y)
    im = plt.imshow(data[1].T,extent=extent, origin = 'lower', cmap='hot',  aspect = 'auto')
    plt.xlabel(data[1].index.name)
    plt.ylabel(data[1].columns.name)
    plt.colorbar()
    plt.tight_layout()
    button1pos= plt.axes([0.79, 0.0, 0.1, 0.075]) #posx, posy, width, height in %
    button2pos = plt.axes([0.9, 0.0, 0.1, 0.075])
    button3pos = plt.axes([0.9, 0.1, 0.1, 0.075])
    bcut1 = Button(button1pos, 'Int. X', color=buttoncolor)
    bcut2 = Button(button2pos, 'Int. Y', color=buttoncolor)
    bcut3 = Button(button3pos, 'Info', color=buttoncolor)
    bcut1.on_clicked(lambda event: on_clickX(event, data[1]))
    bcut2.on_clicked(lambda event: on_clickY(event, data[1]))
    bcut3.on_clicked(lambda event: on_clickInfo(event, data[0]))
    button1pos._button = bcut1 #otherwise the butten will be killed by carbagcollector
    button2pos._button = bcut2
    button3pos._button = bcut3
    #im = plt.gcf()
    return im

def on_clickInfo(event,data):
    temp = []
    dictlist = []
    for key, value in data.items():
        temp = [key,value]
        dictlist.append(temp)
    # event = sg.Window('Info'). Layout([[sg.Listbox(values=dictlist,size=(40, 20))],[sg.Cancel()] ]).Read()
    # event = sg.Window('Info',auto_size_text=True,font=("Helvetica", 18)). Layout([[sg.Multiline(dictlist,size=(80, 10))],[sg.Cancel()] ]).Read()
    event = sg.Window('Info',auto_size_text=True,font=("Helvetica", 18)). Layout([[sg.Multiline([grab_dic(data)],size=(80, 10))],[sg.Cancel()]]).Read()    
    return event

def grab_dic(data):
    #tmp_list = []
    info_list = []
    for ele in data.values():
        if isinstance(ele,dict):
            for k, v in ele.items():
                info_list.append(k+' : '+v+'\n')
    return ' '.join(info_list)
      
    
def on_clickX(event,data):
    print('Start to Integrate X')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    digis = 3
    ax2.set_title('x:%s y:%s' %((round(x_lim[0],digis),round(x_lim[1],digis)), (round(y_lim[0],digis),round(y_lim[1],digis))))
    slicedData = sliceData(data, xlim = x_lim, ylim = y_lim)
    reducedData = reduceByX(slicedData)
    #print(reducedData.values, type(reducedData), len(reducedData))
    ax2.plot(reducedData, 'bo')
    button3pos = plt.axes([0.9, 0.0, 0.1, 0.075])
    bcut3 = Button(button3pos, 'Save', color=buttoncolor)
    buttonFitpos = plt.axes([0.9, 0.1, 0.1, 0.075])
    buttonFit = Button(buttonFitpos, 'Fit', color=buttoncolor)
    bcut3.on_clicked(lambda event:saveReduceData(event, reducedData))
    buttonFit.on_clicked(lambda event:fitPanel(event, ax2, data))
    button3pos._button = bcut3 #without this the garbage collector destroyes the button
    buttonFitpos._button = buttonFit
    plt.show()
    #plt.legend()

def fermiFct(x,E_f,b,s,T):
    k_b = const.value(u'Boltzmann constant in eV/K')
    return b + s*(1./(np.exp((x-E_f)/(k_b*T))))

def fitPanel(event, ax, data):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    line1 = ax.axvline(x=x_lim[0])
    line2 = ax.axvline(x=x_lim[1])
    x_abstand = abs(x_lim[1]-x_lim[0])/len(data)
    layout = [
        [sg.Text(r'Left'), sg.Slider(key = 'LeftLeft',range=(x_lim[0]*1e5,x_lim[1]*1e5), resolution = 1, orientation='h', size=(34, 20), default_value=x_lim[0]),
         sg.Slider(key = 'LeftRight', range=(x_lim[0]*1e5,x_lim[1]*1e5),resolution = 1, orientation='h', size=(34, 20), default_value=x_lim[0]),],
        [sg.Text(r'Right'), sg.Slider(key = 'RightLeft',range=(x_lim[0]*1e5,x_lim[1]*1e5), resolution = x_abstand, orientation='h', size=(34, 20), default_value=x_lim[1]),
         sg.Slider(key = 'RightRight', range=(x_lim[0]*1e5,x_lim[1]*1e5),resolution = x_abstand, orientation='h', size=(34, 20), default_value=x_lim[1]),],
        [sg.ReadButton('Fit'), sg.Cancel()],
    ]
    window = sg.Window('Fit Parameter for figure ' + str(plt.gcf().number))
    window.Layout(layout)
    window.Finalize()
    while True:
        if event == 'Ok' or event is None:    # be nice to your user, always have an exit from your form
            break
        event, values = window.Read()
        print(values)
        del line1
        line1 = ax.axvline(x=values['LeftLeft']/1e5)
        plt.show()
    return event, values


def fitPanel_old(event, ax, data):
    layout = [[sg.Text(r'$g = B + S\times f(T,E_f,E)$\n $f(T,E_f,E) = [\exp{((E-E_f)/(k_b\cdot T))+1}]^{-1}$')],    
                [sg.Text(r'E_f'), sg.InputText('16.89',key='E_f')],
                [sg.Text(r'B'), sg.InputText('5000',key='B')],
                [sg.Text(r'S'), sg.InputText('200000',key='S')],
                [sg.Text(r'T'), sg.InputText('10',key='T')],      
                [sg.ReadButton('Fit'), sg.Cancel()],
    ]
    window = sg.Window('Fit Parameter',force_top_level = True)
    window.Layout(layout)
    window.Finalize()
    while True:      
        event2, values = window.Read()
        if event2 is None:      
            break
        if event2 == 'Fit':
            try:
                event, values = fitFermi(event, data, ax, values.values())
            except:
                print("Error:", sys.exc_info()[0])
                raise
        E_f, B, S, T = values#['16.89','5000', '200000', '10']
        window.FindElement('E_f').Update(str(E_f))
        window.FindElement('B').Update(str(B))
        window.FindElement('S').Update(str(S))
        window.FindElement('T').Update(str(T))
    return event, values

def fitFermi(event, data, ax, p0):
    mask = (data.index > 16.2) & (data.index <= 17.0)
    # print(data.values[mask][:,0], data.values[mask][:,1])
    # print(len(data.index[mask]), len(data.values[mask]))
    # print(type(data.index[mask]), type(data.values[mask]))
    
    try:
        p0=[float(x) for x in p0]
    except:
        print("Error:", sys.exc_info()[0])
        raise
    try:
        popt, pcov = curve_fit(fermiFct, data.index[mask], data.values[mask][:,0], p0=p0)
    except:
        print("Error:", sys.exc_info()[0])
        raise
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    # if fitPlot:
    #     print(fitPlot)
    #     fitPlot.pop(0).remove()
    fitPlot = ax.plot(data.index[mask], fermiFct(data.index[mask], *popt), 'r-', label='fit: E_f=%5.3f, T=%5.3f,b=%5.3f,c=%5.3f ' % tuple(popt))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.show()
    print('POPT:%s' % (popt))
    return event, p0
        

def on_clickY(event, data):
    print('Start to Integrate Y')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    digis = 3
    ax2.set_title('x:%s y:%s' %((round(x_lim[0],digis),round(x_lim[1],digis)), (round(y_lim[0],digis),round(y_lim[1],digis)))) 
    button4pos = plt.axes([0.9, 0.0, 0.1, 0.075])
    bcut4 = Button(button4pos, 'Save', color=buttoncolor)
    slicedData = sliceData(data, xlim = x_lim, ylim = y_lim)
    reducedData = reduceByY(slicedData)
    ax2.plot(reducedData, 'ko')
    plt.show()
    bcut4.on_clicked(lambda event:saveReduceData(event,reducedData))
    button4pos._button = bcut4

def saveReduceData(event, reddata):
    event, (filename,) = sg.Window('Save data'). Layout([[sg.Text('Filename')], [sg.Input(), sg.SaveAs()], [sg.OK(), sg.Cancel()]]).Read()
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

