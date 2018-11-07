#!/usr/bin/python3
import numpy as np
import sys
from matplotlib.widgets import Button
import pandas as pd
from scipy import constants as const
from scipy.interpolate import interp1d
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
    buttonFit = Button(buttonFitpos, 'Fit-Panel', color=buttoncolor)
    bcut3.on_clicked(lambda event:saveReduceData(event, reducedData))
    buttonFit.on_clicked(lambda event:fitPanel(event, ax2, reducedData))
    button3pos._button = bcut3 #without this the garbage collector destroyes the button
    buttonFitpos._button = buttonFit
    plt.show()

def fitPanel(event, ax, data):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_abstand = abs(x_lim[1]-x_lim[0])/len(data)
    leftbound, rightbound = x_lim[0], x_lim[1]
    leftboundStep = x_lim[0]+abs(x_lim[1]-x_lim[0])*0.05
    rightboundStep = x_lim[1]-abs(x_lim[1]-x_lim[0])*0.05
    faktor = 1e5                # da es in PySimpleGUI der slider nur die int Werte zurueck gibt
    layout = [# ll, lr steht fuer LeftLeft, LeftRight, ... 
        [sg.Text(r'Left'), \
        sg.Slider(key = 'll_slider', change_submits = True, background_color = 'red',\
                  range=(x_lim[0]*faktor,x_lim[1]*faktor), resolution = 1, orientation='h', size=(34, 20), default_value=leftbound),
        sg.Slider(key = 'lr_slider', change_submits = True, background_color = 'red', \
                  range=(x_lim[0]*faktor,x_lim[1]*faktor),resolution = 1, orientation='h', size=(34, 20), default_value=leftboundStep),\
        sg.Spin(data.index,key='ll_spin',size=(10, 20), auto_size_text = True),\
        sg.Spin(data.index,key='lr_spin',size=(10, 20), auto_size_text = True)],
        [sg.Text(r'Right'), sg.Slider(key = 'rl_slider', change_submits = True, background_color = 'green',\
                                      range=(x_lim[0]*faktor,x_lim[1]*faktor), resolution = x_abstand, orientation='h', size=(34, 20), default_value=rightboundStep),
        sg.Slider(key = 'rr_slider', change_submits = True, background_color = 'green',\
                  range=(x_lim[0]*faktor,x_lim[1]*faktor),resolution = x_abstand, orientation='h', size=(34, 20), default_value=rightbound),
        sg.Spin(data.index,key='rl_spin',size=(10, 20), auto_size_text = True),
        sg.Spin(data.index,key='rr_spin',size=(10, 20), auto_size_text = True)],
        [sg.ReadButton('Fit')],
        [sg.ReadButton('Finde Fermi Edge'), sg.Text(r'Fermi edge [eV]'), sg.InputText(size =(10,20), key='fermi_edge'), sg.Text('16%-84% width [eV]'), sg.InputText(size =(10,20),  key = 'resolution')] ,
        [sg.Cancel()],
    ]
   
    window = sg.Window('Fit Parameter for figure ' + str(plt.gcf().number), grab_anywhere=False)
    window.Layout(layout)
    window.Finalize()
    line1, = ax.plot((leftbound,leftbound),y_lim, color = 'r', marker = '>', alpha=0.5)
    line2, = ax.plot((leftboundStep, leftboundStep),y_lim, color = 'r', marker = '<', alpha=0.5)
    line3, = ax.plot((rightboundStep, rightboundStep),y_lim, color = 'g', marker = '>', alpha=0.5)
    line4, = ax.plot((rightbound, rightbound),y_lim, color = 'g', marker = '<', alpha=0.5)
    leftFit =[]
    rightFit = []
    while True:
        event, values = window.Read()
        line1.set_xdata((values['ll_slider']/faktor,values['ll_slider']/faktor))
        line2.set_xdata((values['lr_slider']/faktor,values['lr_slider']/faktor))
        line3.set_xdata((values['rl_slider']/faktor,values['rl_slider']/faktor))
        line4.set_xdata((values['rr_slider']/faktor,values['rr_slider']/faktor))
        window.FindElement('ll_spin').Update(values['ll_slider']/faktor)
        window.FindElement('lr_spin').Update(values['lr_slider']/faktor)
        window.FindElement('rl_spin').Update(values['rl_slider']/faktor)
        window.FindElement('rr_spin').Update(values['rr_slider']/faktor)
        if event == 'Fit':
            try:
                leftFitPara = fitLinear(event, (values['ll_slider']/faktor,values['lr_slider']/faktor), data, ax, 'red')
                rightFitPara = fitLinear(event, (values['rl_slider']/faktor,values['rr_slider']/faktor), data, ax, 'green')
                if leftFit:
                    leftFit[0].set_ydata(LinearFit(data.index,*leftFitPara))
                    rightFiinter_line,inter_line,t[0].set_ydata(LinearFit(data.index,*rightFitPara))
                else:
                    leftFit, = ax.plot(data.index, LinearFit(data.index, *leftFitPara), color = 'red', label='fit: a=%5.3f, b=%5.3f ' % tuple(leftFitPara))
                    rightFit, = ax.plot(data.index, LinearFit(data.index, *rightFitPara), color = 'green', label='fit: a=%5.3f, b=%5.3f ' % tuple(rightFitPara))
                inter = interpolate(data, ax)    
                inter_line, = ax.plot(inter[0], inter[1])      
                inter_dot = ax.scatter(inter[0], inter[1])
            except TypeError as error:
                if str(error) == 'Improper input: N=2 must not exceed M=0':
                    print('Please select the appropriate limits for the fit')
                    pass
                else:
                    print("Error:", sys.exc_info()[0])
                    raise
        if event == 'Finde Fermi Edge':
            try:
                fermi_edge_plot, sexteen_plot, eigthy4_plot = finde_edge(inter,leftFitPara,rightFitPara,ax,window)
            except:
                print("Error:", sys.exc_info()[0]) 
                raise
        plt.draw()
        if event == 'Cancel' or event is None:    # be nice to your user, always have an exit from your form
            line1.remove()
            line2.remove()
            line3.remove()
            line4.remove()
            if inter_line: inter_line.remove()
            if inter_dot: inter_dot.remove()
            if fermi_edge_plot: fermi_edge_plot.remove()
            if sexteen_plot: sexteen_plot.remove()
            if eigthy4_plot: eigthy4_plot.remove()
            if leftFit: leftFit.remove()
            if rightFit: rightFit.remove()
            break
     
    window.Close()
    return event, values

def finde_edge(interPolData, fit1Para, fit2Para, ax, window):
    for i in range(0, len(interPolData[0])):
        diff = 0.5*(abs(LinearFit(interPolData[0][i],*fit1Para)-LinearFit(interPolData[0][i],*fit2Para)))
        fermi_edge = interPolData[1][i] - LinearFit(interPolData[0][i],*fit2Para)
        if fermi_edge <= diff:
            fermi_edge_plot = ax.axvline(x=interPolData[0][i], color = 'k', dashes = (5, 1))
            fermi_edge_x = interPolData[0][i]
            plt.draw()
            window.FindElement('fermi_edge').Update(str(fermi_edge_x))
            break
    for i in range(0, len(interPolData[0])):
        diff = 0.16*(abs(LinearFit(interPolData[0][i],*fit1Para)-LinearFit(interPolData[0][i],*fit2Para)))
        sexteen = interPolData[1][i] - LinearFit(interPolData[0][i],*fit2Para)
        if sexteen <= diff:
            sexteen_x = interPolData[0][i]
            sexteen_plot = ax.axvline(x=interPolData[0][i], color = 'k', dashes = (5, 1))
            plt.draw()
            break
    for i in range(0, len(interPolData[0])):
        diff = 0.84*(abs(LinearFit(interPolData[0][i],*fit1Para)-LinearFit(interPolData[0][i],*fit2Para)))
        eigthy4 = interPolData[1][i] - LinearFit(interPolData[0][i],*fit2Para)
        if eigthy4 <= diff:
            eigthy4_plot = ax.axvline(x=interPolData[0][i], color = 'k', dashes = (5, 1))
            eigthy4_x = interPolData[0][i]
            plt.draw()
            window.FindElement('resolution').Update(str(abs(eigthy4_x-sexteen_x)))
            break
    return fermi_edge_plot, sexteen_plot, eigthy4_plot 

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
                p0 = values.values('E_f','B','S','T')
                print(p0)
                event, values = fitFermi(event, data, ax, p0)
            except:
                print("Error:", sys.exc_info()[0])
                raise
        E_f, B, S, T = values#['16.89','5000', '200000', '10']
        window.FindElement('E_f').Update(str(E_f))
        window.FindElement('B').Update(str(B))
        window.FindElement('S').Update(str(S))
        window.FindElement('T').Update(str(T))
    return event, values

def fermiFct(x,E_f,b,s,T):
    k_b = const.value(u'Boltzmann constant in eV/K')
    return b + s*(1./(np.exp((x-E_f)/(k_b*T))))

def LinearFit(x,a,b):
    return a*x+b

def fitLinear(event, x_range, data, ax, color):
    mask = (data.index > x_range[0]) & (data.index <= x_range[1])
    
    # try:
    #     p0=[float(x) for x in p0]
    # except:
    #     print("Error:", sys.exc_info()[0])
    #     raise
    try:
        popt, pcov = curve_fit(LinearFit, data.index[mask], data.values[mask])
    except:
        print("Error:", sys.exc_info()[0])
        raise
    #fitPlot = ax.plot(data.index, LinearFit(data.index, *popt), color = color, label='fit: a=%5.3f, b=%5.3f ' % tuple(popt))
    return popt

def fitFermi(event, data, ax, p0):
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    mask = (data.index > x_lim[0]) & (data.index <= x_lim[1])
    # print(data.values[mask][:,0], data.values[mask][:,1])
    # print(len(data.index[mask]), len(data.values[mask]))
    # print(type(data.index[mask]), type(data.values[mask]))
    
    try:
        p0=[float(x) for x in p0]
    except:
        print("Error:", sys.exc_info()[0])
        raise
    try:
        popt, pcov = curve_fit(fermiFct, data.index[mask], data.values[mask], p0=p0)
    except:
        print("Error:", sys.exc_info()[0])
        raise
    # if fitPlot:
    #     print(fitPlot)
    #     fitPlot.pop(0).remove()
    fitPlot = ax.plot(data.index[mask], fermiFct(data.index[mask], *popt), 'r-', label='fit: E_f=%5.3f, T=%5.3f,b=%5.3f,c=%5.3f ' % tuple(popt))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    plt.show()
    #print('POPT:%s' % (popt))
    values = {'E_f':popt[0],'B':popt[1],'S':popt[2],'T':popt[3]}
    return event, values

def interpolate(data, ax, xstep = None):
    '''
    xstep: int faktor of the interpolated points. So if xstep = 2, two times more points would be created. Default 10
    '''
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    if x_lim[0]<data.index.min():
        x_lim = data.index.min(), x_lim[1]
    if x_lim[1]>data.index.max():
        x_lim = x_lim[0], data.index.max()
    mask = (data.index > x_lim[0]) & (data.index <= x_lim[1])
    f = interp1d(data.index[mask], data.values[mask], fill_value="extrapolate")
    if xstep == None:
        xstep = 10
    newx = np.linspace(x_lim[0], x_lim[1], num=xstep*len(data.index[mask]), endpoint=True)
    return newx, f(newx)

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

