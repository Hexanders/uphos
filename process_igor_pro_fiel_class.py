#!/usr/bin/python3
import numpy as np
import sys
from matplotlib.widgets import Button
import pandas as pd
from pandas import read_pickle
from scipy import constants as const
from scipy.interpolate import interp1d
#import workingFunctions as wf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import tkinter as Tk 
import matplotlib.backends.tkagg as tkagg
import matplotlib._pylab_helpers


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

sg.SetOptions(auto_size_text = False)
class Uphos:
    def __init__(self, path):
        try:
            if path.endswith('.txt'):
                print('I working on %s' % path)
                self.path = path 
                self.info, self.data = self.readIgorTxt()
            else:
                print('I working on %s' %  path)
                self.path = path
                self.nfo, self.data = read_pickle(path)
        except:
            print ('Can not read file: %s' % path )
            raise
        
    def get_data(self):
        print(data)
    
    def readIgorTxt(self):
        """
        Convert data points from Igor Pro generated .txt file.
        Return: 
        Info - list
        dimOne - 1d numpy array: Kinetic Energy
        dimTwo - 1d numpy array: Coordinates (Transmition Mode) or angles (ARPES) 
        data - 2d numpy array: Each element of array represents a slice for certain 
               Energy along dimTwo (Coordinates or angles). Attention! First element
               is the Energy of a slice.   
        """
        info = []
        dimOne = []
        dimTwo = []
        data = []                       # Each line is 'vertical' slice by one certan Enery 
        data_field_trigger = False
        with open(self.path) as igor_data:
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
                    self.data = list(filter(None, data)) # some how one of the elements is empty, erase them!
                    del self.data[0]             # remove first line because it is a string e.g.'[Data 1]' 
                    dimOne = np.asfarray(dimOne)
                    dimTwo = np.asfarray(dimTwo)
            for i in range(0,len(data)):
                #data[i] = np.asfarray(data[i][1:])
                self.data[i] = np.asfarray(data[i])
                self.data = np.asfarray(data)
                self.data = pd.DataFrame(data=data)
                self.data = data.set_index([0])
                self.data.columns = dimTwo
                #data = data.to_panel()
            info_dic = {}
            for i in info:
                if 'Dimension 1 name' in i:
                    self.data.index.name = i.split('=')[1]
                if 'Dimension 2 name' in i:
                    self.data.columns.name = i.split('=')[1] 
                if i == '':
                    continue
                if i.startswith('[') and i.endswith(']'):
                    info_dic[i] = {} 
                    curent_item = i
                    continue
                sub_item = i.split('=' , 1)
                info_dic[curent_item][sub_item[0]] = sub_item[1]
            self.data = (info_dic, self.data)

    def plotData(self, title = None, interactive = True):
        fig = plt.figure()
        # cid = fig.canvas.mpl_connect('resize_event', onresize)
        global ax
        ax = fig.add_subplot(111)
        if title:
            fig.canvas.set_window_title(title)
        else:
            fig.canvas.set_window_title('Data_Set')
        x,y = self.data.index.values, self.data.columns.values
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        im = plt.imshow(self.data.T,extent=extent, origin = 'lower', cmap='hot',  aspect = 'auto')
        plt.xlabel(self.data.index.name)
        plt.ylabel(self.data.columns.name)
        plt.colorbar()
        plt.tight_layout()
        if interactive:
            button1pos= plt.axes([0.79, 0.0, 0.1, 0.075]) #posx, posy, width, height in %
            button2pos = plt.axes([0.9, 0.0, 0.1, 0.075])
            button3pos = plt.axes([0.9, 0.1, 0.1, 0.075])
            bcut1 = Button(button1pos, 'Int. X', color=buttoncolor)
            bcut2 = Button(button2pos, 'Int. Y', color=buttoncolor)
            bcut3 = Button(button3pos, 'Info', color=buttoncolor)
            bcut1.on_clicked(lambda event: self.on_clickX(event, self.data))
            bcut2.on_clicked(lambda event: self.on_clickY(event, self.data))
            bcut3.on_clicked(lambda event: self.on_clickInfo(event))
            button1pos._button = bcut1 #otherwise the butten will be killed by carbagcollector
            button2pos._button = bcut2
            button3pos._button = bcut3
        return im

    def plotOverviw(self, data = None):
        if data is not None:
            tmp_data = data
        else:
            tmp_data = self.data
        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 4, hspace=0.01, wspace=0.01)       
        main_ax = fig.add_subplot(grid[1:4, :-1])       
        x,y = tmp_data.index.values, tmp_data.columns.values
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        im = plt.imshow(tmp_data.T, extent = extent, origin = 'lower', cmap='hot',  aspect = 'auto')
        plt.xlabel(tmp_data.index.name)
        plt.ylabel(tmp_data.columns.name)
        main_xlim = plt.xlim()
        main_ylim = plt.ylim()
        cbar_ax = fig.add_axes([0.01, 0.95, 0.9, 0.05])
        plt.colorbar(im, cax = cbar_ax, orientation='horizontal')     
        y_int = fig.add_subplot(grid[1:4, -1])
        y_int.get_yaxis().set_ticks([])
        #plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0), useMathText = True)
        plt.ylim(main_ylim)
        yred = self.reduceY(data = tmp_data)
        plt.xticks(rotation='vertical')
        plt.plot(yred.values, yred.index.values)
        #x_int = fig.add_subplot(grid[-1, 1:])
        x_int = fig.add_subplot(grid[0, :-1])
        x_int.get_xaxis().set_ticks([])
        #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText = True)
        plt.xlim(main_xlim)
        xred = self.reduceX(data = tmp_data)
        plt.plot(xred)
        return fig

    def reduceX(self, data = None):
        '''Integriere Daten entlang einzelnen Energiewerten '''
        if data is not None:
            self.XredData = data.apply(np.sum, axis = 1)
        else:
            self.XredData = self.data.apply(np.sum, axis = 1) 
        return self.XredData 

    def reduceY(self, data = None):
        '''Integritmp_dataere Entlang Y.'''
        #return np.add.reduce(data)
        if data is not None:
            self.YredData = data.apply(np.sum, axis = 0)
        else:
            self.YredData = self.data.apply(np.sum, axis = 0) 
        return self.YredData

    def fermiFct(self, x, E_f, b, s, T):
        k_b = const.value(u'Boltzmann constant in eV/K')
        return b + s*(1./(np.exp((x-E_f)/(k_b*T))))

    def fitFermi(self,  p0 =[0.0001,0.00001,0.0001,0.0001], x_lim = None , y_lim = None):
        if x_lim == None:
            x_lim = (self.XredData.index.values.min(), self.XredData.index.values.max())
        if y_lim == None:
            y_lim = (self.XredData.values.min(), self.XredData.values.max())
        if x_lim is not None: mask = (self.XredData .index > x_lim[0]) & (self.XredData.index <= x_lim[1])
        try:
            p0=[float(x) for x in p0]
        except:
            print("Error:", sys.exc_info()[0])
            # raise
        try:
            popt, pcov = curve_fit(self.fermiFct, self.XredData.index[mask], self.XredData.values[mask], p0=p0)
        except:
            print("Error:", sys.exc_info()[0])
            raise
        #fitPlot = ax.plot(data.index[mask], fermiFct(data.index[mask], *popt), 'r-', label='fit: E_f=%5.3f, T=%5.3f,b=%5.3f,c=%5.3f ' % tuple(popt))
        #ax.set_xlim(x_lim)
        #ax.set_ylim(y_lim)
        #plt.show()
        values = {'E_f':popt[0],'B':popt[1],'S':popt[2],'T':popt[3]}
        return values

    
    def reduceByX(self, data):
        '''Integriere Daten entlang einzelnen Energiewerten '''
        #return np.add.reduce(data.T
        return self.data.apply(np.sum, axis = 1)

    def reduceByY(self, data):
        '''Integriere Entlang Y.'''
        #return np.add.reduce(data)
        return self.data.apply(np.sum, axis = 0)

    def on_clickY(self, event, data):
        print('Start to Integrate Y')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        digis = 3
        ax2.set_title('x:%s y:%s' %((round(x_lim[0],digis),round(x_lim[1],digis)), (round(y_lim[0],digis),round(y_lim[1],digis)))) 
        button4pos = plt.axes([0.9, 0.0, 0.1, 0.075])
        bcut4 = Button(button4pos, 'Save', color=buttoncolor)
        slicedData = self.sliceData(data, xlim = x_lim, ylim = y_lim)
        reducedData = self.reduceByY(slicedData)
        ax2.plot(reducedData, 'ko')
        plt.show()
        bcut4.self.on_clicked(lambda event:saveReduceData(event,reducedData))
        button4pos._button = bcut4
        plt.show()
        
    def sliceData(self, xlim = None, ylim = None):
        if xlim:
            x1 = self.data.index.values.argmin() if xlim[0] < self.data.index.values.min() else np.where(self.data.index.values>=xlim[0])[0][0]
            x2 = self.data.index.values.argmax() if xlim[1] > self.data.index.values.max() else np.where(self.data.index.values>=xlim[1])[0][0] 
        if ylim:
            #hier funkt was nicht. Die Columns werden nicht richtig wiedergegeben.
            y1 = self.data.columns.values.argmin() if ylim[0] < self.data.columns.values.min() else np.where(self.data.columns.values>=ylim[0])[0][0]
            y2 = self.data.columns.values.argmax() if ylim[1] > self.data.columns.values.max() else np.where(self.data.columns.values>=ylim[1])[0][0]
        if xlim and ylim:
            self.data = self.data.iloc[x1:x2,y1:y2] 
        elif xlim:
            self.data = self.data.iloc[x1:x2,:]
        elif ylim:
            self.data = self.data.iloc[:,y1:y2]
        #return data

    def on_clickX(self, event,data):
        print('Start to Integrate X')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        digis = 3
        ax2.set_title('x:%s y:%s' %((round(x_lim[0],digis),round(x_lim[1],digis)), (round(y_lim[0],digis),round(y_lim[1],digis))))
        slicedData = self.sliceData(data, xlim = x_lim, ylim = y_lim)
        reducedData = self.reduceX(slicedData)
        #print(reducedData.values, type(reducedData), len(reducedData))
        ax2.plot(reducedData, 'bo')
        button3pos = plt.axes([0.9, 0.0, 0.1, 0.075])
        bcut3 = Button(button3pos, 'Save', color=buttoncolor)
        buttonFitpos = plt.axes([0.9, 0.1, 0.1, 0.075])
        buttonFit = Button(buttonFitpos, 'Fit-Panel', color=buttoncolor)
        bcut3.on_clicked(lambda event:self.saveReduceData(event, reducedData))
        buttonFit.on_clicked(lambda event:fitPanel(event, ax2, reducedData))
        button3pos._button = bcut3 #without this the garbage collector destroyes the button
        buttonFitpos._button = buttonFit
        figures=[manager.canvas.figure
        for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        for i in figures:
            try:
                axies= i.get_axes()
                for j in axies:
                    print(j.get_title())
            except:
                pass
        plt.show()
        
    def saveReduceData(self, event, reddata):
        event, (filename,) = sg.Window('Save data'). Layout([[sg.Text('Filename')], [sg.Input(), sg.SaveAs()], [sg.OK(), sg.Cancel()]]).Read()
        reddata.to_pickle(filename)
        return event
    
    def on_clickInfo(self, event):
        event = sg.Window('Info',auto_size_text=True,font=("Helvetica", 18)). Layout([[sg.Multiline([grab_dic(self.info)],size=(80, 10))],[sg.Cancel()]]).Read()   
        return event

    
def pklImporter(path):
    return



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



def grab_dic(data):
    #tmp_list = []
    info_list = []
    for ele in data.values():
        if isinstance(ele,dict):
            for k, v in ele.items():
                info_list.append(k+' : '+v+'\n')
    return ' '.join(info_list)

    
    

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
    leftFit = None              # Initiate some elements, important for Canceling of fit-panel 
    rightFit = None
    inter_line = None
    inter_dot = None
    fermi_edge_plot = None
    sexteen_plot = None
    eigthy4_plot = None
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
                    leftFit.set_ydata(LinearFit(data.index,*leftFitPara))
                    rightFit.set_ydata(LinearFit(data.index,*rightFitPara))
                    if inter_line: inter_line.remove()
                    if inter_dot: inter_dot.remove()
                    if fermi_edge_plot: fermi_edge_plot.remove()
                    if sexteen_plot: sexteen_plot.remove()
                    if eigthy4_plot: eigthy4_plot.remove()
            
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


def fermiFct(x,E_f,b,s,T):
    k_b = const.value(u'Boltzmann constant in eV/K')
    return b + s*(1./(np.exp((x-E_f)/(k_b*T))))

def LinearFit(x,a,b):
    return a*x+b

def fitLinear(event, x_range, data, ax, color):
    mask = (data.index > x_range[0]) & (data.index <= x_range[1])    
    try:
        popt, pcov = curve_fit(LinearFit, data.index[mask], data.values[mask])
    except:
        print("Error:", sys.exc_info()[0])
        #raise
    return popt


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



def allMethodsOf(object):
    return [method_name for method_name in dir(object)
            if callable(getattr(object, method_name))]
# def main():
#     """
#     Proceding of ARPES data sets from OMICON SES Software.
#     """
#     __author__ = "Alexander Kononov"
#     __copyright__ = "Royalty-free"
#     __credits__ = ""
#     __license__ = ""
#     __version__ = "1.5"
#     __maintainer__ = "Alexander Kononov"
#     __email__ = "alexander.kononov@tu-dortmund.de"
#     __status__ = "Production"

#      # ------ Menu Definition ------ #      
#     menu_def = [['File', ['Open', 'Exit'  ]],      
#                 ['Help', 'About...'], ]      

#     # ------ GUI Defintion ------ #      
#     layout = [      
#         [sg.Menu(menu_def, )],      
#         [sg.Output(size=(60, 20))]      
#              ]      

#     window = sg.Window("UPhoS", default_element_size=(15, 1), auto_size_text=False, auto_size_buttons=False, default_button_element_size=(15, 1)).Layout(layout)
#     win = window.Finalize()
#     # ------ Loop & Process button menu choices ------ #      
#     while True:      
#         event, values = window.Read()      
#         if event == None or event == 'Exit':      
#             break      
#         # ------ Process menu choices ------ #      
#         if event == 'About...':      
#             sg.Popup(main.__doc__+'\n Author: '+__author__+'\n E-mail: '+__email__+'\n Copyright: '+\
#                      __copyright__+'\n License: '+__license__+'\n Version: '+\
#                      __version__+'\n Status: '+__status__)      
#         elif event == 'Open':      
#             filename = sg.PopupGetFile('file to open', no_window=True, default_path='../Auswertung/Data_for_python/')      
#             try:
#                 if filename: print(filename)
#                 plt.ion()
#                 experiment = Uphos(filename)
#                 experiment.plotData()
#                 plt.show()
#                 # data = read_pickle(filename)
#                 # plotData(data, title = filename.split('/')[-1:])#, title = filename.split('/')[:-2])
#             except AttributeError:
#                 print('Open file function was aborted.')
#                 raise
#                 #pass
# if __name__ == '__main__':
#     main()



