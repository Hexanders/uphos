#!/usr/bin/python3
import numpy as np
import traceback
import sys
from matplotlib.widgets import Button
import pandas as pd
from pandas import read_pickle
from scipy import constants as const
from scipy.interpolate import interp1d
import dataToPickle as dtp
#import workingFunctions as wf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
import tkinter as Tk 
#import matplotlib.backends.tkagg as tkagg
import matplotlib._pylab_helpers
import pprint
import lmfit
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
# root = Tk.Tk()
# screen_width = root.winfo_screenwidth()
# screen_height = root.winfo_screenheight()

class Uphos:
    def __init__(self, path = None):
        if path:
            try:
                self.path = path
                self.name = self.path.split('/')[-1][:-4]
                self.workingPath = path.split('/')[-2]+'/'+path.split('/')[-1]
                if path.endswith('.txt'):
                    print('Processing: %s' % self.workingPath)
                    self.info, self.data = dtp.readIgorTxt(path)
                else:
                    print('Processing: %s' %  self.workingPath)
                    self.info, self.data = read_pickle(path)
            except Exception as err:
                print ('Can not read file: %s' % path )
                traceback.print_tb(err.__traceback__)
                pass
        else:
            self.path = ''
            self.info = ''
            self.data = None
            pass
        
    # def __init__(self, path = None):
    #     if path:
    #         try:
    #             self.path = path
    #             self.name = self.path.split('/')[-1][:-4]
    #             self.workingPath = path.split('/')[-2]+'/'+path.split('/')[-1]
    #             if path.endswith('.txt'):
    #                 print('Processing: %s' % self.workingPath)
    #                 self.info, self.data = dtp.readIgorTxt(path)
    #             else:
    #                 print('Processing: %s' %  self.workingPath)
    #                 self.info, self.data = read_pickle(path)
    #         except Exception as err:
    #             print ('Can not read file: %s' % path )
    #             traceback.print_tb(err.__traceback__)
    #             pass
    #     else:
    #         self.path = ''
    #         self.info = ''
    #         self.data = None
    #         pass
        
    def get_data(self):
        return self.data

    def get_INFO(self):
        return self.info
    
    def exportCSV(self, path, data = None):
        # if path.endswith('/'):
        #     path = path + self.path.split('/')[-1]
        # else:
        #     path = path + '/' + self.path.split('/')[-1]
        f = open(path, 'a')
        info = grab_dic(self.info)
        for i in info:
            f.write('# '+i)
        if data:
            data.to_csv(f)
        else:
            self.data.to_csv(f)
        f.close()
    # def plotAll(self):
    #     for i in self.data[1]:
    #         plotData
    
    def plotData(self, xy_label = (True,True), axExtern = None,  title = None, interactive = True):
        '''
        Plots data of an object into an Image.
        Data format ist (name,pandas.DataFrame)
        Parameters:
            xy_label: Boolean : Display or not x or/and y Labels from pandas Dataframe Column/Index names
            axExtern: Matplotlib axes:  for External ploting
            Title: String: Title of the subplot
            interactive: Bollean: Display or not the buttons for data Proccecing
        Returns:
            imshow Object
        '''
        # cid = fig.canvas.mpl_connect('resize_event', onresize)
        for current_data in self.data:
            if not axExtern:
                fig = plt.figure()
                global ax
                ax = fig.add_subplot(111)
                if title:
                    fig.canvas.set_window_title(title)
                else:
                    fig.canvas.set_window_title(current_data[0])
            y,x = current_data[1].index.values, current_data[1].columns.values
            extent = np.min(x), np.max(x), np.min(y), np.max(y)
            if axExtern:
                fig = axExtern
                im = axExtern.imshow(curent_data[1],extent=extent, origin = 'lower', cmap='hot',  aspect = 'auto')
                plt.colorbar(im, ax=axExtern)
                # if xy_label[0] == True:
                #     axExtern.set_xlabel(current_data.index.name)
                # if xy_label[1] == True:
                #     axExtern.set_ylabel(current_data.columns.name)   
            else:
                im = plt.imshow(current_data[1],extent=extent, origin = 'lower', cmap='hot',  aspect = 'auto')
                plt.colorbar()
                # if xy_label[0] == True:
                #     plt.xlabel(current_data.index.name)
                # if xy_label[1] == True:
                #     plt.ylabel(current_data.columns.name)
                plt.tight_layout()
            if interactive:
                button1pos= plt.axes([0.79, 0.0, 0.1, 0.075]) #posx, posy, width, height in %
                button2pos = plt.axes([0.9, 0.0, 0.1, 0.075])
                button3pos = plt.axes([0.9, 0.1, 0.1, 0.075])
                bcut1 = Button(button1pos, 'Int. X', color=buttoncolor)
                bcut2 = Button(button2pos, 'Int. Y', color=buttoncolor)
                bcut3 = Button(button3pos, 'Info', color=buttoncolor)
                bcut1.on_clicked(lambda event: self.on_click(event, current_data[1], ax = ax))
                bcut2.on_clicked(lambda event: self.on_click(event, current_data[1], ax = ax,axes ='y'))
                bcut3.on_clicked(lambda event: self.on_clickInfo(event))
                button1pos._button = bcut1 #otherwise the butten will be killed by carbagcollector
                button2pos._button = bcut2
                button3pos._button = bcut3
        return im
    
    def sumAllDataFrames(self, data =None):
        '''
        Sum all panda.dataFrames element wise
        '''
        if data: data = data
        else: data = self.data
        cnt = 0
        for i in data:
            if cnt == 0:
                data_frame_sum = i[1]
            else:
                data_frame_sum = data_frame_sum.add(i[1], axis='index', fill_value=0.0)
                #else: data_frame_sum = data_frame_sum.merge(i[1])
            cnt +=1
        return data_frame_sum
    
    def plotOverview(self, data = None):
        '''
        Plots DataFrame and Integratet Plots alon x an y axes
        '''
        if data is not None:
            tmp_data = data
        else:
            tmp_data = self.data
        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 4, hspace=0.01, wspace=0.01)       
        main_ax = fig.add_subplot(grid[1:4, :-1])       
        x,y = tmp_data.index.values, tmp_data.columns.values
        extent = np.min(x), np.max(x), np.min(y), np.max(y)
        im = plt.imshow(tmp_data, extent = extent, origin = 'lower', cmap='hot',  aspect = 'auto')
        main_xlim = plt.xlim()
        main_ylim = plt.ylim()
        plt.xlabel(tmp_data.index.name)
        plt.ylabel(tmp_data.columns.name)
        cbar_ax = fig.add_axes([0.01, 0.95, 0.9, 0.05])
        plt.colorbar(im, cax = cbar_ax, orientation='horizontal')     
        y_int = fig.add_subplot(grid[1:4, -1])
        y_int.get_yaxis().set_ticks([])
        #plt.ticklabel_format(axis='x', style='sci',scilimits=(0,0), useMathText = True)
        #plt.ylim(main_ylim)
        yred = self.reduceY(tmp_data)
        plt.xticks(rotation='vertical')
        y_int.plot(yred.values, yred.index.values,'o')
        #x_int = fig.add_subplot(grid[-1, 1:])
        x_int = fig.add_subplot(grid[0, :-1])
        x_int.set_title(self.workingPath)
        x_int.get_xaxis().set_ticks([])
        #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText = True)
        #plt.xlim(main_xlim)
        xred = self.reduceX(tmp_data)
        x_int.plot(xred,'o')
        fig.tight_layout()
        return fig
    
    def reduceX(self, data = None):
        '''Integriere Daten entlang einzelnen Energiewerten '''
        if data is not None:
            self.XredData = data.apply(np.sum, axis = 0)
            new_name ='summed over ('+str(data.columns.min()) +' : '+str(data.columns.max()) +') ' + data.columns.name 
        else:
            self.XredData = self.data.apply(np.sum, axis = 0)
            new_name ='summed over ('+str(self.data.columns.min()) +' : '+str(self.data.columns.max()) +') ' + self.data.columns.name 
        self.XredData.name = self.name +'_'+new_name
        return self.XredData 

    def reduceY(self, data = None):
        '''Integritmp_dataere Entlang Y.'''
        #return np.add.reduce(data)
        if data is not None:
            self.YredData = data.apply(np.sum, axis = 1)
        else:
            self.YredData = self.data.apply(np.sum, axis = 1) 
        return self.YredData

    def fermiFct(self, x, E_f, b, s, T):
        k_b = const.value(u'Boltzmann constant in eV/K')
        return b + s*(1./(np.exp((x-E_f)/(k_b*T))+1))    

    def fitFermi(self, a=16.9,b = 1.,c = 1. ,d =70.):
        x = self.XredData.index
        y = self.XredData.values
        #mod = lmfit.models.ExponentialModel()
        mod = lmfit.Model(self.fermiFct)
        #pars = mod.guess(y, x=x) ###(x,E_f,b,s,T):
        out = mod.fit(y,E_f = a, b = b, s = c , T = d, x=x)
        print(out.fit_report())
        #plt.plot(x, out.best_fit, 'k-')
        #values = {'E_f':popt[0],'B':popt[1],'S':popt[2],'T':popt[3]}
        print(out.best_fit)
        return (x,out.best_fit)

    
    def fitFermi_old(self,  p0 =[0.0001,0.00001,0.0001,0.0001], x_lim = None , y_lim = None):
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
    
    def sliceData(self, xlim = None, ylim = None):
        # print(len(self.data))
        # print(self.data[0])
        # if xlim and ylim:
        #     self.data = self.data.iloc[x1:x2,y1:y2] 
        # elif xlim:
        #     self.data = self.data.iloc[x1:x2,:]
        # elif ylim:
        #     self.data = self.data.iloc[:,y1:y2]
        return self.data

    def on_click(self, event, data, ax = None, axes = 'x'):
        print('Start to Integrate X')
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        if ax:
            x_lim = ax.get_xlim()
            y_lim = ax.get_ylim()
            slicedData = self.sliceData(xlim = x_lim, ylim = y_lim)
            print(slicedData)
        else:
            slicedData = data
        digis = 3
        ax2.set_title('x:%s y:%s' %((round(x_lim[0],digis),round(x_lim[1],digis)), (round(y_lim[0],digis),round(y_lim[1],digis))))
        
        if axes == 'x':
            #reducedData = self.reduceX(slicedData)
            reducedData = self.reduceX(data)
        else:
            #reducedData = self.reduceY(slicedData)
            reducedData = self.reduceY(data)
            #print(reducedData.values, type(reducedData), len(reducedData))
        ax2.plot(reducedData, 'bo')
        #ax2.plot(data, 'bo')
        button3pos = plt.axes([0.9, 0.0, 0.1, 0.075])
        bcut3 = Button(button3pos, 'Save', color=buttoncolor)
        buttonFitpos = plt.axes([0.9, 0.1, 0.1, 0.075])
        buttonFit = Button(buttonFitpos, 'Fit-Panel', color=buttoncolor)
        buttonEXPpos = plt.axes([0.9, 0.2, 0.1, 0.075])
        buttonEXP = Button(buttonEXPpos, 'Export as csv', color=buttoncolor)
        buttonEXP.on_clicked(lambda event:self.exportCSVtrigger(event, reducedData))
        bcut3.on_clicked(lambda event:self.saveReduceData(event, reducedData))
        buttonFit.on_clicked(lambda event:fitPanel(event, ax2, reducedData))
        button3pos._button = bcut3 #without this the garbage collector destroyes the button
        buttonFitpos._button = buttonFit
        buttonEXPpos._button = buttonEXP
        figures=[manager.canvas.figure
        for manager in matplotlib._pylab_helpers.Gcf.get_all_fig_managers()]
        for i in figures:
            try:
                axies= i.get_axes()
                for j in axies:
                    print(j.get_title())
            except:
                pass
        plt.draw()
        
    def exportCSVtrigger(self, event, reddata):
        event, (filename,) = sg.Window('Export to csv'). Layout([[sg.Text('Filename')], [sg.Input(), sg.SaveAs()], [sg.OK(), sg.Cancel()]]).Read()
        self.exportCSV(filename, reddata)
        return event
    
    def saveReduceData(self, event, reddata):
        event, (filename,) = sg.Window('Save data'). Layout([[sg.Text('Filename')], [sg.Input(), sg.SaveAs()], [sg.OK(), sg.Cancel()]]).Read()
        if filename.endswith('.pkl'):reddata.to_pickle(filename)
        elif filename.endswith('.csv'):
            f = open(filename, 'a')
            f.write(pprint.pformat(self.info))
            reddata.to_csv(f, header = True , sep = ' ')
            f.close()
        else: print('Only .pkl or .csv are implemented.')
        return event
    
    def on_clickInfo(self, event):
        event = sg.Window('Info',auto_size_text=True,font=("Helvetica", 18)). Layout([[sg.Multiline(pprint.pformat(self.info),size=(80, 10))],[sg.Cancel()]]).Read()   
        return event


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

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def grab_dic(data):
    info_list = []
    for ele in data.values():
        if isinstance(ele,dict):
            for k, v in ele.items():
                tmp_list = []
                tmp_list = [k+' : '+v.replace(';', '\n#\t\t')+'\n']
                info_list.append(' '.join(tmp_list))
    return info_list

    
    
def exportCSV(path, data, info = None):
    # if path.endswith('/'):
    #     path = path + self.path.split('/')[-1]
    # else:
    #     path = path + '/' + self.path.split('/')[-1]
    f = open(path, 'a')
    if info:
        info = grab_dic(self.info)
        for i in info:
            f.write('# '+i)
    data.to_csv(f)
    f.close()

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
        [sg.ReadButton('Finde Fermi Edge'), sg.Text(r'Fermi edge [eV]'), sg.InputText(size =(10,10), key='fermi_edge'), sg.Text('16%-84% width [eV]'), sg.InputText(size =(10,10),  key = 'resolution')] ,
        [sg.ReadButton('Fit Fermi Function'), sg.Text(r'E_f:'),sg.InputText(size =(10,20), key = 'E_f', default_text= '16.9'), sg.Text(r'b:'), sg.InputText(size =(10,20), default_text = '20000', key = 'b'),sg.Text(r's:'),sg.InputText(size =(10,20),default_text = '100', key = 's'),sg.Text(r'T:'),sg.InputText(size =(10,20),default_text = '300', key = 'T'),],
        [sg.Cancel()],
    ]
   
    window = sg.Window('Fit Parameter for figure ' + str(plt.gcf().number),  grab_anywhere=False, auto_size_text=True)
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
        if event == 'Fit Fermi Function':
            try:
                xFit, yFit, out = experiment.fitFermi(float(values['E_f']), float(values['b']), float(values['s']), float(values['T']))
                fitParam = out.params.valuesdict()
                print(fitParam)
                window.FindElement('E_f').Update(str(fitParam['E_f']))
                window.FindElement('b').Update(str(fitParam['b']))
                window.FindElement('s').Update(str(fitParam['s']))
                window.FindElement('T').Update(str(fitParam['T']))
                ax.plot(xFit,yFit)
            except:
                print("Error:", sys.exc_info()[0]) 
                raise
            plt.draw()
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
        popt, pcov = curve_fit(LinearFit, data.index.values[mask], data.values[mask])
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
def main():
    """
    Proceding of ARPES data sets from OMICON SES Software.
    """
    __author__ = "Alexander Kononov"
    __copyright__ = "Royalty-free"
    __credits__ = ""
    __license__ = ""
    __version__ = "2.0"
    __maintainer__ = "Alexander Kononov"
    __email__ = "alexander.kononov@tu-dortmund.de"
    __status__ = "Production"

     # ------ Menu Definition ------ #      
    menu_def = [['File', ['Open', 'Exit'  ]],      
                ['Help', 'About...'], ]      

    # ------ GUI Defintion ------ #      
    layout = [      
        [sg.Menu(menu_def, )],      
        [sg.Output(size=(60, 20))]      
             ]      
    window = sg.Window("UPhoS", default_element_size=(15, 1), auto_size_text=False, auto_size_buttons=False, location=(250, 250), default_button_element_size=(15, 1)).Layout(layout)
    win = window.Finalize()
    # ------ Loop & Process button menu choices ------ #      
    while True:      
        event, values = window.Read()      
        if event == None or event == 'Exit':      
            break      
        # ------ Process menu choices ------ #      
        if event == 'About...':      
            sg.Popup(main.__doc__+'\n Author: '+__author__+'\n E-mail: '+__email__+'\n Copyright: '+\
                     __copyright__+'\n License: '+__license__+'\n Version: '+\
                     __version__+'\n Status: '+__status__)      
        elif event == 'Open':      
            filename = sg.PopupGetFile('file to open', no_window=True, keep_on_top =True, default_extension='txt', default_path='../../Data/')      
            try:
                if filename: print(filename)
                plt.ion()
                global experiment 
                experiment = Uphos(filename)
                name = filename.split('/')
                #experiment.plotData(title = name[-2]+'/'+name[-1][:-4])
                #for i in experiment.data:
                experiment.plotData()
                    
                plt.show()
                # data = read_pickle(filename)
                # plotData(data, title = filename.split('/')[-1:])#, title = filename.split('/')[:-2])
            except AttributeError:
                print('Open file function was aborted.')
                raise
                #pass
if __name__ == '__main__':
    main()



