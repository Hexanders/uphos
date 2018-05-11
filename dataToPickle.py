import numpy as np
import pandas as pd
import time
import workingFunctions as wf

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
        info_dic = {}
        for i in info:
            if 'Dimension 1 name' in i:
                data.index.name = i.split('=')[1]
            if 'Dimension 2 name' in i:
                data.columns.name = i.split('=')[1] 
            if i == '':
                continue
            if i.startswith('[') and i.endswith(']'):
                info_dic[i] = {} 
                curent_item = i
                continue
            sub_item = i.split('=' , 1)
            info_dic[curent_item][sub_item[0]] = sub_item[1]
        return(info_dic, data)

def convertAll(originPath, aimPath, ending = '.txt', exeptions = None, overwrite = False):
    """
    Converts all .txt files in the originPath directory into .pkl and puts them in seid aimPath Directory 
    """
    listOfDirs = wf.folderListAll(originPath, ending = ending)
    counter = 0
    total_length = len(listOfDirs)
    for files in listOfDirs:
        counter += 1
        wf.print_progress(counter, total_length)
        if exeptions:
            if any(x in files for x in exeptions):
                continue
        try:
            if overwrite == False and wf.os.path.isfile(aimPath+files[:-3]+'pkl'):
                continue
            obj = readIgorTxt(originPath + files)
            wf.dumpObj(obj,aimPath+files[:-3]+'pkl')
        except IndexError:
            print 'Some .txt files could be in REDME, trash or tmp directories.\n \
                   Put them in to exeptions argument and try again.\n \
                   Better in this way in order to finde errors!\n'+files

            
    
# sourcePath = '/run/media/hexander/main_drive/hexander/Documents/Uni/Promotion/UPS/Data'
# aimPath = '/run/media/hexander/main_drive/hexander/Documents/Uni/Promotion/UPS/Data_pkl'
# convertAll(sourcePath, aimPath, exeptions = ['README', 'tmp', 'Trash'])








# ##### Test the speed of .pkl ########
# dataPath = '/home/kononovdesk/Documents/Promotion/UPS/Data/180412/120180412133606.txt' # 400 MB
# dest = '/home/kononovdesk/Documents/Promotion/UPS/Auswertung/Data_for_python/120180412133606.pkl'
# # dataPath = '/home/kononovdesk/Documents/Promotion/UPS/Data/180412/120180412144122.txt' # 11MB
# # dest = '/home/kononovdesk/Documents/Promotion/UPS/Auswertung/Data_for_python/120180412144122.pkl'
# # dumpObj(readIgorTxt(dataPath),dest)
# start = time.time()
# readIgorTxt(dataPath)
# end = time.time()
# read = (end - start)
# start = time.time()
# loadObj(dest)
# end = time.time()
# pikle = (end - start)
# print read, pikle, read - pikle # result: 23.4932851791 0.231850147247 23.2614350319
