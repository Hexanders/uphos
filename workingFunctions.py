import cPickle as pickle
import os
import sys


def loadObj(path):
    with open(path, 'rb') as input:
        obj = pickle.load(input)
    return obj

def dumpObj(obj,path):
    #if not os.path.exists(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError:
        pass
    with open(path, 'wb') as output:
        pickle.dump(obj, output, -1)

def folderListAll(path, ending = None):
    data_list =[]
    for (dirpath, dirnames, filenames) in os.walk(path):
        for i in filenames:
            if ending:
                if i.endswith(ending):data_list.append(dirpath.replace(path,'')+'/'+i)
            else:
                data_list.append(dirpath.replace(path,'')+'/'+i)
    data_list.sort()
    return data_list

def fileList(path, ending = None):
    data_list =[]
    for (dirpath, dirnames, filenames) in os.walk(path):
        if ending is not None:
            for i in filenames:
                if i.endswith(ending):
                    data_list.append(i)
        else:
            for i in filenames:
                data_list.append(i)
    data_list.sort()
    return data_list

def print_progress(count, total, status=''):
    sys.stdout.flush()
    bar_len = 15
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    if count == total:
        print  '[%s] %s%s ...%s\r' % (bar, percents, '%', status)
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    
