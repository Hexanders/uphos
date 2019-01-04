import _pickle as pickle
import os
import sys
import numpy as np


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
    print('Dumped: %s' %path.split('/')[-1]  )

def folderListAll(path, ending = None):
    data_list =[]
    for (dirpath, dirnames, filenames) in os.walk(path):
        for i in filenames:
            if ending:
                if i.endswith(ending):data_list.append(dirpath+i)
            else:
                data_list.append(dirpath+i)
    data_list.sort()
    return data_list

# def folderListAll(path, ending = None):
#     data_list =[]
#     for (dirpath, dirnames, filenames) in os.walk(path):
#         for i in filenames:
#             if ending:
#                 if i.endswith(ending):data_list.append(dirpath.replace(path,'')+'/'+i)
#             else:
#                 data_list.append(dirpath.replace(path,'')+'/'+i)
#     data_list.sort()
#     return data_list

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
        print('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()  # As suggested by Rom Ruben (see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    
def smooth(y, window_size, order, deriv=0, rate=1):
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
    except ValueError:
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


####### Colors

def color_dict(gradient):
  ''' Takes in a list of RGB sub-lists and returns dictionary of
    colors in RGB and hex form for use in a graphing function
    defined later on '''
  return {"hex":[RGB_to_hex(RGB) for RGB in gradient],
      "r":[RGB[0] for RGB in gradient],
      "g":[RGB[1] for RGB in gradient],
      "b":[RGB[2] for RGB in gradient]}

def hex_to_RGB(hex):
    ''' "#FFFFFF" -> [255,255,255] '''
    # Pass 16 to the integer function for change of base
    return [int(hex[i:i+2], 16) for i in range(1,6,2)]

def RGB_to_hex(RGB):
  ''' [255,255,255] -> "#FFFFFF" '''
  # Components need to be integers for hex to make sense
  RGB = [int(x) for x in RGB]
  return "#"+"".join(["0{0:x}".format(v) if v < 16 else
            "{0:x}".format(v) for v in RGB])


def RGB_tree_colors():
    return [
        RGB_to_hex([165,42,42]), #darkred
        RGB_to_hex([255,0,0]), #red
        RGB_to_hex([218,165,32]), #sand
        RGB_to_hex([255,225,0]), #yellow
        #RGB_to_hex([255,165,0]), #sand
        RGB_to_hex([50,205,50]), #green
        RGB_to_hex([135,206,250]),   #faint blue
        RGB_to_hex([0,0,255]),   #blue
        RGB_to_hex([0,0,100])   #navi
        
    ]


def linear_gradient(start_hex, finish_hex="#FFFFFF", n=10):
  ''' returns a gradient list of (n) colors between
    two hex colors. start_hex and finish_hex
    should be the full six-digit color string,
    inlcuding the number sign ("#FFFFFF") '''
  # Starting and ending colors in RGB form
  s = hex_to_RGB(start_hex)
  f = hex_to_RGB(finish_hex)
  # Initilize a list of the output colors with the starting color
  RGB_list = [s]
  # Calcuate a color at each evenly spaced value of t from 1 to n
  for t in range(1, n):
    # Interpolate RGB vector for color at the current value of t
    curr_vector = [
      int(s[j] + (float(t)/(n-1))*(f[j]-s[j]))
      for j in range(3)
    ]
    # Add it to our list of output colors
    RGB_list.append(curr_vector)

  return color_dict(RGB_list)


def polylinear_gradient(colors, n):
  ''' returns a list of colors forming linear gradients between
      all sequential pairs of colors. "n" specifies the total
      number of desired output colors '''
  # The number of colors per individual linear gradient
  #n_out = int(float(n) / (len(colors) - 1))
  n_out = int(float(n) / (len(colors)-1))
  #print n_out
  # returns dictionary defined by color_dict()
  gradient_dict = linear_gradient(colors[0], colors[1], n_out)

  #if len(colors) > 1:
  for col in range(1, len(colors)-1):
      next = linear_gradient(colors[col], colors[col+1], n_out)
      for k in ("hex", "r", "g", "b"):
        # Exclude first point to avoid duplicates
        gradient_dict[k] += next[k][1:]

  return gradient_dict
#######Colors ENDE
