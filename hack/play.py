
# coding: utf-8

# # Utils

# In[543]:

from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import scipy
import IPython
from scipy.interpolate import interp1d
from scipy.signal import spectrogram
import pandas as pd
from PIL import Image
import colorsys
from copy import copy

def setup_graph(title='', x_label='', y_label='', fig_size=None):
    fig = plt.figure()
    if fig_size != None:
        fig.set_size_inches(fig_size[0], fig_size[1])
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

def scalar_linear_transform(s, s_range=[0,1], map_range=[380,750]):
    transformer = interp1d(s_range,map_range)
    return transformer(s)
    
def array_linear_transform(arr, arr_range=None, map_range=[380,750]):
    if arr_range is None:
        arr_range = [np.min(arr), np.max(arr)]
    transformer = interp1d(arr_range,map_range)
    return transformer(arr)

'''
    == A few notes about color ==

    Color   Wavelength(nm) Frequency(THz)
    Red     620-750        484-400
    Orange  590-620        508-484
    Yellow  570-590        526-508
    Green   495-570        606-526
    Blue    450-495        668-606
    Violet  380-450        789-668

    f is frequency (cycles per second)
    l (lambda) is wavelength (meters per cycle)
    e is energy (Joules)
    h (Plank's constant) = 6.6260695729 x 10^-34 Joule*seconds
                         = 6.6260695729 x 10^-34 m^2*kg/seconds
    c = 299792458 meters per second
    f = c/l
    l = c/f
    e = h*f
    e = c*h/l

    List of peak frequency responses for each type of 
    photoreceptor cell in the human eye:
        S cone: 437 nm
        M cone: 533 nm
        L cone: 564 nm
        rod:    550 nm in bright daylight, 498 nm when dark adapted. 
                Rods adapt to low light conditions by becoming more sensitive.
                Peak frequency response shifts to 498 nm.

'''

def wavelength_to_rgb(wavelength, gamma=0.8):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
#     R *= 255
#     G *= 255
#     B *= 255
#     return (int(R), int(G), int(B))
    return (R, G, B)

def array_to_rgb(arr):    
    return list(map(wavelength_to_rgb, arr))

def discretize_transform(arr, out_len, transform):
    bin_size = len(arr)//out_len
    temp = []
    s = 0
    for i in range(out_len):
        temp.append(transform(arr[s:s+bin_size]))
        s = s+bin_size
    return temp



def jpg_image_to_array(image_path):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with Image.open(image_path) as image:         
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))                                   
  return im_arr

def im_mask_by_wavelength(im, im_wavelength, wave_range):
    im_new = copy(im)
    new_alphas = im_new[...,-1] * 1
    wave_min, wave_max = wave_range
    new_alphas[np.logical_or((im_wavelength < wave_min),(im_wavelength > wave_max))] = 0.3
    im_new[...,-1] = new_alphas
    return im_new


# # Music transform

# In[139]:


(sample_rate, input_signal) = scipy.io.wavfile.read("./btv.wav")
time_array = np.arange(0, len(input_signal)/sample_rate, 1/sample_rate)
# input_signal = input_signal


# In[140]:


# setup_graph(title='amplitutde visual', x_label='time (in seconds)', y_label='amplitude', fig_size=(14,7))
# _ = plt.scatter(time_array, input_signal, c=array_to_rgb(array_linear_transform(input_signal)))


# In[412]:


x= np.linspace(380,750,750-380)
y=np.linspace(380,750,750-380)
cs = np.linspace(380,750,750-380)
rgbs = array_to_rgb(array_linear_transform(cs))
plt.scatter(x,y,c=rgbs)


# In[413]:


rgbs_approx = array_to_rgb(array_linear_transform(list(map(lambda x: colorsys.rgb_to_hsv(*x)[0], rgbs))))
plt.scatter(x,y,c=rgbs)


# In[520]:


f, t, Sxx = spectrogram(input_signal,sample_rate)
dis_amplitude = discretize_transform(input_signal, len(t), lambda x: np.max(x))
cs = array_to_rgb(array_linear_transform(dis_amplitude))
setup_graph(title='frequency visual', x_label='time (in seconds)', y_label='frequency', fig_size=(14,7))
freqs = np.max(Sxx, axis=0)
_ = plt.scatter(t, freqs, c=cs)


# # Image Transform

# In[506]:


im = jpg_image_to_array('./brain.jpg')
im_wavelength = np.apply_along_axis(lambda x: scalar_linear_transform(colorsys.rgb_to_hsv(*tuple(x))[0]), 2, im)
im = np.dstack((im/255, 1 * np.ones(im.shape[:2])))


# In[507]:


plt.imshow(imshow)


# In[527]:


def create_wave(wave_range=(300, 700)):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_facecolor('xkcd:black')
    im_new = im_mask_by_wavelength(im, im_wavelength, wave_range)    
    ax.imshow(im_new)
    plt.show()


# In[529]:


create_wave((300, 500))


# In[538]:


freqs_wave_range = discretize_transform(array_linear_transform(freqs), 100, lambda x: np.max(x))


# In[539]:


plt.plot(freqs_wave_range)


# In[544]:


for f in freqs_wave_range:
    create_wave((f-50, f))

