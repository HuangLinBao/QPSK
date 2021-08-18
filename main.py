from scipy import signal, special
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
from matplotlib.font_manager import FontProperties
font = FontProperties(family='Roboto')

#Modulation Proccess

T = 1               #Baseband signal width, which is frequency
bitNum= 100            #Define the number of bits transmitted
delta_T = T/200     #sampling interval
fs = 1/delta_T      #Sampling frequency
fc = 10/T           #Carrier frequency
SNR = 0             #Signal to noise ratio

t = np.arange(0, bitNum*T, delta_T)
N = len(t)

# Generate baseband signal
data = [1 if x > 0.5 else 0 for x in np.random.randn(1, bitNum)[0]]  #Call the random function to generate any 1*nb matrix from 0 to 1, which is greater than 0.5 as 1 and less than 0.5 as 0
data0 = []                             #Create a 1*nb/delta_T zero matrix
for q in range(bitNum):
    data0 += [data[q]]*int(1/delta_T)  #Convert the baseband signal into the corresponding waveform signal

# Modulation signal generation
data1 = []      #Create a 1*nb/delta_T zero matrix
datanrz = np.array(data)*2-1              #Convert the baseband signal into a polar code, mapping
for q in range(bitNum):
    data1 += [datanrz[q]]*int(1/delta_T)  #Change the polarity code into the corresponding waveform signal
    
idata = datanrz[0:(bitNum-1):2]       #Serial and parallel conversion, separate the odd and even bits, the interval is 2, i is the odd bit q is the even bit
qdata = datanrz[1:bitNum:2]         
ich = []                          #Create a 1*nb/delta_T/2 zero matrix to store parity data later
qch = []         
for i in range(int(bitNum/2)):
    ich += [idata[i]]*int(1/delta_T)    #Odd bit symbols are converted to corresponding waveform signals
    qch += [qdata[i]]*int(1/delta_T)    #Even bit symbols are converted to corresponding waveform signals

a = []     #Cosine function carrier
b = []     #Sine function carrier
for j in range(int(N/2)):
    a.append(np.math.sqrt(2/T)*np.math.cos(2*np.math.pi*fc*t[j]))    #Cosine function carrier
    b.append(np.math.sqrt(2/T)*np.math.sin(2*np.math.pi*fc*t[j]))    #Sine function carrier
idata1 = np.array(ich)*np.array(a)          #Odd-digit data is multiplied by the cosine function to get a modulated signal
qdata1 = np.array(qch)*np.array(b)          #Even-digit data is multiplied by the cosine function to get another modulated signal
s = idata1 + qdata1      #Combine the odd and even data, s is the QPSK modulation signal


#Drawing Proccess

plt.figure(figsize=(14,12))
plt.subplot(3,1,1)
plt.plot(idata1)
plt.title('Odd',fontproperties=font, fontsize=20)
plt.axis([0,500,-3,3])
plt.subplot(3,1,2)
plt.plot(qdata1)
plt.title('Even',fontproperties=font, fontsize=20)
plt.axis([0,500,-3,3])
plt.subplot(3,1,3)
plt.plot(s)
plt.title('Modulated signal',fontproperties=font, fontsize=20)
plt.axis([0,500,-3,3])
plt.show()