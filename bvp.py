import numpy as np
from scipy.signal import find_peaks, stft, lfilter, butter, welch

import torch

nFFT = 3600 # freq. resolution for STFTs
step = 1       # step in seconds




def getBPM(data, fs, winsize=5,p=False):
    """
    Compute the BVP signal spectrogram restricted to the
    band 42-240 BPM by using winsize (in sec) samples.
    """
#     if len(data.shape) == 2:
    data = data.reshape(1, -1)  # 2D array raw-wise
    # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
    if p:
        data=np.array(data.cpu().detach().numpy())
    else:
        data=np.array(data.cpu().numpy())
    
    F, T, Z = stft(data,
                   fs,
                   nperseg=fs * winsize,
                   noverlap=fs * (winsize - step),
                   boundary='even',
                   nfft=nFFT)
    Z = np.squeeze(Z, axis=0)

    # -- freq subband (0.75 Hz - 4.0 Hz)
    minHz = 0.75
    maxHz = 2.0
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    spect = np.abs(Z[band, :])  # spectrum magnitude
    freqs = 60 * F[band]  # spectrum freq in bpm
 
    bpm = freqs[np.argmax(spect, axis=0)]
    return torch.Tensor(bpm).float()
def gthrcula(data,lenth):
    data=np.array(data.cpu().numpy())
    gt_hr=[]
    data_=np.average(data[0:25])
    gt_hr.append(data_)
    i=1
    while(i<lenth):
        data_=np.average(data[25*i+1:25*i+25])
        gt_hr.append(data_)
        i=i+1
    
    return torch.Tensor(gt_hr).float()



def getBPM1(data, fs, winsize=5,p=False):
    """
    Compute the BVP signal spectrogram restricted to the
    band 42-240 BPM by using winsize (in sec) samples.
    """
#     if len(data.shape) == 2:
    data = data.reshape(1, -1)  # 2D array raw-wise
    # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
    if p:
        data=np.array(data.cpu().detach().numpy())
    else:
        data=np.array(data.cpu().numpy())
    
    F, T, Z = stft(data,
                   fs,
                   nperseg=fs * winsize,
                   noverlap=fs * (winsize - step),
#                    boundary='even',
                   nfft=nFFT)
    Z = np.squeeze(Z, axis=0)

    # -- freq subband (0.75 Hz - 4.0 Hz)
    minHz = 0.75
    maxHz = 2.5
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    spect = np.abs(Z[band, :])  # spectrum magnitude
    freqs = 60 * F[band]  # spectrum freq in bpm
 
    bpm = freqs[np.argmax(spect, axis=0)]
    bpm=np.average(bpm)
    return torch.Tensor(bpm.reshape(1,1)).float()