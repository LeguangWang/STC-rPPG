import copy
import datetime
import json
import time
import os
import sys
from scipy import signal
import cv2
import plotly.graph_objects as go
import plotly.io as pio
# sys.path.append("mnt/Pytorch_rppgs/")
# sys.path.append("mnt/Pytorch_rppgs/nets/models/")
# sys.path.append("mnt/Pytorch_rppgs/utils/")
#
# sys.path.append("mnt/")

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
# from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm
import pandas as pd
from scipy.stats import pearsonr
sys.path.append(r"mnt/Pytorch_rppgs/dataset/")
from dataset.dataset_loader import dataset_loader
from log import log_info_time
from loss import loss_fn
from models import is_model_support, get_model, summary
from optim import optimizers
from torch.optim import lr_scheduler
from utils.dataset_preprocess import preprocessing
from utils.funcs import normalize,detrend

from nets.models.Meta import Meta
from nets.models.PhysNet import PhysNet
from nets.models.Deepnew import Deepnew
from PIL import Image

import torchvision.transforms as transforms

ToTensor_transform = transforms.Compose([transforms.ToTensor()])
toPIL = transforms.ToPILImage()

with open('rnet1.json') as f:
    jsonObject = json.load(f)
    __PREPROCESSING__ = jsonObject.get("__PREPROCESSING__")
    __TIME__ = jsonObject.get("__TIME__")
    __MODEL_SUMMARY__ = jsonObject.get("__MODEL_SUMMARY__")
    options = jsonObject.get("options")
    params = jsonObject.get("params")
    hyper_params = jsonObject.get("hyper_params")
    model_params = jsonObject.get("model_params")
    meta_params = jsonObject.get("meta_params")

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
print('Count of using GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())

"""
Check Model Support
"""
is_model_support(model_params["name"], model_params["name_comment"])
'''
Setting Learning Model
'''
if __TIME__:
    start_time = time.time()
model = get_model(model_params["name"])
if meta_params["pre_trained"] == 1:
    new_state_dict = OrderedDict()
    #     for key, value in torch.load("mnt/Pytorch_rppgs-main/PhysNetUBFC_50_100.0.pth").items():
    #         name = key[7:]
    #         new_state_dict[name] = value
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    print('Using pre-trained on all ALL AFRL!')

    model.load_state_dict(torch.load("RnetPURE_5_93.5.pth"), strict=False)
else:
    print('Not using any pretrained models')

model = model.cuda()
model = nn.DataParallel(model).to(device)

'''
if torch.cuda.is_available():
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3, 4, 5, 6, 7, 8, 9'
    # TODO: implement parallel training
    # if options["parallel_criterion"] :
    #     print(options["parallel_criterion_comment"])
    #     model = DataParallelModel(model, device_ids=[0, 1, 2])
    # else:
    #     model = DataParallel(model, output_device=0)
    device = torch.device('cuda:9')
    model.to(device=device)
else:
    model = model.to('cpu')
'''

if __MODEL_SUMMARY__:
    summary(model, model_params["name"])

if __TIME__:
    log_info_time("model initialize time \t: ", datetime.timedelta(seconds=time.time() - start_time))
'''
Generate preprocessed data hpy file 
'''
# if __PREPROCESSING__:
#     if __TIME__:
#         start_time = time.time()
#
#     preprocessing(save_root_path= params["save_root_path"],
#                   model_name=model_params["name"],
#                   data_root_path="UBFC/",
#                   dataset_name="UBFC",
#                   train_ratio=params["train_ratio"])
#     if __TIME__:
#         log_info_time("preprocessing time \t:", datetime.timedelta(seconds=time.time() - start_time))

'''
Load dataset before using Torch DataLoader
'''
if __TIME__:
    start_time = time.time()

train_dataset = dataset_loader(save_root_path=params["save_root_path"],
                         model_name=model_params["name"],
                         dataset_name=params["dataset_name"],
                         option="train",

                         num_shots=meta_params["num_shots"],
                         num_test_shots=meta_params["num_test_shots"],
                         unsupervised=meta_params["unsupervised"]
                         )
# train_lenen=len(train_dataset)
# test_ = [i for i in range(train_lenen) if i<=train_lenen*0.2]
#
#
# train_= [i for i in range(train_lenen) if i>train_lenen*0.2]
# test_dataset = torch.utils.data.Subset(train_dataset, test_)pip
# train_dataset = torch.utils.data.Subset(train_dataset, train_)
if __TIME__:
    log_info_time("load train hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

if __TIME__:
    start_time = time.time()
test_dataset = dataset_loader(save_root_path=params["save_root_path"],
                              model_name=model_params["name"],
                              dataset_name="PURE",
                              option="test",

                              num_shots=meta_params["num_shots"],
                              num_test_shots=meta_params["num_test_shots"],
                              unsupervised=meta_params["unsupervised"]
                              )
if __TIME__:
    log_info_time("load test hpy time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
    Call dataloader for iterate dataset
'''
if __TIME__:
    start_time = time.time()
if model_params["name"] in ['MetaPhys', "MetaPhysNet"]:
    train_loader = train_dataset, batch_size=params["train_batch_size"],

#     validation_loader = BatchMetaDataLoader(validation_dataset, batch_size=params["train_batch_size"],
#                                             shuffle=params["train_shuffle"])
else:
    train_loader = DataLoader(train_dataset, batch_size=params["train_batch_size"],
                              shuffle=params["train_shuffle"],
                         drop_last=True)

    test_loader = DataLoader(test_dataset, batch_size=params["test_batch_size"],
                             shuffle=params["test_shuffle"],
                         drop_last=True)
if __TIME__:
    log_info_time("generate dataloader time \t: ", datetime.timedelta(seconds=time.time() - start_time))

'''
Setting Loss Function
'''
if __TIME__:
    start_time = time.time()
criterion = loss_fn(hyper_params["loss_fn"])
inner_criterion = loss_fn(meta_params["inner_loss"])
outer_criterion = loss_fn(meta_params["outer_loss"])
# if torch.cuda.is_available():
# TODO: implement parallel training
# if options["parallel_criterion"] :
#     print(options["parallel_criterion_comment"])
#     criterion = DataParallelCriterion(criterion,device_ids=[0, 1, 2])

if __TIME__:
    log_info_time("setting loss func time \t: ", datetime.timedelta(seconds=time.time() - start_time))
import numpy as np
from scipy.signal import find_peaks, stft, lfilter, butter, welch

import torch

nFFT = 2048  # freq. resolution for STFTs
step = 1  # step in seconds


def getBPM(data, fs=30, winsize=10,f11=0):
    """
    Compute the BVP signal spectrogram restricted to the
    band 42-240 BPM by using winsize (in sec) samples.
    """

    data = data.reshape(1, -1)  # 2D array raw-wise
    # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
    #     data=np.array(data.cpu().numpy())

    F, T, Z = stft(data,
                   fs,
                   nperseg=fs * winsize,
                   noverlap=fs * (winsize - step),

                   nfft=nFFT)
    Z = np.squeeze(Z, axis=0)

    # -- freq subband (0.75 Hz - 4.0 Hz)
    minHz = 0.75
    maxHz = 2.5
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    spect = np.abs(Z[band, :])  # spectrum magnitude
    freqs = 60 * F[band]  # spectrum freq in bpm

    bpm = freqs[np.argmax(spect, axis=0)]
    #     bpm=np.average(bpm)

    t = T
    f = freqs
    S = spect
    if f11==0 or f11==1:
        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=S, x=t, y=f, colorscale="viridis"))
        # pio.write_image(fig, '222.png')
        # fig.add_trace(go.Scatter(
        #     x=t, y=bpm, name='心率信号', line=dict(color='red', width=2)))
        # # pio.write_image(fig, '33.png')
        fig.update_layout(autosize=False, height=420, showlegend=False,
                          # title='BVP信号的频谱分布',
                          xaxis_title='时间（s）',
                          yaxis_title='BPM (60*Hz)',
                          legend=dict(
                              x=0,
                              y=1,
                              traceorder="normal",
                              font=dict(
                                  family="sans-serif",
                                  size=15,
                                  color="black"),
                              bgcolor="LightSteelBlue",
                              bordercolor="Black",
                              borderwidth=2)
                          )
        if f11==0:
            pio.write_image(fig, 'rnet_ubfc/train_pinipu.png')
        else:
            pio.write_image(fig, 'rnet_ubfc/Test_pinipu.png')

    return bpm


def getBPM1(data, fs=30, winsize=4.26):
    """
    Compute the BVP signal spectrogram restricted to the
    band 42-240 BPM by using winsize (in sec) samples.
    """
    #     if len(data.shape) == 2:
    data = data.reshape(1, -1)  # 2D array raw-wise
    # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]

    data = np.array(data.cpu().detach().numpy())
    F, T, Z = stft(data,
                   fs,
                   nperseg=fs * winsize,
                   noverlap=fs * (winsize - step),
                   boundary='even',
                   nfft=nFFT)
    Z = np.squeeze(Z, axis=0)

    # -- freq subband (0.75 Hz - 4.0 Hz)
    minHz = 0.75
    maxHz = 4.0
    band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
    spect = np.abs(Z[band, :])  # spectrum magnitude
    freqs = 60 * F[band]  # spectrum freq in bpm

    bpm = freqs[np.argmax(spect, axis=0)]
    return torch.Tensor(bpm).float()








from matplotlib import pyplot as plt


def plot_graph(start_point, length, target, inference, epoch, ppg=True):
    if ppg:
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.plot(range(len(target[start_point:start_point + length])), target[start_point:start_point + length],
                 label='ground-truth')
        plt.plot(range(len(inference[start_point:start_point + length])), inference[start_point:start_point + length],
                 label='prediction')

        plt.legend(fontsize='x-large', loc='upper right')
        # plt.title('epoch:{}'.format(epoch))
        plt.xlabel('Frames', fontsize='x-large')
        plt.ylabel('rppg', fontsize='x-large')
        plt.savefig("wlg/result_ppg{}.jpg".format(epoch))
        plt.clf()
    else:
        plt.rcParams["figure.figsize"] = (8, 8)
        plt.plot(range(len(target[start_point:start_point + length])), target[start_point:start_point + length],
                 label='ground-truth')
        plt.plot(range(len(inference[start_point:start_point + length])), inference[start_point:start_point + length],
                 label='prediction')
        plt.legend(fontsize='x-large', loc='upper right')
        # plt.title('epoch:{}'.format(epoch))
        plt.xlabel('Frames', fontsize='x-large')
        plt.ylabel('bpm', fontsize='x-large')
        plt.savefig("wlg/result_bpm{}.jpg".format(epoch))
        plt.clf()
def calculate_psd(time_signal, sampling_rate):

    frequencies, psd = signal.welch(time_signal,
                                    fs=sampling_rate,
                                    nperseg=1024,  # Length of each segment
                                    noverlap=512,  # Number of points to overlap
                                    scaling='density')
    return frequencies, psd


# Example usage
def calculate_psd(time_signal, sampling_rate, freq_range=(0, 4)):
    """
    Calculate Power Spectral Density of a time domain signal within a specific frequency range.

    Parameters:
    time_signal : array_like
        Input time domain signal
    sampling_rate : float
        Sampling rate of the signal in Hz
    freq_range : tuple
        Frequency range to retain (low_freq, high_freq)

    Returns:
    frequencies : ndarray
        Array of sample frequencies within the specified range
    psd : ndarray
        Power spectral density of the signal within the specified range
    """
    # Calculate PSD using Welch's method
    frequencies, psd = signal.welch(time_signal,
                                    fs=sampling_rate,
                                    nperseg=1024,  # Length of each segment
                                    noverlap=512,  # Number of points to overlap
                                    scaling='density')

    # Retain only frequencies within the specified range
    freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    frequencies = frequencies[freq_mask]
    psd = psd[freq_mask]

    return frequencies, psd


def plot_time_and_psd(signal1, signal2, sampling_rate, epoch, freq_range=(0, 4)):
    """
    Plot time domain signals and their corresponding PSDs

    Parameters:
    signal1, signal2 : array_like
        Input time domain signals
    sampling_rate : float
        Sampling rate of the signals in Hz
    epoch : int
        Epoch number for file naming
    freq_range : tuple
        Frequency range to retain (low_freq, high_freq)
    """
    # Calculate PSDs
    freq1, psd1 = calculate_psd(signal1, sampling_rate, freq_range)
    freq2, psd2 = calculate_psd(signal2, sampling_rate, freq_range)

    # Plot PSDs
    plt.rcParams["figure.figsize"] = (14, 6)
    plt.plot(freq1, psd1, label="target")
    plt.plot(freq2, psd2, label="inference")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density')
    plt.title('Power Spectral Density (0-6 Hz)')
    plt.legend()
    plt.grid(True)
    plt.savefig("wlg/result_PSD{}.jpg".format(epoch))
    plt.clf()
from scipy.signal import butter
import scipy


def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    ppg_signal = scipy.signal.filtfilt(b, a, np.double(ppg_signal))
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs),
    return hr_peak


def scatter(data1, data2):
    '''

    '''

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    plt.rcParams['font.size'] = '16'

    x = np.arange(60, 140, 1)
    # x = np.arange(55, 100, 1)
    y = x
    plt.scatter(data1, data2, marker='o', edgecolors='b', s=100, alpha=0.5)
    plt.plot(x, y, linewidth=3, color='red')
    plt.xlabel("HR_gt", fontsize=20 )
    plt.ylabel("HR_pre", fontsize=20)
    plt.savefig("wlg/corr{}.jpg".format(epoch))
    plt.clf()


def bland_altman_plot_my(data1, data2, *args, **kwargs):
    '''

    '''

    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2  # Difference between data1 and data2
    md = np.mean(diff)  # Mean of the difference
    sd = np.std(diff, axis=0)  # Standard deviation of the difference

    plt.scatter(mean, diff, marker='o', s=100, *args, **kwargs)
    plt.axhline(md, color='black', linewidth=3, linestyle='--')
    plt.axhline(md + 1.96 * sd, color='red', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='red', linestyle='--')
    plt.xlabel("(HRpre+HRgt)/2", fontsize='x-large')
    plt.ylabel("HRgt-HRpre", fontsize='x-large')
    plt.savefig("wlg/bland_altman_{}.jpg".format(epoch))
    plt.clf()
def featuremap(x,epoch,f=1):
    if f==1:
        org_img=x[0,:,0,:,:].cpu()
        org_img=org_img.data.numpy()*128+127.5
        org_img=org_img.transpose((1,2,0))
        cv2.imwrite("visual{}.jpg".format(epoch),org_img)
    else:
        org_img = x[0, :, 0, :, :].cpu()
        org_img = org_img.data.numpy() *256
        org_img=org_img.transpose((1,2,0))
        cv2.imwrite("visual{}.jpg".format(epoch), org_img)



if __TIME__:
    start_time = time.time()
optimizer = optimizers(model.parameters(),1, "ada_delta")
if __TIME__:
    log_info_time("setting optimizer time \t: ", datetime.timedelta(seconds=time.time() - start_time))
min_val_loss = 30.0
min_val_loss_model = None
train_loss_mean = []
test_loss_mean = []

loss1 = torch.nn.L1Loss()  # 支持自动梯度计算
for epoch in range(100):
    inference_array = []
    target_array = []
    tarout = []
    preout = []
    if __TIME__ and epoch == 0:
        start_time = time.time()
    if model_params["name"] in ['MetaPhys', "MetaPhysNet"]:
        Meta(model, train_loader, inner_criterion)
    else:
        with tqdm(train_loader, desc="Train ", total=len(train_loader)) as tepoch:
            model.train()
            running_loss = 0.0
            i = 0

            for inputs, target in tepoch:
                tepoch.set_description(f"Train Epoch {epoch}")
                outputs,_,_= model(inputs)
                # print(outputs.shape)
                # print(outputs)
                if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "DeepPhys", "Deepnew","Rnet"]:
                    out=getBPM1(outputs)
                    tar=getBPM1(target)
                    loss = 0.5*loss1(out, tar)+criterion(outputs, target)
                else:
                    loss_0 = criterion(outputs[:][0], target[:][0])
                    loss_1 = criterion(outputs[:][1], target[:][1])
                    loss_2 = criterion(outputs[:][2], target[:][2])
                    loss = loss_0 + loss_2 + loss_1

                if ~torch.isfinite(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                running_loss += loss.item()
                optimizer.step()
                tepoch.set_postfix(loss=running_loss)
                # tepoch.set_postfix(loss=loss.item())
            list = [epoch, running_loss / len(tepoch)]
            data = pd.DataFrame([list])
            data.to_csv('wlg/train_huber.csv', mode='a', header=False, index=False)
        if __TIME__ and epoch == 0:
            log_info_time("1 epoch training time \t: ", datetime.timedelta(seconds=time.time() - start_time))


        with tqdm(test_loader, desc="test ", total=len(test_loader)) as tepoch:
            # model.eval()
            running_loss = 0.0

            mae_ = 0
            rmse_ = 0
            corr_ = 0
            sd_ = 0

            mae_hr_ = 0
            rmse_hr_ = 0
            corr_hr_ = 0
            sd_hr_ = 0
            with torch.no_grad():
                for inputs, target in tepoch:
                    tepoch.set_description(f"test")
                    outputs,x1,x2 = model(inputs)
                    if model_params["name"] in ["PhysNet", "PhysNet_GC", "DeepPhys"]:
                        loss = criterion(outputs, target)
                        tar = getBPM(target[0][0], 30, winsize=10)
                        out = getBPM(outputs, 30, winsize=10)
                        error_hr = tar - out
                        sd = torch.std(error_hr)
                        rmse = torch.sqrt(torch.mean(torch.square(error_hr)))
                        mae = torch.mean(torch.abs(error_hr))
                        corr, _ = pearsonr(out.detach().numpy(), tar.detach().numpy())

                    elif model_params["name"] == "Rnet":
                        # loss = criterion(outputs, target)
                        out = getBPM1(outputs)
                        # print(out.requires_grad)  # True，表明梯度链未断开
                        tar = getBPM1(target)
                        loss =0.5*loss1(out, tar) + criterion(outputs, target)


                    else:
                        loss_0 = criterion(outputs[:][0], target[:][0])
                        loss_1 = criterion(outputs[:][1], target[:][1])
                        loss_2 = criterion(outputs[:][2], target[:][2])
                        loss = loss_0 + loss_2 + loss_1

                    if ~torch.isfinite(loss):
                        continue
                    running_loss += loss.item()

                    tepoch.set_postfix(loss=running_loss / params["test_batch_size"])
                    if model_params["name"] in ["PhysNet", "PhysNet_LSTM", "PhysNet_GC", "Rnet"]:
                        inference_array.extend(normalize(outputs.cpu().numpy()[0]))
                        target_array.extend(normalize(target[0].cpu().numpy()))


                    if tepoch.n == 0 and __TIME__:
                        save_time = time.time()

                loss_mean = running_loss / len(tepoch)
                hr_predic_array = getBPM(np.array(inference_array), fs=30, winsize=10,f11=2)
                ppg_hr_array = getBPM(np.array(target_array), fs=30, winsize=10,f11=2)

                error = torch.Tensor(hr_predic_array - ppg_hr_array)
                sd_hr = torch.std(torch.tensor(hr_predic_array))
                rmse_hr = torch.sqrt(torch.mean(torch.square(error)))
                mae_hr = torch.mean(torch.abs(error))

                corr, _ = pearsonr(np.array(hr_predic_array), np.array(ppg_hr_array))
                list = [epoch, loss_mean, sd_hr.item(), mae_hr.item(), rmse_hr.item(), corr]
                data = pd.DataFrame([list])
                data.to_csv('wlg/L1.csv', mode='a', header=False, index=False)


                if min_val_loss > loss_mean:  # save the train model
                    min_val_loss = loss_mean
                    checkpoint = {'Epoch': epoch,
                                  'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict()}
                    torch.save(model.state_dict(), model_params["name"]
                               + params["dataset_name"] + "_" + str(epoch) + "_"
                               + str(min_val_loss) + '.pth')
                    min_val_loss_model = copy.deepcopy(model)
            if __TIME__ and epoch == 0:
                log_info_time("inference time \t: ", datetime.timedelta(seconds=save_time - start_time))

        if epoch:
            plot_graph(0, 300, target_array, inference_array, epoch, ppg=True)
            plot_graph(0, 300, hr_predic_array, ppg_hr_array, epoch, ppg=False)
            bland_altman_plot_my(hr_predic_array, ppg_hr_array)
            scatter(ppg_hr_array, hr_predic_array)
            plot_time_and_psd(target_array, inference_array, 30, epoch)




