import numpy as np
import h5py
# import POS.pos as pos
import scipy.signal
import xml.etree.ElementTree as ET
import pandas as pd

import numpy as np

import pandas as pd
import json



def Deepphys_preprocess_Label(path):
#     '''
#     :param path: label file path
#     :return: delta pulse
#     '''
#     # TODO : need to check length with video frames
#     # TODO : need to implement piecewise cubic Hermite interpolation
#     # Load input

    part = 0
    window = 32
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))

    delta_label = []
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    while part < (len(delta_label) // 32) - 1:
        delta_label[part*window:(part+1)*window] /= np.std(delta_label[part*window:(part+1)*window])
        part += 1
        
    if len(delta_label) % window != 0:
            delta_label[part * window:] /= np.std(delta_label[part * window:])        
    

    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  
    f.close()

    return delta_pulse

def Deepphys_new_Label(path):
#     '''
#     :param path: label file path
#     :return: delta pulse
#     '''
#     # TODO : need to check length with video frames
#     # TODO : need to implement piecewise cubic Hermite interpolation
#     # Load input

    part = 0
    window = 256
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
    label = list(map(float, label))

    delta_label = []
  
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    while part < (len(delta_label) // 256) - 1:
        delta_label[part*window:(part+1)*window] /= np.std(delta_label[part*window:(part+1)*window])
        part += 1
        
    if len(delta_label) % window != 0:
            delta_label[part * window:] /= np.std(delta_label[part * window:])        
    


    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  
    f.close()




    return delta_pulse
       
def PhysNet_preprocess_Label(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    # Load input
    f = open(path, 'r')
    f_read = f.read().split('\n')
    label = ' '.join(f_read[0].split()).split()
#     label_hr = ' '.join(f_read[1].split()).split()
    label = list(map(float, label))
#     label_hr = list(map(float, label_hr))
    label = np.array(label).astype('float32')
#     label_hr = np.array(label_hr).astype('float32')
    split_raw_label = np.zeros((len(label)// 256, 256))
#     split_raw_label_hr = np.zeros(((len(label_hr) // 256), 256))
    index = 0
    for i in range(((len(label))//256)):
        split_raw_label[i] = label[index:index + 256]
#         split_raw_label_hr[i] = label_hr[index:index + 256]
        index = index + 256
    split_raw_label=np.array(split_raw_label).astype('float32')
    split_label=split_raw_label.copy() 
#     split_raw_label_hr=np.array(split_raw_label_hr).astype('float32')
#     split_label_hr=split_raw_label_hr.copy() 
    f.close()

    return split_label

def Deepphys_new_Label_PURE(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    # Load input 
   
    with open(path, 'r') as input_file:
        json_data = json.load(input_file)
        ids = ['/Image', '/FullPackage']
        label = []
        waveforms = [fr["Value"]["waveform"] for fr in json_data['/FullPackage']]
        x_timestamp = [fr["Timestamp"] for fr in json_data['/FullPackage']]
        image_timestamp = [fr["Timestamp"] for fr in json_data['/Image']]
        PPG = [np.interp(im,x_timestamp,waveforms) for im in image_timestamp]
        label.extend(PPG)
        label = list(map(float, label))
    part = 0
    window=256
    delta_label = []
  
    for i in range(len(label) - 1):
        delta_label.append(label[i + 1] - label[i])
    while part < (len(delta_label) // 256) - 1:
        delta_label[part*window:(part+1)*window] /= np.std(delta_label[part*window:(part+1)*window])
        part += 1
        
    if len(delta_label) % window != 0:
            delta_label[part * window:] /= np.std(delta_label[part * window:])        
    


    delta_label = np.array(delta_label).astype('float32')
    delta_pulse = delta_label.copy()  
   
    return delta_pulse

def PhysNet_cohface_Label(path, frame_total):
    f = h5py.File(path, "r")
    label = list(f['pulse'])
    f.close()
    label = np.interp(np.arange(0, frame_total+1),
                      np.linspace(0, frame_total+1, num=len(label)),
                      label)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32

    return split_raw_label

def LGGI_Label(path, frame_total):
    doc = ET.parse(path)
    root = doc.getroot()
    label = []

    for value in root:
        label.append(int(value.findtext('value2')))

    label = np.array(label).astype('float32')
    label = scipy.signal.resample(label, frame_total)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32
    #print(len(split_raw_label))
    return split_raw_label

def V4V_Label(video, framerate):
    label = pos.PPOS(video, framerate)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32

    return split_raw_label

def VIPL_Label(path, frame_total):
    f = pd.read_csv(path)
    label = f['Wave']
    label = np.array(label).astype('float32')
    label = scipy.signal.resample(label, frame_total)

    split_raw_label = np.zeros(((len(label) // 32), 32))
    index = 0
    for i in range(len(label) // 32):
        split_raw_label[i] = label[index:index + 32]
        index = index + 32
    #print(split_raw_label.shape)
    return split_raw_label

#######PURE#########


def convert_timestamp(current_timestamp, first_timestamp):
    return float(current_timestamp - first_timestamp) * 1e-9

def PhysNet_preprocess_Label_PURE(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    # Load input 

    with open(path, 'r') as input_file:
        json_data = json.load(input_file)
        ids = ['/Image', '/FullPackage']
        label = []
        waveforms = [fr["Value"]["waveform"] for fr in json_data['/FullPackage']]
        x_timestamp = [fr["Timestamp"] for fr in json_data['/FullPackage']]
        image_timestamp = [fr["Timestamp"] for fr in json_data['/Image']]
        PPG = [np.interp(im,x_timestamp,waveforms) for im in image_timestamp]
        label.extend(PPG)
#         print(label[0:10])
#         first_timestamp = min([min([entry['Timestamp'] for entry in json_data[id]]) for id in ids])
#         phys_data = {'timestamp': [], 'ppg': []}
#         for entry in json_data['/FullPackage']:
#             phys_data['timestamp'].append(convert_timestamp(entry['Timestamp'], first_timestamp))
#             phys_data['ppg'].append(entry['Value']['waveform'])
#         df = pd.DataFrame(PPGs)
#         df = CorrectIrregularlySampledData(df, 30.0)
#         label = df['ppg'].values.tolist()
        label = np.array(label).astype('float32')
        split_raw_label = np.zeros(((len(label) // 32), 32))
        index = 0
        for i in range(len(label) // 32):
            split_raw_label[i] = label[index:index + 32]
            index = index +32
        split_raw_label = np.array(split_raw_label).astype('float32')
        split_label = split_raw_label.copy()

    return split_label
def CorrectIrregularlySampledData(df, Fs):
    if df.iloc[0]['timestamp'] > 0.0:
        top_row = df.iloc[[0]].copy()
        df = pd.concat([top_row, df], ignore_index=True)
        df.loc[0, 'timestamp'] = 0.0
    new_data = []
    for frame_on, time_on in enumerate(np.arange(0.0, df.iloc[-1]['timestamp'], 1 / Fs)):
        time_diff = (df['timestamp'] - time_on).to_numpy()
        stop_idx = np.argmax(time_diff > 0)
        start_idx = stop_idx - 1
        time_span = time_diff[stop_idx] - time_diff[start_idx]
        rel_time = -time_diff[start_idx]
        stop_weight = rel_time / time_span
        start_weight = 1 - stop_weight
        average_row = pd.concat([df.iloc[[start_idx]].copy() * start_weight, df.iloc[[stop_idx]].copy() * stop_weight]).sum().to_frame().T
        new_data.append(average_row)
    return pd.concat(new_data)


def PhysNet_preprocess_Label_PURE_new(path):
    '''
    :param path: label file path
    :return: wave form
    '''
    # Load input 





    label = []
    label_time = []
    label_hr = []
    time = []

    with open(path, 'r') as input_file:
        json_data = json.load(input_file)
        for data in json_data['/FullPackage']:
            label.append(data['Value']['waveform'])
            label_time.append(data['Timestamp'])
            label_hr.append(data['Value']['pulseRate'])
        for data in json_data['/Image']:
            time.append(data['Timestamp'])
        label_std = label_time[0]
        time_std = time[0]
        if label_std < time_std:
            time = [(i - label_std) / 1000 for i in time]
            label_time = [(i - label_std) / 1000 for i in label_time]
            j = 0
            i = 0
            new_label = []
            new_hr = []

            while i < len(time):
                if j + 1 >= len(label_time):
                    break
                if i == 0:
                    if time[i] <= label_time[j]:
                        new_label.append(0)
                        new_hr.append(0)
                        i += 1

                if label_time[j + 1] >= time[i] >= label_time[j]:
                    term = label_time[j + 1] - label_time[j]
                    head = time[i] - label_time[j]  # 앞에꺼
                    back = label_time[j + 1] - time[i]  # 뒤에꺼
                    new_label.append((label[i] * back + label[i - 1] * head) / term)
#                     new_hr.append((label_hr[i] * back + label_hr[i - 1] * head) / term)
                    i += 1
                else:
                    j += 1

        else:
            label_time = [(i - time_std) / 1000 for i in label_time]
            time = [(i - time_std) / 1000 for i in time]
            
            j = 0
            i = 0
            new_label = []
            new_hr = []

            while i < len(time):
                if j + 1 >= len(label_time):
                    break

                if time[i] <= label_time[j]:
                    new_label.append(0)
                    new_hr.append(0)
                    i += 1
                    continue

                if label_time[j + 1] >= time[i] and time[i] >= label_time[j]:
                    term = label_time[j + 1] - label_time[j]
                    head = time[i] - label_time[j]  # 앞에꺼
                    back = label_time[j + 1] - time[i]  # 뒤에꺼
                    new_label.append((label[i] * back + label[i - 1] * head) / term)
#                     new_hr.append((label_hr[i] * back + label_hr[i - 1] * head) / term)
                    i += 1
                j += 1


        if (len(new_label) - 22) < 0:
            print("negative" + path)
        label = np.array(new_label).astype('float16')
        split_raw_label = np.zeros(((len(label) // 256), 256))
        index = 0
        for i in range(len(label) // 256):
            split_raw_label[i] = label[index:index + 256]
            index = index + 256
        split_raw_label = np.array(split_raw_label).astype('float16')
        split_label = split_raw_label.copy()

    return split_label