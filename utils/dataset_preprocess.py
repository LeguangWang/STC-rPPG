import multiprocessing
import os
import h5py
from  utils.image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video,Deepphys_new,Deepphys_new_pure,PhysNet_preprocess_Video_pure
# from image_preprocess import Deepphys_preprocess_Video, PhysNet_preprocess_Video#, RTNet_preprocess_Video
# from utils.seq_preprocess import PPNet_preprocess_Mat
from  utils.text_preprocess import *
import signal
signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号


def preprocessing(save_root_path: str = "/media/hdd1/dy_dataset/",
                  model_name: str = "MetaPhys",
                  data_root_path: str = "/media/hdd1/",
                  dataset_name: str = "UBFC",
                  train_ratio: float = 0.8):
    """
    :param save_root_path: save file destination path
    :param model_name: select preprocessing method
    :param data_root_path: data set root path
    :param dataset_name: data set name(ex. UBFC, COFACE)
    :param train_ratio: data split [ train ratio : 1 - train ratio]
    :return:
    """
    dataset_root_path = data_root_path 

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
#     signal(SIGPIPE, SIG_IGN)

    if dataset_name == "UBFC":
        data_list = [dataset_root_path+"/"+data for data in os.listdir(dataset_root_path) if data.__contains__("subject")]
    elif dataset_name == "cuff_less_blood_pressure":
        data_list = [data for data in os.listdir(dataset_root_path) if data.__contains__("part")]
    elif dataset_name == "cohface":
        data_list = [data for data in os.listdir(dataset_root_path) if data.isdigit()]
    elif dataset_name == "V4V":
        dataset_root_path = data_root_path + dataset_name + '/train_val/Videos'
        data_list = ["train",'valid']
    elif dataset_name == "VIPL_HR":
        dataset_root_path = data_root_path + dataset_name + '/data'
        data_list = [data for data in os.listdir(dataset_root_path)] #p1-p107
    
    if dataset_name == "PURE":
        data_list=[]
        for data in os.listdir(dataset_root_path):
            if data.__contains__("json"):
                continue
            else:  
                data_list.append(dataset_root_path+"/"+data) 
                

    process = []
    print(data_list)
    # multiprocessing
    for index, data_path in enumerate(data_list):
        proc = multiprocessing.Process(target=preprocess_Dataset,
                                       args=(data_path, 1, model_name, dataset_name, return_dict))
        # flag 0 : pass
        # flag 1 : detect face
        # flag 2 : remove nose
        process.append(proc)
        proc.start()
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
    for proc in process:
        proc.join()

#     train = int(len(return_dict.keys()) *train_ratio)  # split dataset

    train_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")

    if model_name in ["DeepPhys", "PhysNet", "PhysNet_LSTM", "MetaPhys", "MetaPhysNet", "Deepnew"]:

        for index, data_path in enumerate(return_dict.keys()):
            print(data_path)
            dset = train_file.create_group(data_path)
            dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
            dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
#             dset['preprocessed_label_hr'] = return_dict[data_path]['preprocessed_label_hr']
        train_file.close()

#         test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
#         for index, data_path in enumerate(return_dict.keys()[train:]):
#             dset = test_file.create_group(data_path)
#             dset['preprocessed_video'] = return_dict[data_path]['preprocessed_video']
#             dset['preprocessed_label'] = return_dict[data_path]['preprocessed_label']
#         test_file.close()

    # elif model_name in ["PPNet"]:
    #
    #     for index, data_path in enumerate(return_dict.keys()[:train]):
    #         dset = train_file.create_group(data_path)
    #         dset['ppg'] = return_dict[data_path]['ppg']
    #         dset['sbp'] = return_dict[data_path]['sbp']
    #         dset['dbp'] = return_dict[data_path]['dbp']
    #         dset['hr'] = return_dict[data_path]['hr']
    #     train_file.close()
    #
    #     test_file = h5py.File(save_root_path + model_name + "_" + dataset_name + "_test.hdf5", "w")
    #     for index, data_path in enumerate(return_dict.keys()[train:]):
    #         dset = test_file.create_group(data_path)
    #         dset['ppg'] = return_dict[data_path]['ppg']
    #         dset['sbp'] = return_dict[data_path]['sbp']
    #         dset['dbp'] = return_dict[data_path]['dbp']
    #         dset['hr'] = return_dict[data_path]['hr']
    #     test_file.close()


def preprocess_Dataset(path, flag, model_name, dataset_name, return_dict):
    """
    :param path: dataset path
    :param flag: face detect flag
    :param model_name: select preprocessing method
    :param return_dict: : preprocessed image, label
    """

    # Video Based
    if dataset_name == 'UBFC':
        if model_name == "DeepPhys" or model_name == "MetaPhys":
            rst, preprocessed_video = Deepphys_preprocess_Video(path + "/vid.avi", flag)
        elif model_name in ["PhysNet", "PhysNet_LSTM", "MetaPhysNet"]:
            rst, preprocessed_video = PhysNet_preprocess_Video(path + "/vid.avi", flag)
        elif model_name =="Deepnew":
            rst, preprocessed_video = Deepphys_new(path + "/vid.avi", flag)
        # elif model_name == "RTNet":
        #     rst, preprocessed_video = RTNet_preprocess_Video(path + "/vid.avi", flag)
        # elif model_name == "PPNet":    # Sequence data based
        #     ppg, sbp, dbp, hr  = PPNet_preprocess_Mat(path)

        if model_name in ["DeepPhys","MTTS","PhysNet","PhysNet_LSTM","Deepnew"]:  # can't detect face
            if not rst:
                return

        if model_name == "DeepPhys" or model_name == "MetaPhys":
            preprocessed_label = Deepphys_preprocess_Label(path + "/ground_truth.txt")
        elif model_name == "Deepnew":
            preprocessed_label = Deepphys_new_Label(path + "/ground_truth.txt")
        elif model_name in ["PhysNet", "PhysNet_LSTM", "MetaPhysNet"]:
            preprocessed_label = PhysNet_preprocess_Label(path + "/ground_truth.txt")

        # ppg, sbp, dbp, hr
        
        if model_name in ["DeepPhys","MTTS","PhysNet","PhysNet_LSTM","MetaPhys","MetaPhysNet","Deepnew"]:
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
            return_dict[path.split("/")[-1]] = {'preprocessed_video': preprocessed_video,
                                                'preprocessed_label': preprocessed_label
                                               }
        # elif model_name in ["PPNet"]:
        #     return_dict[path.split("/")[-1]] = {'ppg': ppg,'sbp': sbp,'dbp': dbp,'hr' : hr}
    if dataset_name == 'PURE':
        if model_name == "PhysNet" :
            print(path)
            rst, preprocessed_video=PhysNet_preprocess_Video_pure(path+"/",flag)
            if not rst:
                return
            preprocessed_label=PhysNet_preprocess_Label_PURE(path +".json")
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略SIGPIPE信号
            return_dict[path.split("/")[-1]] = {'preprocessed_video': preprocessed_video,
                                                'preprocessed_label': preprocessed_label}



if __name__ == '__main__':
    preprocessing(save_root_path="../PURE/",
                      model_name="PhysNet",
                      data_root_path="../PURE/PURE_test/1/",
                      dataset_name="PURE",
                      train_ratio=0.8)