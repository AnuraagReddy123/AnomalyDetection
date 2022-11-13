import numpy as np
import pandas as pd
from copy import deepcopy
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
import sys
import joblib
import matplotlib.pyplot as plt
import pickle

from F16_AnomalyHandler import AnomalyHandler
from F1_BaseAnomalyGenerator import BaseAnomalyGenerator
from F2_BaseFeature import BaseFeature
from F3_BaseAnomalyClassifier import BaseAnomalyClassifier
from F4_SpikeAnomalyGenerator import SpikeAnomalyGenerator
from F5_SpikeFeature import SpikeFeature
from F6_SpikeClassifier import SpikeClassifier
from F7_LNAnomalyGenerator import LNAnomalyGenerator
from F8_LNFeature import LNFeature
from F9_LNClassifier import LNClassifier
from F10_PMSAnomalyGenerator import PMSAnomalyGenerator
from F11_PMSFeature import PMSFeature
from F12_PMSClassifier import PMSClassifier
from F13_PSDAnomalyGenerator import PSDAnomalyGenerator
from F14_PSDFeature import PSDFeatures
from F15_PSDClassifier import PSDClassifier
from sklearn.ensemble import RandomForestClassifier
from Segmenter import hmm_segmentation
from Constants import *
from Utils import *

join = os.path.join

# Make dataset for each given sensor
def make_dataset(sensor, anomaly_type, alpha):
    dataset = []
    y = []
    anomaly_dict = {'spike': SpikeAnomalyGenerator(), 'pms': PMSAnomalyGenerator(),
                    'psd': PSDAnomalyGenerator(), 'ln': LNAnomalyGenerator()}

    for i in range(len(sensor)):  # Each row of sensor
        if sensor[i].shape[0] <= 10:
          continue
        # Positive data
        x = sensor[i]
        y.append(0)
        dataset.append(x)

        # Negative data
        x = anomaly_dict[anomaly_type].transform(
            sensor[i], alpha)
        y.append(1)
        dataset.append(x)

    return np.array(dataset, dtype='object'), np.array(y)

#_________________________________________Setup Code________________________________________________#
# Paths
raw_data_path = join(os.getcwd(), 'WADI')
normal_data = pickle.load(open(join(raw_data_path, 'dfn_wadi_pp.pkl'), 'rb'))
anomaly_data = pickle.load(open(join(raw_data_path, 'dfa_wadi_pp.pkl'), 'rb'))
print(normal_data.shape, anomaly_data.shape)

# Save path
save_path = join(os.getcwd(), 'WADI', 'Sensors')
if not os.path.exists(save_path):
    os.mkdir(save_path)

data_path = join(os.getcwd(), 'WADI', 'Datasets')
if not os.path.exists(data_path):
    os.mkdir(data_path)

spike_path = join(os.getcwd(), 'WADI', 'Datasets', 'Spike')
if not os.path.exists(spike_path):
    os.mkdir(spike_path)

pms_path = join(os.getcwd(), 'WADI', 'Datasets', 'PMS')
if not os.path.exists(pms_path):
    os.mkdir(pms_path)

psd_path = join(os.getcwd(), 'WADI', 'Datasets', 'PSD')
if not os.path.exists(psd_path):
    os.mkdir(psd_path)

ln_path = join(os.getcwd(), 'WADI', 'Datasets', 'LN')
if not os.path.exists(ln_path):
    os.mkdir(ln_path)

labels_array = np.array(anomaly_data['label'])
anomalous_indices = np.where(labels_array == 1)[0]

normal_data.columns = normal_data.columns.str.replace(r"\\", '', regex=True)
normal_data.columns = normal_data.columns.str.replace('WIN-25J4RO10SBFLOG_DATASUTD_WADILOG_DATA', '')
anomaly_data.columns = anomaly_data.columns.str.replace(r"\\", '', regex=True)
anomaly_data.columns = anomaly_data.columns.str.replace('WIN-25J4RO10SBFLOG_DATASUTD_WADILOG_DATA', '')


#_________________________________________Data Creation________________________________________________#

# Preprocess
normal_data, deleted_sensors = preprocess_normal(normal_data)

sensors = normal_data.columns
for sensor in sensors:
    if sensor in ['Row', 'Date', 'Time']:
        continue
    print('Sensor: ', sensor)
    # if file exists then skip
    if os.path.exists(join(save_path, sensor + '.npy')):
        continue
    decoded_array, segments, motifs, hmm_train = hmm_segmentation(normal_data[sensor], WINDOW_SIZE, N_STATES)
    signal = [x.to_numpy(dtype='float') for x in motifs]
    signal = np.array(signal, dtype='object')
    np.save(join(save_path, sensor), signal)

# Make dataset for each sensor
for sensor in sensors:
    if sensor in ['Row', 'Date', 'Time']:
        continue
    print('Sensor: ', sensor)
    # if file exists then skip
    if os.path.exists(join(spike_path, sensor + '.npy')):
        continue
    signal = np.load(join(save_path, sensor + '.npy'), allow_pickle=True)
    X, y = make_dataset(signal, 'spike', 0.1)
    np.save(join(spike_path, sensor + '_dataset'), X)
    np.save(join(spike_path, sensor + '_y'), y)