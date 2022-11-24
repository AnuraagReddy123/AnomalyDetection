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
import subprocess

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
import Constants
from Utils import *

join = os.path.join

import subprocess
import sys
import os


def run(cmd):
    os.environ['PYTHONUNBUFFERED'] = "1"
    proc = subprocess.Popen(cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )
    stdout, stderr = proc.communicate()

    return proc.returncode, stdout, stderr


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

classifier_path = join(os.getcwd(), 'WADI', 'Classifiers')
if not os.path.exists(classifier_path):
    os.mkdir(classifier_path)

spike_classifier_path = join(os.getcwd(), 'WADI', 'Classifiers', 'Spike')
if not os.path.exists(spike_classifier_path):
    os.mkdir(spike_classifier_path)

pms_classifier_path = join(os.getcwd(), 'WADI', 'Classifiers', 'PMS')
if not os.path.exists(pms_classifier_path):
    os.mkdir(pms_classifier_path)

psd_classifier_path = join(os.getcwd(), 'WADI', 'Classifiers', 'PSD')
if not os.path.exists(psd_classifier_path):
    os.mkdir(psd_classifier_path)

ln_classifier_path = join(os.getcwd(), 'WADI', 'Classifiers', 'LN')
if not os.path.exists(ln_classifier_path):
    os.mkdir(ln_classifier_path)

labels_array = np.array(anomaly_data['label'])
anomalous_indices = np.where(labels_array == 1)[0]

normal_data.columns = normal_data.columns.str.replace(r"\\", '', regex=True)
normal_data.columns = normal_data.columns.str.replace('WIN-25J4RO10SBFLOG_DATASUTD_WADILOG_DATA', '')
anomaly_data.columns = anomaly_data.columns.str.replace(r"\\", '', regex=True)
anomaly_data.columns = anomaly_data.columns.str.replace('WIN-25J4RO10SBFLOG_DATASUTD_WADILOG_DATA', '')


#_________________________________________Data Creation________________________________________________#

# Preprocess
normal_data, deleted_sensors = preprocess_normal(normal_data)

print('-----------------Segmenting-----------------')

# Get the output of the subprocess Segmenter.py and store it in log.txt
if len(os.listdir(save_path)) == 0:
    code, out, err = run([sys.executable, 'Segmenter.py'])
    # Write out to log.txt
    with open('log.txt', 'w') as f:
        f.write(out.decode())

# Call subprocess on parse_problem_sensors.py
subprocess.call(['python', 'Parse_Problem_Sensors.py'])

# Load problem sensors and append to deleted sensors
with open('problem_sensors.txt', 'r') as f:
    for line in f:
        deleted_sensors.append(line[9:-1])
        normal_data = normal_data.drop(line[9:-1], axis=1)


# Make dataset for each sensor
print('-----------------Making Datasets-----------------')
sensors = normal_data.columns
for sensor in sensors:
    if sensor in ['Row', 'Date', 'Time']:
        continue
    print('Sensor: ', sensor)
    signal = np.load(join(save_path, sensor + '.npy'), allow_pickle=True)
    
    if not os.path.exists(join(spike_path, sensor + '_dataset.npy')):
        X, y = make_dataset(signal, 'spike', Constants.ALPHA_SPIKE)
        np.save(join(spike_path, sensor + '_dataset'), X)
        np.save(join(spike_path, sensor + '_y'), y)

    if not os.path.exists(join(pms_path, sensor + '_dataset.npy')):
        X, y = make_dataset(signal, 'pms', Constants.ALPHA_PMS)
        np.save(join(pms_path, sensor + '_dataset'), X)
        np.save(join(pms_path, sensor + '_y'), y)

    if not os.path.exists(join(psd_path, sensor + '_dataset.npy')):
        X, y = make_dataset(signal, 'psd', Constants.ALPHA_PSD)
        np.save(join(psd_path, sensor + '_dataset'), X)
        np.save(join(psd_path, sensor + '_y'), y)

    if not os.path.exists(join(ln_path, sensor + '_dataset.npy')):
        X, y = make_dataset(signal, 'ln', Constants.ALPHA_LN)
        np.save(join(ln_path, sensor + '_dataset'), X)
        np.save(join(ln_path, sensor + '_y'), y)

#_________________________________________Traning________________________________________________#

# Build classifiers for each sensor and train
print('-----------------Training-----------------')
for sensor in sensors:
    if sensor in ['Row', 'Date', 'Time']:
        continue
    print('Sensor: ', sensor)
    if not os.path.exists(join(spike_classifier_path, sensor + '.pkl')):
        x = np.load(join(spike_path, sensor + '_dataset.npy'), allow_pickle=True)
        y = np.load(join(spike_path, sensor + '_y.npy'), allow_pickle=True)
        clf = SpikeClassifier()
        clf.fit(x, y)
        joblib.dump(clf, join(spike_classifier_path, sensor + '.pkl'))
        y_pred = clf.predict(x)
        # Write metrics in file
        print('Spike')
        with open (join('WADI', 'Accuracies', 'Spike.txt') , 'a') as f:
            f.write('Sensor: ' + sensor + '\n')
            f.write('Accuracy: ' + str(accuracy_score(y, y_pred)) + '\n')
            f.write('Precision: ' + str(precision_score(y, y_pred)) + '\n')
            f.write('Recall: ' + str(recall_score(y, y_pred)) + '\n')
            f.write('\n')

    if not os.path.exists(join(pms_classifier_path, sensor + '.pkl')):
        x = np.load(join(pms_path, sensor + '_dataset.npy'), allow_pickle=True)
        y = np.load(join(pms_path, sensor + '_y.npy'), allow_pickle=True)

        clf = PMSClassifier()
        clf.fit(x, y)
        joblib.dump(clf, join(pms_classifier_path, sensor + '.pkl'))
        y_pred = clf.predict(x)
        # Write metrics in file
        print('PMS')
        with open (join('WADI', 'Accuracies', 'PMS.txt') , 'a') as f:
            f.write('Sensor: ' + sensor + '\n')
            f.write('Accuracy: ' + str(accuracy_score(y, y_pred)) + '\n')
            f.write('Precision: ' + str(precision_score(y, y_pred)) + '\n')
            f.write('Recall: ' + str(recall_score(y, y_pred)) + '\n')
            f.write('\n')
    
    # if not os.path.exists(join(psd_classifier_path, sensor + '.pkl')):
    #     x = np.load(join(psd_path, sensor + '_dataset.npy'), allow_pickle=True)
    #     y = np.load(join(psd_path, sensor + '_y.npy'), allow_pickle=True)
    #     clf = PSDClassifier()
    #     clf.fit(x, y)
    #     joblib.dump(clf, join(psd_classifier_path, sensor + '.pkl'))
    #     y_pred = clf.predict(x)
    #     print(f'Accuracy: {accuracy_score(y, y_pred):.3f}')
    #     print(f'Precision: {precision_score(y, y_pred):.3f}')
    #     print(f'Recall: {recall_score(y, y_pred):.3f}')
    #     print()
    
    # if not os.path.exists(join(ln_classifier_path, sensor + '.pkl')):
    #     x = np.load(join(ln_path, sensor + '_dataset.npy'), allow_pickle=True)
    #     y = np.load(join(ln_path, sensor + '_y.npy'), allow_pickle=True)
    #     clf = LNClassifier()
    #     clf.fit(x, y)
    #     joblib.dump(clf, join(ln_classifier_path, sensor + '.pkl'))
    #     y_pred = clf.predict(x)
    #     print(f'Accuracy: {accuracy_score(y, y_pred):.3f}')
    #     print(f'Precision: {precision_score(y, y_pred):.3f}')
    #     print(f'Recall: {recall_score(y, y_pred):.3f}')
    #     print()
    
#_________________________________________Testing________________________________________________#