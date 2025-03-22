# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 14:35:08 2021
Latest edit July 2023

author: Daniel Vázquez Pombo
email: ...
License: https://creativecommons.org/licenses/by/4.0/
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces CPU usage if needed

import numpy as np
import random
import shap
import quantus
import matplotlib.pyplot as plt

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dense, Flatten

##################################
# Funktionen aus "Functions.py"
##################################

def import_PV_WT_data():
    """
    Returns
    -------
    PV : dict
        Holds data regarding the PV string in SYSLAB 715
    WT : dict
        Holds data regarding the Gaia wind turbine
    """
    PV = {
        "Type": "Poly-cristaline",
        "Az": 60,  # deg
        "Estc": 1000,  # W/m^2
        "Tstc": 25,  # °C
        'Pmp_stc': [165, 125],  # W
        'ganma_mp': [-0.478/100, -0.45/100],  # 1/K
        'Ns': [18, 6],  # int
        'Np': [2, 2],   # int
        'a': [-3.56, -3.56],
        'b': [-0.0750, -0.0750],
        'D_T': [3, 3],
        "index": ['A','B'],
        "L": 10,   # array length
        "W": 1.5,  # array width
        "d": 0.1,  # array thickness
        "k_r": 350
    }

    WT = {
        "Type": "Asynchronous",
        "Mode": "Passive, downwind vaning",
        "Pn": 11,  # kW
        "Vn": 400, # V
        'CWs': [3.5, 6, 8, 10, 10.5, 11, 12, 13, 13.4, 14, 16, 18, 20, 22, 24, 25],
        'CP':  [0,   5, 8.5,10.9,11.2,11.3,11.2,10.5,10.5,10,  8.8, 8.7, 8,  7.3, 6.6, 6.3],
        "Cin": 3.5,
        "Cout": 25,
        "HH": 18,
        "D": 13,
        "SA": 137.7,
        "B": 2,
    }

    return PV, WT


def import_SOLETE_data(Control_Var, PVinfo, WTinfo):
    """
    Imports different versions of SOLETE depending on the inputs:
        -resolution
        -SOLETE_builvsimport
    if built it has the option to save the expanded dataset
    """
    print("___The SOLETE Platform___\n")

    # Dummy: For demonstration we skip the real file import
    # In practice, you'd read an HDF or CSV, etc.
    # We'll just build a placeholder DataFrame or arraylike structure.
    print("Simulating 'import_SOLETE_data' ... done.\n")
    # Return a dictionary or DataFrame
    data = np.random.rand(100, 5)  # placeholder
    return data


def PreProcessDataset(data, control):
    """
    A function that does two things:
      1) It adapts the time series to a forecasting problem with supervised learning,
      2) Summons and trains a scaler according to user input.
    """
    print("Simulating 'PreProcessDataset' ... done.\n")
    # For demonstration, just return dummy ML_DATA and a "scaler"
    ML_DATA = {
        'X_TRAIN': np.random.rand(800, control['PRE'], len(control['PossibleFeatures'])),
        'Y_TRAIN': np.random.rand(800, control['H']),
        'X_VAL':   np.random.rand(150, control['PRE'], len(control['PossibleFeatures'])),
        'Y_VAL':   np.random.rand(150, control['H']),
        'X_TEST':  np.random.rand(145, control['PRE'], len(control['PossibleFeatures'])),
        'Y_TEST':  np.random.rand(145, control['H']),
    }
    Scaler = None
    return ML_DATA, Scaler


def PrepareMLmodel(Control_Var, ML_DATA):
    """
    Prepares a model (RF/SVM/LSTM/CNN) either by training or loading.
    """
    print("Simulating 'PrepareMLmodel' ... done.\n")
    # Return a dummy "model" or reference
    return None


def TestMLmodel(Control_Var, ML_DATA, ML, Scaler):
    """
    Takes DATA and ML_DATA to test the trained model in the testing set
    """
    print("Simulating 'TestMLmodel' ... done.\n")
    results = {'dummy': 'res'}
    return results


def post_process(Control_Var, results):
    """
    Some post-processing of the results
    """
    print("Simulating 'post_process' ... done.\n")
    return {'analysis': 'done'}

##################################
# 1) Control The Script
##################################
Control_Var = {
    '_description_': 'Holds variables that define the behaviour of the algorithm.',
    'resolution': '60min',
    'SOLETE_builvsimport': 'Build',
    'SOLETE_save': True,
    'trainVSimport': True,
    'saveMLmodel': True,
    'Train_Val_Test': [70, 20, 10],
    'Scaler': 'MinMax01',
    'IntrinsicFeature': 'P_Solar[kW]',
    'PossibleFeatures': [
        'TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]', 'WIND_DIR[deg]',
        'GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Gaia[kW]', 'P_Solar[kW]',
        'Pressure[mbar]', 'Pac', 'Pdc', 'TempModule', 'TempCell',
        'HoursOfDay', 'MeanPrevH', 'StdPrevH',
        'MeanWindSpeedPrevH', 'StdWindSpeedPrevH'
    ],
    'MLtype': 'CNN',       # 'RF', 'SVM', 'LSTM', 'CNN', 'CNN_LSTM'
    'H': 10,               # multi-step horizon
    'PRE': 5,              # number of previous steps
}


##################################
# 2) ML Config & Hyperparams
##################################
RF = {'n_trees': 1, 'random_state': 32}
SVM = {'kernel': 'rbf','degree': 3,'gamma': 'scale','coef0': 0,'C': 3,'epsilon': 0.1}
LSTM_params = {
    'n_batch': 16, 'epo_num': 1000,
    'Neurons': [15, 15, 15], 'Dense': [0, 0],
    'ActFun': 'tanh', 'LossFun': 'mean_absolute_error',
    'Optimizer': 'adam'
}
CNN_params = {
    'n_batch': 16, 'epo_num': 3,
    'filters': 32, 'kernel_size': 2,'pool_size': 3,
    'Dense': [10, 10],'ActFun': 'tanh','LossFun': 'mean_absolute_error','Optimizer': 'adam'
}
CNN_LSTM_params = {
    'n_batch': 16,'epo_num': 1000,
    'filters': 32,'kernel_size': 3,'pool_size': 2,
    'Dense': [0, 0],
    'CNNActFun': 'tanh',
    'Neurons': [10, 15, 10],
    'LSTMActFun': 'sigmoid',
    'LossFun': 'mean_absolute_error',
    'Optimizer': 'adam'
}
Control_Var['RF'] = RF
Control_Var['SVM'] = SVM
Control_Var['LSTM'] = LSTM_params
Control_Var['CNN'] = CNN_params
Control_Var['CNN_LSTM'] = CNN_LSTM_params

# ---------------------------------------------------------------------
# Datensatz importieren, preprocess, train/test
PVinfo, WTinfo = import_PV_WT_data()
DATA = import_SOLETE_data(Control_Var, PVinfo, WTinfo)
ML_DATA, Scaler = PreProcessDataset(DATA, Control_Var)
ML = PrepareMLmodel(Control_Var, ML_DATA)
results = TestMLmodel(Control_Var, ML_DATA, ML, Scaler)
analysis = post_process(Control_Var, results)

# ---------------------------------------------------------------------
# Für Demonstration: simulierte / Mock-Daten (1095 Samples)
num_samples = 1095
PRE = Control_Var['PRE']
num_features = len(Control_Var['PossibleFeatures'])
X_full = np.random.rand(num_samples, PRE, num_features)
Y_full = np.random.rand(num_samples, Control_Var['H'])

ML_DATA = {
    'X_TRAIN': X_full[:800],
    'Y_TRAIN': Y_full[:800],
    'X_VAL':   X_full[800:950],
    'Y_VAL':   Y_full[800:950],
    'X_TEST':  X_full[950:],
    'Y_TEST':  Y_full[950:]
}

##################################
# 3) (Re-)Build a CNN model example
##################################
def build_model(Control_Var):
    ml_type = Control_Var['MLtype']
    input_shape = (Control_Var['PRE'], len(Control_Var['PossibleFeatures']))
    
    if ml_type == 'CNN':
        params = Control_Var['CNN']
        model = Sequential()
        model.add(Conv1D(filters=params['filters'],
                         kernel_size=params['kernel_size'],
                         activation=params['ActFun'],
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=params['pool_size']))
        model.add(Flatten())
        for dense_neurons in params['Dense']:
            model.add(Dense(dense_neurons, activation=params['ActFun']))
        model.add(Dense(Control_Var['H']))
        model.compile(loss=params['LossFun'], optimizer=params['Optimizer'])
        return model
    
    raise ValueError("Example only covers CNN for demonstration.")

model = build_model(Control_Var)
model.fit(
    ML_DATA['X_TRAIN'], ML_DATA['Y_TRAIN'],
    epochs=3, batch_size=16,
    validation_data=(ML_DATA['X_VAL'], ML_DATA['Y_VAL'])
)

##################################
# 4) SHAP DeepExplainer
##################################
def shap_deepexplainer_2D(model, X_test_3D, feature_names, background_samples=100):
    """
    Compute SHAP with DeepExplainer for a Keras model (CNN/LSTM), handling 3D or 4D arrays.
    Returns shap_values, and does a summary_plot of the 2D-aggregated results.
    """
    # 1) Background set
    n_samples = X_test_3D.shape[0]
    if background_samples < n_samples:
        background_indices = random.sample(range(n_samples), background_samples)
    else:
        background_indices = range(n_samples)
    background_data = X_test_3D[background_indices]

    # 2) DeepExplainer
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(X_test_3D)  # -> possibly a list

    # 3) Falls multi-output => shap_values[0]
    if isinstance(shap_values, list):
        shap_array = shap_values[0]
    else:
        shap_array = shap_values
    
    print("shap_array shape =", shap_array.shape)
    
    # 4) 3D => (N, PRE, F), 4D => (N, PRE, F, H)
    if shap_array.ndim == 4:
        # => (N, PRE, Features, H)
        shap_2D = shap_array.mean(axis=(1, 3))
        print("Detected 4D shap_array => aggregated axis=(1,3). shape:", shap_2D.shape)
    elif shap_array.ndim == 3:
        # => (N, PRE, Features)
        shap_2D = shap_array.mean(axis=1)
        print("Detected 3D shap_array => aggregated axis=1. shape:", shap_2D.shape)
    else:
        raise ValueError(f"Unexpected shape {shap_array.shape} for shap_array.")
    
    # 5) Input aggregation -> (N, Features)
    X_test_agg = X_test_3D.mean(axis=1)
    print("X_test_agg shape =", X_test_agg.shape)
    print("feature_names count =", len(feature_names))

    # 6) Debug checks
    if shap_2D.shape[0] != X_test_agg.shape[0]:
        print(f"WARNING: shap_2D has {shap_2D.shape[0]} samples, but X_test_agg has {X_test_agg.shape[0]}!")
    if shap_2D.shape[1] != X_test_agg.shape[1]:
        print(f"WARNING: shap_2D has {shap_2D.shape[1]} features, but X_test_agg has {X_test_agg.shape[1]}!")
    if shap_2D.shape[1] != len(feature_names):
        print(f"WARNING: shap_2D has {shap_2D.shape[1]} features, but feature_names has {len(feature_names)}!")

    # 7) Beeswarm-Plot
    shap.summary_plot(
        shap_values=shap_2D,
        features=X_test_agg,
        feature_names=feature_names,
        plot_type='dot'
    )
    return shap_values


# ----------------------------------------------
# Anwendung: SHAP (nur bei CNN/LSTM)
# ----------------------------------------------
if Control_Var['MLtype'] in ['CNN', 'LSTM', 'CNN_LSTM']:
    shap_vals = shap_deepexplainer_2D(
        model=model,
        X_test_3D=ML_DATA['X_TEST'],
        feature_names=Control_Var['PossibleFeatures'],
        background_samples=100
    )
else:
    print("DeepExplainer usage not supported for RF/SVM here. Use shap.TreeExplainer or shap.KernelExplainer.")


##################################
# 5) Restliche Funktionen (Counterfactuals, PDP, etc.)
##################################

def generate_counterfactuals(ML_DATA, reduction_factor=0.5):
    """
    Generate counterfactual data by reducing certain solar-related features.
    """
    new_ML_DATA = {}
    solar_feats = ['GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Solar[kW]']
    feature_list = Control_Var['PossibleFeatures']

    solar_indices = [feature_list.index(f) for f in solar_feats if f in feature_list]
    for key in ML_DATA.keys():
        if key.startswith('X_'):
            arr = ML_DATA[key].copy()
            arr[..., solar_indices] *= reduction_factor
            new_ML_DATA[key] = arr
        else:
            new_ML_DATA[key] = (ML_DATA[key].copy() 
                                if isinstance(ML_DATA[key], np.ndarray) else ML_DATA[key])
    return new_ML_DATA

def generate_counterfactuals_targeted(ML_DATA, Control_Var, feature_changes, sample_indices):
    """
    Manipulate only certain samples in X_... sets.
    """
    new_ML_DATA = {}
    feature_list = Control_Var['PossibleFeatures']
    for key, value in ML_DATA.items():
        if key.startswith('X_'):
            arr = value.copy()
            for idx in sample_indices:
                if 0 <= idx < arr.shape[0]:
                    for feat_name, change_val in feature_changes.items():
                        if feat_name in feature_list:
                            feat_idx = feature_list.index(feat_name)
                            if arr.ndim == 3:
                                # CNN shape: (N, PRE, F)
                                # => arr[idx, :, feat_idx]
                                if (isinstance(change_val, (int, float)) 
                                    and 0 < change_val < 2):
                                    arr[idx, :, feat_idx] *= change_val
                                else:
                                    arr[idx, :, feat_idx] += change_val
                            else:
                                # 2D shape: (N, F)
                                if (isinstance(change_val, (int, float)) 
                                    and 0 < change_val < 2):
                                    arr[idx, feat_idx] *= change_val
                                else:
                                    arr[idx, feat_idx] += change_val
                else:
                    print(f"Index {idx} outside range!")
            new_ML_DATA[key] = arr
        else:
            new_ML_DATA[key] = (value.copy() 
                                if isinstance(value, np.ndarray) else value)
    return new_ML_DATA

print("Done!")
