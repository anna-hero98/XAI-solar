# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces CPU usage if needed

from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process
from counterfactualMethods import *
import numpy as np
import random
import shap
import lime
import joblib

import lime.lime_tabular
import quantus
from lime.lime_tabular import LimeTabularExplainer
from functools import partial
#import timeshap.explainer as tse
#import timeshap.plot as tsp



# TensorFlow/Keras imports
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Conv1D, MaxPooling1D, Dense, Flatten
)

# Matplotlib for plotting
import matplotlib.pyplot as plt

# Example: your local module containing custom functions
# from Functions import (import_SOLETE_data, import_PV_WT_data, 
#                       PreProcessDataset, PrepareMLmodel, 
#                       TestMLmodel, post_process)

##################################
# 1) Control The Script
###################################
Control_Var = {
    '_description_' : 'Holds vari   ables that define the behaviour of the algorithm.',
    'resolution' : '60min', # 1sec, 1min, 5min, or 60min
    'SOLETE_builvsimport': 'Import', # 'Build' to expand the dataset, 'Import' to load an existing expansion
    'SOLETE_save': True, # if True and 'Build', saves the expanded SOLETE dataset
    'trainVSimport' : False, # True to train ML model, False to import a saved model
    'saveMLmodel' : True, # if True and trainVSimport is True, saves the trained model
    'Train_Val_Test' : [70, 20, 10], # Train-Validation-Test division in percentages
    'Scaler': 'MinMax01', # 'MinMax01', 'MinMax11', or 'Standard'
    'IntrinsicFeature' : 'P_Solar[kW]', # feature to be predicted
    'PossibleFeatures': [
        'TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]', 'WIND_DIR[deg]',
        'GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Gaia[kW]', 
        'P_Solar[kW]', #comment out for rf and svm
        'Pressure[mbar]', 'Pac', 'Pdc', 'TempModule', 'TempCell',
        'TempModule_RP',  # Typically commented out due to heavy computation
        'HoursOfDay', 'MeanPrevH', 'StdPrevH',
        'MeanWindSpeedPrevH', 'StdWindSpeedPrevH'
    ],
    'MLtype' : 'CNN',      # One of: 'RF', 'SVM', 'LSTM', 'CNN', or 'CNN_LSTM'
    'H' : 10,              # Forecast horizon in number of samples
    'PRE' : 5,             # Number of previous samples used as input
}

##################################
# 2) Machine Learning Configuration & Hyperparameters
##################################
RF = {
    '_description_' : 'Random Forest parameters',
    'n_trees' : 1,
    'random_state' : 32,
}

SVM = {
    '_description_' : 'SVM parameters',
    'kernel' : 'rbf',
    'degree' : 3,
    'gamma' : 'scale',
    'coef0' : 0,
    'C' : 3,
    'epsilon' : 0.1,
}

LSTM_params = {'_description_' : 'Holds the values related to LSTM ANN design',
        'n_batch' : 16, #int <-> # number of samples fed together - helps with paralelization  (smaller takes longer, improves performance carefull with overfitting)
        'epo_num' : 65,# - epoc number of iterations of each batch - same reasoning as for the batches'
        'Neurons' : [15,15,15], #number of neurons per layer <-> you can feed up to three layers using list e.g. [15, 10] makes two layers of 15 and 10 neurons, respectively.
        'Dense'  : [32, 16], #number of dense layers and neurons in them. If left as 0 they are not created.
        'ActFun' : 'tanh', #sigmoid, tanh, elu, relu - activation function as a str 
        'LossFun' : 'mean_absolute_error', #mean_absolute_error or mean_squared_error
        'Optimizer' : 'adam' # adam RMSProp - optimization method adam is the default of the guild 
        }

CNN = {'_description_' : 'Holds the values related to LSTM NN design',
        'n_batch' : 16, #see note in LSTM
        'epo_num' : 60, #see note in LSTM adapted
        'filters' : 32, #number of nodes per layer, usually top layers have higher values
        'kernel_size' : 3, #size of the filter used to extract features adapted
        'pool_size' : 2, #down sampling feature maps in order to gain robustness to changes
        'Dense'  : [64, 32],#see note in LSTM
        'ActFun' : 'tanh', #see note in LSTM
        'LossFun' : 'mean_absolute_error', #see note in LSTM
        'Optimizer' : 'adam' #see note in LSTM
        }

CNN_LSTM = {'_description_' : 'Holds the values related to LSTM NN design',
        'n_batch' : 16, #see note in LSTM
        'epo_num' : 100, #see note in LSTM        
        'filters' : 32, #see note in CNN
        'kernel_size' : 3, #see note in CNN
        'pool_size' : 2, #see note in CNN
        'Dense'  : [32, 16], #see note in LSTM
        'CNNActFun' : 'tanh', #see note in CNN
        
        'Neurons' : [15,15,15], #see note in LSTM
        'LSTMActFun' : 'tanh', #see note in LSTM
        
        'LossFun' : 'mean_absolute_error', #see note in LSTM
        'Optimizer' : 'adam' #see note in LSTM
        }

# Store all in Control_Var
Control_Var['RF'] = RF
Control_Var['SVM'] = SVM
Control_Var['LSTM'] = LSTM_params
Control_Var['CNN'] = CNN
Control_Var['CNN_LSTM'] = CNN_LSTM

PVinfo, WTinfo = import_PV_WT_data()
DATA=import_SOLETE_data(Control_Var, PVinfo, WTinfo)

#%% Generate Time Periods
ML_DATA, Scaler = PreProcessDataset(DATA, Control_Var)
# ML_DATA, Scaler = TimePeriods(DATA, Control_Var) 

#%% Train, Evaluate, Test
model = PrepareMLmodel(Control_Var, ML_DATA) #train or import model
results = TestMLmodel(Control_Var, ML_DATA, model, Scaler)

#%% Post-Processing
analysis = post_process(Control_Var, results)


### from here on my code
print("==== Logging ML_DATA  ====")
for key, arr in ML_DATA.items():
    if isinstance(arr, np.ndarray):
        print(f"{key} shape: {arr.shape}")
    else:
        print(f"{key}: (non-array) {arr}")

ml_type = Control_Var['MLtype']
is_keras_model = ml_type in ['LSTM', 'CNN', 'CNN_LSTM']

feature_names = Control_Var['PossibleFeatures'].copy()


print("Basic information dataset")
X_train = ML_DATA["X_TRAIN"]  
X_test = ML_DATA["X_TEST"]    
y_train = ML_DATA["Y_TRAIN"] 
y_test = ML_DATA["Y_TEST"]   
feature_names = ML_DATA["xcols"]  

# Ensure data has correct shape
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Number of features:", len(feature_names))
print("Number of Features Expected by LIME:", X_train.shape[2])
print("Number of Feature Names:", len(feature_names))

BG_PATH = "bg_indices.npy"
num_bg_samples = 100
X_test = ML_DATA['X_TEST']

if os.path.exists(BG_PATH):
    print(f"✅ Lade bestehende Hintergrund-Indizes aus '{BG_PATH}'")
    bg_indices = np.load(BG_PATH)
else:
    print(f"📦 Erstelle neue Hintergrund-Indizes (samples: {num_bg_samples}) und speichere sie unter '{BG_PATH}'")
    random.seed(42)
    bg_indices = random.sample(range(X_test.shape[0]), num_bg_samples)
    np.save(BG_PATH, bg_indices)
# Extract IntrinsicFeature in our case always P_Solar[kW]
idx_remove = Control_Var['IntrinsicFeature']  
MLtype = Control_Var['MLtype']
# The intrinsic feature should be removed from the complete feature list nothing happens here apart from index
#if idx_remove in Control_Var['PossibleFeatures']:
#    idx_remove = Control_Var['PossibleFeatures'].index(idx_remove)  
#else:
#    idx_remove = None 
# 


#shap_vals = get_explanations_2D(model, ML_DATA, ML_DATA['X_TEST'], feature_names=feature_names, background_samples=100, Control_Var=Control_Var, bg_indices=bg_indices)

#S, globals_dict = ffc_explanation(model= model,X            = ML_DATA["X_TEST"],feature_names= ML_DATA["xcols"],ml_type      = Control_Var["MLtype"],n_samples    = 30,random_state = 42,make_plots   = True)
plot_actual_pv_output(
        ML_DATA=ML_DATA,
        Control_Var=Control_Var,
        horizon_index=10,  
        bg_idx=bg_indices
    )

Yscaler = joblib.load(f"{MLtype}/Yscaler.pkl")

for feature in feature_names:

    grid_counterfactual_plots_unscaled_all_timesteps(
        ML_DATA        = ML_DATA,
        model          = model,
        Scaler_y       = Yscaler,         # dein Ziel-Scaler
        feature_names  = feature_names,
        feature        = feature,
        change_factors = [0.5, 0.75, 1.25, 1.50],
        Control_Var    = Control_Var,
        bg_indices     = bg_indices,     # optional: Subset
        horizon_index  = -1,             # letzter Forecast-Schritt
        max_cols       = 2
    )
    grid_cf_unscaled_with_inverse(
        ML_DATA        = ML_DATA,
        model          = model,
        feature_names  = feature_names,
        feature        = feature,
        change_factors = [0.5, 0.75, 1.25, 1.5],
        Control_Var    = Control_Var,
        scaler_y       = Yscaler,        # ← unbedingt übergeben!
        bg_indices     = bg_indices,      # optional
        horizon_index  = -1,              # letzter Forecast-Schritt
        max_cols       = 2
    )
    grid_counterfactual_plots_pct(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        change_factors=[0.5, 0.75, 1.25, 1.5],
        Control_Var=Control_Var,
        bg_indices=bg_indices
    )
    
    """
    grid_cf_unscaled_direct(
        ML_DATA        = ML_DATA,
        model          = model,
        feature_names  = feature_names,
        feature        = feature,
        change_factors = [0.5, 0.75, 1.25, 1.5],
        Control_Var    = Control_Var,
        bg_indices     = bg_indices,     # optional: dein Subset
        horizon_index  = -1,             # letzter Forecast‐Schritt
        max_cols       = 2
    )



    global_input_scaling_sensitivity(
    ML_DATA        = ML_DATA,
    model          = model,
    factors        = [0.50, 0.75, 1.25, 1.50],
    Control_Var    = Control_Var,
    bg_indices     = bg_indices,               # optional: dein Subset
    horizon_index  = Control_Var['H'] - 1      # letzter Forecast-Zeitschritt
)

    grid_counterfactual_plots_all_timesteps(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        change_factors=[0.5, 0.75, 1.25, 1.5],
        Control_Var=Control_Var,
        bg_indices=bg_indices,
        horizon_index=-1,   # Mittelwert über H oder z.B. 0…H-1
        max_cols=2
    )
    """
"""
    cf_scatter_percent(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        factors=(0.5, 0.75, 1.25, 1.5),
        Control_Var=Control_Var,
        timestep=-1,
        bg_idx=bg_indices,       
        jitter=0.3
    )

    scatter_cf_grid(ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        change_factors=[0.5, 0.75, 1.25, 1.5],
        Control_Var=Control_Var,
                    timestep=-1,
                    min_baseline=1e-3,
                    bg_indices=bg_indices,
                    max_cols=2)

"""
"""

train_sax, test_sax = generate_sax_for_dataset(
        ML_DATA=ML_DATA,
        Control_Var=Control_Var,
        n_segments=10,
        alphabet_size=5
    )
    
    # Beispiel: Auswertung der ersten 3 Training-SAX
print("Train-SAX[0..2]:")
for i in range(3):
    print(f"Sample {i}: {train_sax[i]}")

X_train_re = X_train.reshape(X_train.shape[0], -1)
X_test_re = X_test.reshape(X_test.shape[0], -1)

#print("idx_remove:", idx_remove)  

# Initialize LIME Explainer
explainer = LimeTabularExplainer(
    training_data=X_train.reshape(X_train_re.shape[0], -1),  # Flatten only for LIME explainer
    feature_names=[f"{col}_{i}" for col in feature_names for i in range(X_train.shape[1])],  
    mode='regression',
    discretize_continuous=False
)

selected_ids = generate_lime_explanations(
    model=model,
    X_train=X_train,
    X_test=X_test,
    feature_names=feature_names,
    ml_type=ml_type,
    lime_predict_fn=lime_predict
)

for feature in feature_names:
    print(f"\n📊 Plotting ICE for feature: {feature}")
    sample_indices = bg_indices[:30]  # nur die ersten 30 Hintergrund-Indizes verwenden
    save_combined_pdp_ice_all_timesteps(
    model=model,
    ML_DATA=ML_DATA,
    feature_names=feature_names,
    feature=feature,
    Control_Var=Control_Var,
    sample_indices=sample_indices
)


plot_combined_pdp_ice(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature='GHI[kW1m2]',timestep=5)

plot_ice_timeseries_feature(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature='GHI[kW1m2]',time_index=5)  # z. B. Mittag


save_pdp_plots_to_pdf(model=model,ML_DATA=ML_DATA,feature_names=feature_names,features_to_plot=feature_names,Control_Var=Control_Var)

plot_pdp_keras_model(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature='GHI[kW1m2]')




#if Control_Var['IntrinsicFeature'] in feature_names:
 #   feature_names.remove(Control_Var['IntrinsicFeature'])
 #Step 4: Debugging Outputs
#print("Updated feature_names:", feature_names)
# ----------------------------------------------
#is_keras_model = (ml_type in ['LSTM', 'CNN', 'CNN_LSTM'])

#if is_keras_model:
    # Rufen Sie die obige Funktion auf


    
#else:
 #   print("SHAP DeepExplainer usage is not directly supported for non-Keras models.\n"
  #        "Use shap.TreeExplainer or shap.KernelExplainer for e.g. RF/SVM.")

"""
##################################
# 8) Example Counterfactual Generation
##################################
def generate_counterfactuals(ML_DATA, feature_list, features_to_be_reduced, reduction_factor=0.5):
  
    # Features that correspond to sun are adapted
    # here: 'GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Solar[kW]'
    new_ML_DATA = {}

    # Indices of solar features
    solar_indices = [feature_list.index(f) for f in features_to_be_reduced if f in feature_list]

    for key in ML_DATA.keys():
        if key.startswith('X_'):
            # copy array
            arr = ML_DATA[key].copy()
            # reduce solar features
            arr[..., solar_indices] = arr[..., solar_indices] * reduction_factor
            new_ML_DATA[key] = arr
        else:
            new_ML_DATA[key] = ML_DATA[key].copy() if isinstance(ML_DATA[key], np.ndarray) else ML_DATA[key]
    return new_ML_DATA
print("Counterfactuals for Solar decrease")
solar_feats = ['GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Solar[kW]']
# Calls the counterfactual
counterfactual_data = generate_counterfactuals(ML_DATA, feature_names, solar_feats, reduction_factor=0.5)



#has to be set to zero, since otherwise it causes issues with the original logics. That is why it can't be deleted
# Gemeinsamer Counterfactual für alle Solar-Features
solar_feats = ['GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Solar[kW]']

generate_and_plot_single_counterfactual(
    ML_DATA=ML_DATA,
    model=model,
    feature_names=feature_names,
    feature=solar_feats,  
    change_factor=0.5,
    idx_remove=idx_remove,
    is_keras_model=is_keras_model,
    Control_Var=Control_Var,
    bg_indices=bg_indices
)

generate_and_plot_single_counterfactual(
    ML_DATA=ML_DATA,
    model=model,
    feature_names=feature_names,
    feature=solar_feats,  
    change_factor=1.5,
    idx_remove=idx_remove,
    is_keras_model=is_keras_model,
    Control_Var=Control_Var,
    bg_indices=bg_indices
)



# Jetzt alle Features aus feature_names jeweils -50 % und +50 %
for feature in feature_names:
    for factor in [0.5, 1.5]:
        generate_and_plot_single_counterfactual(ML_DATA, model, feature_names, feature, factor, Control_Var, idx_remove=None, is_keras_model=True, bg_indices=bg_indices)


# Apply the counterfactual transformation
counterfactual_data, modified_indices = generate_counterfactuals_highest_values(ML_DATA, column_index=4, increase_factor=1.5, num_samples=100)

MLtype = Control_Var['MLtype']
is_keras = MLtype in ['LSTM', 'CNN', 'CNN_LSTM']

if is_keras:
    original_preds = model.predict(ML_DATA['X_TEST'])
    counterfactual_preds = model.predict(counterfactual_data['X_TEST'])
else:
    X_test_2D = ML_DATA['X_TEST'].reshape((ML_DATA['X_TEST'].shape[0], -1))
    original_preds = model.predict(X_test_2D)

    X_test_cf_2D = counterfactual_data['X_TEST'].reshape((counterfactual_data['X_TEST'].shape[0], -1))
    counterfactual_preds = model.predict(X_test_cf_2D)



# Call the function to generate plots
plot_counterfactual_comparison(original_preds, counterfactual_preds, modified_indices, ControlVar=Control_Var)

S, globals_dict = ffc_explanation(model= model,X            = ML_DATA["X_TEST"],feature_names= ML_DATA["xcols"],ml_type      = Control_Var["MLtype"],n_samples    = 30,random_state = 42,make_plots   = True)


shap_values = explain_model(Control_Var, model, X_test, feature_names, idx_remove)
print_feature_indices(feature_names)
