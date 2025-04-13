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
    '_description_' : 'Holds variables that define the behaviour of the algorithm.',
    'resolution' : '60min', # 1sec, 1min, 5min, or 60min
    'SOLETE_builvsimport': 'Build', # 'Build' to expand the dataset, 'Import' to load an existing expansion
    'SOLETE_save': True, # if True and 'Build', saves the expanded SOLETE dataset
    'trainVSimport' : True, # True to train ML model, False to import a saved model
    'saveMLmodel' : True, # if True and trainVSimport is True, saves the trained model
    'Train_Val_Test' : [70, 20, 10], # Train-Validation-Test division in percentages
    'Scaler': 'MinMax01', # 'MinMax01', 'MinMax11', or 'Standard'
    'IntrinsicFeature' : 'P_Solar[kW]', # feature to be predicted
    'PossibleFeatures': [
        'TEMPERATURE[degC]', 'HUMIDITY[%]', 'WIND_SPEED[m1s]', 'WIND_DIR[deg]',
        'GHI[kW1m2]', 'POA Irr[kW1m2]', 'P_Gaia[kW]', 
        'P_Solar[kW]', #comment out for rf and svm
        'Pressure[mbar]', 'Pac', 'Pdc', 'TempModule', 'TempCell',
        #'TempModule_RP',  # Typically commented out due to heavy computation
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
        'epo_num' : 145,# - epoc number of iterations of each batch - same reasoning as for the batches'
        'Neurons' : [15,15,15], #number of neurons per layer <-> you can feed up to three layers using list e.g. [15, 10] makes two layers of 15 and 10 neurons, respectively.
        'Dense'  : [0, 0], #number of dense layers and neurons in them. If left as 0 they are not created.
        'ActFun' : 'tanh', #sigmoid, tanh, elu, relu - activation function as a str 
        'LossFun' : 'mean_absolute_error', #mean_absolute_error or mean_squared_error
        'Optimizer' : 'adam' # adam RMSProp - optimization method adam is the default of the guild 
        }

CNN = {'_description_' : 'Holds the values related to LSTM NN design',
        'n_batch' : 16, #see note in LSTM
        'epo_num' : 3, #see note in LSTM
        'filters' : 32, #number of nodes per layer, usually top layers have higher values
        'kernel_size' : 2, #size of the filter used to extract features
        'pool_size' : 3, #down sampling feature maps in order to gain robustness to changes
        'Dense'  : [10, 10],#see note in LSTM
        'ActFun' : 'tanh', #see note in LSTM
        'LossFun' : 'mean_absolute_error', #see note in LSTM
        'Optimizer' : 'adam' #see note in LSTM
        }

CNN_LSTM = {'_description_' : 'Holds the values related to LSTM NN design',
        'n_batch' : 16, #see note in LSTM
        'epo_num' : 232, #see note in LSTM        
        'filters' : 32, #see note in CNN
        'kernel_size' : 3, #see note in CNN
        'pool_size' : 2, #see note in CNN
        'Dense'  : [0, 0], #see note in LSTM
        'CNNActFun' : 'tanh', #see note in CNN
        
        'Neurons' : [10,15,10], #see note in LSTM
        'LSTMActFun' : 'sigmoid', #see note in LSTM
        
        'LossFun' : 'mean_absolute_error', #see note in LSTM
        'Optimizer' : 'adam' #see note in LSTM
        }

# Store all in Control_Var
Control_Var['RF'] = RF
Control_Var['SVM'] = SVM
Control_Var['LSTM'] = LSTM
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

# Einmalig die Indizes berechnen (z.B. in einem Setup-Skript)
random.seed(42)
bg_indices = random.sample(range(X_test.shape[0]), 100)

# Extract IntrinsicFeature in our case always P_Solar[kW]
idx_remove = Control_Var['IntrinsicFeature']  

# The intrinsic feature should be removed from the complete feature list nothing happens here apart from index
#if idx_remove in Control_Var['PossibleFeatures']:
#    idx_remove = Control_Var['PossibleFeatures'].index(idx_remove)  
#else:
#    idx_remove = None 
# 

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
    ml_type="CNN",
    lime_predict_fn=lime_predict
)

for feature in feature_names:
    print(f"\n📊 Plotting ICE for feature: {feature}")
    save_combined_pdp_ice_all_timesteps(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature=feature, Control_Var=Control_Var)


"""
kurz auskommntirt
plot_combined_pdp_ice(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature='GHI[kW1m2]',timestep=5)

plot_ice_timeseries_feature(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature='GHI[kW1m2]',time_index=5)  # z. B. Mittag


save_pdp_plots_to_pdf(model=model,ML_DATA=ML_DATA,feature_names=feature_names,features_to_plot=feature_names,Control_Var=Control_Var)

plot_pdp_keras_model(model=model,ML_DATA=ML_DATA,feature_names=feature_names,feature='GHI[kW1m2]')
"""



#if Control_Var['IntrinsicFeature'] in feature_names:
 #   feature_names.remove(Control_Var['IntrinsicFeature'])
 #Step 4: Debugging Outputs
#print("Updated feature_names:", feature_names)
# ----------------------------------------------
#is_keras_model = (ml_type in ['LSTM', 'CNN', 'CNN_LSTM'])

#if is_keras_model:
    # Rufen Sie die obige Funktion auf

shap_vals = get_explanations_2D(model, ML_DATA, ML_DATA['X_TEST'], feature_names=feature_names, background_samples=100, Control_Var=Control_Var, bg_indices=bg_indices)

    
#else:
 #   print("SHAP DeepExplainer usage is not directly supported for non-Keras models.\n"
  #        "Use shap.TreeExplainer or shap.KernelExplainer for e.g. RF/SVM.")


##################################
# 8) Example Counterfactual Generation
##################################
def generate_counterfactuals(ML_DATA, feature_list, features_to_be_reduced, reduction_factor=0.5):
    """
    Generate counterfactual data by reducing certain solar-related features.
    """
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

"""
if idx_remove is not None:
    ML_DATA['X_TEST'][..., idx_remove] = 0  # Set feature to zero instead of deleting
    counterfactual_data['X_TEST'][..., idx_remove] = 0  # Same for counterfactual data
    X_test = ML_DATA['X_TEST'][..., idx_remove] = 0 
    X_Test = ML_DATA['X_TEST'][..., idx_remove] = 0 

    X_test = ML_DATA['X_TEST'].copy()
    X_test[..., idx_remove] = 0

    X_Test = ML_DATA['X_TEST'].copy()
    X_test[..., idx_remove] = 0
"""

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

"""
# Make predictions on original vs. counterfactual if desired:
if is_keras_model:
    original_preds = model.predict(ML_DATA['X_TEST'])
    counterfactual_preds = model.predict(counterfactual_data['X_TEST'])

    # Plot comparison for demonstration
    plt.figure(figsize=(10,5))
    plt.plot(original_preds.mean(axis=1), label='Original Predictions (mean across horizon)', alpha=0.7)
    plt.plot(counterfactual_preds.mean(axis=1), label='Counterfactual Predictions (reduced sunlight)', alpha=0.7)
    plt.xlabel('Test Samples')
    plt.ylabel('Power Prediction (kW)')
    plt.title('Original vs Counterfactual Predictions')
    plt.legend()
    plt.show()
else:
    print("Skipping counterfactual predictions for non-Keras model (RF/SVM) in this demo.")

print("Done!")
"""
def generate_counterfactuals_targeted(ML_DATA, Control_Var, feature_changes, sample_indices):
    """
    Erzeugt Counterfactual-Daten, indem ausgewählte Merkmale (Features) nur
    bei bestimmten Samples im Test- bzw. Validierungsdatensatz manipuliert werden.
    
    Parameters
    ----------
    ML_DATA : dict
        Ihr bereits vorbereitetes Datenwörterbuch mit Schlüsseln wie 'X_TEST', 'X_VAL' usw.
        Enthält die (samples, PRE+1, features)-Arrays für LSTM/CNN-Modelle bzw.
        (samples, features) für RF/SVM (angepasst an Ihr Vorgehen).
    Control_Var : dict
        Enthält Informationen wie 'PossibleFeatures' etc.
    feature_changes : dict
        Key: Name des Features (z. B. 'TEMPERATURE[degC]', 'GHI[kW1m2]').
        Value: Änderung, entweder als Multiplikationsfaktor (z. B. 0.8 für -20%)
               oder als additiver Wert (z. B. +2.0 für +2 Grad).
        Beispiele:
            {'TEMPERATURE[degC]': +5.0, 'GHI[kW1m2]': 0.5}
    sample_indices : list
        Liste der Sample-Indizes, an denen wir die Features manipulieren wollen.
        (Z. B. [10, 15, 100, 101])

    Returns
    -------
    new_ML_DATA : dict
        Kopie von ML_DATA, in dem lediglich an den angegebenen sample_indices
        die gewünschten feature_changes ausgeführt wurden.
    
    Hinweise
    --------
    - Achten Sie darauf, dass diese Funktion nur dann sauber läuft, wenn das ML_DATA-Format
      dem in Ihrem Skript entspricht. Für LSTM/CNN-Batches ist meist die Form
      (N, PRE+1, F) nötig, für RF/SVM ggf. (N, F).
    - Passen Sie ggf. den Zugriff auf die letzte Achse an, falls das Feature-Array
      anders geordnet ist.
    """
    # Kopie anlegen, damit das Original nicht verändert wird
    new_ML_DATA = {}
    
    # Liste aller Feature-Namen
    feature_list = feature_names
    
    # Wir greifen nur auf Eingabe-Arrays (X_...) zu,
    # da Ausgänge (Y_...) in der Regel unverändert bleiben
    for key, value in ML_DATA.items():
        if key.startswith('X_'):
            arr = value.copy()  # sichert, dass das Original unberührt bleibt
            
            # Wir prüfen die Dimension. Bei LSTM/CNN:
            # arr.shape == (samples, PRE+1, #features)
            # Bei RF/SVM: arr.shape == (samples, #features)
            # Die Logik unten geht davon aus, dass Achse=-1 die Feature-Achse ist:
            #   arr[..., feat_idx]
            # Wenn Ihr Array anders strukturiert ist, bitte entsprechend anpassen!
            
            for idx in sample_indices:
                # Safety-Check: Index im zulässigen Bereich?
                if 0 <= idx < arr.shape[0]:
                    for feat_name, change_val in feature_changes.items():
                        if feat_name in feature_list:
                            feat_idx = feature_list.index(feat_name)
                            
                            # Änderung festlegen
                            # 1) Multiplikationsfaktor, wenn 0 < change_val < ~2
                            if isinstance(change_val, (float, int)) and 0 < change_val < 2:
                                arr[idx, :, feat_idx] *= change_val
                            else:
                                # 2) Sonst addieren wir den Wert
                                arr[idx, :, feat_idx] += change_val
                else:
                    # Optional: Warnung ausgeben, wenn Index zu groß/negativ ist
                    print(f"Achtung: sample_indices={idx} liegt außerhalb des zulässigen Bereichs.")
            
            new_ML_DATA[key] = arr
        else:
            # Für alle nicht-X_-Schlüssel (z. B. Y_TEST) nur kopieren, unmodifiziert
            if isinstance(value, np.ndarray):
                new_ML_DATA[key] = value.copy()
            else:
                new_ML_DATA[key] = value
    return new_ML_DATA

# Beispiel: Wir wollen bei einigen Zeitpunkten (z.B. Indizes 10 und 15) 
# die Temperatur um +5 Grad anheben und die GHI auf 80% reduzieren:
feature_changes = {
    'TEMPERATURE[degC]': 5.0,  # => +5 Grad
    'GHI[kW1m2]': 0.8         # => -20%
}
sample_indices = [10, 15]  # nur diese Samples werden manipuliert

#counterfactual_data = generate_counterfactuals_targeted(ML_DATA, Control_Var,feature_changes,sample_indices)


MLtype = Control_Var['MLtype']
is_keras = MLtype in ['LSTM', 'CNN', 'CNN_LSTM']

# Originale Vorhersage
if is_keras:
    original_preds = model.predict(ML_DATA['X_TEST'])
else:
    # Bei RF/SVM müssen Sie das Array 2D flatten, z.B. (N, PRE*features) oder so ähnlich
    X_test_2D = ML_DATA['X_TEST'].reshape((ML_DATA['X_TEST'].shape[0], -1))
    original_preds = model.predict(X_test_2D)

# Counterfactual-Vorhersage
if is_keras:
    cf_preds = model.predict(counterfactual_data['X_TEST'])
else:
    X_test_cf_2D = counterfactual_data['X_TEST'].reshape((counterfactual_data['X_TEST'].shape[0], -1))
    cf_preds = model.predict(X_test_cf_2D)

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))

# Beispiel: Wir nehmen den Mittelwert über den Vorhersagehorizont (falls H > 1)
orig_mean = original_preds.mean(axis=1) if original_preds.ndim == 2 else original_preds
cf_mean = cf_preds.mean(axis=1) if cf_preds.ndim == 2 else cf_preds

plt.plot(orig_mean, label='Originalvorhersage', alpha=0.7)
plt.plot(cf_mean, label='Counterfactual (manipuliert)', alpha=0.7)

plt.xlabel('Test-Sample')
plt.ylabel('P_Solar Prognose')
plt.title('Vergleich Original vs. Counterfactual Prediction')
plt.legend()
plt.show()



def analyze_counterfactuals(
    original_preds, 
    cf_preds, 
    manipulated_idx=None, 
    observed=None, 
    focus_range=None,
    title="Vergleich Original vs. Counterfactual Prediction"
):
    """
    Stellt Original- und Counterfactual-Prognosen dar, hebt manipulierte Indizes hervor
    und erlaubt die Gegenüberstellung mit echten Messwerten (observed).
    
    Parameter
    ---------
    original_preds : np.ndarray
        Array mit den Vorhersagen des Modells ohne Manipulation
        (z.B. shape (N,) oder (N,H). Falls (N,H), wird mittlerer Wert gebildet).
    cf_preds : np.ndarray
        Array mit den Counterfactual-Vorhersagen
        (gleiche Dimensionierung wie original_preds).
    manipulated_idx : list or np.ndarray, optional
        Indizes, an denen wirklich manipuliert wurde. Diese Punkte werden hervorgehoben.
        Standard: None -> kein Hervorheben.
    observed : np.ndarray, optional
        Echte Messwerte (falls verfügbar) zum Vergleich. Gleiche Dimension wie preds.
        Standard: None -> keine Observed-Linie.
    focus_range : tuple, optional
        (start, end) zur Beschränkung des Plots auf einen Teil des Datensatzes.
        Beispiel: (0, 50) -> Zeige nur Samples 0 bis 50.
        Standard: None -> zeige gesamten Bereich.
    title : str, optional
        Plot-Titel. Standard: "Vergleich Original vs. Counterfactual Prediction".
    
    Hinweise
    --------
    - Wenn original_preds und cf_preds mehrdimensional sind (z.B. (N,H)),
      wird zur Darstellung jeweils der Mittelwert über Achse 1 gebildet.
    - Sie können die "manipulated_idx" explizit angeben, damit die Abweichung
      nur dort sichtbar markiert wird.
    - Mit "observed" können Sie die realen Messwerte plotten, um zu sehen,
      wie groß die Abweichung zum Ground Truth ist.
    - Mit "focus_range" beschränken Sie den Plot auf einen Teilbereich.
    """
    
    # Sicherstellen, dass beide Arrays gleichartig sind
    if original_preds.ndim > 1:
        # z.B. (samples, horizon) -> Mittelwert über horizon bilden
        orig_mean = original_preds.mean(axis=1)
    else:
        orig_mean = original_preds
        
    if cf_preds.ndim > 1:
        cf_mean = cf_preds.mean(axis=1)
    else:
        cf_mean = cf_preds
    
    # Echte Messwerte ebenfalls mitteln, falls nötig
    if observed is not None:
        if observed.ndim > 1:
            obs_mean = observed.mean(axis=1)
        else:
            obs_mean = observed
    else:
        obs_mean = None
    
    # Fokus auf Teilbereich
    n_samples = len(orig_mean)
    if focus_range is not None:
        start, end = focus_range
        start = max(0, start)
        end = min(n_samples, end)
    else:
        start, end = 0, n_samples
    
    # Plot: Original vs. Counterfactual
    plt.figure(figsize=(10, 5))
    plt.title(title)
    
    x_axis = np.arange(n_samples)
    
    # Geschnittener Bereich
    x_plot = x_axis[start:end]
    orig_plot = orig_mean[start:end]
    cf_plot = cf_mean[start:end]
    
    plt.plot(x_plot, orig_plot, label='Originalvorhersage', alpha=0.7)
    plt.plot(x_plot, cf_plot, label='Counterfactual (manipuliert)', alpha=0.7)
    
    # Markierung der manipulierten Punkte
    if manipulated_idx is not None:
        # Nur Punkte markieren, die im focus_range liegen
        manipulated_idx_in_range = [idx for idx in manipulated_idx if start <= idx < end]
        
        if len(manipulated_idx_in_range) > 0:
            # Original
            plt.scatter(
                manipulated_idx_in_range,
                orig_mean[manipulated_idx_in_range],
                color='red',
                s=50,
                zorder=5,
                label='Manipulierte (Original)'
            )
            # Counterfactual
            plt.scatter(
                manipulated_idx_in_range,
                cf_mean[manipulated_idx_in_range],
                color='green',
                s=50,
                zorder=6,
                label='Manipulierte (CF)'
            )
    
    # Observed (falls vorhanden)
    if obs_mean is not None:
        obs_plot = obs_mean[start:end]
        plt.plot(x_plot, obs_plot, '--', color='black', label='Observed')
    
    plt.xlabel('Test-Sample')
    plt.ylabel('P_Solar Prognose')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Zweiter Plot: Differenz
    delta = cf_mean - orig_mean
    delta_plot = delta[start:end]
    
    plt.figure(figsize=(10, 3))
    plt.title("Abweichung (CF - Original)")
    plt.plot(x_plot, delta_plot, label='Delta (CF - Original)')
    plt.axhline(y=0, color='k', linestyle='--')
    if manipulated_idx is not None:
        manipulated_idx_in_range = [idx for idx in manipulated_idx if start <= idx < end]
        if len(manipulated_idx_in_range) > 0:
            plt.scatter(manipulated_idx_in_range, delta[manipulated_idx_in_range],
                        color='red', s=50, zorder=5, label='Manipulierte Delta')
    plt.xlabel('Test-Sample')
    plt.ylabel('Abweichung')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Optionale Rückgabe: delta-Werte
    return delta

# Originalvorhersage
MLtype = Control_Var['MLtype']
is_keras = MLtype in ['LSTM', 'CNN', 'CNN_LSTM']



def custom_partial_dependence(
    model,
    X,
    feature_indices,
    grid_resolution=50,
    sample_fraction=0.3,
    agg_function='mean'
):
    """
    Berechnet eine einfache Partielle Abhängigkeit (PDP) für beliebige Modelle,
    indem ein oder mehrere Features systematisch von min -> max durchlaufen werden.
    
    Parameter
    ---------
    model : object
        Ihr trainiertes Modell mit einer Methode predict(X_2D).
    X : np.ndarray
        Ausgangsdaten in 2D-Form [Samples, Features].
        Für Keras/CNN müssen Sie vorher selbst reshape übernehmen.
    feature_indices : list
        Liste mit einem oder mehreren Feature-Indizes, z. B. [0] oder [0, 1].
        - Geben Sie eine einzelne Zahl an, erhalten Sie eine 1D-PDP.
        - Geben Sie zwei Zahlen als [0, 1] an, können Sie eine 2D-PDP (Interaktion) berechnen.
    grid_resolution : int, optional
        Wie viele Stützstellen werden pro Feature gebildet? Standard 50.
    sample_fraction : float, optional
        Anteil (0..1) der Datensätze, die Sie für die Berechnung verwenden.
        Bei großen Daten kann man so beschleunigen.
    agg_function : str, optional
        'mean' oder 'median' – wie werden die Vorhersagen über die Proben gemittelt?

    Returns
    -------
    dict
        Enthält:
         - 'values'   : Liste von Arrays/Koordinaten für die Feature-Grids
         - 'pd_values': N-dimensionales Array mit den PDP-Werten
                        1D-Fall -> shape (grid_resolution,)
                        2D-Fall -> shape (grid_resolution, grid_resolution)
         - 'features' : die Feature-Indizes
         """

    # 1) Subset der Daten
    N = X.shape[0]
    n_samples = int(N * sample_fraction)
    if n_samples < 1:
        n_samples = 1
    idx = np.random.choice(np.arange(N), size=n_samples, replace=False)
    X_sub = X[idx, :].copy()
    
    # Eruieren, ob 1D- oder 2D-PDP
    if len(feature_indices) == 1:
        # Ein einzelnes Feature
        f_idx = feature_indices[0]
        feat_min, feat_max = X_sub[:, f_idx].min(), X_sub[:, f_idx].max()
        grid_vals = np.linspace(feat_min, feat_max, grid_resolution)

        pd_vals = []
        for val in grid_vals:
            # Kopie anlegen
            X_temp = X_sub.copy()
            X_temp[:, f_idx] = val
            preds = model.predict(X_temp)
            
            # Falls Ausgabe 2D, auf 1D reduzieren
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds.ravel()
            elif preds.ndim == 2 and preds.shape[1] > 1:
                # Multi-Output: Beispielhaft nur 0. Spalte
                preds = preds[:, 0]
            
            if agg_function == 'mean':
                pd_vals.append(preds.mean())
            else:
                pd_vals.append(np.median(preds))
        
        pd_vals = np.array(pd_vals)
        return {
            'values': [grid_vals],
            'pd_values': pd_vals,
            'features': feature_indices
        }
    
    elif len(feature_indices) == 2:
        # Zwei Features -> 2D-Gitter
        f1, f2 = feature_indices
        feat1_min, feat1_max = X_sub[:, f1].min(), X_sub[:, f1].max()
        feat2_min, feat2_max = X_sub[:, f2].min(), X_sub[:, f2].max()

        grid1 = np.linspace(feat1_min, feat1_max, grid_resolution)
        grid2 = np.linspace(feat2_min, feat2_max, grid_resolution)

        pd_vals_2d = np.zeros((grid_resolution, grid_resolution), dtype=float)
        
        for i, val1 in enumerate(grid1):
            for j, val2 in enumerate(grid2):
                X_temp = X_sub.copy()
                X_temp[:, f1] = val1
                X_temp[:, f2] = val2
                preds = model.predict(X_temp)
                
                if preds.ndim == 2 and preds.shape[1] == 1:
                    preds = preds.ravel()
                elif preds.ndim == 2 and preds.shape[1] > 1:
                    preds = preds[:, 0]
                
                if agg_function == 'mean':
                    pd_vals_2d[i, j] = preds.mean()
                else:
                    pd_vals_2d[i, j] = np.median(preds)
        
        return {
            'values': [grid1, grid2],
            'pd_values': pd_vals_2d,
            'features': feature_indices
        }
    else:
        raise ValueError("custom_partial_dependence demo unterstützt derzeit nur 1 oder 2 Features.")
if isinstance(X_test, list):
    print(" x test is converted to array")
    X_test = np.array(X_test)  
# === 1D-PDP für ein bestimmtes Feature (z.B. Feature 0) ===
pdp_result = custom_partial_dependence(
    model,
    X_test,
    feature_indices=[0],
    grid_resolution=40,
    sample_fraction=0.2
)
grid_vals = pdp_result['values'][0]
pd_vals = pdp_result['pd_values']

# Plot:
import matplotlib.pyplot as plt
plt.figure(figsize=(6,4))
plt.plot(grid_vals, pd_vals, label='PDP Feature #0')
plt.xlabel("Feature #0 Wert (z.B. Einstrahlung)")
plt.ylabel("Mittlere Modellvorhersage")
plt.title("1D Partial Dependence")
plt.grid(True)
plt.legend()
plt.show()


# === 2D-PDP für zwei Features (z.B. Feature 0 & 1) ===
pdp2d_result = custom_partial_dependence(
    model,
    X_test,
    feature_indices=[0,4],  # Indizes für z.B. (GHI, TEMPERATURE)
    grid_resolution=20,
    sample_fraction=0.2
)
grid1, grid2 = pdp2d_result['values']
pd_vals_2d = pdp2d_result['pd_values']

# 2D-Darstellung als Heatmap
plt.figure(figsize=(6,5))
plt.imshow(
    pd_vals_2d,
    origin='lower',
    extent=(grid2[0], grid2[-1], grid1[0], grid1[-1]),
    aspect='auto',
    cmap='viridis'
)
plt.colorbar(label='Mittlere Modellvorhersage')
plt.xlabel("Feature #1")
plt.ylabel("Feature #0")
plt.title("2D Partial Dependence Plot")
plt.show()


##################################################################################################################
# bin nicht mehr sicher was ich hier machen wollte
import shap
import random

#X_test_3D = ML_DATA['X_TEST'][..., idx_remove] = 0 
import shap
import numpy as np
import random

def explain_model(Control_Var, model, X_test, feature_names, idx_remove=None, background_samples=100):
    """
    Explain ML models (CNN, LSTM, RF, SVM) using SHAP.
    
    - CNN, LSTM, CNN_LSTM => Uses SHAP GradientExplainer
    - RF                  => Uses SHAP TreeExplainer
    - SVM                 => Uses SHAP KernelExplainer (approximate, slower)

    Parameters:
    -----------
    Control_Var : dict
        Dictionary containing the ML model type ('MLtype')
    model : trained ML model
        Keras (CNN/LSTM), RandomForest, or SVM model
    X_test : np.array
        Test dataset (3D for CNN/LSTM, 2D for RF/SVM)
    feature_names : list of str
        Feature names
    idx_remove : int, optional
        Index of a feature to remove (default: None)
    background_samples : int, optional
        Number of samples for SHAP background data (default: 100)
    """

    if Control_Var['MLtype'] in ['CNN', 'LSTM', 'CNN_LSTM']:
        print("Using GradientExplainer for CNN/LSTM...")

        # 1️⃣ **Background Data Selection**
        background_samples = min(background_samples, X_test.shape[0])
        background_indices = random.sample(range(X_test.shape[0]), background_samples)
        background_data = X_test[background_indices]

        # 2️⃣ **Initialize SHAP GradientExplainer**
        explainer = shap.GradientExplainer(model, background_data)
        
        # 3️⃣ **Compute SHAP Values**
        shap_values = explainer.shap_values(X_test)
        shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values

        print("Raw SHAP shape:", shap_array.shape)

        # 4️⃣ **Feature Removal (Optional)**
        if idx_remove is not None:
            print(f"Removing feature at index {idx_remove}...")
            shap_array = np.delete(shap_array, idx_remove, axis=2)
            X_test = np.delete(X_test, idx_remove, axis=2)
            #feature_names.pop(idx_remove) #adapt depending on model

        # 5️⃣ **SHAP Aggregation**
        if shap_array.ndim == 4:
            shap_2D = shap_array.mean(axis=(1, 3))  # Average over PRE and H
        elif shap_array.ndim == 3:
            shap_2D = shap_array.mean(axis=1)  # Average over PRE
        else:
            raise ValueError(f"Unexpected SHAP array dimension: {shap_array.shape}")

        print("Final SHAP shape:", shap_2D.shape)

        # 6️⃣ **Aggregate X_test to match SHAP (over PRE)**
        X_test_agg = X_test.mean(axis=1)

    elif Control_Var['MLtype'] == 'RF':
        print("Using TreeExplainer for Random Forest...")

        # 1️⃣ **Initialize SHAP TreeExplainer**
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # 2️⃣ **Handle Multi-output**
        if isinstance(shap_values, list):
            shap_2D = np.mean(np.array(shap_values), axis=0)  # Average across outputs
        else:
            shap_2D = shap_values

        X_test_agg = X_test  # RF does not need PRE aggregation

    elif Control_Var['MLtype'] == 'SVM':
        print("Using KernelExplainer for SVM (may be slow)...")

        # 1️⃣ **Select a subset of data for KernelExplainer**
        background_samples = min(background_samples, X_test.shape[0])
        background_indices = random.sample(range(X_test.shape[0]), background_samples)
        background_data = X_test[background_indices]

        # 2️⃣ **Initialize SHAP KernelExplainer**
        explainer = shap.KernelExplainer(model.predict, background_data)
        shap_values = explainer.shap_values(X_test[:50])  # Limit samples for speed

        shap_2D = shap_values  # KernelExplainer outputs (N, Features)
        X_test_agg = X_test[:50]  # Reduce to match SHAP computation

    else:
        print("Unsupported ML model. Supported: CNN, LSTM, RF, SVM.")
        return None

    # 7️⃣ **Final Shape Check**
    if shap_2D.shape[0] != X_test_agg.shape[0]:
        raise ValueError(
            f"Shape mismatch: SHAP ({shap_2D.shape[0]}) != X_test ({X_test_agg.shape[0]})!"
        )
    if shap_2D.shape[1] != len(feature_names):
        raise ValueError(
            f"Shape mismatch: SHAP ({shap_2D.shape[1]}) != Features ({len(feature_names)})!"
        )

    # 8️⃣ **SHAP Summary Plot**
    shap.summary_plot(
        shap_values=shap_2D,
        features=X_test_agg,
        feature_names=feature_names,
        plot_type='dot'
    )

    return shap_values

shap_values = explain_model(Control_Var, model, X_test, feature_names, idx_remove)







def print_feature_indices(feature_names):
    """
    Print the index and corresponding feature name.

    Parameters:
    ----------
    feature_names : list
        List of feature names from the dataset.
    """
    print("\nFeature Index Mapping:")
    for idx, feature in enumerate(feature_names):
        print(f"Index {idx}: {feature}")

# Call function
print_feature_indices(feature_names)

# %%
