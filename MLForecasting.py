# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Forces CPU usage if needed
#from sklearn.externals import joblib

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


##################################
# 1) Control The Script
###################################
Control_Var = {
    '_description_' : 'Holds variables that define the behaviour of the algorithm.',
    'resolution' : '60min', # 1sec, 1min, 5min, or 60min
    'SOLETE_builvsimport': 'Build', # 'Build' to expand the dataset, 'Import' to load an existing expansion
    'SOLETE_save': False, # if True and 'Build', saves the expanded SOLETE dataset
    'trainVSimport' : False, # True to train ML model, False to import a saved model
    'saveMLmodel' : False, # if True and trainVSimport is True, saves the trained model
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
    'PRE' : 5,             # Number of previous samples used as input -- total PRE + 1
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
#performance mit pombo vergleichen
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
p_solar_col_idx = feature_names.index('P_Solar[kW]')
print(f"Feature 'P_Solar[kW]' found at index: {p_solar_col_idx}")
# Ensure data has correct shape
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Number of features:", len(feature_names))
print("Number of Features Expected by LIME:", X_train.shape[2])
print("Number of Feature Names:", len(feature_names))

BG_PATH = "bg_indices.npy"
BG_PATH_pos = "bg_indices_pos.npy"
MLtype = Control_Var['MLtype']

Yscaler = joblib.load(f"{MLtype}/Yscaler.pkl")

Xscaler = joblib.load(f"{MLtype}/Xscaler.pkl")
num_bg_samples = 100
X_test = ML_DATA['X_TEST']
Y_test = ML_DATA['Y_TEST']

if os.path.exists(BG_PATH):
    print(f"✅ Lade bestehende Hintergrund-Indizes aus '{BG_PATH}'")
    bg_indices = np.load(BG_PATH)
else:
    print(f"📦 Erstelle neue Hintergrund-Indizes (samples: {num_bg_samples}) und speichere sie unter '{BG_PATH}'")
    random.seed(42)
    bg_indices = random.sample(range(X_test.shape[0]), num_bg_samples)
    np.save(BG_PATH, bg_indices)

import os, numpy as np
import tensorflow as tf

# Für TF2 im Kompatibilitätsmodus:
tf.compat.v1.disable_eager_execution()

# Alle Zwischen-Ausgaben in while-Schleifen freigeben
tf.compat.v1.experimental.output_all_intermediates(True)

# ------------------------------------------------------------------
# Ordner‐ & Dateinamen automatisch aus dem Modell ableiten
model_name   = MLtype

idx_dir      = os.path.join(".", model_name, "bg_indices")   
os.makedirs(idx_dir, exist_ok=True)
BG_PATH_pos  = os.path.join(idx_dir, "top100_pred_h9.npy")   # pro Modell einzigartig
# ------------------------------------------------------------------

num_samples  = 100
h_out        = 9
t_in         = 5
p_solar_col  = 7

# ------------------------------------------------------------------
# Vorhersage einmal berechnen
y_pred_scaled = model.predict(X_test)
if y_pred_scaled.ndim == 1:
    y_pred_scaled = y_pred_scaled[:, None]
y_pred_h9 = Yscaler.inverse_transform(y_pred_scaled)[:, h_out]

# ------------------------------------------------------------------
# 1) Laden, falls bereits vorhanden
if os.path.exists(BG_PATH_pos):
    print(f"✅ Lade Hintergrund-Indizes aus '{BG_PATH_pos}'")
    bg_indices_pos = np.load(BG_PATH_pos)

    print("min / max y_pred(H9) der geladenen Indizes:",
          y_pred_h9[bg_indices_pos].min(), y_pred_h9[bg_indices_pos].max())

# ------------------------------------------------------------------
# 2) Neu erzeugen, falls Datei fehlt
else:
    print(f"📦 Erstelle Top-{num_samples}-Indizes für '{model_name}' "
          f"und speichere unter '{BG_PATH_pos}'")

    # Top-100 nach Modell-Vorhersage
    top_idx = np.argsort(y_pred_h9)[-num_samples:][::-1]
    bg_indices_pos = top_idx.copy()

    # Debug-Ausgabe (erste 10 Zeilen)
    print("Rang Sample  Input_t5[kW]  y_pred_h9[kW]  y_true_h9[kW]")
    for rank, idx in enumerate(bg_indices_pos[:10], 1):
        x5_scaled = X_test[idx, t_in, p_solar_col]
        x5_kW     = (x5_scaled - Xscaler.min_[p_solar_col]) / Xscaler.scale_[p_solar_col]
        y_true_h9 = Yscaler.inverse_transform(Y_test[idx, h_out, 0].reshape(1,-1))[0,0]
        print(f"{rank:>4} {idx:>6}   {x5_kW:>7.3f}        {y_pred_h9[idx]:>7.3f}       {y_true_h9:>7.3f}")

    # Speichern
    np.save(BG_PATH_pos, bg_indices_pos)
    print(f"✅ Hintergrund-Indizes gespeichert unter '{BG_PATH_pos}'")

# ------------------------------------------------------------------
# Ab hier können Sie bg_indices_pos an cf_scatter_percentP_ übergeben:
# cf_scatter_percentP_(..., bg_idx=bg_indices_pos, ...)


sample_indices = bg_indices[:30] 

idx_file = 'selected_indices.txt'
if not os.path.exists(idx_file):
    raise FileNotFoundError(f"Datei '{idx_file}' nicht gefunden")

with open(idx_file, 'r') as f:
    selected_indices = [int(line.strip()) for line in f if line.strip().isdigit()]
"""
print(f"Verwende folgende sample_id(s): {selected_indices}")
# 2) Für jede index eine Counterfactual-Tabelle erzeugen
all_cfs = {}
horizons = 10                      # Zahl der Forecast-Schritte

for idx in selected_indices:
    print(f"\n=== Generiere CFs für sample_id = {idx} ===")

for h in range(horizons):
    cf_df = run_counterfactuals(
        model,
        ML_DATA,
        Control_Var,
        sample_id=idx,
        total_CFs=2,             # oder anpassen
        desired_range=(4.0, 7.0) # oder anpassen
    )
    all_cfs[idx] = cf_df
    cf_preds  = model.predict(cf_df[feature_cols]).reshape(len(cf_df), horizons)
    cf_df[f"new_pred_t{h+1}"] = cf_preds[:, h]

    pred_cols = [f"new_pred_t{h+1}" for h in range(horizons)]


# 3) Optional: Ergebnisse speichern
import pandas as pd
with pd.ExcelWriter('counterfactuals_batch.xlsx') as writer:
    for idx, df in all_cfs.items():
        sheet_name = f"CF_{idx}"
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Fertig: Alle CFs in 'counterfactuals_batch.xlsx' gespeichert.")
delta_matrix, feat_ranking = analyze_counterfactuals(
    cf_df, X_test, Xscaler,
    sample_id=0,
    feature_names=ML_DATA["xcols"],
    PRE=Control_Var["PRE"]
)

shap_vals = get_explanations_2D(model, ML_DATA, X_test, feature_names, Control_Var=Control_Var, horizon_steps=list(range(10)))
"""
# ──────────────────────────────────────────────────────────────
# 0) Parameter
# ──────────────────────────────────────────────────────────────
desired_range_orig = (6.0, 7.0)   # Zielwert-Intervall (kW)
n_cf_per_sample    = 2             # zwei Counterfactuals je Sample
target_horizon     = 9             # 0-basiert: Position t+10

# ──────────────────────────────────────────────────────────────
# 1) Initialisierung
# ──────────────────────────────────────────────────────────────
all_cfs = {}          # Counterfactual-Ergebnisse pro Sample
rows    = []          # Übersicht: Original- & CF-Zielwerte
max_cfs = 0           # größte CF-Anzahl pro Sample

# ------------------------------------------------------------------
# 2) Counterfactuals generieren und sammeln
# ------------------------------------------------------------------
for sample_id in bg_indices_pos[:1]:
    print(f"\n=== Counterfactuals für sample_id = {sample_id} ===")

    res = compute_ts_counterfactual_dice(
        model         = model,
        ML_DATA       = ML_DATA,
        feature_names = feature_names,
        idx           = sample_id,
        total_CFs     = n_cf_per_sample,
        desired_range = desired_range_orig,
        horizon       = target_horizon,
        method        = "random",
        x_scaler      = Xscaler,
        y_scaler      = Yscaler
    )
    all_cfs[sample_id] = res

# ------------------------------------------------------------------
# 3) Übersichtstabellen (Zielwerte)
# ------------------------------------------------------------------
for sid, info in all_cfs.items():
    cf_vals = info["y_cfs"]
    max_cfs = max(max_cfs, len(cf_vals))

    row = {"sample_id": sid, "orig_t10": info["y_orig"]}
    for i, v in enumerate(cf_vals, 1):
        row[f"CF{i}_t10"] = v
    rows.append(row)

df_all = pd.DataFrame(rows)
ordered_cols = ["sample_id", "orig_t10"] + [f"CF{i}_t10" for i in range(1, max_cfs + 1)]
df_all = df_all.reindex(columns=ordered_cols)

# Skalierte Zielwerte
df_scaled_rows = []
for sid, info in all_cfs.items():
    row = {"sample_id": sid, "orig_t10_scaled": info["y_orig_scaled"]}
    for i, v in enumerate(info["y_cfs_scaled"], 1):
        row[f"CF{i}_t10_scaled"] = v
    df_scaled_rows.append(row)

df_scaled = pd.DataFrame(df_scaled_rows)
ordered_cols_scaled = ["sample_id", "orig_t10_scaled"] + [f"CF{i}_t10_scaled" for i in range(1, max_cfs + 1)]
df_scaled = df_scaled.reindex(columns=ordered_cols_scaled)

# ------------------------------------------------------------------
# 4) Zielpfad vorbereiten
# ------------------------------------------------------------------
from pathlib import Path
file_path = Path(model_name) / "data" / "cf_summary.xlsx"
file_path.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 5) Alle Tabellen in **eine** Excel-Datei schreiben
# ------------------------------------------------------------------
with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
    # 5-a) Unskalierte Zielwerte
    df_all.to_excel(writer, sheet_name="CF_unscaled", index=False)

    # 5-b) Skalierte Zielwerte
    df_scaled.to_excel(writer, sheet_name="CF_scaled", index=False)

    # 5-c) CF-Input (flattened)
    cf_feat_rows = []
    for sid, info in all_cfs.items():
        T, F      = info["cf_examples_unscaled"][0].shape
        col_names = [f"{fn}_{t+1}" for t in range(T) for fn in feature_names]

        for cf_id, cf_arr in enumerate(info["cf_examples_unscaled"], 1):
            row = dict(zip(col_names, cf_arr.reshape(-1)))
            row.update({"sample_id": sid, "cf_id": cf_id})
            cf_feat_rows.append(row)

    pd.DataFrame(cf_feat_rows).to_excel(
        writer, sheet_name="CF_examples_unscaled", index=False
    )

    # 5-d) Original-Input + CF-Input + jeweilige Ausgabe
    io_rows = []
    for sid, info in all_cfs.items():
        T, F      = info["cf_examples_unscaled"][0].shape
        col_names = [f"{fn}_{t+1}" for t in range(T) for fn in feature_names]

        # Original
        orig_seq = ML_DATA["X_TEST"][sid]                            # skaliert
        if Xscaler is not None:
            orig_seq = Xscaler.inverse_transform(
                orig_seq.reshape(-1, F)
            ).reshape(T, F)
        row_orig = dict(zip(col_names, orig_seq.reshape(-1)))
        row_orig.update({"sample_id": sid, "cf_id": 0, "y_output": info["y_orig"]})
        io_rows.append(row_orig)

        # Counterfactuals
        for cf_id, (cf_arr, ycf) in enumerate(
            zip(info["cf_examples_unscaled"], info["y_cfs"]), 1
        ):
            row_cf = dict(zip(col_names, cf_arr.reshape(-1)))
            row_cf.update({"sample_id": sid, "cf_id": cf_id, "y_output": ycf})
            io_rows.append(row_cf)

    pd.DataFrame(io_rows).to_excel(
        writer, sheet_name="Input_Output", index=False
    )

    # 5-e) (optional) Top-3-Tabelle beibehalten
    if "top3_df" in globals():
        top3_df.to_excel(writer, sheet_name="Top3_First_CF", index=False)

print(f"✅ Excel gespeichert unter: {file_path}")

"""

for idx in selected_indices:
    print(f"\n=== Generiere CFs für sample_id = {idx} ===")

    res = compute_ts_counterfactual_dice(
        model         = model,
        ML_DATA       = ML_DATA,
        feature_names = feature_names,
        idx           = idx,
        total_CFs     = 2,
        desired_range = desired_range_orig,
        method        = "random",
        x_scaler      = Xscaler,   
        y_scaler      = Yscaler    
    )

    print(f"Vorheriger Vorhersagewert (t+1): {res['y_orig']:.3f}")
    for i, ycf in enumerate(res['y_cfs'], 1):
        print(f"CF{i}-Vorhersage (t+1):        {ycf:.3f}")

    all_cfs[idx] = res['cf_examples_unscaled']



# Beispielaufruf:
result = compute_ts_counterfactual(
    model=model,
    ML_DATA={'X_TEST': X_test},
    feature_names=feature_names,
    feature='GHI[kW1m2]',
    idx=0,
    y_target=0.5,
    norm='l1',
    per_timestep=False,
    bounds=(0, None),
    max_iter=100
)
x_cf, y_orig, y_cf = result['x_cf'], result['y_orig'], result['y_cf']

X = ML_DATA['X_TEST'] if bg_indices is None else ML_DATA['X_TEST'][bg_indices]
feat_idx = feature_names.index(feature_names[0])  # Index of the feature to be modified
y_target = 0.5  # Target value for the feature
norm = 0.1  # Normalization factor for the feature

    # Compute CF for each sample
X_cf = np.array([
    generate_ts_counterfactual(x, model, feat_idx, y_target, norm)
    for x in X
    ])
y_orig = model.predict(X).ravel()
y_cf   = model.predict(X_cf).ravel()
    # Plot differences
plt.figure(figsize=(6,4))
plt.scatter(np.arange(len(y_orig)), y_orig, label='Original')
plt.scatter(np.arange(len(y_cf)), y_cf, label='Counterfactual')
plt.legend()
plt.xlabel('Sample index')
plt.ylabel('Prediction')
plt.title(f'Counterfactual to target {y_target}')
plt.grid(True)
plt.show()


X_train_re = X_train.reshape(X_train.shape[0], -1)
X_test_re = X_test.reshape(X_test.shape[0], -1)
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
    ml_type=ml_type)
selected_ids = generate_lime_explanations(
    model=model,
    X_train=X_train,
    X_test=X_test,
    feature_names=feature_names,
    ml_type=ml_type,
    horizon_step=9,  
    input_time_step=5
)

for feature in feature_names:

    cf_scatter_percent_zufällig(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        factors=(0.5, 0.75, 1.25, 1.5),
        Control_Var=Control_Var,
        bg_idx=bg_indices,
        jitter=0.3,
        x_scaler      = Xscaler,
        y_scaler      = Yscaler,
        debug=True

    )

    cf_scatter_percent_zufällig(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        factors=(0.5, 0.75, 1.25, 1.5),
        Control_Var=Control_Var,
        bg_idx=bg_indices,       
        jitter=0.3,
        timestep_idx_forecast=9,
        timestep_idx_input=5,
        x_scaler= Xscaler,
        y_scaler= Yscaler,
        aggregate_input_timesteps=False,
        aggregate_output_timesteps=False,
        debug=True
    )
    cf_scatter_percent_max(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        factors=(0.5, 0.75, 1.25, 1.5),
        Control_Var=Control_Var,
        bg_idx=bg_indices_pos,
        #bg_idx=bg_indices_pos,  # Verwenden Sie bg_indices_pos für Top-100 Indizes       
        jitter=0.3,
        x_scaler      = Xscaler,
        y_scaler      = Yscaler,
        debug=True

    )

    cf_scatter_percent_max(
        ML_DATA=ML_DATA,
        model=model,
        feature_names=feature_names,
        feature=feature,
        factors=(0.5, 0.75, 1.25, 1.5),
        Control_Var=Control_Var,
        bg_idx=bg_indices_pos,       
        jitter=0.3,
        timestep_idx_forecast=9,
        timestep_idx_input=5,
        x_scaler= Xscaler,
        y_scaler= Yscaler,
        aggregate_input_timesteps=False,
        aggregate_output_timesteps=False,
        debug=True
    )
    save_combined_pdp_ice_all_inputs_horizon_output(
        model, ML_DATA, feature_names, feature, Control_Var, Yscaler,
        scaler_x=Xscaler,  
        mode="aggregate"
    )

    save_combined_pdp_ice_all_inputs_horizon_output(
        model, ML_DATA, feature_names, feature, Control_Var, Yscaler,
        scaler_x=Xscaler,
        mode="single",
        timestep=5
    )


    save_combined_pdp_ice_all_inputs_horizon_output(
        model            = model,
        ML_DATA          = ML_DATA,
        feature_names    = feature_names,
        feature          = feature,
        Control_Var      = Control_Var,
        scaler_y         = Yscaler,
        scaler_x         = Xscaler,
        mode             = "aggregate",
        num_horizon_steps= 10,            
        aggregate_output = True           
    )
"""