import shap
import numpy as np
import random
import lime
import lime.lime_tabular
import quantus
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt

from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process

def lime_predict(input_data, X_train, model):
    """
    Reshape LIME's 2D input back to the expected 3D format.
    """
    input_reshaped = input_data.reshape((input_data.shape[0], X_train.shape[1], X_train.shape[2]))
    return model.predict(input_reshaped)




def get_explanations_2D(model, ML_DATA,X_test_3D, feature_names, background_samples=100, Control_Var=None, idx_remove=None):
    """
    Obtain SHAP explanations for Keras (CNN, LSTM) or scikit-learn (RF, SVM) models.

    - CNN/LSTM  => SHAP GradientExplainer (Deep SHAP)
    - RF        => SHAP TreeExplainer
    - SVM       => SHAP KernelExplainer (approximate, slower)

    Parameters:
    -----------
    model : trained ML model
        Keras (CNN/LSTM) or scikit-learn (RF/SVM) model
    X_test_3D : np.array
        Test dataset (3D for CNN/LSTM, 2D for RF/SVM)
    feature_names : list of str
        Feature names (length = #Features)
    background_samples : int, optional
        Number of samples for SHAP background data (default: 100)
    Control_Var : dict, optional
        Dictionary containing 'MLtype' key (default: None)
    idx_remove : int, optional
        Index of a feature to remove before SHAP analysis (default: None)

    Returns:
    --------
    shap_values : list or np.array
        Raw SHAP values from the chosen explainer
    """

    if Control_Var is None:
        raise ValueError("Control_Var dictionary is required to identify the model type!")

    if Control_Var['MLtype'] in ['CNN', 'LSTM', 'CNN_LSTM']:
        print("Using GradientExplainer for CNN/LSTM models...")

        # 1️⃣ **Select Background Data**
        background_samples = min(background_samples, X_test_3D.shape[0])
        background_indices = random.sample(range(X_test_3D.shape[0]), background_samples)
        background_data = X_test_3D[background_indices]

        print("Feature Names from X_TRAIN:", ML_DATA["xcols"])
        print("Feature Names Used for SHAP:", feature_names)
        print("X_TRAIN.shape:", ML_DATA["X_TRAIN"].shape)

        # 2️⃣ **Initialize SHAP GradientExplainer**
        explainer = shap.GradientExplainer(model, background_data)

        # 3️⃣ **Compute SHAP Values**
        shap_values = explainer.shap_values(X_test_3D)
        shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values

        print("Raw SHAP shape:", shap_array.shape)

        # 4️⃣ **Optional Feature Removal**
        if idx_remove is not None:
            print(f"Removing feature at index {idx_remove}...")
            shap_array = np.delete(shap_array, idx_remove, axis=2)
            X_test_3D = np.delete(X_test_3D, idx_remove, axis=2)
           # feature_names.pop(idx_remove) #sometimes remove, depends on model SVM yes, other no

        # 5️⃣ **SHAP Aggregation**
        if shap_array.ndim == 4:
            shap_2D = shap_array.mean(axis=(1, 3))  # Aggregate over PRE and H
            print("Detected 4D SHAP -> Aggregated via axis=(1,3). Final shape:", shap_2D.shape)
        elif shap_array.ndim == 3:
            shap_2D = shap_array.mean(axis=1)  # Aggregate over PRE
            print("Detected 3D SHAP -> Aggregated via axis=1. Final shape:", shap_2D.shape)
        else:
            raise ValueError(f"Unexpected SHAP array dimension: {shap_array.shape}")

        # 6️⃣ **Aggregate X_test over PRE to match SHAP**
        X_test_agg = X_test_3D.mean(axis=1)

    elif Control_Var['MLtype'] == 'RF':
        print("Using TreeExplainer for Random Forest...")

        # 1️⃣ **Initialize SHAP TreeExplainer**
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_3D)

        # 2️⃣ **Handle Multi-output**
        if isinstance(shap_values, list):
            shap_2D = np.mean(np.array(shap_values), axis=0)  # Average across outputs
        else:
            shap_2D = shap_values

        X_test_agg = X_test_3D  # RF does not need PRE aggregation

    elif Control_Var['MLtype'] == 'SVM':
        print("Using KernelExplainer for SVM (may be slow)...")

        # 1️⃣ **Select a subset of data for KernelExplainer**
        background_samples = min(background_samples, X_test_3D.shape[0])
        background_indices = random.sample(range(X_test_3D.shape[0]), background_samples)
        background_data = X_test_3D[background_indices]

        # 2️⃣ **Initialize SHAP KernelExplainer**
        explainer = shap.KernelExplainer(model.predict, background_data)
        shap_values = explainer.shap_values(X_test_3D[:50])  # Limit samples for speed

        shap_2D = shap_values  # KernelExplainer outputs (N, Features)
        X_test_agg = X_test_3D[:50]  # Reduce to match SHAP computation

    else:
        raise ValueError("Unsupported ML model. Supported: CNN, LSTM, RF, SVM.")

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


"""
##################################
# 7) Evaluate Explanations with Quantus
if is_keras_model:
    # Aggregation of SHAP values
    a_batch = shap_2D  # a_batch has initially the shape (samples, time_steps, features, horizon)
    
        # Aggregate over the horizon dimension (last dimension, which contains 10 values)
    a_batch = a_batch.mean(axis=-1)  # (samples, time_steps, features)
    print(f"Shape of a_batch after horizon aggregation: {a_batch.shape}")
        
        # Aggregate over the time axis (if needed)
    #a_batch = a_batch.mean(axis=1)  # (samples, features)
    #print(f"Shape of a_batch after time aggregation: {a_batch.shape}")

    # Check that the shape of a_batch is now (samples, features)
    assert a_batch.shape == (145, 18), f"Shape mismatch: {a_batch.shape} vs (145, 18)"

    # x_batch_2D as aggregated input (samples, features)
    x_batch_2D = X_test_agg  # (samples, features)

    # Assuming y_batch is the ground truth values from 'Y_TEST'
    y_test_single = ML_DATA['Y_TEST'][:, 0]  # Example: Just the first horizon

    # Ensure subset_size is smaller than the number of samples
    num_samples = X_test_agg.shape[0]  # Number of samples in X_test_agg
    print(num_samples)
    # Set a valid subset_size (smaller than num_samples)
    subset_size = min(1, num_samples)  # Example: Use a subset of 100 samples, or the entire dataset if it's smaller

    # Now, pass subset_size to the Faithfulness metric, ensuring it's valid
    faithfulness = quantus.FaithfulnessCorrelation(subset_size=subset_size)
    # Metric calculation: Faithfulness
    #faithfulness = quantus.FaithfulnessCorrelation()
    
    faithfulness_score = faithfulness(
        model=model,
        x_batch=x_batch_2D,
        y_batch=y_test_single,
        a_batch=a_batch,
        device='cpu'  # For Keras model, CPU or GPU
    )
    print("Faithfulness Score:", faithfulness_score)

    # Another metric: SensitivityN
    sensitivityN = quantus.SensitivityN()
    sensitivity_score = sensitivityN(
        model=model,
        x_batch=x_batch_2D,
        y_batch=y_test_single,
        a_batch=a_batch,
        device='cpu'
    )
    print("SensitivityN Score:", sensitivity_score)
else:
    print("Skipping Quantus metrics for non-Keras model types (RF/SVM).")

    """


def generate_and_plot_single_counterfactual(
    ML_DATA, model, feature_names,
    feature, change_factor,
    Control_Var,
    idx_remove=None,
    is_keras_model=True
    
):
    """
    Erzeugt Counterfactuals für ein einzelnes Feature mit einem bestimmten Faktor
    und speichert einen Vergleichsplot.
    """
    def generate_counterfactuals(data, feature_list, feature_to_change, factor, Control_Var):
        new_data = {}
        if feature_to_change not in feature_list:
            raise ValueError(f"Feature '{feature_to_change}' not found in feature list.")

        index = feature_list.index(feature_to_change)
        for key in data:
            if key.startswith('X_'):
                arr = data[key].copy()
                arr[..., index] = arr[..., index] * factor
                new_data[key] = arr
            else:
                new_data[key] = data[key].copy() if isinstance(data[key], np.ndarray) else data[key]
        return new_data

    counterfactual_data = generate_counterfactuals(ML_DATA, feature_names, feature, change_factor,Control_Var)

    if idx_remove is not None:
        ML_DATA['X_TEST'][..., idx_remove] = 0
        counterfactual_data['X_TEST'][..., idx_remove] = 0

    if is_keras_model:
        original_preds = model.predict(ML_DATA['X_TEST'])
        counterfactual_preds = model.predict(counterfactual_data['X_TEST'])

        plt.figure(figsize=(10, 5))
        plt.plot(original_preds.mean(axis=1), label='Original Predictions', alpha=0.7)
        label = f"{'increased' if change_factor > 1.0 else 'reduced'} by {int(abs((change_factor - 1) * 100))}%"
        plt.plot(counterfactual_preds.mean(axis=1), label=f'{feature} {label}', alpha=0.7)
        plt.xlabel('Test Samples')
        plt.ylabel('Power Prediction (kW)')
        plt.title(f'Original vs Counterfactual: {feature} {label}')
        plt.legend()

        model_name = Control_Var['MLtype']
        safe_feature = feature.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "")
        direction = 'plus' if change_factor > 1 else 'minus'
        pct = int(abs((change_factor - 1.0) * 100))
        file_name = f'{model_name}_counterfactual_{safe_feature}_{direction}{pct}pct.png'
        plt.savefig(file_name)
        plt.close()
        print(f"Saved plot: {file_name}")
    else:
        print(f"Skipping prediction for {feature} (non-Keras model).")
