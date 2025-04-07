import shap
import numpy as np
import random
import lime
import lime.lime_tabular
import quantus
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process

import numpy as np
from scipy.stats import norm

# =============================================================================
# 1) SAX (Symbolic Aggregate approXimation)
# =============================================================================
def sax_transform(ts, n_segments=10, alphabet_size=5):
    """
    Transformiert eine 1D-Zeitreihe in eine symbolische Darstellung mittels SAX.
    
    Parameter
    ----------
    ts : np.ndarray
        1D-Zeitreihe (z. B. als np.array).
    n_segments : int, optional
        Anzahl der Segmente für die Piecewise Aggregate Approximation (Standard: 10).
    alphabet_size : int, optional
        Anzahl der Symbole im Alphabet (Standard: 5).
        
    Returns
    -------
    sax_string : list of int
        Liste von Symbolindizes (0 bis alphabet_size-1), die die Zeitreihe repräsentieren.
    """
    ts = np.array(ts, dtype=float)
    n = len(ts)
    
    # Schritt 1: Piecewise Aggregate Approximation (PAA)
    # Teilen in n_segments gleich lange Abschnitte (ggf. letzte Werte interpolieren)
    segment_size = n / n_segments
    paa = []
    for i in range(n_segments):
        # Bestimmen der Start- und Endindizes (mit Rundung)
        start = int(round(i * segment_size))
        end = int(round((i + 1) * segment_size))
        # Falls end <= start, erweitern
        if end <= start:
            end = start + 1
        segment_mean = ts[start:end].mean()
        paa.append(segment_mean)
    paa = np.array(paa)
    
    # Schritt 2: Bestimmen der Breakpoints anhand der Normalverteilung
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    # Beispiel: für alphabet_size=5 ergibt dies 4 Breakpoints
    
    # Schritt 3: Zuordnen der Symbole: 
    # Jeder PAA-Wert wird anhand der Breakpoints einem Symbol (Index) zugeordnet.
    sax_string = []
    for value in paa:
        # np.searchsorted gibt den Einfügeindex zurück
        symbol = np.searchsorted(breakpoints, value)
        sax_string.append(symbol)
    
    return sax_string

# =============================================================================
# 6) Vereinfachter Shapelet Transform
# =============================================================================
def extract_shapelets(data, num_shapelets=5, shapelet_length=20, random_state=None):
    """
    Extrahiert (stark vereinfacht) zufällige Shapelets aus einer Zeitreihen-Datenmenge.
    Hier wird aus den gegebenen 1D-Zeitreihen (z. B. einzelnes Feature) eine
    festgelegte Anzahl zufälliger Shapelets entnommen.
    
    Parameter
    ----------
    data : np.ndarray
        2D-Array der Form (n_samples, time_length) mit den Zeitreihen.
    num_shapelets : int, optional
        Anzahl der zu extrahierenden Shapelets (Standard: 5).
    shapelet_length : int, optional
        Länge der Shapelets (Standard: 20).
    random_state : int oder None, optional
        Für Reproduzierbarkeit (Standard: None).
    
    Returns
    -------
    shapelets : list of np.ndarray
        Liste der extrahierten Shapelets (jeweils 1D-Arrays).
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples, ts_length = data.shape
    shapelets = []
    
    for _ in range(num_shapelets):
        # Zufällige Auswahl eines Samples und eines Startpunkts
        sample_idx = np.random.randint(0, n_samples)
        max_start = ts_length - shapelet_length
        if max_start < 1:
            raise ValueError("Time series too short for the chosen shapelet length.")
        start_idx = np.random.randint(0, max_start)
        shapelet = data[sample_idx, start_idx:start_idx + shapelet_length].copy()
        shapelets.append(shapelet)
    
    return shapelets

def min_distance(ts, shapelet):
    """
    Berechnet die minimale euklidische Distanz zwischen einem Shapelet und 
    allen möglichen Subsequenzen einer Zeitreihe.
    
    Parameter
    ----------
    ts : np.ndarray
        1D-Zeitreihe.
    shapelet : np.ndarray
        1D-Shapelet (gleiche Länge wie das Subsequence-Fenster).
        
    Returns
    -------
    min_dist : float
        Minimale Distanz zwischen dem Shapelet und einem Subsequence von ts.
    """
    ts_length = len(ts)
    sh_length = len(shapelet)
    if ts_length < sh_length:
        raise ValueError("Time series is shorter than the shapelet length.")
    
    min_dist = np.inf
    # Schiebe-Fenster über die Zeitreihe
    for i in range(ts_length - sh_length + 1):
        subseq = ts[i:i + sh_length]
        dist = np.linalg.norm(subseq - shapelet)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def shapelet_transform(data, shapelets):
    """
    Transformiert eine Menge von Zeitreihen in einen neuen Feature-Raum,
    der durch die minimalen Distanzen zu den gegebenen Shapelets charakterisiert wird.
    
    Parameter
    ----------
    data : np.ndarray
        2D-Array der Form (n_samples, time_length) mit den Zeitreihen.
    shapelets : list of np.ndarray
        Liste der Shapelets.
        
    Returns
    -------
    transformed_data : np.ndarray
        2D-Array der Form (n_samples, n_shapelets), wobei jeder Eintrag 
        die minimale Distanz der Zeitreihe zum jeweiligen Shapelet enthält.
    """
    n_samples = data.shape[0]
    n_shapelets = len(shapelets)
    transformed_data = np.zeros((n_samples, n_shapelets))
    
    for i in range(n_samples):
        ts = data[i]
        for j, shapelet in enumerate(shapelets):
            transformed_data[i, j] = min_distance(ts, shapelet)
    return transformed_data

# =============================================================================
# Beispielhafte Anwendung
# =============================================================================
if __name__ == "__main__":
    # Beispiel: Erzeugen einer simulierten Zeitreihe (z. B. für P_Solar)
    ts_example = np.sin(np.linspace(0, 2 * np.pi, 100))  # 1D-Zeitreihe
    
    # SAX-Transformation
    sax_representation = sax_transform(ts_example, n_segments=10, alphabet_size=5)
    print("SAX-Repräsentation:", sax_representation)
    
    # Simulierte Datensätze (zwei Dimensionen: n_samples x time_length)
    np.random.seed(42)
    simulated_data = np.array([np.sin(np.linspace(0, 2 * np.pi, 100)) + 0.1*np.random.randn(100)
                               for _ in range(50)])
    
    # Extrahiere zufällig 5 Shapelets der Länge 20
    shapelets = extract_shapelets(simulated_data, num_shapelets=5, shapelet_length=20, random_state=42)
    print("Extrahierte Shapelets (Längen):", [len(s) for s in shapelets])
    
    # Shapelet-Transformation: Berechnung der minimalen Distanzen
    transformed = shapelet_transform(simulated_data, shapelets)
    print("Shapelet-transformierte Daten (Form):", transformed.shape)

def lime_predict(input_data, X_train, model):
    """
    Reshape LIME's 2D input back to the expected 3D format.
    """
    input_reshaped = input_data.reshape((input_data.shape[0], X_train.shape[1], X_train.shape[2]))
    return model.predict(input_reshaped)

def get_explanations_2D(model, ML_DATA, X_test_3D, feature_names, background_samples=100, Control_Var=None, idx_remove=None, bg_indices=None):
    """
    Obtain SHAP explanations for Keras (CNN, LSTM) or scikit-learn (RF, SVM) models.
    
    For CNN/LSTM models:
      - Computes SHAP values using GradientExplainer.
      - Aggregates the SHAP values over the time dimension to yield a 2D array (samples x features)
        and creates a summary plot (aggregated over time).
      - Additionally, it creates per-timestep SHAP summary plots (without aggregation) and saves
        them in a single multi-page PDF.
    
    Parameters:
    -----------
    model : trained ML model
        Keras (CNN/LSTM) or scikit-learn (RF, SVM) model.
    X_test_3D : np.array
        Test dataset (3D for CNN/LSTM, shape: (samples, time_steps, features)).
    feature_names : list of str
        Feature names (length must equal the number of features).
    background_samples : int, optional
        Number of samples for SHAP background data (default: 100).
    Control_Var : dict, required
        Dictionary containing 'MLtype' key (and possibly 'PossibleFeatures') among others.
    idx_remove : int, optional
        Index of a feature to remove before SHAP analysis (default: None).
    bg_indices : list of int, optional
        List of background indices to use. If provided, these indices are used instead of generating new ones.
    
    Returns:
    --------
    shap_values : list or np.array
        The raw SHAP values computed by the explainer.
    """
    import random
    import numpy as np
    import shap
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Dummy Outline-Klasse, die set_visible unterstützt
    class DummyOutline:
        def set_visible(self, flag):
            pass

    # Dummy Colorbar-Klasse, die set_ticklabels, set_label, set_alpha sowie ein 'ax'-Attribut und ein 'outline' bereitstellt
    class DummyColorbar:
        def __init__(self):
            self.ax = type("DummyAx", (), {"tick_params": lambda self, **kwargs: None})()
            self.outline = DummyOutline()
        def set_ticklabels(self, *args, **kwargs):
            pass
        def set_label(self, *args, **kwargs):
            pass
        def set_alpha(self, alpha):
            pass

    # Speichern der Originalfunktion
    original_colorbar = plt.colorbar
    # Überschreiben von plt.colorbar, sodass ein Dummy zurückgegeben wird
    plt.colorbar = lambda *args, **kwargs: DummyColorbar()

    if Control_Var is None:
        raise ValueError("Control_Var dictionary is required to identify the model type!")
    
    # --- SHAP-Berechnung und Aggregation ---
    if Control_Var['MLtype'] in ['CNN', 'LSTM', 'CNN_LSTM']:
        print("Using GradientExplainer for CNN/LSTM models...")

        # 1. Select Background Data
        if bg_indices is None:
            # Falls keine bg_indices übergeben werden, generiere sie mit festem Seed
            random.seed(42)
            np.random.seed(42)
            background_samples = min(background_samples, X_test_3D.shape[0])
            bg_indices = random.sample(range(X_test_3D.shape[0]), background_samples)
        print("Using background indices:", bg_indices)
        background_data = X_test_3D[bg_indices]

        print("Feature Names from X_TRAIN:", ML_DATA["xcols"])
        print("Feature Names Used for SHAP:", feature_names)
        print("X_TRAIN.shape:", ML_DATA["X_TRAIN"].shape)

        # 2. Initialize SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background_data)

        # 3. Compute SHAP Values
        shap_values = explainer.shap_values(X_test_3D)
        shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
        print("Raw SHAP shape:", shap_array.shape)

        # 4. Optional: Remove a feature
        if idx_remove is not None:
            print(f"Removing feature at index {idx_remove}...")
            shap_array = np.delete(shap_array, idx_remove, axis=2)
            X_test_3D = np.delete(X_test_3D, idx_remove, axis=2)
            # Optional: feature_names.pop(idx_remove)

        # 5. SHAP Aggregation über alle Zeitpunkte
        if shap_array.ndim == 4:
            shap_2D = shap_array.mean(axis=(1, 3))  # Aggregate over time and forecast horizon
            print("Detected 4D SHAP -> Aggregated via axis=(1,3). Final shape:", shap_2D.shape)
        elif shap_array.ndim == 3:
            shap_2D = shap_array.mean(axis=1)  # Aggregate over time steps
            print("Detected 3D SHAP -> Aggregated via axis=1. Final shape:", shap_2D.shape)
        else:
            raise ValueError(f"Unexpected SHAP array dimension: {shap_array.shape}")

        # 6. Aggregate X_test similarly
        X_test_agg = X_test_3D.mean(axis=1)

    elif Control_Var['MLtype'] == 'RF':
        print("Using TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_3D)
        if isinstance(shap_values, list):
            shap_2D = np.mean(np.array(shap_values), axis=0)
        else:
            shap_2D = shap_values
        X_test_agg = X_test_3D  # RF does not need time aggregation

    elif Control_Var['MLtype'] == 'SVM':
        print("Using KernelExplainer for SVM (may be slow)...")
        background_samples = min(background_samples, X_test_3D.shape[0])
        bg_indices = random.sample(range(X_test_3D.shape[0]), background_samples)
        background_data = X_test_3D[bg_indices]
        explainer = shap.KernelExplainer(model.predict, background_data)
        shap_values = explainer.shap_values(X_test_3D[:50])  # Limit samples for speed
        shap_2D = shap_values
        X_test_agg = X_test_3D[:50]
    else:
        raise ValueError("Unsupported ML model. Supported: CNN, LSTM, RF, SVM.")

    # 7. Final Shape Check
    if shap_2D.shape[0] != X_test_agg.shape[0]:
        raise ValueError(f"Shape mismatch: SHAP ({shap_2D.shape[0]}) != X_test ({X_test_agg.shape[0]})!")
    if shap_2D.shape[1] != len(feature_names):
        raise ValueError(f"Shape mismatch: SHAP ({shap_2D.shape[1]}) != Features ({len(feature_names)})!")

    # --- Aggregated SHAP Summary Plot ---
    fig = plt.figure(figsize=(8,6))
    plt.subplots_adjust(right=0.8)  # Reserve space for the Colorbar
    shap.summary_plot(
        shap_values=shap_2D,
        features=X_test_agg,
        feature_names=feature_names,
        plot_type='dot',
        show=False
    )



    agg_filename = f"{Control_Var['MLtype']}_Shap_Aggregated.png"
    fig.savefig(agg_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Aggregated SHAP summary plot saved as: {agg_filename}")

    # --- Per-Timestep SHAP Summary Plots in einem PDF ---
    per_ts_filename = f"{Control_Var['MLtype']}_Shap_PerTimestep.pdf"
    with PdfPages(per_ts_filename) as pdf:
        num_timesteps = shap_array.shape[1]  # Annahme: shap_array hat Form (samples, time_steps, features)
        for t in range(num_timesteps):
            fig = plt.figure(figsize=(8,6))
            plt.subplots_adjust(right=0.8)
            # Temporäres Überschreiben von plt.colorbar in diesem Kontext
            original_cb = plt.colorbar
            plt.colorbar = lambda *args, **kwargs: DummyColorbar()
            shap.summary_plot(
                shap_values=shap_array[:, t, :],
                features=X_test_3D[:, t, :],
                feature_names=feature_names,
                plot_type='dot',
                show=False
            )
                            # Standard-Titel von shap.summary_plot entfernen (falls vorhanden).
            plt.title("")  

            # Titel als "Supertitle" hinzufügen.
            fig.suptitle(f"SHAP Summary at Time Step {t}", fontsize=14)

            # Mit subplots_adjust oder tight_layout genügend Platz für den Titel schaffen.
            # Variante A:
            plt.subplots_adjust(top=0.85)  # je nach Bedarf anpassen (0.8, 0.9 etc.)
            pdf.savefig(fig)
            plt.close(fig)
            plt.colorbar = original_cb
    print(f"Per-timestep SHAP summary plots saved as: {per_ts_filename}")

    # Restore original plt.colorbar (falls noch nicht geschehen)
    plt.colorbar = original_colorbar

    return shap_values




def get_explanations_2DP_odd(model, ML_DATA, X_test_3D, feature_names, background_samples=100, Control_Var=None, idx_remove=None):
    """
    Obtain SHAP explanations for Keras (CNN, LSTM) or scikit-learn (RF, SVM) models.
    
    For CNN/LSTM models, this function:
      - Computes SHAP values using the GradientExplainer.
      - Aggregates the SHAP values over the time dimension to yield a 2D array (samples x features)
        and creates a summary plot (aggregated over time).
      - Additionally, it generates separate SHAP summary plots per time step and saves these plots
        as a multi-page PDF.
    
    Parameters:
    -----------
    model : trained ML model
        Keras (CNN/LSTM) or scikit-learn (RF, SVM) model.
    X_test_3D : np.array
        Test dataset (3D for CNN/LSTM, shape: (samples, time_steps, features)).
    feature_names : list of str
        Feature names (length must equal the number of features).
    background_samples : int, optional
        Number of samples for SHAP background data (default: 100).
    Control_Var : dict, required
        Dictionary containing 'MLtype' (and possibly 'PossibleFeatures') among others.
    idx_remove : int or str, optional
        Index (or name, which is then converted to index) of a feature to remove before SHAP analysis.
    
    Returns:
    --------
    shap_values : list or np.array
        The raw SHAP values computed by the explainer.
    """


    if Control_Var is None:
        raise ValueError("Control_Var dictionary is required to identify the model type!")
    
    # Falls idx_remove als String übergeben wurde, konvertieren Sie diesen in einen Integer-Index.
    if idx_remove is not None and isinstance(idx_remove, str):
        if idx_remove in Control_Var.get('PossibleFeatures', []):
            idx_remove = Control_Var['PossibleFeatures'].index(idx_remove)
        else:
            idx_remove = None

    if Control_Var['MLtype'] in ['CNN', 'LSTM', 'CNN_LSTM']:
        print("Using GradientExplainer for CNN/LSTM models...")

        # 1️⃣ Select Background Data
        background_samples = min(background_samples, X_test_3D.shape[0])
        background_indices = random.sample(range(X_test_3D.shape[0]), background_samples)
        background_data = X_test_3D[background_indices]

        print("Feature Names from X_TRAIN:", ML_DATA["xcols"])
        print("Feature Names Used for SHAP:", feature_names)
        print("X_TRAIN.shape:", ML_DATA["X_TRAIN"].shape)

        # 2️⃣ Initialize SHAP GradientExplainer
        explainer = shap.GradientExplainer(model, background_data)

        # 3️⃣ Compute SHAP Values
        shap_values = explainer.shap_values(X_test_3D)
        shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
        print("Raw SHAP shape:", shap_array.shape)

        # 4️⃣ Optional: Remove a feature
        if idx_remove is not None:
            print(f"Removing feature at index {idx_remove}...")
            shap_array = np.delete(shap_array, idx_remove, axis=2)
            X_test_3D = np.delete(X_test_3D, idx_remove, axis=2)
            # Optional: Auch feature_names anpassen, falls benötigt.
            feature_names.pop(idx_remove)

        # 5️⃣ Aggregation: Für die globale Übersicht über alle Zeitpunkte
        if shap_array.ndim == 3:
            shap_2D = shap_array.mean(axis=1)  # Aggregation über die Zeitachse
            print("Detected 3D SHAP -> Aggregated via axis=1. Final shape:", shap_2D.shape)
        else:
            raise ValueError(f"Unexpected SHAP array dimension: {shap_array.shape}")

        # 6️⃣ Aggregate X_test similarly to match SHAP aggregation
        X_test_agg = X_test_3D.mean(axis=1)

    elif Control_Var['MLtype'] == 'RF':
        print("Using TreeExplainer for Random Forest...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_3D)
        if isinstance(shap_values, list):
            shap_2D = np.mean(np.array(shap_values), axis=0)
        else:
            shap_2D = shap_values
        X_test_agg = X_test_3D  # No time aggregation needed for RF

    elif Control_Var['MLtype'] == 'SVM':
        print("Using KernelExplainer for SVM (may be slow)...")
        background_samples = min(background_samples, X_test_3D.shape[0])
        background_indices = random.sample(range(X_test_3D.shape[0]), background_samples)
        background_data = X_test_3D[background_indices]
        explainer = shap.KernelExplainer(model.predict, background_data)
        shap_values = explainer.shap_values(X_test_3D[:50])  # Limit samples for speed
        shap_2D = shap_values
        X_test_agg = X_test_3D[:50]
    else:
        raise ValueError("Unsupported ML model. Supported: CNN, LSTM, RF, SVM.")

    # Final shape check
    if shap_2D.shape[0] != X_test_agg.shape[0]:
        raise ValueError(f"Shape mismatch: SHAP ({shap_2D.shape[0]}) != X_test ({X_test_agg.shape[0]})!")
    if shap_2D.shape[1] != len(feature_names):
        raise ValueError(f"Shape mismatch: SHAP ({shap_2D.shape[1]}) != Features ({len(feature_names)})!")
    for i in range(X_test_agg.shape[1]):
        col = X_test_agg[:, i]
        if np.allclose(col, col[0]):
            print(f"Feature {i} ({feature_names[i]}) ist konstant.")
    indices_to_remove = []
    for i in range(X_test_agg.shape[1]):
        col = X_test_agg[:, i]
        if np.allclose(col, col[0]):
            indices_to_remove.append(i)

    if indices_to_remove:
        print("Removing constant features at indices:", indices_to_remove)
        # Entferne die entsprechenden Spalten aus feature_names, X_test_agg und shap_2D
        for idx in sorted(indices_to_remove, reverse=True):
            feature_names.pop(idx)
            X_test_agg = np.delete(X_test_agg, idx, axis=1)
            shap_2D = np.delete(shap_2D, idx, axis=1)
    # 7️⃣ Create aggregated SHAP summary plot (global over time)
        # 7️⃣ Create aggregated SHAP summary plot (global over time)
    # Aggregierter Summary-Plot (global über die Zeit)
# Aggregierter Summary-Plot (global über die Zeit)
    fig, ax = plt.subplots(figsize=(8,6))
# Erzeuge einen Dummy-Scatter, um ein mappable zu erzeugen:
    dummy = ax.scatter([], [], c=[], cmap='viridis')
    plt.subplots_adjust(right=0.8)  # Platz für den Colorbar reservieren
    shap.summary_plot(shap_values=shap_2D, features=X_test_agg, feature_names=feature_names, 
                        plot_type='dot', show=False)
    agg_filename = f"{Control_Var['MLtype']}_Shap_Aggregated.png"
    fig.savefig(agg_filename, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Aggregated SHAP summary plot saved as: {agg_filename}")



    # 8️⃣ Create per-timestep SHAP summary plots and save as multi-page PDF
    per_ts_filename = f"{Control_Var['MLtype']}_Shap_PerTimestep.pdf"
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(per_ts_filename) as pdf:
        num_timesteps = shap_array.shape[1]  # Annahme: shap_array hat Form (samples, time_steps, features)
        for t in range(num_timesteps):
            fig, ax = plt.subplots(figsize=(8,6))
            # Dummy-Scatter erzeugen, um ein mappable zu liefern
            dummy = ax.scatter([], [], c=[], cmap='viridis')
            plt.subplots_adjust(right=0.8)
            # SHAP-Werte und entsprechende Feature-Werte für den Zeitschritt t
            shap_t = shap_array[:, t, :]
            X_t = X_test_3D[:, t, :]
            shap.summary_plot(shap_values=shap_t, features=X_t, feature_names=feature_names, 
                                plot_type='dot', show=False)
            plt.title(f"SHAP Summary at Time Step {t}")
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Per-timestep SHAP summary plots saved as: {per_ts_filename}")


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

def generate_and_plot_single_counterfactual(ML_DATA, model, feature_names, feature, change_factor, Control_Var,
                                              idx_remove=None, is_keras_model=True, bg_indices=None):
    """
    Erzeugt Counterfactuals für ein oder mehrere Features mit einem bestimmten Faktor
    und speichert einen Vergleichsplot.
    
    Hier wird die Funktion so angepasst, dass – falls bg_indices übergeben wird –
    nur das entsprechende Subset der Testdaten verwendet wird.
    
    Parameters:
    ----------
    ML_DATA : dict
        Dictionary mit den Testdaten (z.B. "X_TEST" in der Form (Samples, TimeSteps, Features))
        und den Feature-Namen ("xcols").
    model : trainiertes Modell
        Das verwendete Vorhersagemodell (z.B. ein Keras-Modell).
    feature_names : list of str
        Liste der Feature-Namen.
    feature : str oder list of str
        Name(n) der Features, die verändert werden sollen.
    change_factor : float
        Multiplikationsfaktor für die zu ändernden Features (z.B. 1.5 oder 0.5).
    Control_Var : dict
        Dictionary mit Modellparametern, u.a. unter dem Schlüssel 'MLtype'.
    idx_remove : int, optional
        Index eines Features, das vor der Analyse entfernt werden soll.
    is_keras_model : bool, optional
        Flag, ob es sich um ein Keras-Modell handelt (Standard: True).
    bg_indices : list of int, optional
        Falls vorhanden, wird nur dieser Subset der Testdaten (X_TEST) für die Vorhersagen genutzt.
    
    Returns:
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Wähle für die Vorhersagen das Subset der Testdaten, falls bg_indices übergeben wurde.
    if bg_indices is not None:
        X_test = ML_DATA['X_TEST'][bg_indices]
    else:
        X_test = ML_DATA['X_TEST']
    
    # Generiere Counterfactual-Daten auf Basis der Originaldaten.
    def generate_counterfactuals(data, feature_list, features_to_change, factor, Control_Var):
        new_data = {}
        # Falls nur ein Feature als String übergeben wird, in eine Liste umwandeln.
        if isinstance(features_to_change, str):
            features_to_change = [features_to_change]
        indices = [feature_list.index(f) for f in features_to_change if f in feature_list]
        for key in data:
            if key.startswith('X_'):
                arr = data[key].copy()
                # Hier werden alle Zeitschritte beibehalten – die Änderung erfolgt entlang der Features.
                arr[..., indices] = arr[..., indices] * factor
                new_data[key] = arr
            else:
                new_data[key] = data[key].copy() if isinstance(data[key], np.ndarray) else data[key]
        return new_data

    # Erzeuge für X_TEST das Counterfactual-Subset
    counterfactual_data = generate_counterfactuals({'X_TEST': X_test}, feature_names, feature, change_factor, Control_Var)

    # Modellvorhersagen für Original- und Counterfactual-Daten
    original_preds = model.predict(X_test)
    counterfactual_preds = model.predict(counterfactual_data['X_TEST'])

    # Falls das Modell mehrdimensionale Ausgaben liefert (z.B. Forecast über mehrere Zeitpunkte),
    # wird der Mittelwert über die entsprechende Achse genommen.
    if original_preds.ndim > 1:
        orig_mean = original_preds.mean(axis=1)
    else:
        orig_mean = original_preds
    if counterfactual_preds.ndim > 1:
        cf_mean = counterfactual_preds.mean(axis=1)
    else:
        cf_mean = counterfactual_preds

    # Erstelle den Vergleichsplot
    plt.figure(figsize=(10, 5))
    plt.plot(orig_mean, label='Originale Vorhersage', alpha=0.7)
    label_text = f"{'Erhöht' if change_factor > 1.0 else 'Reduziert'} um {int(abs((change_factor - 1)*100))}%"
    feat_label = " + ".join(feature) if isinstance(feature, list) else feature
    plt.plot(cf_mean, label=f'{feat_label} {label_text}', alpha=0.7)
    plt.xlabel('Test-Samples')
    plt.ylabel('Solarstromenergie-Vorhersage (kW)')
    plt.title(f'Original vs. Counterfactual: {label_text}')
    plt.legend()

    model_name = Control_Var['MLtype']
    safe_name = "_".join([f.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "") 
                           for f in (feature if isinstance(feature, list) else [feature])])
    direction = 'plus' if change_factor > 1 else 'minus'
    pct = int(abs((change_factor - 1)*100))
    file_name = f'{model_name}_counterfactual_combined_{safe_name}_{direction}{pct}pct.png'
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved combined plot: {file_name}")


import numpy as np
import matplotlib.pyplot as plt

def plot_pdp_keras_model(model, ML_DATA, feature_names, feature, value_range=None, is_keras_model=True):
    """
    Erstellt einen 1D-PDP für ein Feature mit einem Keras-Modell.
    """

    if not is_keras_model:
        print("PDP für Nicht-Keras-Modelle hier nicht implementiert.")
        return

    X_test = ML_DATA["X_TEST"].copy()
    feature_index = feature_names.index(feature)

    # Automatischer Wertebereich (optional überschreibbar)
    if value_range is None:
        f_values = X_test[..., feature_index].flatten()
        value_range = np.linspace(np.percentile(f_values, 1), np.percentile(f_values, 99), 50)

    mean_predictions = []

    for val in value_range:
        X_temp = X_test.copy()
        X_temp[..., feature_index] = val  # ersetze überall durch denselben Wert
        preds = model.predict(X_temp)
        mean_pred = preds.mean()
        mean_predictions.append(mean_pred)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(value_range, mean_predictions, label=f'PDP for {feature}')
    plt.xlabel(f"{feature}")
    plt.ylabel("Durchschnittliche Vorhersage (kW)")
    plt.title(f"Partial Dependence Plot – {feature}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_pdp_plots_to_pdf(model, ML_DATA, feature_names, features_to_plot, Control_Var, ):
    """
    Erstellt PDPs für mehrere Features und speichert sie gemeinsam in einem PDF.
    """

    X_test = ML_DATA["X_TEST"].copy()
    model_name = Control_Var['MLtype']
    filename ="{model_name} pdp plot to pdf"
    with PdfPages(filename) as pdf:
        for feature in features_to_plot:
            if feature not in feature_names:
                print(f"Feature '{feature}' nicht in feature_names – wird übersprungen.")
                continue

            idx = feature_names.index(feature)
            f_values = X_test[..., idx].flatten()
            value_range = np.linspace(np.percentile(f_values, 1), np.percentile(f_values, 99), 50)
            mean_preds = []

            for val in value_range:
                X_temp = X_test.copy()
                X_temp[..., idx] = val
                preds = model.predict(X_temp)
                mean_preds.append(preds.mean())

            # Plot erstellen
            plt.figure(figsize=(8, 5))
            plt.plot(value_range, mean_preds, label=f'PDP for {feature}', color='tab:blue')
            plt.xlabel(feature)
            plt.ylabel('Durchschnittliche Vorhersage (kW)')
            plt.title(f'Partial Dependence Plot – {feature}')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()

            # Zum PDF hinzufügen
            pdf.savefig()
            plt.close()
            print(f"PDP for '{feature}' added to PDF.")

    print(f"\n✅ Alle PDPs gespeichert in: {filename}")


import numpy as np
import matplotlib.pyplot as plt

def plot_ice_timeseries_feature(
    model, ML_DATA, feature_names,
    feature, time_index=12, sample_indices=None,
    num_points=30, is_keras_model=True
):
    """
    Erstellt ICE-Plots für einzelne Samples (bei gegebenem Zeitschritt) für ein bestimmtes Feature.
    """

    X_test = ML_DATA["X_TEST"].copy()
    feature_idx = feature_names.index(feature)

    # Sample-Wahl
    if sample_indices is None:
        sample_indices = np.random.choice(X_test.shape[0], size=10, replace=False)

    # Wertebereich (z. B. GHI bei Stunde 12)
    values = X_test[:, time_index, feature_idx]
    vmin, vmax = np.percentile(values, [1, 99])
    value_range = np.linspace(vmin, vmax, num_points)

    plt.figure(figsize=(8, 6))

    for i in sample_indices:
        preds = []
        for val in value_range:
            X_temp = X_test[i:i+1].copy()
            X_temp[0, time_index, feature_idx] = val
            y_pred = model.predict(X_temp, verbose=0)
            preds.append(y_pred.mean())
        plt.plot(value_range, preds, alpha=0.6)

    plt.title(f"ICE-Plot für '{feature}' bei Zeit t={time_index}")
    plt.xlabel(f"{feature} (Zeitpunkt {time_index})")
    plt.ylabel("Modellvorhersage (kW)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt

def plot_combined_pdp_ice(
    model, ML_DATA, feature_names,
    feature, timestep=0, sample_indices=None,
    num_points=30, is_keras_model=True,
    show_legend=False
):
    """
    Combines PDP and ICE in one plot for a given feature and time step.
    PDP = average of ICE curves.
    """

    X_test = ML_DATA["X_TEST"].copy()
    feature_idx = feature_names.index(feature)

    if sample_indices is None:
        sample_indices = np.random.choice(X_test.shape[0], size=20, replace=False)

    # Value range to test
    values = X_test[:, timestep, feature_idx]
    vmin, vmax = np.percentile(values, [1, 99])
    value_range = np.linspace(vmin, vmax, num_points)

    ice_matrix = []  # Will hold shape (num_samples, num_points)

    plt.figure(figsize=(9, 6))

    for i in sample_indices:
        preds = []
        for val in value_range:
            X_temp = X_test[i:i+1].copy()
            X_temp[0, timestep, feature_idx] = val
            pred = model.predict(X_temp, verbose=0).mean()
            preds.append(pred)
        ice_matrix.append(preds)
        plt.plot(value_range, preds, alpha=0.4, linewidth=1, label=f"Sample {i}" if show_legend else None)

    # Add PDP (average ICE)
    ice_matrix = np.array(ice_matrix)
    pdp = ice_matrix.mean(axis=0)
    plt.plot(value_range, pdp, color="black", linewidth=2.5, label="PDP (mean)")

    plt.title(f"Kombiniertes PDP + ICE für '{feature}' und den Zeithorizont t={timestep}")
    plt.xlabel(f"{feature} (während t={timestep})")
    plt.ylabel("Model Prediction (kW)")
    plt.grid(True)
    if show_legend:
        plt.legend()
    plt.tight_layout()
    plt.show()


from matplotlib.backends.backend_pdf import PdfPages

def save_combined_pdp_ice_all_timesteps(
    model, ML_DATA, feature_names,
    feature, Control_Var, num_time_steps=6,
    sample_indices=None, num_points=30, is_keras_model=True, filename=None
):
    """
    Speichert PDP + ICE für alle Zeitschritte eines Features als PDF.
    """

    X_test = ML_DATA["X_TEST"].copy()
    feature_idx = feature_names.index(feature)
    model_name = Control_Var['MLtype']
    if filename is None:
        filename = f"{model_name},PDP_ICE_{feature.replace('[','').replace(']','').replace(' ', '_')}.pdf"

    if sample_indices is None:
        sample_indices = np.random.choice(X_test.shape[0], size=20, replace=False)

    with PdfPages(filename) as pdf:
        for timestep in range(num_time_steps):
            values = X_test[:, timestep, feature_idx]
            vmin, vmax = np.percentile(values, [1, 99])
            value_range = np.linspace(vmin, vmax, num_points)

            ice_matrix = []

            plt.figure(figsize=(9, 6))

            for i in sample_indices:
                preds = []
                for val in value_range:
                    X_temp = X_test[i:i+1].copy()
                    X_temp[0, timestep, feature_idx] = val
                    pred = model.predict(X_temp, verbose=0).mean()
                    preds.append(pred)
                ice_matrix.append(preds)
                plt.plot(value_range, preds, alpha=0.4, linewidth=1)
            for i in sample_indices:
                x_val = X_test[i, timestep, feature_idx]
                y_val = model.predict(X_test[i:i+1])[0]
                print("x_val shape:", np.array(x_val).shape)
                print("y_val shape:", np.array(y_val).shape)

                plt.scatter(x_val, y_val, color='black', s=15, alpha=0.6)
            for idx, i in enumerate(sample_indices):
                x_val = X_test[i, timestep, feature_idx]
                # Finde das y_val aus der ICE-Kurve, die du eben erzeugt hast
                # Nimm den Wert in value_range, der dem Originalwert am nächsten kommt
                closest_index = np.argmin(np.abs(value_range - x_val))
                y_val = ice_matrix[idx][closest_index]
                plt.scatter(x_val, y_val, color='black', s=15, alpha=0.6)
            # PDP (Mittelwert aller ICE-Kurven)
            ice_matrix = np.array(ice_matrix)
            pdp = ice_matrix.mean(axis=0)
            plt.plot(value_range, pdp, color='black', linewidth=2.5, label='PDP (mean)')

            plt.title(f"PDP + ICE für '{feature}' zum Timestep t={timestep}")
            plt.xlabel(f"{feature} (t = {timestep})")
            plt.ylabel("Modelvorhersage Solarenergie (kW)")
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            print(f"Added PDP+ICE plot for timestep {timestep} to PDF.")

    print(f"\n✅ PDF saved as: {filename}")


def generate_counterfactuals_highest_values(ML_DATA, column_index=7, increase_factor=1.5, num_samples=100):
    """
    Generate counterfactuals by increasing the selected feature (column_index)
    by a specified factor for the top N highest values.

    Parameters:
    ----------
    ML_DATA : dict
        Dictionary containing ML input data, e.g., 'X_TEST'.
    column_index : int
        The index of the feature column to modify.
    increase_factor : float
        Factor by which to increase the feature values (default: 1.5 = +50%).
    num_samples : int
        Number of highest values to modify (default: 100).

    Returns:
    --------
    counterfactual_data : dict
        New dataset with modified feature values.
    modified_indices : list
        Indices of modified samples for tracking.
    """

    X_test = ML_DATA['X_TEST'].copy()  # Copy to avoid modifying the original data

    # Identify top N highest values in the selected column
    top_indices = np.argsort(X_test[:, column_index])[-num_samples:]

    # Modify the selected feature
    X_test[top_indices, column_index] *= increase_factor  # Increase by specified factor

    # Create a new dataset dictionary
    counterfactual_data = ML_DATA.copy()
    counterfactual_data['X_TEST'] = X_test  # Replace only the test data

    return counterfactual_data, top_indices

def plot_counterfactual_comparison(original_preds, counterfactual_preds, modified_indices, ControlVar):
    """
    Plot the difference between original and counterfactual predictions.

    Parameters:
    ----------
    original_preds : np.ndarray
        Model predictions on the original dataset.
    counterfactual_preds : np.ndarray
        Model predictions on the counterfactual dataset.
    modified_indices : list
        Indices where modifications were applied.
    """
    plt.figure(figsize=(10,5))

    # Compute mean if predictions are 2D (multi-horizon outputs)
    orig_mean = original_preds.mean(axis=1) if original_preds.ndim == 2 else original_preds
    cf_mean = counterfactual_preds.mean(axis=1) if counterfactual_preds.ndim == 2 else counterfactual_preds

    plt.plot(orig_mean, label='Originale Vorhersage', alpha=0.7)
    plt.plot(cf_mean, label='Counterfactual Vorhersage (+50% Spalte XY)', alpha=0.7)

    # Highlight modified samples
    plt.scatter(modified_indices, orig_mean[modified_indices], color='red', label='Modifizierte Inputs (Original)', zorder=5)
    plt.scatter(modified_indices, cf_mean[modified_indices], color='green', label='Modifizierte Inputs (CF)', zorder=6)

    plt.xlabel('Test Sample Index')
    plt.ylabel('Predicted Output')
    plt.title('Vergleich: Original vs. Counterfactual Vorhersagen')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot differences
    plt.figure(figsize=(10,3))
    delta = cf_mean - orig_mean
    plt.plot(delta, label='Unterschied (CF - Original)', color='blue')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.scatter(modified_indices, delta[modified_indices], color='red', label='Modified Samples', zorder=5)
    plt.xlabel('Test Sample Index')
    plt.ylabel('Differenz der Vorhersage')
    plt.title('Auswirkung von 50% Erhöhung din Column 7')
    plt.legend()
    plt.grid(True)
    plt.show()
    model_name = ControlVar['MLtype']

    file_name = f'{model_name}_top_100_counterfactual.png'
    plt.savefig(file_name)
    plt.close()
    print(f"Counterfactual comparision.")