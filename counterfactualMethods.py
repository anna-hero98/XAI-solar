import shap
import numpy as np
import random
import lime
import lime.lime_tabular
import quantus
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_pdf import PdfPages
from functools import partial

from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process

import numpy as np
from scipy.stats import norm


# =============================================================================
# 1) SAX-Funktion (unverändert)
# =============================================================================
def sax_transform(ts, n_segments=10, alphabet_size=5):
    """
    Transformiert eine 1D-Zeitreihe in eine symbolische Darstellung mittels SAX.
    
    Parameter
    ----------
    ts : np.ndarray
        1D-Zeitreihe (z. B. als np.array).
    n_segments : int, optional
        Anzahl der Segmente für die PAA (Standard: 10).
    alphabet_size : int, optional
        Anzahl der Symbole im Alphabet (Standard: 5).
        
    Returns
    -------
    sax_string : list of int
        Liste von Symbolindizes (0 bis alphabet_size-1).
    """
    ts = np.array(ts, dtype=float)
    n = len(ts)
    
    # Schritt 1: Piecewise Aggregate Approximation (PAA)
    segment_size = n / n_segments
    paa = []
    for i in range(n_segments):
        start = int(round(i * segment_size))
        end = int(round((i + 1) * segment_size))
        if end <= start:
            end = start + 1
        segment_mean = ts[start:end].mean()
        paa.append(segment_mean)
    paa = np.array(paa)
    
    # Schritt 2: Bestimmen der Breakpoints anhand der Normalverteilung
    breakpoints = norm.ppf(np.linspace(0, 1, alphabet_size + 1)[1:-1])
    
    # Schritt 3: Zuordnen der Symbole
    sax_string = []
    for value in paa:
        symbol = np.searchsorted(breakpoints, value)
        sax_string.append(symbol)
    return sax_string


# =============================================================================
# 2) Funktion, die Ihr 3D-Dataset (n_samples, time_steps, n_features)
#    für jede Instanz in eine 1D-Sequenz flattenet und SAX anwendet.
# =============================================================================
def generate_sax_for_dataset(ML_DATA, Control_Var, n_segments=10, alphabet_size=5):
    """
    Wendet SAX auf ein mehrdimensionales (3D) Datenset an, indem jede Instanz
    in eine 1D-Zeitreihe transformiert wird. Anschließend werden die SAX-Indizes
    für jede Instanz gespeichert.

    Parameter
    ----------
    ML_DATA : dict
        Enthält 'X_TRAIN', 'X_TEST' und ggf. weitere Felder.
        X_TRAIN und X_TEST haben die Form (n_samples, time_steps, n_features).
    Control_Var : dict
        Enthält u.a. 'MLtype' und evtl. 'PossibleFeatures'. Hier primär für Logging.
    n_segments : int, optional
        Anzahl der Segmente für die PAA in SAX (Standard: 10).
    alphabet_size : int, optional
        Anzahl der Symbole im Alphabet (Standard: 5).

    Returns
    -------
    train_sax : list of list of int
        Für jeden Trainingssample eine Liste von SAX-Symbolen.
    test_sax : list of list of int
        Für jeden Testsample eine Liste von SAX-Symbolen.
    """
    ml_type = Control_Var.get('MLtype', 'Unknown')
    print(f"=== Generiere SAX-Repräsentationen für Modelltyp '{ml_type}' ===")
    
    X_train_3D = ML_DATA["X_TRAIN"]  # (7660, 6, 18) z. B.
    X_test_3D = ML_DATA["X_TEST"]    # (1095, 6, 18)
    
    print("X_train shape:", X_train_3D.shape)
    print("X_test shape:", X_test_3D.shape)
    n_train = X_train_3D.shape[0]
    n_test = X_test_3D.shape[0]
    
    # Jedes Sample in 1D umwandeln: (time_steps * n_features)
    # Danach via sax_transform -> Symbolische Repräsentation
    train_sax = []
    for i in range(n_train):
        # Flatten (6,18)->(108)
        ts_1D = X_train_3D[i].reshape(-1)
        sax_rep = sax_transform(ts_1D, n_segments=n_segments, alphabet_size=alphabet_size)
        train_sax.append(sax_rep)
    
    test_sax = []
    for i in range(n_test):
        ts_1D = X_test_3D[i].reshape(-1)
        sax_rep = sax_transform(ts_1D, n_segments=n_segments, alphabet_size=alphabet_size)
        test_sax.append(sax_rep)
    
    # Beispielhafter Output
    print(f"Erste Training-SAX-Repräsentation (Sample 0): {train_sax[0]}")
    print(f"Erste Test-SAX-Repräsentation (Sample 0): {test_sax[0]}")
    print("Beispielhafte Länge einer SAX-Repräsentation:", len(train_sax[0]))
    
    return train_sax, test_sax

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
    
    FIXED_XLIM = (-0.5, 0.5)  # Set fixed x-axis limits for all plots

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
                # Titel als "Supertitle" hinzufügen.
    ml_type = Control_Var['MLtype']
    fig.suptitle(f"SHAP für das {ml_type}-Modell",fontsize=14)

            # Mit subplots_adjust oder tight_layout genügend Platz für den Titel schaffen.
            # Variante A:
    plt.subplots_adjust(top=0.85)  # je nach Bedarf anpassen (0.8, 0.9 etc.)
    plt.subplots_adjust(right=0.8)  # Reserve space for the Colorbar
    shap.summary_plot(
        shap_values=shap_2D,
        features=X_test_agg,
        feature_names=feature_names,
        plot_type='dot',
        show=False,
        sort=False,
        xlim=FIXED_XLIM
    )


    model_folder = f"./{Control_Var['MLtype']}"
    os.makedirs(model_folder, exist_ok=True)

    agg_filename = f"{Control_Var['MLtype']}_Shap_Aggregated.png"
    agg_filepath = os.path.join(model_folder, agg_filename)
    fig.savefig(agg_filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Aggregated SHAP summary plot saved as: {agg_filename}")

    # --- Per-Timestep SHAP Summary Plots in einem PDF ---
    per_ts_filename = f"{Control_Var['MLtype']}_Shap_PerTimestep.pdf"
    per_ts_filepath  = os.path.join(model_folder, per_ts_filename) 
    with PdfPages(per_ts_filepath) as pdf:
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
                show=False,
                sort=False,
                xlim=FIXED_XLIM
            )
                            # Standard-Titel von shap.summary_plot entfernen (falls vorhanden).
            plt.title("")  

            # Titel als "Supertitle" hinzufügen.
            fig.suptitle(f"SHAP Diagramm zum Zeitpunkt t = {t} für das {ml_type}-Modell", fontsize=14)

            # Mit subplots_adjust oder tight_layout genügend Platz für den Titel schaffen.
            # Variante A:
            plt.subplots_adjust(top=0.95)  # je nach Bedarf anpassen (0.8, 0.9 etc.)
            model_folder = f"./{Control_Var['MLtype']}"
            agg_filepath = os.path.join(model_folder, per_ts_filename)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)            # CHANGED
            plt.close(fig)
            plt.colorbar = original_cb
    print(f"Per-timestep SHAP summary plots saved as: {per_ts_filename}")

    # Restore original plt.colorbar (falls noch nicht geschehen)
    plt.colorbar = original_colorbar

    return shap_values


def ffc_explanation(
        model,
        X,
        feature_names,
        ml_type: str,
        n_samples: int = 30,
        random_state: int = 42,
        make_plots: bool = True):
    """
    Liefert lokale FFC‑Relevanzen S und erzeugt (optional) globale Plots.

    Zusätzliche Parameter
    ---------------------
    feature_names : list[str]
        Namen der Eingangssignale (Länge = n_features).
    ml_type : str
        Modelltyp; wird als Ordner‑/Dateipräfix verwendet.
    make_plots : bool, default=True
        Erzeugt PNG‑Plots der globalen Kennzahlen.
    """
    rng = np.random.default_rng(random_state)
    n, T, F = X.shape
    S = np.zeros((n, T, F), dtype=float)

    # ---------- Lokale Relevanzen berechnen (identisch wie zuvor) ----------
    for i in range(n):
        x_orig = X[i].copy()
        pred_orig = model.predict(x_orig[None, ...])
        y_orig    = float(np.asarray(pred_orig).mean())        
        for t in range(T):
            for f in range(F):
                idx_pool = rng.choice(np.delete(np.arange(n), i), size=n_samples)
                repl_vals = X[idx_pool, t, f]
                delta = 0.0
                for val in repl_vals:
                    x_cf = x_orig.copy()
                    x_cf[t, f] = val
                    pred_cf = model.predict(x_cf[None, ...])
                    y_cf    = float(np.asarray(pred_cf).mean())       # ⇒ Skalar
                    delta  += abs(y_cf - y_orig)
                    S[i, t, f] = delta / n_samples

    # ---------- Globale Kennzahlen ----------
    feature_global = S.mean(axis=(0, 1))   # (F,)
    time_global    = S.mean(axis=(0, 2))   # (T,)
    instance_sum   = S.sum(axis=(1, 2))    # (n,)

    # ---------- Plots ----------
    if make_plots:
        out_dir = f"./{ml_type}"
        os.makedirs(out_dir, exist_ok=True)

        # 1) Feature‑Global – Balkendiagramm
        plt.figure(figsize=(8, 4))
        plt.bar(range(F), feature_global)
        plt.xticks(range(F), feature_names, rotation=90)
        plt.ylabel("Mittlere Relevanz")
        plt.title(f"{ml_type} – FFC Feature Global")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{ml_type}_FFC_FeatureGlobal.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # 2) Time‑Global – Liniendiagramm
        plt.figure(figsize=(6, 4))
        plt.plot(range(T), time_global, marker="o")
        plt.xlabel("Zeitschritt")
        plt.ylabel("Mittlere Relevanz")
        plt.title(f"{ml_type} – FFC Time Global")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{ml_type}_FFC_TimeGlobal.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        # 3) Instance‑Sum – Boxplot
        plt.figure(figsize=(4, 4))
        plt.boxplot(instance_sum, vert=True, patch_artist=True)
        plt.ylabel("Σ Relevanz pro Instanz")
        plt.title(f"{ml_type} – FFC Instance Sum")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{ml_type}_FFC_InstanceSum.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    return S, {"feature_global": feature_global,
               "time_global": time_global,
               "instance_sum": instance_sum}




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
def read_indices_from_file(file_path):
    """
    Liest Indizes aus einer Textdatei (eine Zeile pro Index).
    Gibt eine Liste von Integer-Indizes zurück.
    """
    with open(file_path, 'r') as f:
        lines = f.read().strip().splitlines()
    return [int(line) for line in lines]

def write_indices_to_file(file_path, indices):
    """
    Schreibt die Indizes zeilenweise in eine Textdatei.
    Beispiel: 
      12
      37
      98
    """
    with open(file_path, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")

def generate_lime_explanations(
    model,
    X_train,
    X_test,
    feature_names,
    ml_type,
    lime_predict_fn,
    selected_indices=None,
    selected_indices_file_path='selected_indices.txt',
    num_instances=5,
    seed=42
):
    """
    Erzeugt LIME-Erklärungen für ausgewählte Testinstanzen und speichert sie in einer PDF-Datei.
    Wenn 'selected_indices' nicht explizit übergeben wird, versucht die Funktion, diese 
    aus 'selected_indices_file_path' zu lesen. Gelingt dies nicht, werden neue Zufallsindizes 
    gezogen und in der Datei gespeichert.

    Parameter:
    ----------
    model : trainiertes ML-Modell
        Das zu erklärende Modell.
    X_train : np.ndarray
        Trainingsdaten (3D-Format).
    X_test : np.ndarray
        Testdaten (3D-Format).
    feature_names : list of str
        Die ursprünglichen Feature-Namen.
    ml_type : str
        Modelltyp, z. B. 'CNN', 'LSTM', ...
    lime_predict_fn : callable
        Eine Funktion (Wrapper) für die Vorhersage, damit LIME die Daten im richtigen Format erhält.
    selected_indices : list of int, optional
        Manuell übergebene Indizes. Falls nicht gesetzt, wird versucht, sie aus einer Datei zu lesen.
    selected_indices_file_path : str
        Pfad zu einer Textdatei, in der die Indizes gespeichert werden oder aus der sie gelesen werden.
    num_instances : int
        Anzahl an Instanzen, die zufällig ausgewählt werden, falls keine Indizes übergeben 
        oder in der Datei gefunden werden.
    seed : int
        Zufalls-Seed zur Reproduzierbarkeit.

    Returns:
    --------
    used_indices : list of int
        Die tatsächlich verwendeten Testinstanz-Indizes.
    """

    # Ordner für das Modell anlegen (falls nicht vorhanden)
    os.makedirs(f"./{ml_type}", exist_ok=True)

    # Flatten: 3D -> 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Erweiterte Feature-Namen für LIME (jeder Zeitschritt für jedes Feature)
    lime_feature_names = [
        f"{col}_{i}" for col in feature_names for i in range(X_train.shape[1])
    ]

    # Initialisieren des LIME-Explainers
    explainer = LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=lime_feature_names,
        mode='regression',
        discretize_continuous=False
    )

    # Logik für 'selected_indices':
    np.random.seed(seed)
    used_indices = None

    if selected_indices is not None:
        # Falls manuell übergeben, direkt verwenden.
        used_indices = selected_indices
        print("Verwende übergebene Indizes:", used_indices)
    else:
        # Falls keine Indizes übergeben, versuchen wir, sie aus der Datei zu lesen:
        if os.path.exists(selected_indices_file_path):
            try:
                used_indices = read_indices_from_file(selected_indices_file_path)
                print("Verwende zuvor gespeicherte Indizes aus Datei:", used_indices)
            except Exception as e:
                print("Fehler beim Lesen der Datei. Generiere neue Zufallsindizes.")
        
        # Falls immer noch keine Indizes vorhanden (Datei nicht vorhanden oder fehlerhaft)
        if not used_indices:
            used_indices = np.random.choice(
                range(X_test.shape[0]), 
                num_instances, 
                replace=False
            ).tolist()
            print("Keine gültigen Indizes gefunden. Verwende neue Zufallsindizes:", used_indices)
            # Neue Indizes speichern
            write_indices_to_file(selected_indices_file_path, used_indices)

    # Ausgabe-Datei für die LIME-Erklärungen
    pdf_path = f"./{ml_type}/{ml_type}_LIME_Explanations.pdf"
    with PdfPages(pdf_path) as pdf:
        for idx in used_indices:
            test_instance = X_test_flat[idx].reshape(1, -1)
            # Partial-Funktion für LIME, damit X_train und model nicht immer wieder neu übergeben werden
            predict_fn = partial(lime_predict_fn, X_train=X_train, model=model)
            explanation = explainer.explain_instance(test_instance[0], predict_fn)
            
            fig = explanation.as_pyplot_figure()
            fig.suptitle(f"{ml_type} LIME Explanation for Test Instance {idx}", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"LIME-Erklärungen wurden in '{pdf_path}' gespeichert.")
    return used_indices

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
    model_folder = f"./{Control_Var['MLtype']}"
    agg_filepath = os.path.join(model_folder, file_name)
    plt.savefig(agg_filepath, bbox_inches='tight', dpi=300)
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
        X_temp[..., feature_index] = val  
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


    model_folder = f"./{model_name}"  # e.g. "./CNN"
    os.makedirs(model_folder, exist_ok=True)  # Ensure folder exists

    pdf_path = os.path.join(model_folder, filename)
    with PdfPages(pdf_path) as pdf:
        for timestep in range(num_time_steps):
            values = X_test[:, timestep, feature_idx]
            vmin, vmax = np.percentile(values, [1, 99])
            value_range = np.linspace(vmin, vmax, num_points)

            ice_matrix = []

            fig = plt.figure(figsize=(9, 6)) 
            for i in sample_indices:
                preds = []
                for val in value_range:
                    X_temp = X_test[i:i+1].copy()
                    X_temp[0, timestep, feature_idx] = val
                    pred = model.predict(X_temp, verbose=0)[0][timestep]
                    preds.append(pred)
                ice_matrix.append(preds)
                plt.plot(value_range, preds, alpha=0.4, linewidth=1)
            for i in sample_indices:
                x_val = X_test[i, timestep, feature_idx]
                y_val = model.predict(X_test[i:i+1], verbose=0)[0][timestep]
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

            plt.title(f"PDP + ICE für '{feature}' zum Zeitpunkt t={timestep}")
            plt.xlabel(f"{feature} (t = {timestep})")
            plt.ylabel("Modelvorhersage Solarenergie (kW)")
            plt.grid(True)
            plt.tight_layout()
            model_folder = f"./{Control_Var['MLtype']}"
            #agg_filepath = os.path.join(model_folder, filename)
            agg_filepath = os.path.join(model_folder,f"{model_name}_PDP_ICE_{feature.replace('[','').replace(']','')}_t{timestep}.png"
    )
            plt.savefig(agg_filepath, bbox_inches='tight', dpi=300)   # CHANGED

            #pdf.savefig(agg_filepath, bbox_inches='tight', dpi=300)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)            # CHANGED
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
    model_folder = f"./{ControlVar['MLtype']}"
    agg_filepath = os.path.join(model_folder, file_name)
    plt.savefig(agg_filepath, bbox_inches='tight', dpi=300)
    #plt.savefig(file_name)
    plt.close()
    print(f"Counterfactual comparision.")


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

# %%
