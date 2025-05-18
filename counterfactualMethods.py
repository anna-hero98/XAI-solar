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
from matplotlib import ticker as mticker
from matplotlib.ticker import FuncFormatter
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re

from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process

import numpy as np
from scipy.stats import norm
from matplotlib.lines import Line2D

from pathlib import Path
from typing import Sequence, Union, Optional, Iterable, List, Literal

def lime_predict(input_data, X_train, model):
    """
    Reshape LIME's 2D input back to the expected 3D format.
    """
    input_reshaped = input_data.reshape((input_data.shape[0], X_train.shape[1], X_train.shape[2]))
    return model.predict(input_reshaped)

import os
import random
import re
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.backends.backend_pdf import PdfPages       # nur noch für RF/SVM‑Zweig

# ---------------------------------------------------------------------------
#  get_explanations_2D
# ---------------------------------------------------------------------------


def get_explanations_2D(
        model,
        ML_DATA: dict,
        X_test_3D: np.ndarray,
        feature_names: Sequence[str],
        *,
        horizon_steps: Sequence[int] = (9,),
        input_steps:   Optional[Sequence[int]] = None,
        background_samples: int = 100,
        Control_Var: dict,
        idx_remove: Optional[int] = None,
        bg_indices: Optional[Sequence[int]] = None,
    ):
    """
    ------------------------------------------------------------------------
    SHAP‑Erklärungen für beliebige Prognose‑HORIZONTE **und** frei wählbare
    INPUT‑TIMESTEPS.

    ▸  horizon_steps   – Iterable über die gewünschten Ausgabeschritte
                         (0‑basiert), z. B.  (0,3,9)

    ▸  input_steps     – Iterable über Input‑Timesteps (ebenfalls 0‑basiert)
                         * None  →  alle Timesteps
                         * ()    →  kein Einzel‑Timestep‑Plot, nur Aggregat

    Rückgabe:  *raw* SHAP‑Werte (Liste oder Array, wie vom jeweiligen
               SHAP‑Explainer geliefert).  Die erzeugten PNGs landen im
               Ordner  "./<MLtype>/".
    ------------------------------------------------------------------------
    """
    # --------------------------------------------------------------------- #
    # 0) Vor‑/Grund­lagen
    # --------------------------------------------------------------------- #
    ml_type = Control_Var["MLtype"]                
    out_dir = f"./{ml_type}"
    os.makedirs(out_dir, exist_ok=True)

    # Fixe x‑Achsen‑Grenzen, damit mehrere Plots visuell vergleichbar sind
    X_LIM = (-0.27, 0.27)

    # Farbleiste bei shap.summary_plot unterdrücken
    class _DummyCbar:
        def __init__(self):
            # Dummy‑Achse, damit shap() kein Attribute‑Error wirft
            self.ax       = type("A", (), {"tick_params": lambda *_a, **_k: None})()
            self.outline  = type("B", (), {"set_visible": lambda *_a, **_k: None})()
        def set_ticklabels(self,*_a,**_k): pass
        def set_label(self,*_a,**_k):      pass
        def set_alpha(self,*_a,**_k):      pass
    _orig_cbar = plt.colorbar
    plt.colorbar = lambda *_a, **_k: _DummyCbar()       # type: ignore

    # --------------------------------------------------------------------- #
    # 1)  Hintergrund­daten bestimmen
    # --------------------------------------------------------------------- #
    if bg_indices is None:
        rng = np.random.default_rng(42)
        bg_indices = rng.choice(X_test_3D.shape[0],
                                size=min(background_samples,
                                         X_test_3D.shape[0]),
                                replace=False)

    # --------------------------------------------------------------------- #
    # 2)  SHAP berechnen – abhängig vom Modell­typ
    # --------------------------------------------------------------------- #
    if ml_type in ("CNN", "LSTM", "CNN_LSTM"):
        # ---- GradientExplainer erwartet ein Tensorflow‑Tensor oder ndarray
        explainer   = shap.GradientExplainer(model, X_test_3D[bg_indices])
        shap_vals   = explainer.shap_values(X_test_3D)          # Liste oder Array

        # SHAP gibt bei mehreren Targets eine Liste zurück → zu 4‑D Array stapeln
        shap_arr    = (np.stack(shap_vals, axis=-1)
                       if isinstance(shap_vals, list) else shap_vals)
        # Form: (N  , T_in, F, H)
    # Nicht genutzt
    elif ml_type == "RF":
        explainer   = shap.TreeExplainer(model)
        shap_vals   = explainer.shap_values(X_test_3D)          # Liste (pro Target)
        shap_arr    = np.stack(shap_vals, axis=-1)              # (N, F, H)
        shap_arr    = shap_arr[:, None, :, :]                   # Dummy‑Time‑Achse

    elif ml_type == "SVM":
        explainer   = shap.KernelExplainer(model.predict,
                                           X_test_3D[bg_indices])
        shap_vals   = explainer.shap_values(X_test_3D,
                                            nsamples=background_samples)
        shap_arr    = np.stack(shap_vals, axis=-1)              # (N, F, H)
        shap_arr    = shap_arr[:, None, :, :]                   # Dummy‑Time‑Achse
    else:
        raise ValueError(f"Unbekannter ML‑Typ: {ml_type}")

    # ---- Dimensionen benennen
    N, T_in, F, H = shap_arr.shape

    # --------------------------------------------------------------------- #
    # 3)  Optional: Feature entfernen (Index bekannt)
    # --------------------------------------------------------------------- #
    if idx_remove is not None:
        shap_arr   = np.delete(shap_arr, idx_remove, axis=2)
        X_test_3D  = np.delete(X_test_3D,  idx_remove, axis=2)
        feature_names = list(feature_names)        # kopieren → bearbeitbar
        feature_names.pop(idx_remove)
        F -= 1

    # --------------------------------------------------------------------- #
    # 4)  Welche Input‑Timesteps sollen berücksichtigt werden?
    # --------------------------------------------------------------------- #
    if input_steps is None:                        # alle Timesteps
        input_steps_sel = range(T_in)
    else:
        input_steps_sel = [t for t in input_steps if 0 <= t < T_in]

    # --------------------------------------------------------------------- #
    # 5)  Haupt­schleife über gewünschte Horizon­te
    # --------------------------------------------------------------------- #
    for h in horizon_steps:
        if not 0 <= h < H:
            print(f"[Skip] Forecast‑Index {h} existiert nicht (H={H})")
            continue

        shap_h = shap_arr[..., h]                  # (N, T_in, F)

        # ---------------- 5a) Aggregiert über gewählte Timesteps ----------
        shap_h_agg = shap_h[:, input_steps_sel, :].mean(axis=1)     # (N, F)
        x_h_agg    = X_test_3D[:, input_steps_sel, :].mean(axis=1)  # (N, F)

        fig = plt.figure(figsize=(8, 6))
        shap.summary_plot(shap_h_agg, x_h_agg,
                          feature_names=feature_names,
                          plot_type="dot", sort=False, show=False)
        plt.xlim(*X_LIM)
        plt.title(f"{ml_type} - SHAP  Prognosehorizont t={h}  (Eingabeschritte: {list(input_steps_sel)})")
        f_out = os.path.join(out_dir, f"{ml_type}_Shap_t{h}_Input_{input_steps_sel}.png")
        fig.savefig(f_out, dpi=300, bbox_inches="tight");  plt.close(fig)
        print("✅", f_out)
        del fig


        # ---------------- 5b) Einzel‑Timesteps (falls erwünscht) ----------
      # ----------------------------------------------------------------- #
    # 5b) Für den letzten Horizont (t=0) werden die SHAP-Werte für alle einzelnen Inputs berechnet.
    # ----------------------------------------------------------------- #
    if input_steps is not () and len(input_steps_sel):
        for t in input_steps_sel:
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(
                shap_h[:, t, :],
                X_test_3D[:, t, :],
                feature_names=feature_names,
                plot_type="dot", sort=False, show=False
            )
            plt.xlim(*X_LIM)

            # ---------- Titel & Dateiname ----------------------------------
            titel = (
                f"{ml_type} - SHAP Prognosehorizont t ={h}  |  "
                f"Eingabeschritt t = {t}"
            )
            plt.gca().set_title(titel, fontsize=14, pad=12)

            f_t = os.path.join(
                out_dir, f"{ml_type}_Shap_t{h}_input-step{t:02d}.png"
            )
            plt.tight_layout()
            plt.savefig(f_t, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print("✅", f_t)
            del fig



    # --------------------------------------------------------------------- #
    # 6)  Zusätzlich: komplett aggregiertes Übersicht‑Plot
    #     (über *alle* Horizonte und gewählte Timesteps)
    # --------------------------------------------------------------------- #
    shap_agg_total = shap_arr[:, input_steps_sel, :, :].mean(axis=(1, 3))  # (N, F)
    x_agg_total    = X_test_3D[:, input_steps_sel, :].mean(axis=1)         # (N, F)

    fig = plt.figure(figsize=(8,6))
    fig.suptitle(f"{ml_type}  – SHAP (voll aggregiert)", fontsize=14)
    plt.subplots_adjust(top=0.87, right=0.8)
    shap.summary_plot(shap_agg_total, x_agg_total, feature_names,
                      plot_type="dot", sort=False, show=False)
    plt.xlim(*X_LIM)
    f_tot = os.path.join(out_dir, f"{ml_type}_Shap_Aggregated.png")
    fig.savefig(f_tot, dpi=300, bbox_inches="tight"); plt.close(fig)
    print("✅", f_tot)
    del fig


    # --------------------------------------------------------------------- #
    # 7)  Aufräumen & Rückgabe
    # --------------------------------------------------------------------- #
    plt.colorbar = _orig_cbar           # ursprüngliche Funktion wiederherstellen
    return shap_vals                    # *roh* zurückgeben




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
    """
    with open(file_path, 'w') as f:
        for idx in indices:
            f.write(f"{idx}\n")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from lime.lime_tabular import LimeTabularExplainer

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
    Erzeugt LIME-Erklärungen und speichert jede Instanz als einzelne PNG-Datei
    mit dem Titel der Abbildung als Dateinamen.

    Rückgabe
    --------
    used_indices : list[int]
        Die verwendeten Testinstanz-Indizes.
    """

    # Zielordner anlegen
    out_dir = f"./{ml_type}"
    os.makedirs(out_dir, exist_ok=True)

    # 3D → 2D
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat  = X_test.reshape(X_test.shape[0],  -1)

    # Feature-Namen erweitern (jede Zeitstufe pro Ursprungsfeature)
    lime_feature_names = [
        f"{col}_{i}" for col in feature_names for i in range(X_train.shape[1])
    ]

    # LIME-Explainer initialisieren
    explainer = LimeTabularExplainer(
        training_data=X_train_flat,
        feature_names=lime_feature_names,
        mode='regression',
        discretize_continuous=False
    )

    # Zufalls- / Benutzer­indizes wählen
    np.random.seed(seed)
    if selected_indices is not None:
        used_indices = selected_indices
    else:
        if os.path.exists(selected_indices_file_path):
            try:
                with open(selected_indices_file_path) as f:
                    used_indices = [int(line.strip()) for line in f]
            except Exception:
                used_indices = []
        if not used_indices:
            used_indices = np.random.choice(
                range(X_test.shape[0]), num_instances, replace=False
            ).tolist()
            with open(selected_indices_file_path, 'w') as f:
                f.write('\n'.join(map(str, used_indices)))

    # Schleife über ausgewählte Instanzen
    for idx in used_indices:
        test_instance = X_test_flat[idx].reshape(1, -1)
        predict_fn = partial(lime_predict_fn, X_train=X_train, model=model)

        explanation = explainer.explain_instance(test_instance[0], predict_fn)
        fig = explanation.as_pyplot_figure()

        title = f"{ml_type} LIME Erklärung für Testinstanz {idx}"
        fig.suptitle(title, fontsize=14)

        # Gültigen Dateinamen aus Überschrift ableiten
        safe_title = re.sub(r'[^\w\-\. ]', '_', title).replace(' ', '_')
        file_path = os.path.join(out_dir, f"{safe_title}.png")

        fig.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    print(f"LIME-Einzelbilder liegen in '{out_dir}'")
    return used_indices





###############################################################################
# 1) Generische Counterfactual‑Routine                                        #
###############################################################################
import numpy as np
from scipy.optimize import minimize

import numpy as np
from scipy.optimize import minimize

# Beispiel mit der Alibi-Bibliothek für Kontrafaktische Erklärungen (Zeitreihen)
# Installation: pip install alibi

import numpy as np
import tensorflow as tf
from alibi.explainers import Counterfactual
# Wrapper-Funktion für Zeitreihen-Kontrafaktoren mit Alibi
# Installation: pip install alibi

import numpy as np
import tensorflow as tf
from alibi.explainers import Counterfactual
import matplotlib.pyplot as plt

# Wrapper-Funktion für Zeitreihen-Kontrafaktoren mit Alibi
# Installation: pip install alibi

import numpy as np
import tensorflow as tf
from alibi.explainers import Counterfactual
import matplotlib.pyplot as plt


# Wrapper-Funktion für Zeitreihen-Kontrafaktoren mit Alibi
# Basierend auf dem Optimierungsansatz für kontrafaktische Erklärungen nach
# Wachter, S., Mittelstadt, B., & Russell, C. (2017). "Counterfactual Explanations Without Opening the Black Box".
# Installation: pip install alibi

import numpy as np
import tensorflow as tf
# Alibi nutzt tf.placeholder; dafür muss Eager-Execution deaktiviert sein:
tf.compat.v1.disable_eager_execution()

from alibi.explainers import Counterfactual
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_ts_counterfactual(
    x: np.ndarray,
    model,
    feature_idx: int,
    y_target: float,
    norm: str = 'l1',
    per_timestep: bool = False,
    bounds: tuple = (None, None),
    max_iter: int = 100
) -> np.ndarray:
    """
    Berechnet ein kontrafaktisches Gegenbeispiel für Zeitreihen-Daten (Timesteps × Features)
    mittels numerischer Optimierung gemäß dem kontrafaktischen Erklärungsansatz von Wachter et al. (2017):

        \min_{x_cf}\; ||x_cf - x||_p + \lambda \,(f(x_cf) - y_target)^2

    Hier wird der Minimierungsparameter \lambda fest als Strafterm-Gewicht (1000) im Code implementiert.

    Args:
        x: Original-Eingabe als Array der Form (Timesteps, Features).
        model: Black-Box-Modell mit .predict(), das Eingaben der Form
               (1, Timesteps, Features) verarbeitet.
        feature_idx: Index des Merkmals, das als Freiheitsgrad in die Optimierung eingeht.
        y_target: Gewünschter Vorhersagewert (z. B. Klassenwahrscheinlichkeit oder Regressionsziel).
        norm: 'l1' oder 'l2' zur Auswahl der Distanzmetrik ||x_cf - x||_p.
        per_timestep: Optimiere individuelle Delta-Werte pro Timestep (True) oder einen globalen Delta-Scalar (False).
        bounds: (min, max) Schranken für zulässige Werte des Merkmals beim Gegenfaktum.
        max_iter: Maximale Anzahl von Optimierungsschritten.

    Returns:
        x_cf: Kontrafaktische Eingabe mit Form (Timesteps, Features).
    """
    timesteps, _ = x.shape
    orig = x[:, feature_idx]

    # Initialisierung der Optimierungsvariablen
    if per_timestep:
        delta0 = np.zeros(timesteps)
        bnds = [bounds] * timesteps
    else:
        delta0 = np.array([0.0])
        bnds = [bounds]

    def objective(delta):
        # Erzeuge x_cf gemäß Delta
        x_cf = x.copy()
        if per_timestep:
            x_cf[:, feature_idx] = orig + delta
            dist = np.linalg.norm(delta, ord=1 if norm=='l1' else 2)
        else:
            x_cf[:, feature_idx] = orig + delta[0]
            dist = np.abs(delta[0]) if norm=='l1' else delta[0]**2
        # Black-Box-Vorhersage
        y_pred = model.predict(x_cf[np.newaxis, ...]).ravel()
        y_val = y_pred.mean() if y_pred.ndim > 1 else y_pred[0]
        # Kontrafakt-Verlust: Distanz + Strafterm für Zielabweichung
        return dist + 1000.0 * (y_val - y_target) ** 2

    # Minimierung
    res = minimize(objective, delta0, bounds=bnds, options={'maxiter': max_iter})
    best = res.x

    # Kontrafaktische Eingabe erzeugen
    x_cf = x.copy()
    if per_timestep:
        x_cf[:, feature_idx] = orig + best
    else:
        x_cf[:, feature_idx] = orig + best[0]
    return x_cf


# Custom TS Counterfactual Implementation based on Wachter et al. (2017)
# This SciPy-based solver is a simple, single-feature optimizer.
# For multi-feature, categorical data, or advanced regularization,
# consider established libraries like Alibi (https://github.com/SeldonIO/alibi) or DiCE.
# They provide more robust optimization heuristics, support for multiple perturbations,
# sparsity, and categorical handling out of the box.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_ts_counterfactual(
    x: np.ndarray,
    model,
    feature_idx: int,
    y_target: float,
    norm: str = 'l1',
    per_timestep: bool = False,
    bounds: tuple = (None, None),
    max_iter: int = 100
) -> np.ndarray:
    """
    Berechnet ein kontrafaktisches Gegenbeispiel für Zeitreihen-Daten (Timesteps × Features)
    mittels numerischer Optimierung gemäß dem kontrafaktischen Erklärungsansatz von Wachter et al. (2017).

    Hinweis: Dieses Beispiel optimiert nur ein Feature, nutzt eine feste Zielstrafe (λ=1000),
    und unterstützt keine kategorialen Daten oder multiple Merkmalsänderungen simultan.
    Für komplexere Anforderungen bieten Alibi und DiCE umfangreichere Funktionalität.

    Args:
        x: Original-Eingabe als Array der Form (Timesteps, Features).
        model: Black-Box-Modell mit .predict(), das Eingaben der Form
               (1, Timesteps, Features) verarbeitet.
        feature_idx: Index des Merkmals, das als Freiheitsgrad in die Optimierung eingeht.
        y_target: Gewünschter Vorhersagewert (z. B. Klassenwahrscheinlichkeit oder Regressionsziel).
        norm: 'l1' oder 'l2' zur Auswahl der Distanzmetrik ||x_cf - x||ₚ.
        per_timestep: Optimiere individuelle Delta-Werte pro Timestep (True) oder einen globalen Delta-Scalar (False).
        bounds: (min, max) Schranken für zulässige Werte des Merkmals beim Gegenfaktum.
        max_iter: Maximale Anzahl von Optimierungsschritten.

    Returns:
        x_cf: Kontrafaktische Eingabe mit Form (Timesteps, Features).
    """
    timesteps, _ = x.shape
    orig = x[:, feature_idx]

    # Initialisierung der Optimierungsvariablen
    if per_timestep:
        delta0 = np.zeros(timesteps)
        bnds = [bounds] * timesteps
    else:
        delta0 = np.array([0.0])
        bnds = [bounds]

    def objective(delta):
        # Erzeuge x_cf gemäß Delta
        x_cf = x.copy()
        if per_timestep:
            x_cf[:, feature_idx] = orig + delta
            dist = np.linalg.norm(delta, ord=1 if norm=='l1' else 2)
        else:
            x_cf[:, feature_idx] = orig + delta[0]
            dist = np.abs(delta[0]) if norm=='l1' else delta[0]**2
        # Black-Box-Vorhersage
        y_pred = model.predict(x_cf[np.newaxis, ...]).ravel()
        y_val = y_pred.mean() if y_pred.ndim > 1 else y_pred[0]
        # Kontrafakt-Verlust: Distanz + Strafterm für Zielabweichung
        return dist + 1000.0 * (y_val - y_target) ** 2

    # Minimierung
    res = minimize(objective, delta0, bounds=bnds, options={'maxiter': max_iter})
    best = res.x

    # Kontrafaktische Eingabe erzeugen
    x_cf = x.copy()
    if per_timestep:
        x_cf[:, feature_idx] = orig + best
    else:
        x_cf[:, feature_idx] = orig + best[0]
    return x_cf


def compute_ts_counterfactual(
    model,
    ML_DATA: dict,
    feature_names: list,
    feature: str,
    idx: int,
    y_target: float,
    norm: str = 'l1',
    per_timestep: bool = False,
    bounds: tuple = (None, None),
    max_iter: int = 100
) -> dict:
    """
    Wrapper-Funktion für zeitreihenbasierte Kontrafakt-Analyse.

    Diese Implementierung folgt dem methodischen Rahmen von Wachter et al. (2017),
    indem sie das oben definierte Optimierungsproblem (Distanz + Zielabweichungsstrafe)
    für ein ausgewähltes Testbeispiel löst.

    Hinweis: Alibi und DiCE implementieren erweitere Varianten (mehr Features, Sparsity,
    Classification, Mixed Data Types) sowie optimierte Solver (Genetische Algorithmen,
    Gradientenverfahren) und bieten entsprechende APIs.

    Args:
        model: Trainiertes Black-Box-Modell (Keras/PyTorch) für Zeitreihen.
        ML_DATA: Dictionary mit mindestens 'X_TEST': np.ndarray (n_samples, T, F).
        feature_names: Liste der Featurebezeichner.
        feature: Zu manipulierendes Feature (Name aus feature_names).
        idx: Index des Testbeispiels.
        y_target: Gewünschter Zielwert der Vorhersage.
        norm: Norm für die Distanz ('l1' | 'l2').
        per_timestep: Ob pro Timestep ein separates Delta optimiert wird.
        bounds: Schranken für Feature-Delta.
        max_iter: Maximale Solver-Iteration.

    Returns:
        dict mit:
          - 'x_cf': die kontrafaktische Eingabe (T, F)
          - 'y_orig': Original-Vorhersage (array)
          - 'y_cf': Gegenfakt-Vorhersage (array)
    """
    # Ausgangsdaten
    X_test = ML_DATA['X_TEST']
    x = X_test[idx]
    feat_idx = feature_names.index(feature)

    # Gegenfaktische Eingabe berechnen
    x_cf = generate_ts_counterfactual(
        x, model, feat_idx, y_target,
        norm=norm, per_timestep=per_timestep,
        bounds=bounds, max_iter=max_iter
    )

    # Vorhersagen
    y_orig = model.predict(x[np.newaxis, ...]).ravel()
    y_cf = model.predict(x_cf[np.newaxis, ...]).ravel()

    # Plot: Vorhersagen
    plt.figure(figsize=(6,4))
    # Zuerst Counterfactual, dann Original oben drüber, damit beides sichtbar ist
    plt.plot(y_cf, label='Counterfactual', linestyle='--', marker='x')
    plt.plot(y_orig, label='Original', linestyle='-', marker='o')
    plt.title(f'Vorhersage: Original vs. CF für {feature} (idx={idx})')
    plt.xlabel('Zeitschritt')
    plt.ylabel('Vorhersage')
    plt.legend()
    plt.grid(True)
    plt.show()

        # Plot: Zeitreiheninput
    fig, axs = plt.subplots(2,1,figsize=(10,6), sharex=True)
    for f in range(x.shape[-1]):
        axs[0].plot(x[:, f], alpha=0.3)
        axs[1].plot(x_cf[:, f], '--', alpha=0.3)
        axs[0].set_title('Original Input')
        axs[1].set_title('Kontrafaktische Input')
        axs[1].set_xlabel('Timesteps')
        plt.legend(feature_names, ncol=4)
        plt.tight_layout()
        plt.show()

    return {'x_cf': x_cf, 'y_orig': y_orig, 'y_cf': y_cf}


# Implementierung des kontrafaktischen Optimierungsansatzes nach
# Wachter, S., Mittelstadt, B. & Russell, C. (2017). Counterfactual Explanations Without Opening the Black Box.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def generate_ts_counterfactual(
    x: np.ndarray,
    model,
    feature_idx: int,
    y_target: float,
    norm: str = 'l1',
    per_timestep: bool = False,
    bounds: tuple = (None, None),
    max_iter: int = 100
) -> np.ndarray:
    """
    Löst das kontrafaktische Optimierungsproblem
        min ||x_cf - x||_p + λ (f(x_cf) - y_target)^2
    analog zu Wachter et al. (2017) für Zeitreihen-Daten.

    Args:
        x: Eingabe (Timesteps, Features).
        model: Black-Box-Vorhersager mit .predict((1,T,F)).
        feature_idx: Index des zu ändernden Merkmals.
        y_target: Zielvorhersage.
        norm: 'l1' oder 'l2'.
        per_timestep: eigene Delta pro Zeitschritt.
        bounds: (min, max)-Grenzen für das Feature.
        max_iter: Maximale Optimierungsiterationen.

    Returns:
        Kontrafaktische Eingabe (Timesteps, Features).
    """
    timesteps, _ = x.shape
    orig = x[:, feature_idx]

    # Initialisierung
    if per_timestep:
        delta0 = np.zeros(timesteps)
        bnds = [bounds] * timesteps
    else:
        delta0 = np.array([0.0])
        bnds = [bounds]

    def objective(delta):
        x_cf = x.copy()
        if per_timestep:
            x_cf[:, feature_idx] = orig + delta
            dist = np.linalg.norm(delta, ord=(1 if norm=='l1' else 2))
        else:
            x_cf[:, feature_idx] = orig + delta[0]
            dist = np.abs(delta[0]) if norm=='l1' else delta[0]**2
        # Vorhersage
        y_pred = model.predict(x_cf[np.newaxis, ...]).ravel()
        y_val = y_pred.mean() if y_pred.ndim > 1 else y_pred[0]
        # Verlust = Distanz + Strafe
        return dist + 1000.0 * (y_val - y_target)**2

    res = minimize(objective, delta0, bounds=bnds, options={'maxiter': max_iter})
    best = res.x

    # Erzeuge Gegenfaktum
    x_cf = x.copy()
    if per_timestep:
        x_cf[:, feature_idx] = orig + best
    else:
        x_cf[:, feature_idx] = orig + best[0]
    return x_cf


def compute_ts_counterfactual(
    model,
    ML_DATA: dict,
    feature_names: list,
    feature: str,
    idx: int,
    y_target: float,
    norm: str = 'l1',
    per_timestep: bool = False,
    bounds: tuple = (None, None),
    max_iter: int = 100
) -> dict:
    """
    Wrapper gemäß Wachter et al. (2017):
    Wendet das Optimierungsproblem auf ein Testbeispiel an,
    plottet die Vorhersungen und Input-Zeitreihen.
    """
    X_test = ML_DATA['X_TEST']
    x = X_test[idx]
    feat_idx = feature_names.index(feature)

    x_cf = generate_ts_counterfactual(
        x, model, feat_idx, y_target,
        norm=norm, per_timestep=per_timestep,
        bounds=bounds, max_iter=max_iter
    )

    y_orig = model.predict(x[np.newaxis, ...]).ravel()
    y_cf = model.predict(x_cf[np.newaxis, ...]).ravel()

    # Plot Vorhersagen
    plt.figure(figsize=(6,4))
    plt.plot(y_cf, '--x', label='Counterfactual')
    plt.plot(y_orig, '-o', label='Original')
    plt.title(f'Vorhersage: Original vs. CF für {feature} (idx={idx})')
    plt.xlabel('Zeitschritt')
    plt.ylabel('Vorhersage')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Input
    fig, axs = plt.subplots(2,1,figsize=(10,6), sharex=True)
    for f in range(x.shape[-1]):
        axs[0].plot(x[:, f], alpha=0.3)
        axs[1].plot(x_cf[:, f], '--', alpha=0.3)
    axs[0].set_title('Original Input')
    axs[1].set_title('Kontrafakt Input')
    axs[1].set_xlabel('Timesteps')
    plt.legend(feature_names, ncol=4)
    plt.tight_layout()
    plt.show()

    return {'x_cf': x_cf, 'y_orig': y_orig, 'y_cf': y_cf}


def compute_ts_counterfactual_dice(
    model,
    ML_DATA: dict,
    feature_names: list,
    feature: str,
    idx: int,
    total_CFs: int = 1,
    desired_range: tuple = (0.5, 0.5),
    method: str = 'random',
    features_to_vary: list = None,
    horizon: int = -1
) -> dict:
    """
    Berechnet und speichert zeitserielle Gegenfaktoren mit DiCE für einen spezifischen Forecast-Horizont.

    Plots werden im Ordner basierend auf dem Modellnamen angelegt.
    """
    import os
    import pandas as pd
    import dice_ml
    import numpy as np
    import matplotlib.pyplot as plt

    # Erstelle Ordner nach Modell-Typ
    ml_type = getattr(model, 'name', model.__class__.__name__)
    os.makedirs(ml_type, exist_ok=True)

    # 1. Originaldaten & Vorhersage
    X = ML_DATA['X_TEST']
    x_seq = X[idx:idx+1]
    preds = model.predict(x_seq).ravel()
    H = preds.shape[0]
    h = horizon if horizon >= 0 else H + horizon
    h = max(0, min(h, H-1))
    y_orig = preds[h]

    # 2. Flatten für DiCE
    T, F = x_seq.shape[1], x_seq.shape[2]
    flat = x_seq.reshape(1, T*F)
    col_names = [f"{fn}_{t}" for t in range(T) for fn in feature_names]
    df = pd.DataFrame(flat, columns=col_names)
    df['target'] = y_orig

    # 3. DiCE DataInterface
    data_dice = dice_ml.Data(
        dataframe=df,
        continuous_features=col_names,
        outcome_name='target'
    )

    # 4. Wrapper für flache Predict-Funktion
    def predict_fn_sklearn(X_flat):
        arr = X_flat.values if hasattr(X_flat, 'values') else np.array(X_flat)
        Xr = arr.reshape(-1, T, F)
        p = model.predict(Xr).ravel()
        return np.array([p[h] for _ in range(Xr.shape[0])])

    class ModelWrapper:
        def __init__(self, func): self.func = func
        def predict(self, X_flat): return self.func(X_flat)

    model_dice = dice_ml.Model(
        model=ModelWrapper(predict_fn_sklearn),
        backend='sklearn',
        model_type='regressor'
    )
    exp = dice_ml.Dice(data_dice, model_dice, method=method)

    # 5. Query-Instance
    if features_to_vary is None:
        features_to_vary = col_names
    query_instance = df[features_to_vary].iloc[[0]]

    # 6. Generate CFs
    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=total_CFs,
        desired_range=desired_range,
        features_to_vary=features_to_vary
    )

    # 7. Extract CF examples robust
    cfs_obj = dice_exp.cf_examples_list[0]
    # Versuche direkt
    try:
        cfs_df = cfs_obj.final_cfs_df
    except Exception:
        # Introspektion aller DataFrame-Attribute
        df_found = None
        for attr in dir(cfs_obj):
            val = getattr(cfs_obj, attr)
            if isinstance(val, pd.DataFrame):
                df_found = val
                break
        if df_found is None:
            raise ValueError(f"Keine DataFrame-Attribute im CF-Objekt gefunden. Verfügbare: {[a for a in dir(cfs_obj) if not a.startswith('_') ]}")
        cfs_df = df_found

    cf_flat = cfs_df[features_to_vary].values
    cf_examples = [cf.reshape(T, F) for cf in cf_flat]
    y_cfs = [model.predict(cf.reshape(1, T, F)).ravel()[h] for cf in cf_examples]

    # 8. Speichern & Plots
    # 8.1 Balkendiagramm der CF-Vorhersagen
    plt.figure(figsize=(6,4))
    labels = ['Original'] + [f'CF{i+1}' for i in range(len(y_cfs))]
    values = [y_orig] + y_cfs
    plt.bar(labels, values)
    plt.title(f'DiCE CF-Vorhersagen (Horizon={h}) für {feature} (idx={idx})')
    plt.ylabel('Vorhersage')
    plt.savefig(os.path.join(ml_type, 'dice_predictions_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8.2 Zeitreihen-Overlay
    feat_idx = feature_names.index(feature)
    plt.figure(figsize=(8,4))
    plt.plot(x_seq[0,:,feat_idx], '-o', color='black', label='Original')
    for i, cf in enumerate(cf_examples):
        plt.plot(cf[:,feat_idx], '--', label=f'CF{i+1}')
    plt.title(f'TimeSeries Overlay: {feature}')
    plt.xlabel('Timesteps')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ml_type, 'dice_timeseries_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8.3 Parallelkoordinaten-Plot
    try:
        import pandas.plotting as pd_plot
        df_pc = pd.DataFrame([cf.reshape(-1) for cf in cf_examples] + [flat.reshape(-1)], columns=col_names)
        df_pc['Typ'] = [f'CF{i+1}' for i in range(len(cf_examples))] + ['Original']
        plt.figure(figsize=(10,4))
        pd_plot.parallel_coordinates(df_pc, 'Typ')
        plt.title('Parallel Coordinates: CF vs Original')
        plt.savefig(os.path.join(ml_type, 'dice_parallel_coordinates.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception:
        pass

    # 8.4 Scatter-Plot der ersten beiden Features
    idx_x, idx_y = 0, 1
    ox = x_seq[0,:,idx_x].mean(); oy = x_seq[0,:,idx_y].mean()
    plt.figure(figsize=(5,5))
    plt.scatter(ox, oy, c='black', label='Original')
    for i, cf in enumerate(cf_examples):
        cx = cf[:,idx_x].mean(); cy = cf[:,idx_y].mean()
        plt.scatter(cx, cy, label=f'CF{i+1}')
        plt.arrow(ox, oy, cx-ox, cy-oy, head_width=0.01, length_includes_head=True)
    plt.xlabel(feature_names[idx_x]); plt.ylabel(feature_names[idx_y])
    plt.title('2D Scatter CFs'); plt.legend()
    plt.savefig(os.path.join(ml_type, 'dice_scatter2d.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8.5 Heatmap der Deltas
    deltas = np.array([cf.reshape(-1) - flat.reshape(-1) for cf in cf_examples])
    plt.figure(figsize=(8,4))
    plt.imshow(deltas, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Δ Value')
    plt.title('Heatmap der Δs (CF vs Original)')
    plt.savefig(os.path.join(ml_type, 'dice_heatmap_deltas.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8.6 Diversity vs. Proximity
    from itertools import combinations
    # Compute pairwise distances
    flats = [cf.reshape(-1) for cf in cf_examples]
    # Diversity: mean of pairwise L2 norms
    div_list = [
        np.linalg.norm(flats[i] - flats[j])
        for i, j in combinations(range(len(flats)), 2)
    ]
    diversity = float(np.mean(div_list)) if div_list else 0.0
    # Proximity: mean L2 norm to original flat
    prox_list = [
        np.linalg.norm(f - flat.reshape(-1)) for f in flats
    ]
    proximity = float(-np.mean(prox_list)) if prox_list else 0.0
    # Plot and save
    plt.figure(figsize=(4,3))
    plt.bar(['Diversity','Proximity'], [diversity, proximity])
    plt.title('Diversity vs Proximity')
    plt.savefig(os.path.join(ml_type, 'dice_diversity_proximity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return {'cf_examples': cf_examples, 'y_orig': y_orig, 'y_cfs': y_cfs} 

def generate_counterfactuals(
    data: dict,
    feature_list: Sequence[str],
    features_to_change: Union[str, Sequence[str]],
    factor: float,
    control_var: Optional[dict] = None,
) -> dict:
    """Erstellt eine manipulierte Kopie eines ``ML_DATA``‑ähnlichen Dicts.

    Alle Keys, die mit ``"X_"`` beginnen, werden dupliziert; die angegebenen
    Features werden um ``factor`` skaliert (Multiplikation). So bleibt das
    Original‑Dict unverändert.
    """
    if isinstance(features_to_change, str):
        features = [features_to_change]
    else:
        features = list(features_to_change)

    indices = [feature_list.index(f) for f in features if f in feature_list]

    new_data: dict = {}
    for key, val in data.items():
        if key.startswith("X_") and isinstance(val, np.ndarray):
            arr = val.copy()
            arr[..., indices] = arr[..., indices] * factor
            new_data[key] = arr
        else:
            new_data[key] = val.copy() if isinstance(val, np.ndarray) else val
    return new_data


def grid_counterfactual_plots_pct(
    ML_DATA,
    model,
    feature_names,
    feature,               
    change_factors,        
    Control_Var,
    bg_indices=None,
    max_cols=2
):
    """
    Baut auf grid_counterfactual_plots auf, plottet aber die %-Änderung der Vorhersage.
    """

    # Hilfsfunktion für Beschriftung
    def _factor_label_pct(f):
        pct = int(abs((f - 1) * 100))
        dir_text = "↑" if f > 1 else "↓"
        return f"{dir_text}{pct}%"

    feat_label = feature if isinstance(feature, str) else "+".join(feature)
    safe_name  = feat_label.replace("[","").replace("]","").replace(" ","_")
    model_name = Control_Var["MLtype"]

    # Raster-Größe bestimmen
    n_plots = len(change_factors)
    n_cols  = min(max_cols, n_plots)
    n_rows  = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5*n_cols, 4*n_rows),
                             sharex=True, sharey=True)
    axes = axes.flatten()

    # Original-Vorhersage einmal berechnen
    X_test = ML_DATA["X_TEST"][bg_indices] if bg_indices is not None else ML_DATA["X_TEST"]
    orig_preds = model.predict(X_test)
    orig_mean  = orig_preds.mean(axis=1) if orig_preds.ndim>1 else orig_preds

    for i, factor in enumerate(change_factors):
        ax = axes[i]

        # Counterfactual erzeugen
        # (hier nehme ich an, Du hast schon eine passende Funktion dafür)
        cf_data   = generate_counterfactuals(
                        {"X_TEST": X_test},
                        feature_names,
                        feature,
                        factor,
                        Control_Var
                    )
        cf_preds  = model.predict(cf_data["X_TEST"])
        cf_mean   = cf_preds.mean(axis=1) if cf_preds.ndim>1 else cf_preds

        # Prozentuale Änderung berechnen
        # Achtung: orig_mean kann 0 sein! ggf. Maske setzen oder small_eps addieren
        delta_pct = (cf_mean - orig_mean) / (orig_mean + 1e-6) * 100

        # Plot
        ax.scatter(
            np.arange(len(delta_pct)),
            delta_pct,
            s=10,
            alpha=0.7,
            label=_factor_label_pct(factor)
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"Faktor {factor:.2f} ({_factor_label_pct(factor)})")
        ax.set_xlabel("Testdatensatz")
        if i % n_cols == 0:
            ax.set_ylabel("Δ Vorhersage [%]")
        ax.grid(True, ls=":", lw=0.5)
        for ax in axes.flatten():
            ax.tick_params(axis='x', labelbottom=True)
 
    fig.suptitle(f"{model_name}: %-Änderung der Vorhersage bei {feat_label}", y=1.02)
    fig.tight_layout()

    # Speichern im Modell-Verzeichnis
    outdir = f"./{model_name}"
    os.makedirs(outdir, exist_ok=True)
    fname = f"{model_name}_counterfactual_{safe_name}_cf_pct_grid_{'-'.join(str(int((f-1)*100)) for f in change_factors)}.png"
    fig.savefig(os.path.join(outdir, fname), dpi=300, bbox_inches="tight")
    plt.close(fig)
    del fig

    print(f"✅ Raster mit %-Änderungen gespeichert als: {fname}")




def grid_cf_unscaled_with_inverse(
    ML_DATA: dict,
    model,
    feature_names: List[str],
    feature: str,
    change_factors: List[float],
    Control_Var: dict,
    scaler_y,
    bg_indices: Optional[List[int]] = None,
    horizon_index: Optional[int] = None,
    max_cols: int = 2
):
    """
    Erstellt CF-Rasterplots, diesmal auf unskalierten kW-Achsen,
    indem wir sowohl original- als auch CF-Predictions inversetransformieren.
    """
    import os
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    # Forecast-Horizon der Roh-Targets
    Y = ML_DATA["Y_TEST"]
    H = Y.shape[1]

    # welche Zeitschritte?
    if horizon_index is None:
        horizons = list(range(H))
    else:
        hi = horizon_index if horizon_index >= 0 else H + horizon_index
        hi = max(0, min(hi, H-1))
        horizons = [hi]

    # Ordner anlegen
    ml_name = Control_Var["MLtype"]
    out_dir = os.path.join(".", ml_name)
    os.makedirs(out_dir, exist_ok=True)

    # Index des Features
    feat_idx = feature_names.index(feature)

    # Inputs (ggf. Subset)
    X_base = ML_DATA["X_TEST"]
    if bg_indices is not None:
        X_base = X_base[bg_indices]

    # Für jeden Zeitschritt:
    for hi in horizons:
        # Original-Prediction
        y_orig_scaled = model.predict(X_base)
        if y_orig_scaled.ndim == 3:
            y_orig_scaled = y_orig_scaled[:, hi, 0]
        else:
            y_orig_scaled = y_orig_scaled.ravel()
        y_orig = scaler_y.inverse_transform(y_orig_scaled.reshape(-1, 1)).ravel()

        # Setup
        n = len(change_factors)
        cols = min(n, max_cols)
        rows = int(math.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(5*cols, 4*rows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(rows, cols)

        for idx, fac in enumerate(change_factors):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            X_cf = X_base.copy()
            X_cf[..., feat_idx] *= fac

            y_cf_scaled = model.predict(X_cf)
            if y_cf_scaled.ndim == 3:
                y_cf_scaled = y_cf_scaled[:, hi, 0]
            else:
                y_cf_scaled = y_cf_scaled.ravel()
            y_cf = scaler_y.inverse_transform(y_cf_scaled.reshape(-1, 1)).ravel()

            ax.plot(y_orig,
                    color="tab:blue", lw=1.5, alpha=0.7, label="Original [kW]")
            ax.plot(y_cf,
                    color="tab:orange", ls="--", lw=1.5, alpha=0.8,
                    label="{:+d}%".format(int((fac-1)*100)))
            ax.set_title(f"Faktor {fac:.2f}")            
            ax.set_ylabel("produzierter PV-Strom [kW] (unskaliert)")
            ax.set_xlabel("Testdatensatz")
            ax.grid(alpha=0.3)

        # Legende unten, gemeinsam
        handles = [
            Line2D([0], [0], color="tab:blue", linestyle="-", lw=1.5, label="Original"),
            Line2D([0], [0], color="tab:orange", linestyle="--", lw=1.5, label="Counterfactual")
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

        # Titel, Layout
        fig.suptitle("{}: Unskaliertes CF-Raster für '{}' (t={})".format(ml_name, feature, hi),
                     fontsize=14, y=1.02)
        fig.tight_layout()

        # Speichern
        safe = feature.replace("[", "").replace("]", "").replace(" ", "_")
        fac_str = "-".join("{:+d}".format(int((f-1)*100)) for f in change_factors)
        fn = "{}_counterfactual_{}_unscaled_inv_t{}_{}.png".format(ml_name, safe, hi, fac_str)
        path = os.path.join(out_dir, fn)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("✅ Unskaliertes CF-Raster gespeichert: {}".format(path))
        del fig


def kw_formatter(x, pos):
    return f"{x:.1f} kW"

def grid_cf_unscaled_direct(
    ML_DATA: dict,
    model,
    feature_names: List[str],
    feature: str,
    change_factors: List[float],
    Control_Var: dict,
    bg_indices: Optional[List[int]] = None,
    horizon_index: Optional[int] = None,
    max_cols: int = 2
):
    """
    Zeichnet für jeden Faktor ein Raster von Original- vs. CF-Vorhersagen
    *unskaliert*, indem es direkt Y_TEST verwendet.
    """
    # Roh-Zielwerte
    Y = ML_DATA["Y_TEST"]    # shape (n_samples, H, 1)
    H = Y.shape[1]

    # welche Zeitschritte?
    if horizon_index is None:
        horizons = list(range(H))
    else:
        hi = horizon_index if horizon_index >= 0 else H + horizon_index
        hi = max(0, min(hi, H-1))
        horizons = [hi]

    # Ordner
    ml_name = Control_Var["MLtype"]
    out_dir = os.path.join(".", ml_name)
    os.makedirs(out_dir, exist_ok=True)

    # Index des manipulierten Features
    feat_idx = feature_names.index(feature)

    # Basis-Inputs (ggf. Subset)
    X_base = ML_DATA["X_TEST"]
    if bg_indices is not None:
        X_base = X_base[bg_indices]

    # Schleife über Zeitschritte
    for hi in horizons:
        # Original: direkt aus Y_TEST
        if bg_indices is not None:
            y_orig = Y[bg_indices, hi, 0]
        else:
            y_orig = Y[:, hi, 0]

        # Setup Raster
        n = len(change_factors)
        cols = min(max_cols, n)
        rows = math.ceil(n/cols)
        fig, axes = plt.subplots(rows, cols,
                                 figsize=(5*cols, 4*rows),
                                 sharex=True, sharey=True)
        axes = np.array(axes).reshape(rows, cols)

        for idx, fac in enumerate(change_factors):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            # Counterfactual
            X_cf = X_base.copy()
            X_cf[..., feat_idx] *= fac

            # Vorhersage skalierte Modell-Ausgabe → bleibt skalierte Werte,
            # aber wir plotten ja nur zur Gegenüberstellung:
            y_cf_full = model.predict(X_cf)
            # falls Forecast-Horizon ausgegeben wird, indexieren:
            if y_cf_full.ndim == 3:
                y_cf = y_cf_full[:, hi, 0]
            else:
                y_cf = y_cf_full.ravel()

            # Plot
            ax.plot(y_orig,
                    color="tab:blue",
                    lw=1.5,
                    alpha=0.7,
                    label="Y_TEST (kW)")
            ax.plot(y_cf,
                    color="tab:orange",
                    ls="--",
                    lw=1.5,
                    alpha=0.8,
                    label=f"{int((fac-1)*100):+d}%")
            ax.set_title(f"Faktor = {fac:.2f}")
            if c == 0:
                ax.set_ylabel("produzierter PV-Strom [kW] (unskaliert)")
            ax.set_xlabel("Testdatensatz")
            ax.grid(alpha=0.3)
            for ax in axes.flatten():
                ax.tick_params(axis='x', labelbottom=True)

        # Legende
        h, l = axes[0,0].get_legend_handles_labels()
        handles = [
            plt.Line2D([0], [0], color='tab:blue', linestyle='-', label='Original'),
            plt.Line2D([0], [0], color='tab:orange', linestyle='--', label='Counterfactual')
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

        fig.suptitle(f"{ml_name}: CF-Raster für '{feature}' (t={hi})",
                     fontsize=14, y=1.02)
       
        fig.tight_layout()

        # Speichern
        feat_safe = feature.replace("[","").replace("]","").replace(" ","_")
        fac_str = "-".join(f"{int((f-1)*100):+d}" for f in change_factors)
        fn = f"{ml_name}_counterfactual_{feat_safe}_unscaled_d_t{hi}_{fac_str}_.png"
        path = os.path.join(out_dir, fn)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ Unskaliertes CF-Raster gespeichert: {path}")
        del fig



def cf_scatter_percent(
    ML_DATA,
    model,
    feature_names,                    
    feature,                          
    factors=(0.50, 0.75, 1.25, 1.50), 
    Control_Var=None,
    timestep=0,                      
    bg_idx=None,                      
    jitter=0.3,                       
    lin_thresh=20.0               
):
    """
    Δ-Vorhersage [%] für verschiedene Multiplikationsfaktoren eines Features,
    mit symlog-Skalierung und festen Ticks, Label ‚10^0‘ nur einmal.
    """
    ml_name = (Control_Var or {})["MLtype"] if Control_Var else "model"
    out_dir = os.path.join(".", ml_name)
    os.makedirs(out_dir, exist_ok=True)

    # Basis-Daten
    X_all  = ML_DATA["X_TEST"]
    X_base = X_all[bg_idx] if bg_idx is not None else X_all
    y_pred = model.predict(X_base)
    y_base = y_pred.mean(axis=1) if y_pred.ndim > 1 else y_pred.ravel()

    # Feature-Index
    feat_idx = feature_names.index(feature)

    # Subplots anlegen
    n_fac = len(factors)
    fig, axes = plt.subplots(1, n_fac, figsize=(5*n_fac,4),
                             sharey=True, sharex=True)
    if n_fac == 1:
        axes = [axes]

    rng = np.random.RandomState(0)

    # die gewünschten y-Ticks
    fixed_ticks = [-1000, -100, -10, 0, 10, 100, 1000]

    # scalar formatter konfigurieren
    sf = mticker.ScalarFormatter()
    sf.set_scientific(False)
    sf.set_useOffset(False)

    # Plot-Schleife
    for ax, fac in zip(axes, factors):
        # Counterfactual erzeugen
        X_cf = X_base.copy()
        X_cf[:, timestep, feat_idx] *= fac

        # Vorhersage
        y_cf_pred = model.predict(X_cf)
        y_cf      = y_cf_pred.mean(axis=1) if y_cf_pred.ndim > 1 else y_cf_pred.ravel()

        # Δ in Prozent, clamp
        eps       = 1e-6
        delta_pct = (y_cf - y_base)/(np.abs(y_base)+eps)*100
        delta_pct = np.clip(delta_pct, -1000, 1000)

        # nur Samples mit Basis >1e-3
        mask = np.abs(y_base) > 1e-3
        idxs = np.arange(len(delta_pct))[mask]
        vals = delta_pct[mask]
        xs   = idxs + rng.normal(0, jitter, size=idxs.shape)

        # Label
        arrow = "↑" if fac > 1 else "↓"
        pct   = int(abs((fac-1)*100))
        lbl   = f"{arrow}{pct}%"

        # Plot
        ax.scatter(xs, vals, s=20, alpha=0.7, label=lbl)
        ax.axhline(0, color="gray", linewidth=1)
        ax.set_yscale("symlog", linthresh=lin_thresh)
        ax.set_ylim(-1000, 1000)

        # **Tick-Konfiguration**
        ax.set_yticks(fixed_ticks)
        ax.yaxis.set_major_formatter(sf)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        # Entferne doppelte '10^0'-Labels
        for label in ax.get_yticklabels():
            if label.get_text() == "10^0":
                label.set_text("1")

        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.set_title(f"Faktor {fac:.2f} ({lbl})")
        ax.set_xlabel("Testdatensatz")

    axes[0].set_ylabel("Δ Vorhersage [%]")

    # Supertitle
    h = (Control_Var or {}).get("H")
    h=h-1
    if h is not None:
        fig.suptitle(f"{ml_name}: %-Änderung der Vorhersage bei {feature} (t={h})",
                     y=1.02, fontsize=12)
    else:
        fig.suptitle(f"{ml_name}: %-Änderung der Vorhersage bei {feature}",
                     y=1.02, fontsize=12)

    # Gemeinsame Legende
    all_h, all_l = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        all_h += h; all_l += l
    uniq_h, uniq_l, seen = [], [], set()
    for hndl, lbl in zip(all_h, all_l):
        if lbl not in seen:
            uniq_h.append(hndl); uniq_l.append(lbl); seen.add(lbl)


    fig.tight_layout(rect=[0, 0, 1, 0.95])
    feat_safe = feature.replace("[", "").replace("]", "").replace("/", "_")
    fname = f"{ml_name}_counterfactual_{feat_safe}_cf_scatter_pct" + \
            "-".join(str(int((f-1)*100)) for f in factors) + ".png"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("✅ Scatter-Raster gespeichert:", fpath)
    del fig


###############################################################################
#Scatter‑Plot (Prozentuale Änderung EIN‑ vs. AUS‑Größe)                   #
###############################################################################


def plot_actual_pv_output(
    ML_DATA: dict,
    Control_Var: dict,
    horizon_index: Optional[int] = None,
    bg_idx: Optional[np.ndarray] = None
) -> None:
    """
    Plottet die gemessene AC-Leistung P_Solar für einen gegebenen Forecast-Zeitschritt.

    ML_DATA must contain 'Y_TEST' mit Shape (n_samples, H, 1).
    horizon_index: Index im Forecast-Horizon (0…H-1). 
                   None oder negative Zahl → letzter Schritt.
    bg_idx: Optional[np.ndarray] → nur diese Test-Indizes verwenden.
    """
    # 1) Gesamtzahl der Forecast-Horizonte
    y_full = ML_DATA["Y_TEST"]  # (n_samples, H, 1)
    H = y_full.shape[1]

    # 2) horizon_index auf gültigen Bereich bringen
    if horizon_index is None:
        hi = H - 1
    else:
        hi = horizon_index if horizon_index >= 0 else H + horizon_index
        # z.B. -1 → H-1, -2 → H-2, etc.
    hi = max(0, min(hi, H - 1))

    # 3) Auswahl der Werte und Flatten auf 1-D
    if bg_idx is not None:
        y_sel = y_full[bg_idx, hi, 0]
    else:
        y_sel = y_full[:, hi, 0]

    # 4) Plot erzeugen
    ml_name = Control_Var["MLtype"]
    out_dir = os.path.join(".", ml_name)
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(y_sel, linewidth=1.5, label=f"Gemessene PV-Leistung t={hi}")
    plt.xlabel("Testdatensatz")
    plt.ylabel("produzierter PV-Strom [kW]")
    plt.title(f"{ml_name} – Gemessene PV-Leistung (Forecast-Horizon {hi})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fname = f"{ml_name}_actual_pv_t{hi}.png"
    path  = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"✅ Gespeichert unter: {path}")



def grid_counterfactual_plots_unscaled_all_timesteps(
    ML_DATA,
    model,
    Scaler_y,
    feature_names,
    feature,
    change_factors,
    Control_Var,
    bg_indices=None,
    horizon_index=None,
    max_cols=2,
    colors=None
):
    """
    Rasterplots der Counterfactual-Vorhersagen **unskaliert** für alle oder einen ausgewählten Forecast-Zeitschritt.

    Args:
        ML_DATA (dict): Enthält "X_TEST" und "Y_TEST"
        model (obj): ML Modell mit .predict()
        Scaler_y (obj): y-Scaler mit inverse_transform()
        feature_names (list): Feature-Namen
        feature (str): Zu manipulierendes Feature
        change_factors (list): Multiplikationsfaktoren für Feature
        Control_Var (dict): Infos wie "MLtype"
        bg_indices (list, optional): Subset von Indices
        horizon_index (int, optional): Forecast-Horizont
        max_cols (int): Maximalanzahl Spalten im Raster
        colors (dict, optional): Farben, z.B. {"original": "tab:blue", "cf": "tab:orange"}
    """

    Y_test_full = ML_DATA["Y_TEST"]
    H = Y_test_full.shape[1]

    if horizon_index is None:
        horizons = list(range(H))
    else:
        hi = horizon_index if horizon_index >= 0 else H + horizon_index
        hi = max(0, min(hi, H-1))
        horizons = [hi]

    ml_name = Control_Var["MLtype"]
    out_dir = os.path.join(".", ml_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    feat_idx = feature_names.index(feature)

    X_base = ML_DATA["X_TEST"]
    if bg_indices is not None:
        X_base = X_base[bg_indices]

    y_orig_scaled = model.predict(X_base)
    if y_orig_scaled.ndim == 3:
        y_orig_scaled = y_orig_scaled[..., 0]

    for hi in horizons:
        y_orig_h_scaled = y_orig_scaled[:, hi].reshape(-1, 1)

        if hasattr(Scaler_y, "inverse_transform"):
            y_orig = Scaler_y.inverse_transform(y_orig_h_scaled).ravel()
        else:
            if bg_indices is not None:
                y_orig = Y_test_full[bg_indices, hi, 0]
            else:
                y_orig = Y_test_full[:, hi, 0]

        n = len(change_factors)
        cols = min(max_cols, n)
        rows = int(math.ceil(float(n) / cols))

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(5 * cols, 4 * rows),
                                 sharex=True,
                                 sharey=True)

        axes = np.array(axes).reshape(rows, cols)

        for ax in axes.flatten():
            ax.tick_params(axis='x', labelbottom=True)

        for idx, fac in enumerate(change_factors):
            r, c = divmod(idx, cols)
            ax = axes[r, c]

            # Feature manipulieren
            X_mod = X_base.copy()
            X_mod[:, :, feat_idx] *= fac
            y_mod_scaled = model.predict(X_mod)
            if y_mod_scaled.ndim == 3:
                y_mod_scaled = y_mod_scaled[..., 0]

            y_mod_h_scaled = y_mod_scaled[:, hi].reshape(-1, 1)

            if hasattr(Scaler_y, "inverse_transform"):
                y_mod = Scaler_y.inverse_transform(y_mod_h_scaled).ravel()
            else:
                y_mod = y_mod_h_scaled.ravel()

            # Farben definieren
            if colors is not None:
                color_orig = colors.get("original", "tab:blue")
                color_cf = colors.get("cf", "tab:orange")
            else:
                color_orig = "tab:blue"
                color_cf = "tab:orange"

            ax.plot(np.arange(len(y_orig)), y_orig, color=color_orig, linestyle='-', label="Original")
            ax.plot(np.arange(len(y_mod)), y_mod, color=color_cf, linestyle='--', label="Counterfactual")

            ax.set_title("Faktor = {:.2f}".format(fac))

            ax.set_xlabel("Testdatensatz")
            if c == 0:
                ax.set_ylabel("produzierter PV-Strom [kW] (unskaliert)")

            ax.grid(alpha=0.3)

        # Gesamtüberschrift
        fig.suptitle("{}: Unskaliertes CF-Raster für '{}' (t+{})".format(
            ml_name, feature, hi), fontsize=16)

        # Zentrale Legende unten
        handles = [
            plt.Line2D([0], [0], color=color_orig, linestyle='-', label='Original'),
            plt.Line2D([0], [0], color=color_cf, linestyle='--', label='Counterfactual')
        ]
        fig.legend(handles=handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

        fig.tight_layout(rect=[0, 0.05, 1, 0.95])

        # Sicher speichern
        feat_safe = feature.replace("[", "").replace("]", "").replace(" ", "_")
        fac_strs = []
        for f in change_factors:
            fac_strs.append("{:+d}".format(int((f-1)*100)))
        fac_str = "-".join(fac_strs)

        fname = "{}_counterfactual_{}_unscaled_ts_t{}_{}.png".format(
            ml_name, feat_safe, hi, fac_str)
        path = os.path.join(out_dir, fname)

        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("✅ Unscaled Counterfactual-Raster gespeichert:", path)
        del fig




def save_combined_pdp_ice_all_inputs_horizon_output(
    model,
    ML_DATA,
    feature_names,
    feature,
    Control_Var,
    scaler_y,
    num_horizon_steps: int = 10,
    sample_indices: Optional[np.ndarray] = None,
    num_points: int = 30,
    scaler_x: Optional[Union[MinMaxScaler, StandardScaler]] = None,
    mode: Literal["aggregate", "single"] = "aggregate",   # NEU
    timestep: int = 5,                                    # nur für mode="single"
    filename: Optional[str] = None,
) -> None:
    """
    Erzeugt für jeden Prognosehorizont einen PDP+ICE‑Plot.

    mode="aggregate":  alle Timesteps des Features werden gleichzeitig
                       auf denselben Wert gesetzt (Input‑Mittelwert wird
                       als x‑Koordinate geplottet).

    mode="single":     nur ein spezifischer Timestep (Parameter `timestep`)
                       wird variiert; die x‑Achse zeigt den Wert dieses
                       Timesteps.

    ICE‑Kurven basieren auf einer Stichprobe (sample_indices);
    der PDP wird immer über *alle* Test‑Sequenzen gebildet.
    """

    # ------------------------- Vorbereitung ------------------------------
    X_test        = ML_DATA["X_TEST"].copy()
    feature_idx   = feature_names.index(feature)
    feature_clean = feature.replace("[", "").replace("]", "").replace(" ", "_")
    model_name    = Control_Var["MLtype"]
    H             = num_horizon_steps
    n_samples     = X_test.shape[0]

    if sample_indices is None:
        sample_indices = np.random.choice(n_samples, 30, replace=False)

    model_folder = f"./{model_name}"
    os.makedirs(model_folder, exist_ok=True)

    # ------------------------- Schleife über Horizonte -------------------
    for horizon_step in range(H):
        # --------------------- Wertebereich (skaliert) -------------------
        if mode == "aggregate":
            values = X_test[:, :, feature_idx].flatten()
        else:  # mode == "single"
            values = X_test[:, timestep, feature_idx]
        vmin, vmax  = np.percentile(values, [1, 99])
        value_range = np.linspace(vmin, vmax, num_points)

        # x‑Achse (Original­einheit, falls scaler_x vorhanden)
        if scaler_x is not None:
            dummy = np.zeros((value_range.size, len(feature_names)))
            dummy[:, feature_idx] = value_range
            value_range_plot = scaler_x.inverse_transform(dummy)[:, feature_idx]
        else:
            value_range_plot = value_range

        fig = plt.figure(figsize=(9, 6))
        ice_matrix = []

        # --------------------- ICE‑Kurven (Stichprobe) -------------------
        for idx in sample_indices:
            preds = []
            for val in value_range:
                X_tmp = X_test[idx:idx+1].copy()
                if mode == "aggregate":
                    X_tmp[0, :, feature_idx] = val
                else:  # single timestep
                    X_tmp[0, timestep, feature_idx] = val

                y_hat = model.predict(X_tmp, verbose=0)
                y_hat = y_hat[:, horizon_step, 0] if y_hat.ndim == 3 else y_hat[:, horizon_step]
                preds.append(
                    scaler_y.inverse_transform(y_hat.reshape(-1, 1))[0, 0]
                )

            ice_matrix.append(preds)
            plt.plot(value_range_plot, preds, alpha=0.4, linewidth=1)

        ice_matrix = np.asarray(ice_matrix)

        # --------------------- PDP (alle Test‑Sequenzen) -----------------
        pdp_all = []
        for val in value_range:
            X_mod = X_test.copy()
            if mode == "aggregate":
                X_mod[:, :, feature_idx] = val
            else:
                X_mod[:, timestep, feature_idx] = val

            y_hat = model.predict(X_mod, verbose=0)
            y_hat = y_hat[:, horizon_step, 0] if y_hat.ndim == 3 else y_hat[:, horizon_step]
            y_hat_unscaled = scaler_y.inverse_transform(y_hat.reshape(-1, 1)).ravel()
            pdp_all.append(y_hat_unscaled.mean())

        plt.plot(value_range_plot, pdp_all,
                 color="black", linewidth=2.8,
                 label="PDP (alle Samples)")

        # --------------------- Scatter‑Punkte ----------------------------
        for idx in sample_indices:
            if mode == "aggregate":
                x_val_scaled = X_test[idx, :, feature_idx].mean()
            else:
                x_val_scaled = X_test[idx, timestep, feature_idx]

            if scaler_x is not None:
                temp = np.zeros((1, len(feature_names)))
                temp[0, feature_idx] = x_val_scaled
                x_val_plot = scaler_x.inverse_transform(temp)[0, feature_idx]
            else:
                x_val_plot = x_val_scaled

            y_orig = model.predict(X_test[idx:idx+1], verbose=0)
            y_orig = y_orig[:, horizon_step, 0] if y_orig.ndim == 3 else y_orig[:, horizon_step]
            y_orig = scaler_y.inverse_transform(y_orig.reshape(-1, 1))[0, 0]
            plt.scatter(x_val_plot, y_orig, color="black", s=15, alpha=0.6)

        # --------------------- Layout & Export ---------------------------
        unit_tag = " (Original)" if scaler_x is not None else " (skaliert)"
        mode_tag = (
            f"Eingabemittelwert über {X_test.shape[1]} Timesteps"
            if mode == "aggregate"
            else f"Eingabeschritt t ={timestep}"
        )
        plt.title(
            f"PDP+ICE für '{feature}' ({mode_tag}) — "
            f"Vorhersage t={horizon_step}"
        )
        plt.xlabel(f"{feature} — {mode_tag}{unit_tag}")
        plt.ylabel("Solarstrom‑Vorhersage [kW]")
        plt.ylim(0, 12)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            model_folder,
            f"{feature_clean}_t{horizon_step}_{mode}.png"
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ PDP+ICE (mode={mode}) für Horizon {horizon_step}: {out_path}")
        del fig




def save_combined_pdp_ice_all_inputs_horizon_output(
    model,
    ML_DATA,
    feature_names,
    feature,
    Control_Var,
    scaler_y,
    num_horizon_steps: int = 10,
    sample_indices: Optional[np.ndarray] = None,
    num_points: int = 30,
    scaler_x: Optional[Union[MinMaxScaler, StandardScaler]] = None,
    mode: Literal["aggregate", "single"] = "aggregate",
    timestep: int = 5,
    filename: Optional[str] = None,
) -> None:
    """
    Erstellt für jeden Prognosehorizont (0 … num_horizon_steps‑1) einen
    PDP+ICE‑Plot. Der PDP wird immer über *alle* Test‑Sequenzen berechnet.

    mode="aggregate":  Alle Timesteps des Features erhalten denselben Wert.
                       Die x‑Achse zeigt den Mittelwert dieser Timesteps.

    mode="single":     Nur `timestep` wird variiert; x‑Achse zeigt dessen Wert.

    Wird ein scaler_x übergeben, erscheinen die x‑Werte in Original­einheiten.
    """

    # ------------------------- Vorbereitung ------------------------------
    X_test        = ML_DATA["X_TEST"].copy()
    feature_idx   = feature_names.index(feature)
    feature_clean = feature.replace("[", "").replace("]", "").replace(" ", "_")
    model_name    = Control_Var["MLtype"]
    H             = num_horizon_steps
    n_samples     = X_test.shape[0]

    if sample_indices is None:
        sample_indices = np.random.choice(n_samples, 30, replace=False)

    model_folder = f"./{model_name}"
    os.makedirs(model_folder, exist_ok=True)

    # Klammerinhalt für die x‑Achse (z. B. "°C")
    m = re.search(r"\[(.*?)\]", feature)
    feature_unit = m.group(1) if m else feature

    # ------------------------- Schleife über Horizonte -------------------
    for horizon_step in range(H):
        # --------------------- Wertebereich bestimmen --------------------
        if mode == "aggregate":
            values = X_test[:, :, feature_idx].flatten()
        else:  # mode == "single"
            values = X_test[:, timestep, feature_idx]

        vmin, vmax  = np.percentile(values, [1, 99])
        value_range = np.linspace(vmin, vmax, num_points)            # skaliert

        # x‑Werte ggf. zurückskalieren
        if scaler_x is not None:
            dummy = np.zeros((value_range.size, len(feature_names)))
            dummy[:, feature_idx] = value_range
            value_range_plot = scaler_x.inverse_transform(dummy)[:, feature_idx]
        else:
            value_range_plot = value_range

        fig = plt.figure(figsize=(9, 6))
        ice_matrix = []

        # --------------------- ICE‑Kurven (Stichprobe) -------------------
        for idx in sample_indices:
            preds = []
            for val in value_range:
                X_tmp = X_test[idx : idx + 1].copy()
                if mode == "aggregate":
                    X_tmp[0, :, feature_idx] = val
                else:
                    X_tmp[0, timestep, feature_idx] = val

                y_hat = model.predict(X_tmp, verbose=0)
                y_hat = (
                    y_hat[:, horizon_step, 0]
                    if y_hat.ndim == 3
                    else y_hat[:, horizon_step]
                )
                preds.append(
                    scaler_y.inverse_transform(y_hat.reshape(-1, 1))[0, 0]
                )

            ice_matrix.append(preds)
            plt.plot(value_range_plot, preds, alpha=0.4, linewidth=1)

        ice_matrix = np.asarray(ice_matrix)

        # --------------------- PDP (über alle Samples) -------------------
        pdp_all = []
        for val in value_range:
            X_mod = X_test.copy()
            if mode == "aggregate":
                X_mod[:, :, feature_idx] = val
            else:
                X_mod[:, timestep, feature_idx] = val

            y_hat = model.predict(X_mod, verbose=0)
            y_hat = (
                y_hat[:, horizon_step, 0]
                if y_hat.ndim == 3
                else y_hat[:, horizon_step]
            )
            y_hat_unscaled = scaler_y.inverse_transform(
                y_hat.reshape(-1, 1)
            ).ravel()
            pdp_all.append(y_hat_unscaled.mean())

        plt.plot(
            value_range_plot,
            pdp_all,
            color="black",
            linewidth=2.8,
            label="PDP",
        )

        # --------------------- Scatter‑Punkte ----------------------------
        for idx in sample_indices:
            if mode == "aggregate":
                x_val_scaled = X_test[idx, :, feature_idx].mean()
            else:
                x_val_scaled = X_test[idx, timestep, feature_idx]

            if scaler_x is not None:
                temp = np.zeros((1, len(feature_names)))
                temp[0, feature_idx] = x_val_scaled
                x_val_plot = scaler_x.inverse_transform(temp)[0, feature_idx]
            else:
                x_val_plot = x_val_scaled

            y_orig = model.predict(X_test[idx : idx + 1], verbose=0)
            y_orig = (
                y_orig[:, horizon_step, 0]
                if y_orig.ndim == 3
                else y_orig[:, horizon_step]
            )
            y_orig = scaler_y.inverse_transform(y_orig.reshape(-1, 1))[0, 0]
            plt.scatter(x_val_plot, y_orig, color="black", s=15, alpha=0.6)

        # --------------------- Layout & Export ---------------------------
        mode_tag = (
            f"Eingabemittelwert über {X_test.shape[1]} Timesteps"
            if mode == "aggregate"
            else f"Eingabeschritt t ={timestep}"
        )

        plt.title(
            f"{model_name}: PDP+ICE für '{feature}' ({mode_tag}) — Vorhersagehorizont t = {horizon_step}"
        )
        plt.xlabel(f"{feature_unit}")
        plt.ylabel("Solarstrom‑Vorhersage [kW]")
        plt.ylim(0, 12)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(
            model_folder,
            f"PDP_ICE_{feature_clean}_t{horizon_step}_{mode}.png",
        )
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅ PDP+ICE (mode={mode}) für Horizon {horizon_step}: {out_path}")
        del fig



def save_combined_pdp_ice_all_timesteps(
    model,
    ML_DATA,
    feature_names,
    feature,
    Control_Var,
    num_time_steps: int = 6,
    sample_indices: Optional[np.ndarray] = None,
    num_points: int = 30,
    is_keras_model: bool = True,
) -> None:
    """
    Erstellt für jeden Input‑Timestep t = 0 … num_time_steps‑1 einen
    PDP+ICE‑Plot und speichert *jede Grafik einzeln* als PNG.
    """

    # ---------------------------- Basics ---------------------------------
    X_test      = ML_DATA["X_TEST"].copy()
    feat_idx    = feature_names.index(feature)
    model_name  = Control_Var["MLtype"]
    feat_clean  = feature.replace("[","").replace("]","").replace(" ", "_")

    if sample_indices is None:
        sample_indices = np.random.choice(X_test.shape[0], size=20, replace=False)

    out_dir = f"./{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # ----------------------- Schleife über Timesteps ---------------------
    for ts in range(num_time_steps):
        values = X_test[:, ts, feat_idx]
        vmin, vmax  = np.percentile(values, [1, 99])
        value_range = np.linspace(vmin, vmax, num_points)

        ice_mat = []

        fig = plt.figure(figsize=(9, 6))
        # -------- ICE‑Kurven (Sample‑Subset) -----------------------------
        for idx in sample_indices:
            preds = []
            for val in value_range:
                X_tmp = X_test[idx:idx+1].copy()
                X_tmp[0, ts, feat_idx] = val
                y_hat = model.predict(X_tmp, verbose=0)
                y_val = y_hat[0, ts] if y_hat.ndim == 2 else y_hat[0][ts]
                preds.append(y_val)
            ice_mat.append(preds)
            plt.plot(value_range, preds, alpha=0.4, lw=1)

        # -------- Scatterpunkte Original -------------------------------
        for idx in sample_indices:
            x_val = X_test[idx, ts, feat_idx]
            y_val = model.predict(X_test[idx:idx+1], verbose=0)
            y_val = y_val[0, ts] if y_val.ndim == 2 else y_val[0][ts]
            plt.scatter(x_val, y_val, color="black", s=15, alpha=0.6)

        # -------- PDP ----------------------------------------------------
        ice_mat = np.asarray(ice_mat)
        pdp = ice_mat.mean(axis=0)
        plt.plot(value_range, pdp, color="black", lw=2.5, label="PDP")

        # -------- Layout & Save -----------------------------------------
        plt.title(f"PDP + ICE für '{feature}' (Eingabeschritt t={ts})")
        plt.xlabel(f"{feature} (t = {ts})")
        plt.ylabel("Solarstrom‑Vorhersage (kW)")
        plt.grid(True, ls=":")
        plt.tight_layout()

        fname = f"{model_name}_PDP_ICE_{feat_clean}_t{ts}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"✅  gespeichert: {fname}")
        del fig



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

    plt.xlabel('Testdatensatz')
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
    plt.title('Auswirkung von 50% Erhöhung in Column 7')
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

# Call function

# %%
"""
not used in the end
def sax_transform(ts, n_segments=10, alphabet_size=5):
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
# 2) Funktion, die das 3D-Dataset (n_samples, time_steps, n_features)
#    für jede Instanz in eine 1D-Sequenz flatted und SAX anwendet.
# =============================================================================
def generate_sax_for_dataset(ML_DATA, Control_Var, n_segments=10, alphabet_size=5):
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
    n_samples = data.shape[0]
    n_shapelets = len(shapelets)
    transformed_data = np.zeros((n_samples, n_shapelets))
    
    for i in range(n_samples):
        ts = data[i]
        for j, shapelet in enumerate(shapelets):
            transformed_data[i, j] = min_distance(ts, shapelet)
    return transformed_data



    def ffc_explanation(
        model,
        X,
        feature_names,
        ml_type: str,
        n_samples: int = 30,
        random_state: int = 42,
        make_plots: bool = True):
    Liefert lokale FFC‑Relevanzen S und erzeugt (optional) globale Plots.

    Zusätzliche Parameter
    ---------------------
    feature_names : list[str]
        Namen der Eingangssignale (Länge = n_features).
    ml_type : str
        Modelltyp; wird als Ordner‑/Dateipräfix verwendet.
    make_plots : bool, default=True
        Erzeugt PNG‑Plots der globalen Kennzahlen.
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

def plot_counterfactual_comparison(original_preds, counterfactual_preds, modified_indices, ControlVar):
    Plot the difference between original and counterfactual predictions.

    Parameters:
    ----------
    original_preds : np.ndarray
        Model predictions on the original dataset.
    counterfactual_preds : np.ndarray
        Model predictions on the counterfactual dataset.
    modified_indices : list
        Indices where modifications were applied.
    plt.figure(figsize=(10,5))

    # Compute mean if predictions are 2D (multi-horizon outputs)
    orig_mean = original_preds.mean(axis=1) if original_preds.ndim == 2 else original_preds
    cf_mean = counterfactual_preds.mean(axis=1) if counterfactual_preds.ndim == 2 else counterfactual_preds

    plt.plot(orig_mean, label='Originale Vorhersage', alpha=0.7)
    plt.plot(cf_mean, label='Counterfactual Vorhersage (+50% Spalte XY)', alpha=0.7)

    # Highlight modified samples
    plt.scatter(modified_indices, orig_mean[modified_indices], color='red', label='Modifizierte Inputs (Original)', zorder=5)
    plt.scatter(modified_indices, cf_mean[modified_indices], color='green', label='Modifizierte Inputs (CF)', zorder=6)

    plt.xlabel('Testdatensatz')
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
    plt.title('Auswirkung von 50% Erhöhung in Column 7')
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
    
    plt.xlabel('Testdatensatz')
    plt.ylabel('produzierter PV-Strom [kW] (skaliert)')
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
    plt.xlabel('Testdatensatz')
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


"""