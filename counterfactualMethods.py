from __future__ import annotations

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
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Functions import import_SOLETE_data, import_PV_WT_data, PreProcessDataset  
from Functions import PrepareMLmodel, TestMLmodel, post_process
import pandas as pd
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
    feature_names = [fn.replace("1", "/") for fn in feature_names]
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
            print(f"[Skip] Forecast‑Index {h+1} existiert nicht (H={h+1})")
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
        plt.title(f"{ml_type} - SHAP für aggregierte Eingabeschritte und Vorhersagehorizont h = {h+1}",fontsize=16,pad=14)
        f_out = os.path.join(out_dir, f"{ml_type}_Shap_t{h+1}_Input_{input_steps_sel}.png")
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
                f"{ml_type} - SHAP für Eingabeschritt t = {t+1} und Vorhersagehorizont h = {h+1} "
            )
            plt.gca().set_title(titel, fontsize=14, pad=12)

            f_t = os.path.join(
                out_dir, f"{ml_type}_Shap_t{h+1}_input-step{t:02d}.png"
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
from lime.lime_tabular import LimeTabularExplainer

def generate_lime_explanations(
    model,
    X_train,
    X_test,
    feature_names,
    ml_type,
    selected_indices=None,
    selected_indices_file_path='selected_indices.txt',
    num_instances=5,
    seed=42,
    horizon_step=None,      # Optional: Index des Vorhersagehorizonts (0-basiert); None = alle Ziele
    input_time_step=None     # Optional: Extrahiert nur Input-Features eines einzelnen Zeitpunkts (0-basiert);
):
    """
    Erzeugt LIME-Erklärungen und speichert jede Instanz als einzelne PNG-Datei
    mit dem Titel der Abbildung als Dateinamen.
    """

    import os, re, numpy as np
    from lime.lime_tabular import LimeTabularExplainer
    import matplotlib.pyplot as plt

    # Zielordner
    out_dir = f"./{ml_type}"
    os.makedirs(out_dir, exist_ok=True)

    # Dimensionen
    total_steps, total_feats = X_train.shape[1], X_train.shape[2]
    feature_names = [fn.replace("1", "/") for fn in feature_names]

    # Flattened Daten für LIME
    if input_time_step is not None:
        if not (0 <= input_time_step < total_steps):
            raise ValueError(f"input_time_step muss zwischen 0 und {total_steps-1} liegen.")
        X_train_flat = X_train[:, input_time_step, :]
        X_test_flat  = X_test[:,  input_time_step, :]
        lime_feature_names = feature_names
    else:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat  = X_test.reshape(X_test.shape[0],  -1)
        lime_feature_names = [
            f"{col}_{i}" for col in feature_names for i in range(total_steps)
        ]

    # Initialisiere LIME
    explainer = LimeTabularExplainer(
        training_data          = X_train_flat,
        feature_names          = lime_feature_names,
        mode                   = 'regression',
        discretize_continuous  = False
    )

    # Auswahl der Indizes
    np.random.seed(seed)
    if selected_indices is not None:
        used_indices = selected_indices
    else:
        try:
            used_indices = list(map(int, open(selected_indices_file_path).read().splitlines()))
        except Exception:
            used_indices = []
        if not used_indices:
            used_indices = np.random.choice(
                len(X_test), num_instances, replace=False
            ).tolist()
            open(selected_indices_file_path, 'w').write(
                "\n".join(map(str, used_indices))
            )

    # Erkläre jede Instanz
    for idx in used_indices:
        base_seq = X_test[idx]

        def predict_fn(flat_inst):
            flat = np.atleast_2d(flat_inst)
            n_samp = flat.shape[0]
            if input_time_step is not None:
                X_seq = np.repeat(base_seq[np.newaxis, :], n_samp, axis=0)
                X_seq[:, input_time_step, :] = flat
            else:
                X_seq = flat.reshape(n_samp, total_steps, total_feats)
            preds = model.predict(X_seq)
            return preds[:, horizon_step] if horizon_step is not None else preds[:, 0]

        expl = explainer.explain_instance(
            X_test_flat[idx], predict_fn, num_features=10
        )
        fig = expl.as_pyplot_figure()

        # 1) LIME-Suptitle entfernen
        if fig._suptitle is not None:
            fig._suptitle.remove()

        # 2) Achsentitel "Local explanation" löschen
        for ax in fig.get_axes():
            if ax.get_title().strip().lower().startswith("local explanation"):
                ax.set_title("")

        # 3) x-Achse clippen (korrekt!)
        for ax in fig.axes:
            ax.set_xlim(-0.1, 0.1)

        # Titel und Dateiname
        parts = [ml_type, f" - LIME-Diagramm für Testinstanz {idx} "]
        if input_time_step is not None:
            parts.append(f"für Eingabeschritt t = {input_time_step+1} und ")
        if horizon_step is not None:
            parts.append(f"Vorhersagehorizont h = {horizon_step+1}")
        else:
            parts.append("(voll aggregiert)")
        title = "".join(parts)

        # 4) Eigenen Titel als Figure-Text mit exakt 16 pt
        fig.text(
            0.5, 0.98, title,
            ha='center', va='top',
            fontsize=11
        )
        fig.subplots_adjust(top=0.90)

        # 5) Speichern und Schließen
        fname = re.sub(r'[^\w\-\.]', '_', "_".join(parts)) + '.png'
        fig.savefig(os.path.join(out_dir, fname), bbox_inches='tight', dpi=300)
        plt.close(fig)

    print(f"✅ LIME-Erklärungen in '{out_dir}' erstellt für Indizes: {used_indices}")
    return used_indices

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
    plt.title(f'Vorhersage: Original vs. CF für {feature} (idx={idx})',fontsize=16,pad=14)
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

#from dice_ml.exceptions import UserConfigValidationException
def safe_generate_cfs(
                     
                      horizon,
                      n_cfs=2,
                      model=None,          ML_DATA        = None,
                             feature_names=None,   
                                 idx=None,
                                     total_CFs=None,desired_range=None, method="random",
                                             # ← NEU
                                             # ← NEU
                      x_scaler=None,
                      y_scaler=None):
    """
    Liefert ein Dict mit
      • 'found'      : True/False
      • 'y_orig'     : unskalierter Ausgangswert (t+h)
      • 'y_cfs'      : Liste unskalierter CF-Werte (evtl. leer)
      • 'best_value' : größter verfügbarer Wert (CF oder Original)

    Bricht NICHT mit Exception ab, wenn keine CFs gefunden werden.
    """
    try:
        res = compute_ts_counterfactual_dice(
            model          = model,
            ML_DATA        = ML_DATA,
            feature_names  = feature_names,
            idx            = sample_id,
            total_CFs      = n_cfs,
            desired_range  = desired_range,
            horizon        = horizon,
            x_scaler       = x_scaler,
            y_scaler       = y_scaler
        )
        best_val = max(res["y_cfs"]) if res["y_cfs"] else res["y_orig"]
        return {"found": True,
                "y_orig": res["y_orig"],
                "y_cfs":  res["y_cfs"],
                "best_value": best_val,
                "cf_seq":   res["cf_examples_unscaled"]}
    except UserConfigValidationException:
        # Kein CF – wir geben wenigstens den Originalwert zurück
        orig_scaled = model.predict(ML_DATA["X_TEST"][sample_id:sample_id+1]).ravel()[horizon]
        y_orig = (float(y_scaler.inverse_transform([[orig_scaled]])[0,0])
                  if y_scaler is not None else float(orig_scaled))
        return {"found": False,
                "y_orig": y_orig,
                "y_cfs":  [],
                "best_value": y_orig,
                "cf_seq":   None}
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
    plt.title(f'Vorhersage: Original vs. CF für {feature} (idx={idx})'
    ,fontsize=16,pad=14)
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
    idx: int,
    desired_range: tuple,

    total_CFs: int = 1,
    method: str = "random",
    features_to_vary: list | None = None,
    horizon: int = -1,
    x_scaler=None,
    y_scaler=None,

    feature: str | None = None
) -> dict:
    """
    Erzeugt zeitserielle Counterfactual-Erklärungen (DiCE) für ein einzelnes
    Test-Sample.

    Parameter
    ---------
    model : sklearn-kompatibles Prognosemodell
    ML_DATA : dict
        Muss mindestens 'X_TEST' enthalten (Shape: [n, T, F]).
    feature_names : list[str]
        Namen der F Features in exakt derselben Reihenfolge wie im Modell.
    idx : int
        Index des Test-Samples in X_TEST.
    total_CFs : int, default 1
        Anzahl gewünschter Counterfactuals.
    desired_range : tuple[float, float], default (0.5, 0.5)
        Zielwert-Intervall **in Originaleinheiten**; wird intern skaliert,
        sofern `y_scaler` angegeben ist.
    method : str, default "random"
        DiCE-Erzeugungsmethode.
    features_to_vary : list[str] | None
        Liste der variierbaren Spaltennamen im Flatten-Format.  Wird
        `None` übergeben, darf DiCE alle Features verändern.
    horizon : int, default −1
        Vorhersagehorizont (0-basiert).  Negative Werte zählen rückwärts
        vom letzten Zeitschritt.
    x_scaler, y_scaler
        Instanzen von sklearn-Skalierern oder `None`.
    feature : str | None
        Name eines Features, dessen Verlauf in Overlays geplottet wird.
        Ist `None`, wird automatisch `feature_names[0]` gewählt.

    Rückgabe
    --------
    dict mit u. a.
        cf_examples_unscaled : list[np.ndarray]  # [(T, F), …]
        y_orig                : float           # unskaliert
        y_cfs                 : list[float]     # unskaliert
        … weitere skalierte Varianten
    """

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import dice_ml
    from itertools import combinations

    # ───────────────────────────────────────────────────────────────
    # 0 | allgemeine Vorbereitungen
    # ───────────────────────────────────────────────────────────────
    if feature is None:
        feature = feature_names[0]

    ml_type = getattr(model, "name", model.__class__.__name__)
    os.makedirs(ml_type, exist_ok=True)

    # ───────────────────────────────────────────────────────────────
    # 1 | Ausgangssequenz und skalierte Vorhersage
    # ───────────────────────────────────────────────────────────────
    X_test = ML_DATA["X_TEST"]
    x_seq = X_test[idx : idx + 1]                    # (1, T, F)
    preds_scaled = model.predict(x_seq).ravel()      # (H,)
    H = preds_scaled.shape[0]

    h = horizon if horizon >= 0 else H + horizon
    h = max(0, min(h, H - 1))
    y_orig_scaled = float(preds_scaled[h])

    # unskalierter Ausgangswert
    y_orig = (
        float(y_scaler.inverse_transform([[y_orig_scaled]])[0, 0])
        if y_scaler is not None
        else y_orig_scaled
    )

    # ───────────────────────────────────────────────────────────────
    # 2 | Flatten & DataFrame für DiCE
    # ───────────────────────────────────────────────────────────────
    T, F = x_seq.shape[1], x_seq.shape[2]
    flat = x_seq.reshape(1, T * F)
    col_names = [f"{fn}_{t + 1}" for t in range(T) for fn in feature_names]

    df = pd.DataFrame(flat, columns=col_names)
    df["target"] = y_orig_scaled

    # ───────────────────────────────────────────────────────────────
    # 3 | DiCE DataInterface & Modell-Wrapper
    # ───────────────────────────────────────────────────────────────
    data_dice = dice_ml.Data(
        dataframe=df, continuous_features=col_names, outcome_name="target"
    )

    def _predict_flat(X_flat):
        arr = X_flat.values if hasattr(X_flat, "values") else np.array(X_flat)
        Xr = arr.reshape(-1, T, F)
        out = model.predict(Xr).ravel()
        return np.array([out[h] for _ in range(Xr.shape[0])])

    class _Wrapper:
        def __init__(self, fn): self.fn = fn
        def predict(self, Xf): return self.fn(Xf)

    model_dice = dice_ml.Model(
        model=_Wrapper(_predict_flat),
        backend="sklearn",
        model_type="regressor"
    )
    exp = dice_ml.Dice(data_dice, model_dice, method=method)

    # ───────────────────────────────────────────────────────────────
    # 4 | gewünschter Zielbereich → Skalierungsraum
    # ───────────────────────────────────────────────────────────────
    if y_scaler is not None:
        desired_range_scaled = tuple(
            y_scaler.transform(np.array(desired_range).reshape(-1, 1)).ravel()
        )
    else:
        desired_range_scaled = desired_range

    # ───────────────────────────────────────────────────────────────
    # 5 | Counterfactuals generieren
    # ───────────────────────────────────────────────────────────────
    if features_to_vary is None:
        features_to_vary = col_names

    query_instance = df[features_to_vary].iloc[[0]]

    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=total_CFs,
        desired_range=desired_range_scaled,
        features_to_vary=features_to_vary, 
        sample_size      = 200000
    )

    # DataFrame mit CFs robust extrahieren
    cf_df = getattr(dice_exp.cf_examples_list[0], "final_cfs_df", None)
    if cf_df is None:
        # Fallback: erstes DataFrame-Attribut suchen
        for a in dir(dice_exp.cf_examples_list[0]):
            obj = getattr(dice_exp.cf_examples_list[0], a)
            if isinstance(obj, pd.DataFrame):
                cf_df = obj
                break
        if cf_df is None:
            raise RuntimeError("Counterfactual-DataFrame nicht gefunden.")

    # ───────────────────────────────────────────────────────────────
    # 6 | skalierte CF-Sequenzen & Vorhersagen
    # ───────────────────────────────────────────────────────────────
    cf_flat = cf_df[features_to_vary].values
    cf_examples_scaled = [cf.reshape(T, F) for cf in cf_flat]

    y_cfs_scaled = [
        float(model.predict(cf.reshape(1, T, F)).ravel()[h])
        for cf in cf_examples_scaled
    ]

    # Unskalierte Zielwerte
    if y_scaler is not None:
        y_cfs = [
            float(y_scaler.inverse_transform([[yc]])[0, 0])
            for yc in y_cfs_scaled
        ]
    else:
        y_cfs = y_cfs_scaled

            # ───────────────────────────────────────────────────────────────
    # 6a | Werte auf Trainings-Min/Max begrenzen           ✱ NEU
    # ───────────────────────────────────────────────────────────────
    if x_scaler is not None:                       # nur falls ein Skaler übergeben wurde
        # Grenzen im Originalraum bestimmen
        if hasattr(x_scaler, "data_min_") and hasattr(x_scaler, "data_max_"):
            orig_mins = x_scaler.data_min_
            orig_maxs = x_scaler.data_max_
        elif "X_TRAIN" in ML_DATA:                 # Fallback für z. B. StandardScaler
            flat_train = ML_DATA["X_TRAIN"].reshape(-1, cf_examples_scaled[0].shape[-1])
            orig_mins = flat_train.min(axis=0)
            orig_maxs = flat_train.max(axis=0)
        else:
            raise ValueError(
                "Feature-Grenzen konnten nicht bestimmt werden "
                "(x_scaler ohne data_min_/data_max_ und X_TRAIN fehlt)."
            )

        # Grenzen in den skalierten Raum übertragen
        mins_scaled = x_scaler.transform(orig_mins.reshape(1, -1)).ravel()
        maxs_scaled = x_scaler.transform(orig_maxs.reshape(1, -1)).ravel()

        # Jede CF-Sequenz hart clippen
        cf_examples_scaled = [
            np.clip(cf, mins_scaled, maxs_scaled) for cf in cf_examples_scaled
        ]

    # ───────────────────────────────────────────────────────────────
    # 7 | Unskalierung der Feature-Sequenzen (für Plots)
    # ───────────────────────────────────────────────────────────────
    if x_scaler is not None:
        seq_unscaled = x_scaler.inverse_transform(
            x_seq.reshape(-1, F)
        ).reshape(x_seq.shape)
        cf_examples = [
            x_scaler.inverse_transform(cf.reshape(-1, F)).reshape(T, F)
            for cf in cf_examples_scaled
        ]
    else:
        seq_unscaled = x_seq.copy()
        cf_examples = [cf.copy() for cf in cf_examples_scaled]

    # ───────────────────────────────────────────────────────────────
    # 8 | Visualisierungen (alle unskaliert)
    # ───────────────────────────────────────────────────────────────
    feat_idx = feature_names.index(feature)

    # 8.1 Balkendiagramm
    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Original"] + [f"CF{i+1}" for i in range(len(y_cfs))],
        [y_orig] + y_cfs
    )
    plt.ylabel("Zielwert")
    plt.title(f"DiCE-Vorhersagen (Horizon {h+1})")
    plt.savefig(
        os.path.join(ml_type, "dice_predictions_bar.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 8.2 Zeitreihen-Overlay
    plt.figure(figsize=(8, 4))
    plt.plot(
        seq_unscaled[0, :, feat_idx], "-o",
        color="black", label="Original"
    )
    for i, cf in enumerate(cf_examples):
        plt.plot(cf[:, feat_idx], "--", label=f"CF{i+1}")
    plt.xlabel("Timesteps")
    plt.title(f"Overlay {feature}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(ml_type, "dice_timeseries_overlay.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 8.3 Parallel-Koordinaten
    try:
        import pandas.plotting as pd_plot
        df_pc = pd.DataFrame(
            [cf.reshape(-1) for cf in cf_examples] + [seq_unscaled.reshape(-1)],
            columns=col_names
        )
        df_pc["Typ"] = [f"CF{i+1}" for i in range(len(cf_examples))] + ["Original"]
        plt.figure(figsize=(10, 4))
        pd_plot.parallel_coordinates(df_pc, "Typ")
        plt.title("Parallel Coordinates")
        plt.savefig(
            os.path.join(ml_type, "dice_parallel_coordinates.png"),
            dpi=300, bbox_inches="tight"
        )
        plt.close()
    except Exception:
        pass  # fehlende Abhängigkeit oder zu viele Spalten

    # 8.4 Scatter-Plot (erste zwei Features)
    idx_x, idx_y = 0, 1
    ox = seq_unscaled[0, :, idx_x].mean()
    oy = seq_unscaled[0, :, idx_y].mean()
    plt.figure(figsize=(5, 5))
    plt.scatter(ox, oy, c="black", label="Original")
    for i, cf in enumerate(cf_examples):
        cx = cf[:, idx_x].mean()
        cy = cf[:, idx_y].mean()
        plt.scatter(cx, cy, label=f"CF{i+1}")
        plt.arrow(ox, oy, cx - ox, cy - oy,
                  head_width=0.02, length_includes_head=True)
    plt.xlabel(feature_names[idx_x]); plt.ylabel(feature_names[idx_y])
    plt.title("2D Scatter CFs")
    plt.legend()
    plt.savefig(
        os.path.join(ml_type, "dice_scatter2d.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 8.5 Heatmap der Δ-Werte
    deltas = np.array([cf.reshape(-1) - seq_unscaled.reshape(-1) for cf in cf_examples])
    plt.figure(figsize=(8, 4))
    plt.imshow(deltas, aspect="auto", cmap="coolwarm", interpolation="nearest")
    plt.colorbar(label="Δ Value")
    plt.title("Δ (CF − Original)")
    plt.savefig(
        os.path.join(ml_type, "dice_heatmap_deltas.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # 8.6 Diversity & Proximity
    flats = [cf.reshape(-1) for cf in cf_examples]
    diversity = float(
        np.mean([np.linalg.norm(flats[i] - flats[j])
                 for i, j in combinations(range(len(flats)), 2)])
    ) if len(flats) > 1 else 0.0
    proximity = float(
        -np.mean([np.linalg.norm(f - seq_unscaled.reshape(-1)) for f in flats])
    ) if flats else 0.0
    plt.figure(figsize=(4, 3))
    plt.bar(["Diversity", "Proximity"], [diversity, proximity])
    plt.title("Diversity vs Proximity")
    plt.savefig(
        os.path.join(ml_type, "dice_diversity_proximity.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()

    # ───────────────────────────────────────────────────────────────
    # 9 | Rückgabe
    # ───────────────────────────────────────────────────────────────
    return {
        "cf_examples_scaled":   cf_examples_scaled,
        "cf_examples_unscaled": cf_examples,
        "y_orig_scaled":        y_orig_scaled,
        "y_orig":               y_orig,
        "y_cfs_scaled":         y_cfs_scaled,
        "y_cfs":                y_cfs
    }


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

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import viridis, ScalarMappable

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable, viridis


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable, viridis

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable, viridis


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable, viridis


def cf_scatter_percent(
    ML_DATA,
    model,
    feature_names,
    feature,
    factors=(0.50, 0.75, 1.25, 1.50),
    Control_Var=None,
    timestep_idx_input=None,          # 0–5 | None ⇒ ∅ aller 6 Eingabe-TS
    timestep_idx_forecast=None,       # 0–9 | None ⇒ ∅ aller 10 Output-TS
    bg_idx=None,
    jitter=0.3,
    lin_thresh=20.0,
    aggregate_input_timesteps=True,
    aggregate_output_timesteps=True,
    color_by_distance=True,
    verbose=False
):
    """Scatter-Raster der %-Δ-Vorhersage; ohne Punkt-Legende, nur Colorbar."""
    # ───────────────── Basisdaten ────────────────────────────────────────────
    X_all = ML_DATA["X_TEST"]                      # (N, IN_TS, F)
    N, IN_TS, F = X_all.shape
    X_raw = X_all[bg_idx] if bg_idx is not None else X_all

    y_probe = model.predict(X_raw[:1])
    if y_probe.ndim == 1:
        y_probe = y_probe[:, None]
    OUT_TS = y_probe.shape[1]

    if timestep_idx_input is not None:
        timestep_idx_input = int(np.clip(timestep_idx_input, 0, IN_TS - 1))
    if timestep_idx_forecast is not None:
        timestep_idx_forecast = int(np.clip(timestep_idx_forecast, 0, OUT_TS - 1))

    feat_idx = feature_names.index(feature)
    sigma_feat = X_all[:, :, feat_idx].std(ddof=0)
    if sigma_feat == 0:
        raise ValueError(f"σ({feature}) = 0 – Distanz nicht definiert.")
    eps = 1e-12

    # ───────── Basis-Input in Modell-Shape ───────────────────────────────────
    if aggregate_input_timesteps:
        # 1:1-Übernahme des Originals → Spitzen bleiben erhalten
        X_base_in = X_raw.copy()            # Shape (N, IN_TS, F)
    else:
        # Optional: gezielten Zeitschritt duplizieren (unverändert)
        idx_in = timestep_idx_input if timestep_idx_input is not None else -1
        X_base_in = np.repeat(X_raw[:, [idx_in], :], IN_TS, 1)
    y_base_pred = model.predict(X_base_in)
    if y_base_pred.ndim == 1:
        y_base_pred = y_base_pred[:, None]
    y_base = (y_base_pred.mean(axis=1) if aggregate_output_timesteps
              else y_base_pred[:, timestep_idx_forecast])

    # ───────── Plot-Setup ────────────────────────────────────────────────────
    n_fac = len(factors)
    fig, axes = plt.subplots(1, n_fac, figsize=(5 * n_fac, 4),
                             sharex=True, sharey=True)
    if n_fac == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    fixed_ticks = [-1000, -100, -10, 0, 10, 100, 1000]
    sf = mticker.ScalarFormatter(useOffset=False)
    sf.set_scientific(False)

    norm = plt.Normalize(0, max(factors) * 2)
    sm = ScalarMappable(norm=norm, cmap=viridis)

    # ───────── Counterfactuals ───────────────────────────────────────────────
    for ax, fac in zip(axes, factors):
        X_cf = X_base_in.copy()
        X_cf[:, :, feat_idx] *= fac

        y_cf_pred = model.predict(X_cf)
        if y_cf_pred.ndim == 1:
            y_cf_pred = y_cf_pred[:, None]
        y_cf = (y_cf_pred.mean(axis=1) if aggregate_output_timesteps
                else y_cf_pred[:, timestep_idx_forecast])

        delta_pct = (y_cf - y_base) / (np.abs(y_base) + eps) * 100
        delta_pct = np.clip(delta_pct, -1000, 1000)

        dist = np.abs(X_cf[:, 0, feat_idx] - X_base_in[:, 0, feat_idx]) / (sigma_feat + eps)

        if verbose:
            print(f"Fac {fac:.2f}: Δ-Median {np.median(delta_pct):+.2f}%  "
                  f"dist-Median {np.median(dist):.2f}")

        m = np.abs(y_base) > 1e-3
        xs = np.where(m)[0] + rng.normal(0, jitter, m.sum())
        ys = delta_pct[m]
        colours = sm.to_rgba(dist[m]) if color_by_distance else "tab:blue"

        ax.scatter(xs, ys, s=20, alpha=0.7, c=colours)

        ax.axhline(0, color="gray", linewidth=1)
        ax.set_yscale("symlog", linthresh=lin_thresh)
        ax.set_ylim(-1000, 1000)
        ax.set_yticks(fixed_ticks)
        ax.yaxis.set_major_formatter(sf)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.set_title(f"Faktor {fac:.2f}")

    axes[0].set_ylabel("Δ Vorhersage [%]")
    axes[-1].set_xlabel("Test-Sample")

    # ───────── Titel ─────────────────────────────────────────────────────────
    ml_name = (Control_Var or {}).get("MLtype", "model")
    in_txt  = "Input ∅" if aggregate_input_timesteps else f"Input t={timestep_idx_input}"
    out_txt = "Output ∅" if aggregate_output_timesteps else f"Output t={timestep_idx_forecast}"
    fig.suptitle(f"{ml_name}: %-Δ bei Skalierung von {feature}  ({in_txt}, {out_txt})",
                 y=0.96, fontsize=12)

    # ───────── Layout: Colorbar ohne Punkt-Legende ───────────────────────────
    fig.tight_layout(rect=[0, 0.25, 1, 0.98])

    cax = fig.add_axes([0.25, 0.12, 0.70, 0.05])   # [left, bottom, width, height]
    cb  = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Distanz (σ-normiert)")

    # ───────── Speichern ─────────────────────────────────────────────────────
    feat_safe = feature.replace("[", "").replace("]", "").replace("/", "_")
    fname = (f"{ml_name}_cf_scatter_{feat_safe}_"
             f"{'inAvg' if aggregate_input_timesteps else f'in{timestep_idx_input}'}_"
             f"{'outAvg' if aggregate_output_timesteps else f'out{timestep_idx_forecast}'}.png")
    out_dir = os.path.join(".", ml_name)
    os.makedirs(out_dir, exist_ok=True)
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    print("✅ Scatter-Plot gespeichert:", fpath)

import os, random, numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
def cf_scatter_percent_zufällig(
    ML_DATA, model, feature_names, feature,
    factors=(0.5, 0.75, 1.25, 1.5),
    Control_Var=None,
    timestep_idx_input=None, timestep_idx_forecast=None,
    bg_idx=None,
    jitter=0.3, lin_thresh=20.0,
    aggregate_input_timesteps=True, aggregate_output_timesteps=True,
    x_scaler=None, y_scaler=None,
    verbose=False, debug=False,
    fixed_ymax_abs=None
):
    # ── Plausibilitätsprüfung ──────────────────────────────────────────
    if x_scaler is None or not all(hasattr(x_scaler, a)
                                   for a in ("data_min_", "data_max_", "scale_", "min_")):
        raise ValueError("x_scaler fehlt oder unvollständig.")
    if y_scaler is None or not hasattr(y_scaler, "inverse_transform"):
        raise ValueError("y_scaler (mit inverse_transform) fehlt.")

    # ── Basisdaten ------------------------------------------------------
    X_all = ML_DATA["X_TEST"]
    X_raw = X_all[bg_idx] if bg_idx is not None else X_all
    _, IN_TS, F = X_all.shape
    N= X_raw.shape[0]
    y_probe = model.predict(X_raw[:1])
    if y_probe.ndim == 1: y_probe = y_probe[:, None]
    OUT_TS = y_probe.shape[1]

    # Index-Grenzen absichern
    timestep_idx_input    = int(np.clip(
        timestep_idx_input if timestep_idx_input is not None else -1,
        -IN_TS, IN_TS-1))
    timestep_idx_forecast = int(np.clip(
        timestep_idx_forecast if timestep_idx_forecast is not None else -1,
        -OUT_TS, OUT_TS-1))

    feat_idx = feature_names.index(feature)
    to_orig  = lambda x: (x - x_scaler.min_[feat_idx]) / x_scaler.scale_[feat_idx]
    to_scaled= lambda x: x * x_scaler.scale_[feat_idx] + x_scaler.min_[feat_idx]
    eps = 1e-12

    # ── Basis-Input -----------------------------------------------------
    if aggregate_input_timesteps:
        X_base_in = X_raw.copy()              # komplette Sequenz
    else:
        X_base_in = X_raw.copy()              # Sequenz unverändert

    y_base_scaled = model.predict(X_base_in)
    if y_base_scaled.ndim == 1: y_base_scaled = y_base_scaled[:, None]
    y_base_unscaled = y_scaler.inverse_transform(y_base_scaled)

    # ── Counterfactuals -------------------------------------------------
    y_cf_vals_list, was_clipped_list = [], []
    for fac in factors:
        X_cf = X_base_in.copy()

        # Ganze Sequenz unskaliert holen
        x_orig_full = to_orig(X_cf[:, :, feat_idx])

        # Nur den gewählten Input-Step ändern
        x_mod_step  = np.clip(
            x_orig_full[:, timestep_idx_input] * fac,
            x_scaler.data_min_[feat_idx],
            x_scaler.data_max_[feat_idx]
        )
        x_orig_full[:, timestep_idx_input] = x_mod_step
        X_cf[:, :, feat_idx] = to_scaled(x_orig_full)

        was_clipped = (x_mod_step == x_scaler.data_min_[feat_idx]) | \
                      (x_mod_step == x_scaler.data_max_[feat_idx])
        was_clipped_list.append(was_clipped)

        y_cf_scaled = model.predict(X_cf)
        if y_cf_scaled.ndim == 1: y_cf_scaled = y_cf_scaled[:, None]
        y_cf_vals_list.append(y_scaler.inverse_transform(y_cf_scaled))

    # ── Forecast-Slice --------------------------------------------------
    if aggregate_output_timesteps:
        y_base_vals = y_base_unscaled.mean(axis=1)
        y_cf_vals_list = [ycf.mean(axis=1) for ycf in y_cf_vals_list]
        horizon_tag = ""
    else:
        y_base_vals = y_base_unscaled[:, timestep_idx_forecast]
        y_cf_vals_list = [ycf[:, timestep_idx_forecast] for ycf in y_cf_vals_list]
        horizon_tag = f"und Vorhersagehorizont h = {timestep_idx_forecast+1}"

    # ── Clipping nur für Plot-Kopien ------------------------------------
    ymin = 0
    all_y = np.concatenate([y_base_vals] + y_cf_vals_list)
    ymax = 7.5
    y_base_plot, y_cf_plotlist = y_base_vals, y_cf_vals_list

    # ── Titelbausteine --------------------------------------------------
    model_name = Control_Var.get("MLtype", "model") if Control_Var else "model"
    input_tag  = "(voll aggregiert)" if aggregate_input_timesteps else f"für Eingabeschritt t = {timestep_idx_input+1}"

    # ── Plot 1 – Δ-Prozent-Scatter -------------------------------------
    n_fac = len(factors)
    fig_pct, ax_pct = plt.subplots(
        1, n_fac,
        figsize=(5 * n_fac, 4),
        gridspec_kw={'wspace': 0.4}   # gleicht den Abstand an den Absolut-Plot an
    )
    rng = np.random.default_rng(0)
    sf = mticker.ScalarFormatter(useOffset=False); sf.set_scientific(False)

    handles_cache = None
    for ax, fac, ycf_vals, clipped in zip(ax_pct, factors, y_cf_vals_list, was_clipped_list):
        delta_pct = (ycf_vals - y_base_vals) / (np.abs(y_base_vals)+eps) * 100
        delta_pct = np.clip(delta_pct, -10000, 10000)

        normal  = ~clipped
        sc1 = ax.scatter(np.where(normal)[0] + rng.normal(0,jitter,normal.sum()),
                         delta_pct[normal], s=20, alpha=0.7, c="tab:blue", marker="o")
        sc2 = ax.scatter(np.where(clipped)[0]+ rng.normal(0,jitter,clipped.sum()),
                         delta_pct[clipped], s=40, alpha=0.9, c="red", marker="x")
        if handles_cache is None: handles_cache = [sc1, sc2]

        ax.axhline(0, color="gray", lw=1)
        ax.set_yscale("symlog", linthresh=lin_thresh)
        ax.set_ylim(-1000, 1000)
        ax.grid(ls=":", lw=0.5); ax.set_title(f"Faktor {fac:.2f}")
        ax.set_xlabel("Testinstanz")
        ax.set_ylabel("Δ Vorhersage [%]")
        ax.set_xlim(0, 100)      

    handles_pct = [
        Line2D([0], [0], marker="o", color="tab:blue",   linestyle="none", ms=6, label="nicht geclippt"),
        Line2D([0], [0], marker="x", color="red",        linestyle="none", ms=6, label="geclippt")
    ]

    # 2) Legende exakt wie im Absolut-Plot platzieren und stylen
    fig_pct.subplots_adjust(top=0.85, bottom=0.25)

    # 2) Legende an der Figure, unten mittig einhängen
    fig_pct.legend(
        handles=handles_pct,
        loc='lower center',               # Anker in der Mitte unten
        bbox_to_anchor=(0.5, 0.05),       # x=0.5 (mittel), y=0.05 (5% über der unteren Figure-Grenze)
        ncol=2,
        frameon=False
    )
    feature_disp = feature.replace("1", "/")
    fig_pct.suptitle(f"{model_name} – Prozentuales Counterfactual zufällige Werte: {feature_disp} {input_tag} {horizon_tag}",
                     y=0.97)


    out_dir = os.path.join(".", model_name); os.makedirs(out_dir, exist_ok=True)
    feat_safe = feature.replace("[", "").replace("]", "").replace("/", "1")
    fname_pct = f"{model_name}_cf_scatter_{feat_safe}_{input_tag.replace(' ','')}_{horizon_tag}_perc.png"
    fname_pct_save = fname_pct.replace("/", "1")
    fig_pct.savefig(os.path.join(out_dir, fname_pct_save), dpi=300); plt.close(fig_pct)

    # ── Plot 2 – Absolut-Vorher/Nachher ---------------------------------
    fig_abs, ax_abs = plt.subplots(1, n_fac, figsize=(5*n_fac, 4), sharex=True, gridspec_kw={'wspace': 0.4})
    ax_abs = np.atleast_1d(ax_abs)
    xs = np.arange(len(y_base_plot))

    for ax, fac, ycf_vals, ycf_plot in zip(ax_abs, factors, y_cf_vals_list, y_cf_plotlist):
        ax.plot(np.vstack([xs,xs]), np.vstack([y_base_plot,ycf_plot]),
                color="gray", lw=0.6, alpha=0.5)
        ax.scatter(xs, y_base_plot, s=22, c="tab:blue", marker="o", alpha=0.8)
        ax.scatter(xs, ycf_plot,   s=30, c="tab:orange", marker="^", alpha=0.9)
        ax.set_ylim(ymin, ymax); ax.grid(ls=":", lw=0.5)
        ax.set_title(f"Faktor {fac:.2f}")
        ax.set_xlabel("Testinstanz")
        ax.set_ylabel("Vorhersage (kW, unskaliert)")
        ax.set_xlim(0, N)      

    handles = [
        Line2D([0], [0], marker="o", color="tab:blue",   linestyle="none", ms=6, label="Originale Vorhersage"),
        Line2D([0], [0], marker="^", color="tab:orange", linestyle="none", ms=6, label="Counterfactual Vorhersage")
    ]
    feature_disp = feature.replace("1", "/")

    fig_abs.suptitle(f"{model_name} – Absolutes Counterfactual zufällige Werte: {feature_disp} {input_tag} {horizon_tag}",
                     y=0.97)
    fig_abs.subplots_adjust(top=0.85, bottom=0.30)      # mehr unteren Rand
    fig_abs.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),                     # ggf. Legendensitz weiter nach unten
        frameon=False,
        borderaxespad=1.0                               # Abstand zwischen Legende und Achsen
    )
    
    fig_abs.tight_layout(rect=[0, 0.15, 1, 0.98]) 
    fname_abs = fname_pct.replace("/","1").replace("perc","abs")
    fig_abs.savefig(os.path.join(out_dir, fname_abs), dpi=300); plt.close(fig_abs)

    if debug:
        print("Max unskaliert:", y_base_unscaled.max())
        print("Max Slice    :", y_base_vals.max())
    print("✅ Scatter-Plot gespeichert:", fname_pct)
    print("✅ Absolut-Plot gespeichert:", fname_abs)

def cf_scatter_percent_max(
    ML_DATA, model, feature_names, feature,
    factors=(0.5, 0.75, 1.25, 1.5),
    Control_Var=None,
    timestep_idx_input=None, timestep_idx_forecast=None,
    bg_idx=None,
    jitter=0.3, lin_thresh=20.0,
    aggregate_input_timesteps=True, aggregate_output_timesteps=True,
    x_scaler=None, y_scaler=None,
    verbose=False, debug=False, fixed_ymax_abs=None
):
    # ── Plausibilitätsprüfung ──────────────────────────────────────────
    if x_scaler is None or not all(hasattr(x_scaler, a)
                                   for a in ("data_min_", "data_max_", "scale_", "min_")):
        raise ValueError("x_scaler fehlt oder unvollständig.")
    if y_scaler is None or not hasattr(y_scaler, "inverse_transform"):
        raise ValueError("y_scaler (mit inverse_transform) fehlt.")

    # ── Basisdaten ------------------------------------------------------
    X_all = ML_DATA["X_TEST"]
    X_raw = X_all[bg_idx] if bg_idx is not None else X_all
    _, IN_TS, F = X_all.shape
    N= X_raw.shape[0]
    y_probe = model.predict(X_raw[:1])
    if y_probe.ndim == 1: y_probe = y_probe[:, None]
    OUT_TS = y_probe.shape[1]

    # Index-Grenzen absichern
    timestep_idx_input    = int(np.clip(
        timestep_idx_input if timestep_idx_input is not None else -1,
        -IN_TS, IN_TS-1))
    timestep_idx_forecast = int(np.clip(
        timestep_idx_forecast if timestep_idx_forecast is not None else -1,
        -OUT_TS, OUT_TS-1))

    feat_idx = feature_names.index(feature)
    to_orig  = lambda x: (x - x_scaler.min_[feat_idx]) / x_scaler.scale_[feat_idx]
    to_scaled= lambda x: x * x_scaler.scale_[feat_idx] + x_scaler.min_[feat_idx]
    eps = 1e-12

    # ── Basis-Input -----------------------------------------------------
    if aggregate_input_timesteps:
        X_base_in = X_raw.copy()              # komplette Sequenz
    else:
        X_base_in = X_raw.copy()              # Sequenz unverändert

    y_base_scaled = model.predict(X_base_in)
    if y_base_scaled.ndim == 1: y_base_scaled = y_base_scaled[:, None]
    y_base_unscaled = y_scaler.inverse_transform(y_base_scaled)

    # ── Counterfactuals -------------------------------------------------
    y_cf_vals_list, was_clipped_list = [], []
    for fac in factors:
        X_cf = X_base_in.copy()

        # Ganze Sequenz unskaliert holen
        x_orig_full = to_orig(X_cf[:, :, feat_idx])

        # Nur den gewählten Input-Step ändern
        x_mod_step  = np.clip(
            x_orig_full[:, timestep_idx_input] * fac,
            x_scaler.data_min_[feat_idx],
            x_scaler.data_max_[feat_idx]
        )
        x_orig_full[:, timestep_idx_input] = x_mod_step
        X_cf[:, :, feat_idx] = to_scaled(x_orig_full)

        was_clipped = (x_mod_step == x_scaler.data_min_[feat_idx]) | \
                      (x_mod_step == x_scaler.data_max_[feat_idx])
        was_clipped_list.append(was_clipped)

        y_cf_scaled = model.predict(X_cf)
        if y_cf_scaled.ndim == 1: y_cf_scaled = y_cf_scaled[:, None]
        y_cf_vals_list.append(y_scaler.inverse_transform(y_cf_scaled))

    # ── Forecast-Slice --------------------------------------------------
    if aggregate_output_timesteps:
        y_base_vals = y_base_unscaled.mean(axis=1)
        y_cf_vals_list = [ycf.mean(axis=1) for ycf in y_cf_vals_list]
        horizon_tag = ""
    else:
        y_base_vals = y_base_unscaled[:, timestep_idx_forecast]
        y_cf_vals_list = [ycf[:, timestep_idx_forecast] for ycf in y_cf_vals_list]
        horizon_tag = f"und Vorhersagehorizont h = {timestep_idx_forecast+1}"

    # ── Clipping nur für Plot-Kopien ------------------------------------
    ymin = 0
    ymax = 7.5
    y_base_plot, y_cf_plotlist = y_base_vals, y_cf_vals_list

    # ── Titelbausteine --------------------------------------------------
    model_name = Control_Var.get("MLtype", "model") if Control_Var else "model"
    input_tag  = "(voll aggregiert)" if aggregate_input_timesteps else f"für Eingabeschritt t = {timestep_idx_input+1}"

    # ── Plot 1 – Δ-Prozent-Scatter -------------------------------------
    n_fac = len(factors)
    fig_pct, ax_pct = plt.subplots(
        1, n_fac,
        figsize=(5 * n_fac, 4),
        gridspec_kw={'wspace': 0.4}   # gleicht den Abstand an den Absolut-Plot an
    )
    ax_pct = np.atleast_1d(ax_pct)
    rng = np.random.default_rng(0)
    sf = mticker.ScalarFormatter(useOffset=False); sf.set_scientific(False)

    handles_cache = None
    for ax, fac, ycf_vals, clipped in zip(ax_pct, factors, y_cf_vals_list, was_clipped_list):
        delta_pct = (ycf_vals - y_base_vals) / (np.abs(y_base_vals)+eps) * 100
        delta_pct = np.clip(delta_pct, -10000, 10000)

        normal  = ~clipped
        sc1 = ax.scatter(np.where(normal)[0] + rng.normal(0,jitter,normal.sum()),
                         delta_pct[normal], s=20, alpha=0.7, c="tab:blue", marker="o")
        sc2 = ax.scatter(np.where(clipped)[0]+ rng.normal(0,jitter,clipped.sum()),
                         delta_pct[clipped], s=40, alpha=0.9, c="red", marker="x")
        if handles_cache is None: handles_cache = [sc1, sc2]

        ax.axhline(0, color="gray", lw=1)
        ax.set_yscale("symlog", linthresh=lin_thresh)
        ax.set_ylim(-1000, 1000)
        ax.grid(ls=":", lw=0.5); ax.set_title(f"Faktor {fac:.2f}")
        ax.set_xlabel("Testinstanz")
        ax.set_ylabel("Δ Vorhersage [%]")
        ax.set_xlim(0, 100)      

    handles_pct = [
        Line2D([0], [0], marker="o", color="tab:blue",   linestyle="none", ms=6, label="nicht geclippt"),
        Line2D([0], [0], marker="x", color="red",        linestyle="none", ms=6, label="geclippt")
    ]

    # 2) Legende exakt wie im Absolut-Plot platzieren und stylen
    fig_pct.subplots_adjust(top=0.85, bottom=0.25)

    # 2) Legende an der Figure, unten mittig einhängen
    fig_pct.legend(
        handles=handles_pct,
        loc='lower center',               # Anker in der Mitte unten
        bbox_to_anchor=(0.5, 0.05),       # x=0.5 (mittel), y=0.05 (5% über der unteren Figure-Grenze)
        ncol=2,
        frameon=False
    )
    feature_disp = feature.replace("1", "/")
    fig_pct.suptitle(f"{model_name} – Prozentuales Counterfactual maximale Werte: {feature_disp} {input_tag} {horizon_tag}",
                     y=0.97)


    out_dir = os.path.join(".", model_name); os.makedirs(out_dir, exist_ok=True)
    feat_safe = feature.replace("[", "").replace("]", "").replace("/", "1")
    fname_pct = f"{model_name}_cf_scatter_max_{feat_safe}_{input_tag.replace(' ','')}_{horizon_tag}_perc.png"
    fname_pct_save = fname_pct.replace("/", "1")
    fig_pct.savefig(os.path.join(out_dir, fname_pct_save), dpi=300); plt.close(fig_pct)

    # ── Plot 2 – Absolut-Vorher/Nachher ---------------------------------
    fig_abs, ax_abs = plt.subplots(1, n_fac, figsize=(5*n_fac, 4), sharex=True, gridspec_kw={'wspace': 0.4})
    ax_abs = np.atleast_1d(ax_abs)
    xs = np.arange(len(y_base_plot))

    for ax, fac, ycf_vals, ycf_plot in zip(ax_abs, factors, y_cf_vals_list, y_cf_plotlist):
        ax.plot(np.vstack([xs,xs]), np.vstack([y_base_plot,ycf_plot]),
                color="gray", lw=0.6, alpha=0.5)
        ax.scatter(xs, y_base_plot, s=22, c="tab:blue", marker="o", alpha=0.8)
        ax.scatter(xs, ycf_plot,   s=30, c="tab:orange", marker="^", alpha=0.9)
        ax.set_ylim(ymin, ymax); ax.grid(ls=":", lw=0.5)
        ax.set_title(f"Faktor {fac:.2f}")
        ax.set_xlabel("Testinstanz")
        ax.set_ylabel("Vorhersage (kW, unskaliert)")
        ax.set_xlim(0, N)      

    handles = [
        Line2D([0], [0], marker="o", color="tab:blue",   linestyle="none", ms=6, label="Originale Vorhersage"),
        Line2D([0], [0], marker="^", color="tab:orange", linestyle="none", ms=6, label="Counterfactual Vorhersage")
    ]
    feature_disp = feature.replace("1", "/")

    fig_abs.suptitle(f"{model_name} – Absolutes Counterfactual maximale Werte: {feature_disp} {input_tag} {horizon_tag}",
                     y=0.97)
    fig_abs.subplots_adjust(top=0.85, bottom=0.30)      # mehr unteren Rand
    fig_abs.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),                     # ggf. Legendensitz weiter nach unten
        frameon=False,
        borderaxespad=1.0                               # Abstand zwischen Legende und Achsen
    )
    
    fig_abs.tight_layout(rect=[0, 0.15, 1, 0.98]) 
    fname_abs = fname_pct.replace("/","1").replace("perc","abs")
    fig_abs.savefig(os.path.join(out_dir, fname_abs), dpi=300); plt.close(fig_abs)

    if debug:
        print("Max unskaliert:", y_base_unscaled.max())
        print("Max Slice    :", y_base_vals.max())
    print("✅ Scatter-Plot gespeichert:", fname_pct)
    print("✅ Absolut-Plot gespeichert:", fname_abs)

###############################################################################
#Scatter‑Plot (Prozentuale Änderung EIN‑ vs. AUS‑Größe)                   #
###############################################################################
# ───────────────────────────────── what_if.py ──────────────────────────────────
def run_counterfactuals(model,
                        ML_DATA: dict,
                        Control_Var: dict,
                        sample_id: int = 0,
                        total_CFs: int = 3,
                        desired_range: tuple = (0.9, 1.0),
                        pr_low: float = 0.1,
                        pr_high: float = 0.9):
    """
    Erstellt Counterfactuals für das letzte Prognose-Intervall (t = H–1)
    und gibt sie in unskalierten Einheiten zurück.
    
    Neuer Parameter:
      pr_low, pr_high: untere/obere Grenze im skalierten Raum für alle Features
    """
    # -----------------------------------------------------------------
    # 1  Vorbereitungen
    # -----------------------------------------------------------------
    import numpy as np
    import pandas as pd
    import dice_ml
    import joblib

    PRE           = Control_Var['PRE']
    H             = Control_Var['H']
    feature_names = ML_DATA['xcols']
    MLtype        = Control_Var['MLtype']

    Xscaler = joblib.load(f"{MLtype}/Xscaler.pkl")
    Yscaler = joblib.load(f"{MLtype}/Yscaler.pkl")

    X_train = ML_DATA['X_TRAIN']
    X_test  = ML_DATA['X_TEST']
    n_feat  = X_train.shape[2]

    # -----------------------------------------------------------------
    # 2  Trainings-DataFrame: FLAT & SKALIERT + Dummy-Outcome
    # -----------------------------------------------------------------
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    df_train = pd.DataFrame(
        X_train_flat,
        columns=[f"{f}_t{-τ}" for τ in range(PRE, -1, -1) for f in feature_names]
    )
    df_train["y"] = 0.0
    cont_feats = [c for c in df_train.columns if c != "y"]

    # -----------------------------------------------------------------
    # 3  Wrapper für die Modell-Vorhersage
    # -----------------------------------------------------------------
    def _predict_last_step(arr_2d):
        import pandas as _pd, numpy as _np
        if isinstance(arr_2d, (_pd.DataFrame, _pd.Series)):
            arr_2d = arr_2d.values
        arr_3d        = arr_2d.reshape(-1, PRE + 1, n_feat)
        pred_scaled   = model.predict(arr_3d, verbose=0)
        pred_unscaled = Yscaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).reshape(-1, H, 1)
        return pred_unscaled[:, -1, 0]

    class PredWrapper:
        def __init__(self, fn): self._fn = fn
        def predict(self, X):   return self._fn(X)

    # -----------------------------------------------------------------
    # 4  DiCE-Objekte
    # -----------------------------------------------------------------
    dice_data = dice_ml.Data(
        dataframe           = df_train,
        continuous_features = cont_feats,
        outcome_name        = "y"
    )
    dice_model = dice_ml.Model(
        model      = PredWrapper(_predict_last_step),
        backend    = "sklearn",
        model_type = "regressor"
    )
    exp = dice_ml.Dice(dice_data, dice_model, method="genetic")

    # -----------------------------------------------------------------
    # 5  Query-Instanz (skaliert, ohne 'y')
    # -----------------------------------------------------------------
    query_flat = X_test[sample_id: sample_id + 1].reshape(1, -1)
    query_df   = pd.DataFrame(query_flat, columns=cont_feats)

    # -----------------------------------------------------------------
    # 6  permitted_range für alle Features 0.05–0.95
    # -----------------------------------------------------------------
    permitted_range = {col: (pr_low, pr_high) for col in cont_feats}

    # -----------------------------------------------------------------
    # 7  Gegenbeispiele suchen (skaliert)
    # -----------------------------------------------------------------
    cfs_scaled = exp.generate_counterfactuals(
        query_df,
        total_CFs        = total_CFs,
        desired_range    = list(desired_range),
        features_to_vary = "all",
        permitted_range  = permitted_range,
        maxiterations    = 2000,
    ).cf_examples_list[0].final_cfs_df

    # -----------------------------------------------------------------
    # 8  UNskalieren der CFs
    # -----------------------------------------------------------------
    cf_arr_scaled   = cfs_scaled[cont_feats].to_numpy()
    n_cf            = cf_arr_scaled.shape[0]
    cf_arr3         = cf_arr_scaled.reshape(n_cf, PRE + 1, n_feat)
    cf_unscaled3    = np.empty_like(cf_arr3)
    for τ in range(PRE + 1):
        cf_unscaled3[:, τ, :] = Xscaler.inverse_transform(cf_arr3[:, τ, :])
    cf_arr_unscaled = cf_unscaled3.reshape(n_cf, -1)

    cf_df = pd.DataFrame(cf_arr_unscaled, columns=cont_feats)

    # -----------------------------------------------------------------
    # 9  Ausgabe
    # -----------------------------------------------------------------
    print("\n--- Counterfactuals (unskaliert) -----------------------------")
    print(cf_df.round(3))
    return cf_df

def analyze_counterfactuals(cf_df,
                            X_test: np.ndarray,
                            Xscaler,
                            sample_id: int,
                            feature_names: list,
                            PRE: int):
    """Vergleicht jedes CF mit dem Original-Testpunkt und fasst Änderungen zusammen."""
    
    # ---------- Original-Punkt UNskaliert (als 1-Zeilen-DataFrame) ----------
    orig_unscaled = Xscaler.inverse_transform(
        X_test[sample_id].reshape(-1, len(feature_names))
    ).reshape(PRE + 1, len(feature_names)).flatten()
    
    orig_df = pd.DataFrame([orig_unscaled], columns=cf_df.columns)
    
    # ---------- Delta-Matrix: CF – Original -------------------------------
    delta      = cf_df - orig_df.iloc[0]
    delta_abs  = delta.abs()
    
    # ---------- Aggregation je Feature (über alle Zeitschritte) -----------
    # Spaltennamen: "<Feature>_t-τ" → wir nehmen den Teil vor '_t'
    feat_base_names = [c.split('_t')[0] for c in cf_df.columns]
    delta_abs.columns = feat_base_names
    
# ALT -----------------------------------------------------------------
# feat_mean = delta_abs.groupby(level=0, axis=1).mean().T
# feat_mean.columns = ['mean_abs_change']

    feat_mean = (
        delta_abs.groupby(axis=1, level=0)   # alle Zeit­schritte eines Features
                .mean()                      #   ➜ Mittelwert je Feature & CF
                .mean(axis=0)                #   ➜ Mittelwert über alle CF-Zeilen
                .to_frame('mean_abs_change') #   ➜ einspaltiger DataFrame
                .sort_values('mean_abs_change', ascending=False)
    )
    # ---------------------------------------------------------------------

    feat_mean.sort_values('mean_abs_change', ascending=False, inplace=True)
    
    print("\n--- Feature-Ranking nach mittlerer absoluter Änderung ---")
    print(feat_mean.round(3))
    return delta, feat_mean


def _percent_delta(y_cf: np.ndarray, y_base: np.ndarray) -> np.ndarray:
    eps = 1e-9
    return (y_cf - y_base) / (np.abs(y_base) + eps) * 100.0


def _distance_wit(x_base: np.ndarray, x_cf: np.ndarray, sigma: float) -> np.ndarray:
    """Distanz nach What-If-Tool (nur ein manipuliertes Merkmal)."""
    eps = 1e-12
    return np.abs(x_cf - x_base) / (sigma + eps)


def quick_what_if(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    feature: str,
    factors: tuple[float, ...] = (0.5, 0.75, 1.25, 1.5),
    sample_idx: int = 0,
    input_ts_idx: int | None = None,     # None  ⇒ über IN-Timesteps mitteln
    output_ts_idx: int | None = None,    # None  ⇒ über OUT-Timesteps mitteln
    out_dir: str = ".",
) -> str:
    """
    Erstellt einen Scatter-Plot, der zeigt, wie stark die Vorhersage für
    *ein* Test­sample reagiert, wenn das ausgewählte Feature mit
    unterschiedlichen Faktoren skaliert wird.  
    Zusätzlich wird das What-If-Distanz­maß berechnet und in der Konsole
    aus­gegeben.

    Rückgabewert
    ------------
    Pfad der gespeicherten PNG-Datei.
    """
    # ── Eingaben prüfen ─────────────────────────────────────────────────────────
    if feature not in feature_names:
        raise ValueError(f"'{feature}' ist kein bekanntes Feature.")
    f_idx = feature_names.index(feature)

    N, IN_TS, F = X_test.shape
    sample_idx = int(np.clip(sample_idx, 0, N - 1))

    # Vorhersage prüfen
    y_pred = model.predict(X_test[[sample_idx]])
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]                 # → (1, OUT_TS=1)
    OUT_TS = y_pred.shape[1]

    # Time-Index ggf. einschmiegen
    if input_ts_idx is not None:
        input_ts_idx = int(np.clip(input_ts_idx, 0, IN_TS - 1))
    if output_ts_idx is not None:
        output_ts_idx = int(np.clip(output_ts_idx, 0, OUT_TS - 1))

    # Basiseingabe & -vorhersage ────────────────────────────────────────────────
    x_base_all_ts = X_test[sample_idx]                                   # (IN_TS, F)
    if input_ts_idx is None:
        x_base = x_base_all_ts.mean(axis=0)                              # (F,)
    else:
        x_base = x_base_all_ts[input_ts_idx]                             # (F,)

    if output_ts_idx is None:
        y_base = y_pred.mean(axis=1)[0]                                  # Skalar
    else:
        y_base = y_pred[0, output_ts_idx]                                # Skalar

    # σ für Distanz
    sigma_feat = X_test[:, :, f_idx].std(ddof=0)
    if sigma_feat == 0:
        print(f"⚠ σ({feature}) = 0  → Distanz nicht definiert.")
    print(f"σ({feature}) = {sigma_feat:.4g}")

    # ── Plot vorbereiten ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    sf = mticker.ScalarFormatter()
    sf.set_scientific(False)
    ax.set_yscale("symlog", linthresh=20.0)
    ax.set_yticks([-1000, -100, -10, 0, 10, 100, 1000])
    ax.yaxis.set_major_formatter(sf)
    ax.set_xlabel("Faktor")
    ax.set_ylabel("Δ Vorhersage [%]")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)

    # ── Schleife über Faktoren ─────────────────────────────────────────────────
    for fac in factors:
        x_cf = x_base.copy()
        x_cf[f_idx] *= fac

        # Mini-Batch (1 × IN_TS × F) für Modell bauen
        if input_ts_idx is None:
            X_cf = np.broadcast_to(x_cf, (IN_TS, F)).copy()              # Durchschnitt → replizieren
        else:
            X_cf = x_base_all_ts.copy()
            X_cf[input_ts_idx, f_idx] = x_cf[f_idx]

        y_cf_pred = model.predict(X_cf.reshape(1, IN_TS, F))
        if y_cf_pred.ndim == 1:
            y_cf_pred = y_cf_pred[:, None]

        if output_ts_idx is None:
            y_cf = y_cf_pred.mean(axis=1)[0]
        else:
            y_cf = y_cf_pred[0, output_ts_idx]

        delta_pct = _percent_delta(y_cf, y_base)
        dist      = _distance_wit(x_base[f_idx], x_cf[f_idx], sigma_feat)

        print(f"Faktor {fac:>5}:  Δ = {delta_pct:+7.2f} %   Distanz = {dist:.4f}")

        ax.scatter([fac], [delta_pct],
                   label=f"{fac:>4}  (dist={dist:.2f})",
                   s=40)

    ax.axhline(0, color="gray", linewidth=1)
    ax.legend(title="Skalierungs­faktor")

    title_txt = (f"What-If für '{feature}' – Sample {sample_idx}  "
                 f"[IN {'∅' if input_ts_idx is None else input_ts_idx}, "
                 f"OUT {'∅' if output_ts_idx is None else output_ts_idx}]")
    plt.title(title_txt,
    fontsize=16,pad=14)

    # ── Speichern ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    fname = (f"whatif_{feature}_sample{sample_idx}_"
             f"in{'avg' if input_ts_idx is None else input_ts_idx}_"
             f"out{'avg' if output_ts_idx is None else output_ts_idx}.png")
    fpath = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    print("✅ Plot gespeichert:", fpath)
    return fpath
# ───────────────────────────────────────────────────────────────────────────────


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
    plt.plot(y_sel, linewidth=1.5, label=f"Gemessene PV-Leistung t={h+1}")
    plt.xlabel("Testdatensatz")
    plt.ylabel("produzierter PV-Strom [kW]")
    plt.title(f"{ml_name} – Gemessene PV-Leistung (Forecast-Horizon {hi+1})")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    fname = f"{ml_name}_actual_pv_t{hi+1}.png"
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
        fig.suptitle("{}: Unskaliertes CF-Raster für '{}' (t={})".format(
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
    mode: Literal["aggregate", "single"] = "aggregate",
    aggregate_output: bool = False,
    timestep: int = 5,
    filename: Optional[str] = None,
) -> None:
    """
    Erstellt PDP+ICE-Plots für ein Feature.

    mode="aggregate":  Mittelt über alle Input-Timesteps.
    mode="single":     Variiert nur den angegebenen timestep.
    aggregate_output:   Wenn True, mittelt über alle Horizon-Steps und erzeugt einen einzigen Plot.
                       Sonst je Horizon einen separaten Plot.
    """
    import os, re, numpy as np
    import matplotlib.pyplot as plt

    # ── Vorbereitung ─────────────────────────────────────────────────────────
    X_test      = ML_DATA["X_TEST"].copy()  # (N, IN_TS, F)
    feature_idx = feature_names.index(feature)
    model_name  = Control_Var["MLtype"]
    H           = num_horizon_steps
    N, IN_TS, F = X_test.shape

    # Feature für Dateinamen (Unterstriche statt Sonderzeichen)
    feature_clean = re.sub(r"[^\w]", "_", feature)
    # Feature für Titel behalten [], nur 1->/
    title_feature = feature.replace('1','/')

    # Sample-Auswahl
    if sample_indices is None:
        sample_indices = np.random.choice(N, min(30, N), replace=False)

    out_dir = f"./{model_name}"
    os.makedirs(out_dir, exist_ok=True)

    # Einheit aus Feature-Name
    m = re.search(r"\[(.*?)\]", feature)
    unit = m.group(1) if m else ""
    unit = unit.replace('1','/')
    is_percent = '%' in unit

    # Horizonte bestimmen: None=ein Plot, sonst für jeden
    horizons = [None] if aggregate_output else list(range(H))

    for h in horizons:
        # ── Input-Wertebereich und Tag ─────────────────────────────────────
        if mode == "aggregate":
            vals = X_test[:, :, feature_idx].flatten()
            if aggregate_output:
                mode_tag = "voll aggregiert"
            else:
                mode_tag = "aggregierte Eingabeschritte"
        else:
            vals = X_test[:, timestep, feature_idx]
            mode_tag = f"für Eingabeschritt t = {timestep+1}"
        vmin, vmax = np.percentile(vals, [1, 99])
        value_range = np.linspace(vmin, vmax, num_points)

        # ── Rückskalierung falls Scaler übergeben ─────────────────────────
        if scaler_x is not None:
            dummy = np.zeros((num_points, len(feature_names)))
            dummy[:, feature_idx] = value_range
            vr_plot = scaler_x.inverse_transform(dummy)[:, feature_idx]
        else:
            vr_plot = value_range.copy()
        if is_percent:
            vr_plot *= 100
            unit = '%'

        # ── Plot erstellen ─────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))

        # ICE-Kurven
        for idx in sample_indices:
            preds = []
            for val in value_range:
                X_tmp = X_test[idx:idx+1].copy()
                if mode == "aggregate":
                    X_tmp[0, :, feature_idx] = val
                else:
                    X_tmp[0, timestep, feature_idx] = val
                yhat = model.predict(X_tmp)
                if aggregate_output:
                    yval = yhat.reshape(-1, H).mean()
                else:
                    yval = yhat.reshape(-1, H)[0, h]
                preds.append(scaler_y.inverse_transform([[yval]])[0, 0])
            ax.plot(vr_plot, preds, alpha=0.4, linewidth=1)

        # PDP über alle Samples
        pdp_vals = []
        for val in value_range:
            X_mod = X_test.copy()
            if mode == "aggregate":
                X_mod[:, :, feature_idx] = val
            else:
                X_mod[:, timestep, feature_idx] = val
            yhat = model.predict(X_mod).reshape(-1, H)
            if aggregate_output:
                mean_out = yhat.mean(axis=1)
                y_out = scaler_y.inverse_transform(mean_out.reshape(-1,1)).ravel()
                pdp_vals.append(y_out.mean())
            else:
                ystep = yhat[:, h]
                y_out = scaler_y.inverse_transform(ystep.reshape(-1,1)).ravel()
                pdp_vals.append(y_out.mean())
        ax.plot(vr_plot, pdp_vals, color='black', linewidth=2.8, label='PDP')

        # ── Titel ───────────────────────────────────────────────────────────
        if aggregate_output:
            title = f"{model_name} - PDP+ICE für {title_feature} ({mode_tag})"
        else:
            title = f"{model_name} - PDP+ICE für {title_feature}: {mode_tag} und Vorhersagehorizont h = {h+1}"
        ax.set_title(title, fontsize=16, pad=14)

        from matplotlib.ticker import MaxNLocator,FixedLocator

        # Achsenlimits
        ymin, ymax = 0, 6
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(vr_plot.min(), vr_plot.max())
        ax.set_xlabel(unit)
        ax.set_ylabel('Vorhersage [kW]')
        ax.margins(x=0)

        # Y-Ticks: sicher 0 und 6
        yt = [t for t in ax.get_yticks() if ymin <= t <= ymax]
        if yt[0] > ymin: yt.insert(0, ymin)
        if yt[-1] < ymax: yt.append(ymax)
        ax.set_yticks(yt)

        # ─── Imports (einmal im File) ────────────────────────────────────────────
        from matplotlib.ticker import AutoLocator, FixedLocator
        import numpy as np

        # ─── 1) Limits exakt setzen ─────────────────────────────────────────────
        xmin, xmax = vr_plot.min(), vr_plot.max()
        ax.set_xlim(xmin)
        ax.margins(x=0)

        # ─── 2) „schöne“ Auto-Ticks holen (bereits gerundet) ────────────────────
        ax.xaxis.set_major_locator(AutoLocator())
        auto_ticks = ax.get_xticks().tolist()

        # ─── 3) Liste vorbereiten: xmin + auto + xmax (kein Duplikat) ───────────
        ticks = sorted(set(auto_ticks) | {xmin})

        # ─── 4) Adaptiv ausdünnen – aber Endpunkte NIE entfernen ────────────────
        min_gap = 0.04 * (xmax - xmin)       # 10 % der Spanne
        filtered = [xmin]                    # xmin bleibt auf jeden Fall
        last = xmin
        for t in ticks[1:]:                  # überspringt xmin
            # lasse Tick zu, wenn Lücke ausreichend
            if (t - last) >= min_gap:
                filtered.append(t)
                last = t

        ticks = filtered

        # ─── 5) Locator + Labels setzen ─────────────────────────────────────────
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([f"{t:.2f}" for t in ticks],
                        rotation=45, ha='right', fontsize=8)

        # ─── 6) Feine graue Linien an allen Tick-Positionen ─────────────────────
        for x in ticks:
            ax.axvline(x, color='gray', linestyle=':', linewidth=0.7,
                    alpha=0.5, zorder=0)

        # ─── 7) Y-Grid wie gehabt ───────────────────────────────────────────────
        ax.grid(which='major', axis='y', linestyle=':', color='gray', alpha=0.7)

        # ─── 8) Layout final glätten ────────────────────────────────────────────
        fig.tight_layout()


        # ── Legende unten mittig ────────────────────────────────────────────
        fig.subplots_adjust(bottom=0.2, top=0.85)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels,
                   loc='lower center', bbox_to_anchor=(0.5, 0.03),
                   ncol=1, frameon=False)

        # ── Speichern ───────────────────────────────────────────────────────
        suffix = 'aggAll' if aggregate_output else f'h{h+1}'
        fname = f"PDP_ICE_{feature_clean}_{mode_tag.replace(' ','_')}_{suffix}.png"
        fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches='tight')
        plt.close(fig)

    print(f"✅ Plots in '{out_dir}' erzeugt (aggregate_output={aggregate_output}).")




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
    plt.title('Vergleich: Original vs. Counterfactual Vorhersagen',fontsize=16,pad=14)
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
    plt.title('Auswirkung von 50% Erhöhung in Column 7',fontsize=16,pad=14)
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