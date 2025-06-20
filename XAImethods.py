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
from sklearn.utils import resample

from pathlib import Path
from typing import Sequence, Union, Optional, Iterable, List, Literal
import tensorflow as tf
from alibi.explainers import Counterfactual

tf.compat.v1.disable_eager_execution()
from scipy.optimize import minimize
import dice_ml
from itertools import combinations

import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
#  SHAP Berechnungen für beliebige Prognose-HORIZONTE
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
        n_bootstrap: int = 1000,            
        ci_alpha: float = 0.05,            
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
    plt.colorbar = lambda *_a, **_k: _DummyCbar()  

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

        explainer   = shap.GradientExplainer(model, X_test_3D[bg_indices])
        shap_vals   = explainer.shap_values(X_test_3D)   
        shap_arr    = (np.stack(shap_vals, axis=-1)
                       if isinstance(shap_vals, list) else shap_vals)
        # Form: (N  , T_in, F, H)
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
        feature_names = list(feature_names) 
        feature_names.pop(idx_remove)
        F -= 1

    if isinstance(shap_vals, list):
        shap_arr = np.stack(shap_vals, axis=-1)
    else:
        shap_arr = shap_vals
    if idx_remove is not None:
        shap_arr = np.delete(shap_arr, idx_remove, axis=2)
        feature_names = [fn for i, fn in enumerate(feature_names) if i != idx_remove]
    N, T_in, F, H = shap_arr.shape

    # Wenn ausgewäht, nur bestimmte Input-Steps
    input_sel = range(T_in) if input_steps is None else input_steps

    results = {}

    for h in horizon_steps:
        # 1) Aggregierte SHAP pro Instanz und Feature
        shap_h      = shap_arr[..., h]                  # (N, T_in, F)
        shap_agg    = shap_h[:, input_sel, :].mean(axis=1)   # (N, F)

        # 2) Kennzahlen berechnen
        mean_vals = shap_agg.mean(axis=0)               # (F,)
        se_vals   = shap_agg.std(axis=0, ddof=1) / np.sqrt(N)

        # 3) Bootstrap-CI - aber final nicht genutzt
        cis_lower = np.zeros(F)
        cis_upper = np.zeros(F)
        for f in range(F):
            boot_means = []
            for _ in range(n_bootstrap):
                samp = resample(shap_agg[:, f], replace=True, n_samples=N)
                boot_means.append(samp.mean())
            lo = np.percentile(boot_means, 100 * (ci_alpha/2))
            hi = np.percentile(boot_means, 100 * (1 - ci_alpha/2))
            cis_lower[f], cis_upper[f] = lo, hi

        # 4) DataFrame erstellen
        df = pd.DataFrame({
            "feature": feature_names,
            "mean_shap": mean_vals,
            "se_shap":   se_vals,
            f"ci{int((1-ci_alpha)*100)}_low":  cis_lower,
            f"ci{int((1-ci_alpha)*100)}_high": cis_upper,
        }).sort_values("mean_shap", ascending=False)

        results[h] = df

        # 5) Abspeichern oder zurückgeben
        df.to_csv(f"{Control_Var['MLtype']}_shap_stats_h{h+1}.csv", index=False)
    # --------------------------------------------------------------------- #
    # 4)  Welche Eingabeschritte sollen berücksichtigt werden?
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
    # 5b) Für den letzten Horizont 10 werden die SHAP-Werte für einzelnen Inputs berechnet
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
    # 6)  Zusätzlich: komplett aggregiertes Übersicht‑Plots über alle Horizonte und gewählte Eingabeschritte
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
    return shap_vals                  


# ---------------------------------------------------------------------------
#  LIME Erklärungen für Zeitreihen
# ---------------------------------------------------------------------------

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
    horizon_step=None,       
    input_time_step=None     
):
    """
    Erzeugt LIME-Erklärungen und speichert jede Instanz als einzelne PNG-Datei
    mit dem Titel der Abbildung als Dateinamen.
    """
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

    # Auswahl der Indizes, die übergeben worden sind
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

    # LIME-Erklärung für die aktuelle Instanz und Horizont
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

        # 3) x-Achse auf [-0.1, 0.1] begrenzen
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

# ---------------------------------------------------------------------------
#  Kontrafaktische Erklärungen für Zeitreihen DICE
# ---------------------------------------------------------------------------

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
        sample_size      = 20000
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
                "Feature-Grenzen konnten nicht bestimmt werden"
            )

        # Grenzen in den skalierten Raum übertragen
        mins_scaled = x_scaler.transform(orig_mins.reshape(1, -1)).ravel()
        maxs_scaled = x_scaler.transform(orig_maxs.reshape(1, -1)).ravel()

        # Jede CF-Sequenz hart clippen
        cf_examples_scaled = [
            np.clip(cf, mins_scaled, maxs_scaled) for cf in cf_examples_scaled
        ]

    # ───────────────────────────────────────────────────────────────
    # 7 | Unskalierung der Feature-Sequenzen
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
    # 8 | Rückgabe
    # ───────────────────────────────────────────────────────────────
    return {
        "cf_examples_scaled":   cf_examples_scaled,
        "cf_examples_unscaled": cf_examples,
        "y_orig_scaled":        y_orig_scaled,
        "y_orig":               y_orig,
        "y_cfs_scaled":         y_cfs_scaled,
        "y_cfs":                y_cfs
    }


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
        X_base_in = X_raw.copy()             
    else:
        X_base_in = X_raw.copy()              

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
        gridspec_kw={'wspace': 0.4}   
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

    # 2) Legende Prozent-Plot platzieren
    fig_pct.subplots_adjust(top=0.85, bottom=0.25)
    fig_pct.legend(
        handles=handles_pct,
        loc='lower center',               
        bbox_to_anchor=(0.5, 0.05),       #
        ncol=2,
        frameon=False
    )
    #Anpassung, der Benennung der Featureeinheit
    feature_disp = feature.replace("1", "/")
    fig_pct.suptitle(f"{model_name} – Prozentuales Counterfactual zufällige Werte: {feature_disp} {input_tag} {horizon_tag}",
                     y=0.97)

    #Speichern des Prozent-Plots
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
    fig_abs.subplots_adjust(top=0.85, bottom=0.30)      

    #Legende für Absolut-Plot
    fig_abs.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),                     
        frameon=False,
        borderaxespad=1.0                               
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
        X_base_in = X_raw.copy()              
    else:
        X_base_in = X_raw.copy()              

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
        gridspec_kw={'wspace': 0.4}   
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
        loc='lower center',               
        bbox_to_anchor=(0.5, 0.05),       
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
    fig_abs.subplots_adjust(top=0.85, bottom=0.30)      
    fig_abs.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.04),                     
        frameon=False,
        borderaxespad=1.0                              
    )
    
    fig_abs.tight_layout(rect=[0, 0.15, 1, 0.98]) 
    fname_abs = fname_pct.replace("/","1").replace("perc","abs")
    fig_abs.savefig(os.path.join(out_dir, fname_abs), dpi=300); plt.close(fig_abs)

    if debug:
        print("Max unskaliert:", y_base_unscaled.max())
        print("Max Slice    :", y_base_vals.max())
    print("✅ Scatter-Plot gespeichert:", fname_pct)
    print("✅ Absolut-Plot gespeichert:", fname_abs)


# # ────────────────────────────────────────────────────────────────────
# PDP und ICE für alle Inputs, aggregiert oder einzeln
# # ────────────────────────────────────────────────────────────────────
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

    # ── Vorbereitung ─────────────────────────────────────────────────────────
    X_test      = ML_DATA["X_TEST"].copy()  
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
        ymin, ymax = 0, 7.5
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


