"""Metricas y registro de corridas.

Un solo lugar donde se calculan las metricas y una sola tabla donde se anotan
los resultados, ``experiments/results.csv``, con **una fila por corrida**
(configuracion x semilla x fold). Las medias y los desvios se calculan al leer,
nunca se guardan agregados: si mas adelante hace falta el desvio entre semillas
por separado del desvio entre folds, la informacion sigue estando.

Metrica principal: PR-AUC (``average_precision_score``). Con prevalencia 0.13 el
ROC-AUC es optimista, y ademas el enunciado la pide. Se reporta siempre junto a
la prevalencia de la particion evaluada, que es su piso.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from src.data.load import PROJECT_ROOT

RESULTS_CSV = PROJECT_ROOT / "experiments" / "results.csv"

#: Orden canonico de las columnas de ``results.csv``.
COLUMNAS: tuple[str, ...] = (
    "run_name",
    "config",
    "features",
    "seed",
    "fold",
    "split",
    "n_train",
    "n_eval",
    "prevalence",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
    "epochs_ran",
    "seconds",
    "commit",
    "notes",
)

METRICAS: tuple[str, ...] = ("roc_auc", "pr_auc", "log_loss", "brier")


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Las cuatro metricas, calculadas sobre **probabilidades**, no logits.

    Para las dos AUC daria lo mismo porque son invariantes a transformaciones
    monotonas, pero ``log_loss`` y ``brier`` no: pasarles logits da un numero sin
    sentido que igual se imprime sin quejarse.

    ``pr_auc`` es ``average_precision_score``, no ``auc(recall, precision)``: la
    segunda interpola linealmente entre puntos de la curva PR, que no es una
    aproximacion valida y sobreestima.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_prob.ndim != 1 or len(y_prob) != len(y_true):
        raise ValueError(f"formas incompatibles: y_true {y_true.shape}, y_prob {y_prob.shape}")
    if not np.all((y_prob >= 0) & (y_prob <= 1)):
        raise ValueError(
            "y_prob tiene valores fuera de [0, 1]: parecen logits. Aplicar "
            "sigmoide antes de medir, o log_loss y brier no significan nada."
        )

    constante = float(y_prob.max() - y_prob.min()) < 1e-12
    return {
        # Con predicciones constantes (el baseline del prior) el ROC-AUC no esta
        # definido por ranking: vale 0.5 por convencion y sklearn avisa.
        "roc_auc": 0.5 if constante else float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "prevalence": float(y_true.mean()),
    }


def lift(pr_auc: float, prevalence: float) -> float:
    """Cuantas veces mejor que el azar. Un PR-AUC sin su prevalencia no dice nada."""
    return pr_auc / prevalence


def git_commit() -> str:
    """Hash corto del commit actual, con sufijo ``-dirty`` si hay cambios."""
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        sucio = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{rev}-dirty" if sucio else rev
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido"


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=list(COLUMNAS))


def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    if not path.exists():
        return empty_results()
    return pd.read_csv(path)


def append_results(
    filas: list[dict],
    path: Path = RESULTS_CSV,
    replace_runs: bool = True,
) -> pd.DataFrame:
    """Agrega filas a ``results.csv``.

    Con ``replace_runs`` (el default) las corridas previas que compartan
    ``run_name`` se reemplazan, para que volver a correr un experimento actualice
    su resultado en vez de duplicarlo. El archivo queda ordenado por corrida y
    fold, que es como se lee.
    """
    nuevas = pd.DataFrame(filas)
    faltantes = set(COLUMNAS) - set(nuevas.columns)
    sobrantes = set(nuevas.columns) - set(COLUMNAS)
    if sobrantes:
        raise ValueError(f"columnas no declaradas en COLUMNAS: {sorted(sobrantes)}")
    for col in faltantes:
        nuevas[col] = pd.NA
    nuevas = nuevas[list(COLUMNAS)]

    previas = load_results(path)
    if replace_runs and not previas.empty:
        previas = previas.loc[~previas["run_name"].isin(nuevas["run_name"].unique())]

    todas = pd.concat([previas, nuevas], ignore_index=True)
    todas = todas.sort_values(["split", "run_name", "seed", "fold"], kind="stable")
    path.parent.mkdir(parents=True, exist_ok=True)
    todas.to_csv(path, index=False)
    return todas


def summarize(
    resultados: pd.DataFrame,
    por: str | list[str] = "run_name",
    metricas: tuple[str, ...] = METRICAS,
) -> pd.DataFrame:
    """Media y desvio de cada metrica sobre folds y semillas.

    Sin el desvio, una tabla de ablacion no permite concluir nada: con ~260
    positivos por fold, diferencias de PR-AUC de pocos puntos son ruido.
    """
    if resultados.empty:
        return empty_results()
    grupos = resultados.groupby(por, sort=False)
    filas = grupos.agg(
        corridas=("run_name", "size"),
        features=("features", "first"),
        prevalencia=("prevalence", "mean"),
    )
    for m in metricas:
        filas[f"{m}"] = grupos[m].mean()
        filas[f"{m}_sd"] = grupos[m].std(ddof=1)
    filas["lift"] = filas["pr_auc"] / filas["prevalencia"]
    orden = ["corridas", "features", "prevalencia"]
    for m in metricas:
        orden += [m, f"{m}_sd"]
    return filas[orden + ["lift"]].reset_index()


def format_summary(resumen: pd.DataFrame, metricas: tuple[str, ...] = METRICAS) -> pd.DataFrame:
    """El resumen como texto ``media +- desvio``, para pegar en el informe."""
    salida = pd.DataFrame({resumen.columns[0]: resumen.iloc[:, 0]})
    salida["prevalencia"] = resumen["prevalencia"].map(lambda v: f"{v:.4f}")
    for m in metricas:
        salida[m] = [
            f"{mu:.4f} +- {sd:.4f}" for mu, sd in zip(resumen[m], resumen[f"{m}_sd"])
        ]
    salida["lift"] = resumen["lift"].map(lambda v: f"{v:.2f}x")
    return salida
