"""Tests del calculo de metricas y del registro de corridas (Fase 5).

Cubren los tres errores que el plan marca como los mas faciles de cometer al
reportar: medir sobre logits en vez de probabilidades, usar
``auc(recall, precision)`` en lugar de ``average_precision_score``, y leer un
PR-AUC sin la prevalencia al lado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import evaluate as E


@pytest.fixture()
def y_true():
    y = np.zeros(100, dtype=int)
    y[:13] = 1                      # prevalencia 0.13, como el dataset
    return y


def test_las_metricas_rechazan_logits(y_true):
    logits = np.linspace(-4, 4, len(y_true))
    with pytest.raises(ValueError, match="logits"):
        E.compute_metrics(y_true, logits)


def test_prediccion_constante_da_roc_medio_y_pr_igual_a_la_prevalencia(y_true):
    m = E.compute_metrics(y_true, np.full(len(y_true), 0.13))
    assert m["roc_auc"] == 0.5
    assert m["pr_auc"] == pytest.approx(y_true.mean(), abs=1e-12)
    assert m["prevalence"] == pytest.approx(0.13)


def test_prediccion_perfecta_da_uno(y_true):
    m = E.compute_metrics(y_true, y_true * 0.99 + 0.005)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["pr_auc"] == pytest.approx(1.0)


def test_el_brier_es_el_error_cuadratico_medio(y_true):
    prob = np.full(len(y_true), 0.3)
    assert E.compute_metrics(y_true, prob)["brier"] == pytest.approx(
        np.mean((prob - y_true) ** 2)
    )


def test_formas_incompatibles_fallan(y_true):
    with pytest.raises(ValueError, match="formas incompatibles"):
        E.compute_metrics(y_true, np.full(7, 0.5))


def test_el_lift_es_relativo_a_la_prevalencia():
    assert E.lift(0.78, 0.13) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# results.csv
# --------------------------------------------------------------------------- #

def _fila(run: str, fold: int, pr: float) -> dict:
    return {
        "run_name": run, "config": "test", "features": "x", "seed": 42, "fold": fold,
        "split": "cv_dev", "n_train": 10, "n_eval": 5, "prevalence": 0.13,
        "roc_auc": 0.9, "pr_auc": pr, "log_loss": 0.2, "brier": 0.05,
        "epochs_ran": None, "seconds": 1.0, "commit": "abc123", "notes": "",
    }


def test_append_escribe_las_columnas_canonicas(tmp_path):
    ruta = tmp_path / "results.csv"
    todas = E.append_results([_fila("a", 0, 0.7)], path=ruta)
    assert list(todas.columns) == list(E.COLUMNAS)
    assert list(pd.read_csv(ruta).columns) == list(E.COLUMNAS)


def test_volver_a_correr_una_configuracion_la_reemplaza(tmp_path):
    ruta = tmp_path / "results.csv"
    E.append_results([_fila("a", 0, 0.70), _fila("b", 0, 0.50)], path=ruta)
    todas = E.append_results([_fila("a", 0, 0.99)], path=ruta)
    assert len(todas) == 2, "la corrida repetida se duplico en vez de reemplazarse"
    assert float(todas.loc[todas["run_name"] == "a", "pr_auc"].iloc[0]) == 0.99
    assert float(todas.loc[todas["run_name"] == "b", "pr_auc"].iloc[0]) == 0.50


def test_append_rechaza_columnas_no_declaradas(tmp_path):
    fila = _fila("a", 0, 0.7) | {"inventada": 1}
    with pytest.raises(ValueError, match="no declaradas"):
        E.append_results([fila], path=tmp_path / "results.csv")


def test_el_resumen_trae_media_desvio_y_lift():
    filas = [_fila("a", k, pr) for k, pr in enumerate([0.70, 0.80, 0.90])]
    resumen = E.summarize(pd.DataFrame(filas))
    assert len(resumen) == 1
    fila = resumen.iloc[0]
    assert fila["pr_auc"] == pytest.approx(0.80)
    assert fila["pr_auc_sd"] == pytest.approx(np.std([0.7, 0.8, 0.9], ddof=1))
    assert fila["lift"] == pytest.approx(0.80 / 0.13)
    assert fila["corridas"] == 3


def test_el_resumen_formateado_es_legible():
    filas = [_fila("a", k, pr) for k, pr in enumerate([0.70, 0.80])]
    texto = E.format_summary(E.summarize(pd.DataFrame(filas)))
    assert "+-" in texto["pr_auc"].iloc[0]
    assert texto["lift"].iloc[0].endswith("x")


def test_leer_un_results_inexistente_da_una_tabla_vacia(tmp_path):
    vacio = E.load_results(tmp_path / "no-existe.csv")
    assert vacio.empty and list(vacio.columns) == list(E.COLUMNAS)
