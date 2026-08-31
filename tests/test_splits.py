"""Criterio de aceptacion de la Fase 3, como tests.

El test central es el de solapamiento: ningun ``query_id`` puede aparecer en dos
particiones. Si aparece, las filas de una misma busqueda quedan repartidas entre
train y test, el modelo memoriza el contexto en lugar de aprenderlo, y todas las
metricas quedan infladas sin que ninguna curva se vea rara.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import load as L
from src.data import splits as S

pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def df():
    if not L.CSV_PATH.exists():
        pytest.skip(f"falta {L.CSV_PATH} (el dataset no se versiona)")
    return L.load_raw()


@pytest.fixture(scope="module")
def sp(df):
    return S.make_splits(df)


# --------------------------------------------------------------------------- #
# Solapamiento de grupos: el criterio de aceptacion de la fase
# --------------------------------------------------------------------------- #

def test_ningun_query_id_aparece_en_dos_particiones(df, sp):
    solape = S.group_overlap(df, sp)
    conflictos = solape.loc[solape["queries compartidas"] > 0]
    assert conflictos.empty, f"queries compartidas:\n{conflictos.to_string(index=False)}"


def test_los_folds_de_validacion_particionan_dev(df, sp):
    # Cada fila de dev cae en exactamente un fold de validacion.
    concatenado = np.concatenate([va for _, va in sp.folds])
    assert len(concatenado) == len(sp.dev)
    assert np.array_equal(np.sort(concatenado), np.sort(sp.dev))


def test_cada_fold_usa_todo_dev(sp):
    for k, (tr, va) in enumerate(sp.folds):
        union = np.union1d(tr, va)
        assert np.array_equal(union, np.sort(sp.dev)), f"fold {k} no cubre dev"
        assert np.intersect1d(tr, va).size == 0, f"fold {k} se solapa consigo mismo"


# --------------------------------------------------------------------------- #
# Cobertura y disyuncion
# --------------------------------------------------------------------------- #

def test_dev_y_test_cubren_el_dataset_sin_solaparse(df, sp):
    assert np.intersect1d(sp.dev, sp.test).size == 0
    assert np.array_equal(np.union1d(sp.dev, sp.test), np.arange(len(df)))


def test_train_y_valid_particionan_dev(sp):
    assert np.intersect1d(sp.train, sp.valid).size == 0
    assert np.array_equal(np.union1d(sp.train, sp.valid), np.sort(sp.dev))


def test_no_hay_indices_repetidos(sp):
    for nombre in ("dev", "test", "train", "valid"):
        idx = getattr(sp, nombre)
        assert len(np.unique(idx)) == len(idx), f"{nombre} tiene indices repetidos"


def test_el_test_es_aproximadamente_el_20_por_ciento(df, sp):
    fraccion = len(sp.test) / len(df)
    assert 0.17 <= fraccion <= 0.23, f"test quedo en {fraccion:.1%}"


# --------------------------------------------------------------------------- #
# Prevalencia
# --------------------------------------------------------------------------- #

def test_la_prevalencia_de_cada_particion_esta_dentro_de_15_puntos(df, sp):
    # No se estratifica por bought a nivel fila (agrupando por query no tendria
    # sentido); se verifica a posteriori, que es lo que pide el plan.
    resumen = S.describe(df, sp)
    fuera = resumen.loc[resumen["delta vs global (pp)"].abs() > S.PREVALENCE_TOL * 100]
    assert fuera.empty, f"prevalencia fuera de rango:\n{fuera.to_string(index=False)}"


def test_ninguna_particion_queda_sin_positivos(df, sp):
    y = df[L.TARGET].astype(int)
    for k, (tr, va) in enumerate(sp.folds):
        assert y.iloc[tr].sum() > 0, f"fold {k} train sin positivos"
        assert y.iloc[va].sum() > 0, f"fold {k} valid sin positivos"


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #

def test_dos_generaciones_dan_la_misma_particion(df):
    assert S.fingerprint(S.make_splits(df)) == S.fingerprint(S.make_splits(df))


def test_el_fingerprint_coincide_con_el_congelado(sp):
    # Si esto falla cambio el CSV, la semilla o la version de scikit-learn, y
    # los resultados ya reportados dejan de ser comparables.
    assert S.fingerprint(sp) == S.EXPECTED_FINGERPRINT


def test_cambiar_la_semilla_cambia_la_particion(df):
    assert S.fingerprint(S.make_splits(df, seed=7)) != S.EXPECTED_FINGERPRINT


def test_round_trip_por_disco(df, sp, tmp_path):
    S.save_splits(sp, tmp_path)
    leidos = S.load_splits(tmp_path)
    assert S.fingerprint(leidos) == S.fingerprint(sp)
    assert leidos.n_folds == sp.n_folds
    for nombre in ("dev", "test", "train", "valid"):
        assert np.array_equal(getattr(leidos, nombre), getattr(sp, nombre))


def test_load_splits_falla_con_mensaje_util_si_no_estan_generados(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.data.splits"):
        S.load_splits(tmp_path / "no-existe")


def test_load_splits_detecta_npy_inconsistentes(sp, tmp_path):
    S.save_splits(sp, tmp_path)
    # Se corrompe un indice: el manifiesto deja de coincidir con los .npy.
    corrupto = np.load(tmp_path / "test_idx.npy")
    corrupto[0] = corrupto[0] + 1
    np.save(tmp_path / "test_idx.npy", corrupto)
    with pytest.raises(ValueError, match="no coinciden con su manifiesto"):
        S.load_splits(tmp_path)
