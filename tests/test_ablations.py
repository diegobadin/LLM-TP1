"""Tests de la grilla de ablacion y de la evaluacion final (Fases 9 y 10).

Los dos que protegen las conclusiones del trabajo:

* **cada fila cambia una sola cosa** respecto de la base (si cambiara dos, la
  fila no mide nada atribuible);
* la configuracion final se elige **solo con CV**, con una regla escrita, y el
  test se evalua una unica vez.
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from src import ablations as A
from src import final_eval as F
from src.train import BASE

def _diferencias(config) -> set[str]:
    return {
        f.name for f in fields(BASE)
        if f.name != "name" and getattr(config, f.name) != getattr(BASE, f.name)
    }


# --------------------------------------------------------------------------- #
# La grilla
# --------------------------------------------------------------------------- #

def test_los_nombres_son_unicos():
    nombres = [c.name for c, _ in A.GRILLA]
    assert len(set(nombres)) == len(nombres)
    assert "base" in nombres


def test_cada_fila_cambia_exactamente_lo_que_declara():
    """La grilla admite recetas, pero no cambios silenciosos.

    Antes la regla era "una sola clave por fila". Ahora dos filas mueven tres
    claves a proposito (las recetas de capacidad), asi que la regla pasa a ser
    explicita: cada configuracion declara que toca y la declaracion tiene que
    coincidir **exactamente** con el diff contra la base. Eso cubre los dos
    errores posibles: una clave de mas que nadie note, y una clave declarada que
    en realidad no cambia nada porque ya valia lo mismo que en la base.
    """
    for config, _ in A.GRILLA:
        declarado = A.CAMBIOS_DECLARADOS[config.name]
        real = _diferencias(config)
        assert real == set(declarado), (
            f"{config.name}: declara {sorted(declarado)} y cambia {sorted(real)}"
        )


def test_las_conclusiones_salen_de_filas_de_un_solo_cambio():
    # De estas cuatro se afirma algo atribuible, asi que no pueden ser recetas.
    # sin_marcador es la excepcion justificada: el nivel esta codificado dos
    # veces y apagar una sola codificacion no lo saca del input.
    atribuibles = {
        "solo_texto", "solo_tabular", "con_cart",
        # Con la base en una sola capa y el ratio d_ff/d_model fijo en 4, las
        # familias de arquitectura tambien quedan de una clave: mover d_ff, las
        # cabezas o la profundidad no arrastra nada mas.
        "ratio_1", "ratio_2", "ratio_8",
        "cabezas_1", "cabezas_2", "cabezas_8",
        "capas_2", "capas_4",
        "vocab_256",
    }
    for nombre in atribuibles:
        assert len(A.CAMBIOS_DECLARADOS[nombre]) == 1, f"{nombre} deberia cambiar una sola clave"
    assert A.CAMBIOS_DECLARADOS["sin_marcador"] == frozenset(
        {"keep_suffix", "keep_reputation_sentence"}
    )


def test_la_grilla_entra_en_el_presupuesto():
    # 17 configuraciones x 5 semillas x 5 folds = 425 entrenamientos, ~70 min
    # secuenciales en la GPU de referencia. El presupuesto declarado son 2 h.
    assert 6 <= len(A.GRILLA) <= 18
    assert len(A.GRILLA) * len(A.SEEDS) * 5 <= 450


def test_cada_fila_explica_que_demuestra():
    for config, nota in A.GRILLA:
        assert len(nota) >= 30, f"{config.name} no explica que demuestra"


def test_la_grilla_cubre_lo_que_el_trabajo_tiene_que_responder():
    nombres = {c.name for c, _ in A.GRILLA}
    esperados = {
        "base",                                # referencia
        "sin_marcador",                        # el hallazgo central
        "solo_texto", "solo_tabular",          # que aporta cada rama
        "con_cart",                            # la fuga, medida
        "ratio_1", "ratio_2", "ratio_8",       # la premisa del paper
        "cabezas_1", "cabezas_2", "cabezas_8",  # Table 3(A), params constantes
        "ancho_32", "d48_6cabezas", "ancho_96",  # ancho, con d_head 16
        "capas_2", "capas_4",                  # profundidad
        "vocab_256",                           # el tokenizador
    }
    assert esperados <= nombres, f"faltan: {sorted(esperados - nombres)}"


def test_el_ancho_respeta_el_limite_del_enunciado():
    for config, _ in A.GRILLA:
        assert config.d_model < 100, f"{config.name}: d_model {config.d_model} >= 100"


def test_las_arquitecturas_respetan_el_ratio_del_paper():
    """d_ff = 4 x d_model, salvo en la familia que estudia justamente el ratio.

    "Attention is all you need" usa 512 -> 2048 en el base y 1024 -> 4096 en el
    big: ratio 4 en los dos. La grilla anterior tenia d_ff clavado en 128, con
    lo cual el ratio iba de 4.0 en la receta chica a 1.33 en la grande, al reves
    de lo que se pretendia demostrar.
    """
    familia_del_ratio = {"ratio_1", "ratio_2", "ratio_8"}
    for config, _ in A.GRILLA:
        if config.name in familia_del_ratio:
            continue
        assert config.d_ff == 4 * config.d_model, (
            f"{config.name}: d_ff {config.d_ff} con d_model {config.d_model} "
            f"da ratio {config.d_ff / config.d_model:.2f}, no 4"
        )


def test_la_familia_de_cabezas_es_a_parametros_constantes():
    # Repartir d_model en mas cabezas no crea ni destruye pesos: W_Q/W_K/W_V/W_O
    # siguen siendo d_model x d_model. Es lo que hace interpretable la
    # comparacion, y es el diseno de la Table 3(A) del paper.
    familia = ["base", "cabezas_1", "cabezas_2", "cabezas_8"]
    configs = [c for c, _ in A.GRILLA if c.name in familia]
    assert len(configs) == len(familia)
    assert len({(c.d_model, c.n_layers, c.d_ff) for c in configs}) == 1
    assert {c.n_heads for c in configs} == {1, 2, 4, 8}


def test_la_base_tiene_una_sola_capa():
    from src.train import BASE
    assert BASE.n_layers == 1
    assert BASE.d_ff == 4 * BASE.d_model


def test_todas_las_arquitecturas_son_validas():
    for config, _ in A.GRILLA:
        assert config.d_model % config.n_heads == 0, config.name


def test_el_cache_de_folds_distingue_lo_que_cambia_los_datos():
    # Dos corridas que solo difieren en la arquitectura o en la semilla comparten
    # los tensores; una que cambia el texto o el vocabulario, no.
    claves = set(A.CLAVES_DE_DATOS)
    assert {"keep_suffix", "vocab_size", "text_fields", "include_cart"} <= claves
    assert not ({"d_model", "n_heads", "seed", "lr", "n_layers"} & claves)


def test_el_protocolo_es_de_cinco_semillas():
    assert len(A.SEEDS) == 5 and len(set(A.SEEDS)) == 5


# --------------------------------------------------------------------------- #
# Eleccion de la configuracion final
# --------------------------------------------------------------------------- #

def _resultados_sinteticos(pr_por_config: dict[str, float], sd: float = 0.01) -> pd.DataFrame:
    """25 corridas por configuracion con media y desvio exactos.

    Se construyen deterministas (patron simetrico alrededor de la media) para
    que el test verifique la regla de decision y no el sorteo.
    """
    patron = np.linspace(-1, 1, 25)
    patron = (patron - patron.mean()) / patron.std(ddof=1)
    filas = []
    for nombre, media in pr_por_config.items():
        for i, z in enumerate(patron):
            filas.append({
                "run_name": nombre, "config": "ablacion", "features": "x",
                "seed": 42 + i % 5, "fold": i % 5, "split": "cv_dev",
                "n_train": 6300, "n_eval": 1580, "prevalence": 0.129,
                "roc_auc": 0.97, "pr_auc": float(media + sd * z),
                "log_loss": 0.15, "brier": 0.045, "epochs_ran": 40,
                "seconds": 20.0, "commit": "abc", "notes": "",
            })
    return pd.DataFrame(filas)


def test_gana_la_base_si_la_ventaja_no_supera_el_error_estandar():
    # +0.002 con desvio 0.03 sobre 25 corridas: el error estandar es ~0.008.
    datos = _resultados_sinteticos({"base": 0.800, "capas_2": 0.802}, sd=0.03)
    config, _ = F.elegir_configuracion(datos, verbose=False)
    assert config.name == "base"


def test_gana_la_alternativa_si_la_ventaja_es_clara():
    datos = _resultados_sinteticos({"base": 0.780, "capas_2": 0.830}, sd=0.01)
    config, _ = F.elegir_configuracion(datos, verbose=False)
    assert config.name == "capas_2"


def test_las_ablaciones_diagnosticas_no_pueden_ganar():
    # con_cart usa una columna posterior al momento de la prediccion: por mas
    # metrica que de, no compite.
    datos = _resultados_sinteticos({"base": 0.780, "con_cart": 0.990,
                                    "sin_marcador": 0.999}, sd=0.01)
    config, candidatas = F.elegir_configuracion(datos, verbose=False)
    assert config.name == "base"
    assert "con_cart" not in set(candidatas["run_name"])
    assert "sin_marcador" not in set(candidatas["run_name"])


def test_sin_ablaciones_el_error_es_explicito():
    vacio = pd.DataFrame(columns=["config", "run_name", "pr_auc"])
    with pytest.raises(RuntimeError, match="python -m src.ablations"):
        F.elegir_configuracion(vacio, verbose=False)


# --------------------------------------------------------------------------- #
# Metricas de la Fase 10
# --------------------------------------------------------------------------- #

@pytest.fixture()
def prediccion():
    """Predicciones sinteticas con **correlacion dentro de la query**.

    Es la estructura real del dataset: los items de una busqueda comparten
    filtros y contexto. Sin esa correlacion, agrupar el bootstrap no cambiaria
    nada y el test no probaria lo que dice probar.
    """
    rng = np.random.default_rng(0)
    n_queries, por_query = 120, 5
    calidad = rng.random(n_queries) < 0.3          # queries "buenas" y "malas"
    y = np.concatenate([
        (rng.random(por_query) < (0.4 if buena else 0.02)).astype(int) for buena in calidad
    ])
    p = np.clip(0.1 + 0.6 * y + rng.normal(0, 0.15, len(y)), 0.001, 0.999)
    grupos = np.repeat(np.arange(n_queries), por_query)
    return y, p, grupos


def test_el_bootstrap_devuelve_intervalos_que_contienen_la_media(prediccion):
    y, p, grupos = prediccion
    ic = F.bootstrap_ci(y, p, grupos, n_boot=500).set_index("metrica")
    for metrica in ("roc_auc", "pr_auc", "log_loss", "brier"):
        assert ic.loc[metrica, "IC 2.5%"] < ic.loc[metrica, "media"] < ic.loc[metrica, "IC 97.5%"]


def test_agrupar_por_query_ensancha_el_intervalo():
    # Caso limite y transparente: las 5 impresiones de cada query son identicas.
    # El tamano de muestra efectivo es la cantidad de queries, no la de filas, y
    # remuestrear filas sueltas fingiria 5 veces mas informacion de la que hay.
    rng = np.random.default_rng(0)
    n_queries, por_query = 80, 5
    y_query = (rng.random(n_queries) < 0.3).astype(int)
    p_query = np.clip(0.15 + 0.5 * y_query + rng.normal(0, 0.2, n_queries), 0.01, 0.99)
    y = np.repeat(y_query, por_query)
    p = np.repeat(p_query, por_query)
    grupos = np.repeat(np.arange(n_queries), por_query)

    agrupado = F.bootstrap_ci(y, p, grupos, n_boot=400).set_index("metrica")
    sueltas = F.bootstrap_ci(y, p, np.arange(len(y)), n_boot=400).set_index("metrica")
    ancho = lambda t: t.loc["pr_auc", "IC 97.5%"] - t.loc["pr_auc", "IC 2.5%"]  # noqa: E731
    assert ancho(agrupado) > ancho(sueltas)


def test_la_calibracion_reparte_todas_las_filas(prediccion):
    y, p, _ = prediccion
    tabla, ece = F.curva_calibracion(y, p, n_bins=10)
    assert tabla["n"].sum() == len(y)
    assert 0 <= ece <= 1


def test_una_prediccion_perfectamente_calibrada_tiene_ece_bajo():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.9, 20_000)
    y = (rng.random(20_000) < p).astype(int)
    _, ece = F.curva_calibracion(y, p)
    assert ece < 0.02


def test_las_metricas_de_ranking_estan_en_cero_uno(prediccion):
    y, p, grupos = prediccion
    tabla = F.metricas_de_ranking(y, p, grupos, ks=(1, 2, 3))
    assert len(tabla) == 3
    for col in tabla.columns.drop("k"):
        assert tabla[col].between(0, 1).all()
    # El recall no puede bajar al agrandar k.
    assert tabla["recall@k (por query)"].is_monotonic_increasing


def test_un_ranking_perfecto_da_map_uno():
    y = np.array([1, 0, 0, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.1])
    grupos = np.array([0, 0, 0, 1, 1, 1])
    assert F.map_por_query(y, p, grupos) == pytest.approx(1.0)


def test_un_ranking_invertido_da_map_bajo():
    y = np.array([1, 0, 0, 1, 0, 0])
    p = np.array([0.1, 0.8, 0.9, 0.2, 0.7, 0.9])
    grupos = np.array([0, 0, 0, 1, 1, 1])
    assert F.map_por_query(y, p, grupos) < 0.5


# --------------------------------------------------------------------------- #
# Regresion: el cache de datos no puede congelar la arquitectura
# --------------------------------------------------------------------------- #

def test_el_cache_rearma_la_arquitectura_en_cada_corrida():
    """Dos corridas que comparten datos pero cambian la arquitectura.

    Es la regresion de un bug real: el cache devolvia el ``FoldData`` entero,
    ``model_config`` incluido, asi que la segunda corrida entrenaba la
    arquitectura de la primera y su fila de ablacion salia identica a ella. No
    fallaba nada: la tabla quedaba llena de numeros plausibles y falsos.
    """
    from unittest.mock import patch

    from src.train import FoldData, model_config_de

    class _DatasetFalso:
        num = np.zeros((10, 4), dtype=np.float32)
        y = np.zeros(10, dtype=np.float32)
        y[0] = 1.0

    class _PipelineFalso:
        cardinalidades = (5, 4)

    class _TokenizadorFalso:
        vocab_size, max_len = 300, 32

    base_fold = FoldData(_DatasetFalso(), _DatasetFalso(), None,
                         _TokenizadorFalso(), _PipelineFalso())
    base_fold.model_config = model_config_de(BASE, base_fold)

    cache = A._CacheDeFolds()
    with patch.object(A, "prepare_fold", return_value=base_fold) as preparado:
        a = cache.get(None, None, None, 0, replace(BASE, name="a", n_layers=1))
        b = cache.get(None, None, None, 0, replace(BASE, name="b", n_layers=4, n_heads=8))

    assert preparado.call_count == 1, "los datos se prepararon dos veces"
    assert a.model_config.n_layers == 1 and b.model_config.n_layers == 4
    assert b.model_config.n_heads == 8
    # Y los tensores si se comparten: de eso se trata el cache.
    assert a.train is b.train
