"""Tests del modelo completo y del entrenamiento (Fase 8).

El criterio de aceptacion de la fase es el **test de sobreajuste**: si el modelo
no puede memorizar 64 ejemplos, hay un bug y no tiene sentido entrenar sobre el
dataset entero. Es el diagnostico mas barato que existe y esta abajo, marcado
como lento.

El resto cubre el cableado: que cada rama se pueda apagar, que la fusion use las
dos, que la atencion cross-item respete los limites de la query, y que dos
corridas con la misma semilla den exactamente lo mismo.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from src.data.dataset import Batch, ImpressionsDataset, make_dataloader
from src.model.model import ModelConfig, build_model
from src.train import BASE, FocalLoss, make_optimizer, make_scheduler, prepare_fold, train_model
from src.utils.seed import DEVICE, set_seed

B, L, VOCAB = 12, 16, 40
CARDINALIDADES = (5, 4, 3)
D_NUM = 6


@pytest.fixture(autouse=True)
def semilla():
    set_seed(0)


def config(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=VOCAB, max_len=L, d_model=32, n_heads=4, n_layers=2, d_ff=64,
        cat_cardinalities=CARDINALIDADES, d_num=D_NUM, d_tab=16, dropout=0.0,
        prevalence=0.13,
    )
    return ModelConfig(**(base | kwargs))


def batch(n: int = B, con_texto: bool = True, grupos: np.ndarray | None = None) -> Batch:
    rng = np.random.default_rng(0)
    ids = torch.as_tensor(rng.integers(1, VOCAB, (n, L)))
    mask = torch.ones(n, L, dtype=torch.bool)
    mask[:, -3:] = False
    ids[:, -3:] = 0
    return Batch(
        num=torch.randn(n, D_NUM),
        cat=torch.as_tensor(rng.integers(0, 3, (n, len(CARDINALIDADES)))),
        y=torch.as_tensor(rng.random(n) < 0.13).float(),
        input_ids=ids if con_texto else None,
        attention_mask=mask if con_texto else None,
        group=None if grupos is None else torch.as_tensor(grupos),
    )


# --------------------------------------------------------------------------- #
# Configuracion
# --------------------------------------------------------------------------- #

def test_no_se_puede_apagar_las_dos_ramas():
    with pytest.raises(ValueError, match="al menos una rama"):
        config(use_text=False, use_tabular=False)


def test_la_rama_tabular_necesita_columnas():
    with pytest.raises(ValueError, match="no hay columnas tabulares"):
        config(cat_cardinalities=(), d_num=0)


def test_la_dimension_de_fusion_es_la_suma_de_las_ramas():
    assert config().d_fusion == 32 + 16
    assert config(use_tabular=False).d_fusion == 32
    assert config(use_text=False).d_fusion == 16


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "flags", [{}, {"use_tabular": False}, {"use_text": False}],
    ids=["fusion", "solo_texto", "solo_tabular"],
)
def test_el_forward_devuelve_un_logit_por_impresion(flags):
    modelo = build_model(config(**flags)).eval()
    salida = modelo(batch())
    assert salida.shape == (B,)
    assert torch.isfinite(salida).all()


def test_las_probabilidades_caen_en_cero_uno():
    modelo = build_model(config())
    prob = modelo.predict_proba(batch())
    assert ((prob >= 0) & (prob <= 1)).all()
    assert not modelo.training, "predict_proba tiene que dejar el modelo en eval"


def test_arranca_prediciendo_la_tasa_base():
    # Con el bias de salida en log(p/(1-p)) el modelo arranca cerca de la tasa
    # base en vez de gastar las primeras epocas corrigiendo el intercepto. La
    # capa oculta agrega algo de sesgo, asi que la comparacion util es contra el
    # mismo modelo inicializado con otra prevalencia.
    entrada = batch(256)
    en_la_base = float(build_model(config(prevalence=0.13)).predict_proba(entrada).mean())
    en_la_mitad = float(build_model(config(prevalence=0.50)).predict_proba(entrada).mean())
    assert abs(en_la_base - 0.13) < 0.05
    assert abs(en_la_base - 0.13) < abs(en_la_mitad - 0.13)


def test_un_modelo_con_texto_avisa_si_el_batch_no_lo_trae():
    modelo = build_model(config()).eval()
    with pytest.raises(ValueError, match="no trae input_ids"):
        modelo(batch(con_texto=False))


def test_la_rama_tabular_sola_ignora_el_texto():
    modelo = build_model(config(use_text=False)).eval()
    con = batch()
    sin = Batch(num=con.num, cat=con.cat, y=con.y)      # el mismo batch sin texto
    assert torch.allclose(modelo(con), modelo(sin), atol=1e-6)


def test_las_dos_ramas_aportan_a_la_salida():
    modelo = build_model(config()).eval()
    original = batch()
    otro_tabular = Batch(
        num=original.num + 3.0, cat=original.cat, y=original.y,
        input_ids=original.input_ids, attention_mask=original.attention_mask,
    )
    assert not torch.allclose(modelo(original), modelo(otro_tabular), atol=1e-4)


def test_el_resumen_de_parametros_cierra():
    modelo = build_model(config())
    resumen = modelo.resumen_parametros()
    assert resumen["total"] == sum(v for k, v in resumen.items() if k != "total")
    assert resumen["total"] == modelo.n_parameters()


def test_los_mapas_de_atencion_estan_disponibles():
    modelo = build_model(config()).eval()
    modelo(batch())
    mapas = modelo.attention_maps()
    assert len(mapas) == 2 and mapas[0].shape[:2] == (B, 4)
    assert not build_model(config(use_text=False)).attention_maps()


# --------------------------------------------------------------------------- #
# Atencion cross-item
# --------------------------------------------------------------------------- #

def test_sin_cross_item_los_items_no_se_ven():
    modelo = build_model(config()).eval()
    grupos = np.array([0] * 6 + [1] * 6)
    entrada = batch(grupos=grupos)
    salida = modelo(entrada)
    # Cambiar el vecino de la misma query no cambia nada: la atencion cross-item
    # esta apagada.
    alterado = Batch(**{**entrada.__dict__, "num": entrada.num.clone()})
    alterado.num[0] += 5.0
    assert torch.allclose(salida[1:], modelo(alterado)[1:], atol=1e-6)


def test_con_cross_item_solo_influye_la_misma_query():
    modelo = build_model(config(use_cross_item=True)).eval()
    grupos = np.array([0] * 6 + [1] * 6)
    entrada = batch(grupos=grupos)
    salida = modelo(entrada)

    alterado = Batch(**{**entrada.__dict__, "num": entrada.num.clone()})
    alterado.num[0] += 5.0                      # item de la query 0
    nueva = modelo(alterado)

    misma_query = ~torch.isclose(salida[1:6], nueva[1:6], atol=1e-5)
    otra_query = torch.isclose(salida[6:], nueva[6:], atol=1e-5)
    assert misma_query.any(), "los items de la misma query no se ven entre si"
    assert otra_query.all(), "un item influye en otra query: la mascara esta mal"


def test_el_sampler_por_query_no_parte_queries():
    from src.data.dataset import QueryBatchSampler

    grupos = torch.as_tensor(np.repeat(np.arange(40), 5))
    sampler = QueryBatchSampler(grupos, batch_size=16, shuffle=True, seed=0)
    for lote in sampler:
        queries = grupos[lote].tolist()
        for q in set(queries):
            assert queries.count(q) == 5, "una query quedo partida entre dos batches"
    assert sum(len(lote) for lote in sampler) == len(grupos)


# --------------------------------------------------------------------------- #
# Optimizacion
# --------------------------------------------------------------------------- #

def test_el_weight_decay_no_toca_bias_ni_layernorm():
    modelo = build_model(config())
    grupos = make_optimizer(modelo, BASE).param_groups
    con_decay, sin_decay = grupos[0], grupos[1]
    assert con_decay["weight_decay"] == BASE.weight_decay
    assert sin_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in con_decay["params"])
    assert all(p.ndim <= 1 for p in sin_decay["params"])


def test_el_scheduler_hace_warmup_y_despues_baja():
    modelo = build_model(config())
    opt = make_optimizer(modelo, BASE)
    total = 100
    sched = make_scheduler(opt, total, replace(BASE, warmup_ratio=0.1))
    lrs = []
    for _ in range(total):
        lrs.append(sched.get_last_lr()[0])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9] <= max(lrs)          # sube durante el warmup
    assert np.argmax(lrs) <= 12                 # el pico esta al final del warmup
    assert lrs[-1] < 0.02 * max(lrs)            # el coseno termina cerca de cero


def test_la_focal_loss_castiga_menos_lo_ya_resuelto():
    logits = torch.tensor([4.0, 4.0])
    objetivo = torch.tensor([1.0, 1.0])
    bce = nn.BCEWithLogitsLoss()(logits, objetivo)
    focal = FocalLoss(gamma=2.0)(logits, objetivo)
    assert focal < bce


def test_la_perdida_invalida_falla_temprano():
    with pytest.raises(ValueError, match="loss desconocida"):
        replace(BASE, loss="hinge")
    with pytest.raises(ValueError, match="text_fields desconocido"):
        replace(BASE, text_fields="titulo")


# --------------------------------------------------------------------------- #
# Sobre el dataset real
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def datos():
    from src.data import load as L
    from src.data.splits import load_splits

    if not L.CSV_PATH.exists():
        pytest.skip(f"falta {L.CSV_PATH} (el dataset no se versiona)")
    return L.load_raw(), load_splits()


@pytest.mark.data
def test_prepare_fold_ajusta_todo_solo_con_train(datos):
    df, splits = datos
    run = replace(BASE, epochs=1)
    fold = prepare_fold(df, splits.train, splits.valid, run)

    # El pipeline vio solo train: sus cardinalidades son las de train.
    for col, k in zip(run.categoricas, fold.pipeline.cardinalidades):
        assert k == df.iloc[splits.train][col].fillna("NONE").nunique() + 1
    # El tokenizador tambien: entrenarlo tambien con valid daria otro vocabulario.
    from src.tokenizer.bpe import BPETokenizer

    con_valid = BPETokenizer.train_from_pairs(
        fold.pipeline.transform(df).text_a, fold.pipeline.transform(df).text_b,
        vocab_size=run.vocab_size,
    )
    assert con_valid.fingerprint() != fold.tokenizer.fingerprint()
    # Las dos particiones comparten la misma longitud de secuencia.
    assert fold.train.input_ids.shape[1] == fold.eval.input_ids.shape[1]


@pytest.mark.data
def test_dos_corridas_con_la_misma_semilla_dan_lo_mismo(datos):
    df, splits = datos
    run = replace(BASE, epochs=2, patience=2)
    idx = splits.train[:1500]
    a = train_model(prepare_fold(df, idx, splits.valid[:500], run), run)
    b = train_model(prepare_fold(df, idx, splits.valid[:500], run), run)
    assert a.best["pr_auc"] == pytest.approx(b.best["pr_auc"], abs=1e-9)
    assert np.allclose(a.y_prob, b.y_prob, atol=1e-6)


@pytest.mark.data
def test_la_historia_registra_las_curvas(datos):
    df, splits = datos
    run = replace(BASE, epochs=3, patience=3)
    res = train_model(prepare_fold(df, splits.train[:1500], splits.valid[:500], run), run)
    assert list(res.history.columns) == [
        "epoca", "train_loss", "lr", "roc_auc", "pr_auc", "log_loss", "brier"
    ]
    assert len(res.history) == 3
    assert res.history["train_loss"].iloc[-1] < res.history["train_loss"].iloc[0]


@pytest.mark.slow
@pytest.mark.data
def test_el_modelo_sobreajusta_64_ejemplos(datos):
    # Criterio de aceptacion de la Fase 8.
    from src.train import overfit_test

    assert overfit_test(n=64, epochs=300, verbose=False) < 0.01
