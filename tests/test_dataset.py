"""Tests del ``Dataset`` y el ``DataLoader`` (Fase 4).

Lo que se verifica: que las formas y los dtypes sean los que espera el modelo,
que el orden de los batches sea reproducible desde la semilla, y que la rama de
texto sea realmente opcional (las ablaciones "solo tabular" y "solo texto" de la
Fase 9 se expresan construyendo el dataset sin una de las dos partes).

No necesitan el CSV: se arman con arreglos sinteticos, que ademas hacen el test
mas rapido y mas facil de leer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.dataset import Batch, ImpressionsDataset, collate, make_dataloader

N, D_NUM, D_CAT, L_TOK = 40, 5, 3, 7


@pytest.fixture()
def arrays():
    rng = np.random.default_rng(0)
    num = rng.normal(size=(N, D_NUM)).astype(np.float32)
    cat = rng.integers(0, 4, size=(N, D_CAT))
    y = (rng.random(N) < 0.13).astype(int)
    ids = rng.integers(1, 50, size=(N, L_TOK))
    mask = np.ones((N, L_TOK), dtype=bool)
    mask[:, -2:] = False                      # las dos ultimas posiciones son [PAD]
    return num, cat, y, ids, mask


@pytest.fixture()
def ds(arrays):
    num, cat, y, ids, mask = arrays
    return ImpressionsDataset(num, cat, y, ids, mask)


# --------------------------------------------------------------------------- #
# Formas y tipos
# --------------------------------------------------------------------------- #

def test_dtypes_y_formas(ds):
    assert ds.num.dtype == torch.float32
    assert ds.cat.dtype == torch.int64        # indices para nn.Embedding
    assert ds.y.dtype == torch.float32        # BCEWithLogitsLoss lo pide float
    assert ds.input_ids.dtype == torch.int64
    assert ds.attention_mask.dtype == torch.bool
    assert len(ds) == N


def test_getitem_devuelve_una_fila(ds):
    item = ds[3]
    assert item["num"].shape == (D_NUM,)
    assert item["cat"].shape == (D_CAT,)
    assert item["y"].shape == ()
    assert item["input_ids"].shape == (L_TOK,)


def test_la_rama_de_texto_es_opcional(arrays):
    num, cat, y, _, _ = arrays
    solo_tabular = ImpressionsDataset(num, cat, y)
    assert not solo_tabular.tiene_texto
    assert "input_ids" not in solo_tabular[0]
    batch = collate([solo_tabular[i] for i in range(4)], device=None)
    assert not batch.tiene_texto and batch.input_ids is None


def test_ids_y_mascara_van_juntos(arrays):
    num, cat, y, ids, _ = arrays
    with pytest.raises(ValueError, match="van juntos"):
        ImpressionsDataset(num, cat, y, input_ids=ids)


def test_longitudes_inconsistentes_fallan(arrays):
    num, cat, y, _, _ = arrays
    with pytest.raises(ValueError, match="filas"):
        ImpressionsDataset(num, cat[:10], y)


def test_prevalencia_y_pos_weight(arrays):
    num, cat, _, _, _ = arrays
    y = np.zeros(N, dtype=int)
    y[:10] = 1
    ds = ImpressionsDataset(num, cat, y)
    assert ds.prevalencia == pytest.approx(0.25)
    assert ds.pos_weight == pytest.approx(3.0)   # (N - pos) / pos


# --------------------------------------------------------------------------- #
# collate
# --------------------------------------------------------------------------- #

def test_collate_apila_en_el_orden_recibido(ds):
    indices = [5, 1, 9]
    batch = collate([ds[i] for i in indices], device=None)
    assert len(batch) == 3
    assert torch.equal(batch.num, ds.num[indices])
    assert torch.equal(batch.input_ids, ds.input_ids[indices])
    assert torch.equal(batch.attention_mask, ds.attention_mask[indices])


def test_batch_to_mueve_todos_los_tensores(ds):
    batch = collate([ds[i] for i in range(4)], device=None).to(torch.device("cpu"))
    for tensor in (batch.num, batch.cat, batch.y, batch.input_ids, batch.attention_mask):
        assert tensor.device.type == "cpu"


def test_la_mascara_marca_los_tokens_reales(ds):
    batch = collate([ds[i] for i in range(4)], device=None)
    # Convencion propia: True = token real. La inversa es la de
    # nn.MultiheadAttention y solo aparece en el test de equivalencia.
    assert batch.attention_mask[:, 0].all()
    assert not batch.attention_mask[:, -1].any()


# --------------------------------------------------------------------------- #
# DataLoader reproducible
# --------------------------------------------------------------------------- #

def _orden(loader) -> list[float]:
    return [float(v) for batch in loader for v in batch.num[:, 0]]


def test_el_orden_de_los_batches_es_reproducible(ds):
    a = make_dataloader(ds, batch_size=8, shuffle=True, seed=42, device=None)
    b = make_dataloader(ds, batch_size=8, shuffle=True, seed=42, device=None)
    assert _orden(a) == _orden(b)


def test_cambiar_la_semilla_cambia_el_orden(ds):
    a = make_dataloader(ds, batch_size=8, shuffle=True, seed=42, device=None)
    b = make_dataloader(ds, batch_size=8, shuffle=True, seed=7, device=None)
    assert _orden(a) != _orden(b)


def test_sin_shuffle_el_orden_es_el_del_dataset(ds):
    loader = make_dataloader(ds, batch_size=8, shuffle=False, device=None)
    assert _orden(loader) == [float(v) for v in ds.num[:, 0]]


def test_dos_epocas_con_shuffle_no_ven_el_mismo_orden(ds):
    loader = make_dataloader(ds, batch_size=8, shuffle=True, seed=42, device=None)
    assert _orden(loader) != _orden(loader)


def test_el_loader_cubre_todas_las_filas_una_vez(ds):
    loader = make_dataloader(ds, batch_size=8, shuffle=True, seed=42, device=None)
    vistos = sorted(_orden(loader))
    assert vistos == sorted(float(v) for v in ds.num[:, 0])


def test_no_se_puede_mover_a_cuda_desde_los_workers(ds):
    # El collate corre en el proceso worker y no puede inicializar CUDA.
    with pytest.raises(ValueError, match="num_workers"):
        make_dataloader(ds, num_workers=2, device=torch.device("cuda"))


def test_el_batch_llega_al_dispositivo_pedido(ds):
    loader = make_dataloader(ds, batch_size=8, device=torch.device("cpu"))
    batch = next(iter(loader))
    assert isinstance(batch, Batch)
    assert batch.num.device.type == "cpu"
