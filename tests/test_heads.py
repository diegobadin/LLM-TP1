"""Tests del pooling, la cabeza de clasificacion y la inicializacion (Fase 7).

El pooling promediado sin mascara es un bug que no rompe nada: diluye el vector
segun cuanto relleno tenga cada ejemplo y solo se ve como metricas peores. El
bias de salida mal inicializado tampoco rompe nada: se gasta las primeras epocas
corrigiendo el intercepto. Los dos se cubren aca.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from src.model.heads import ClassifierHead, CLSPooling, MeanPooling, make_pooling, masked_mean
from src.model.init import (
    EMBEDDING_STD,
    init_embedding,
    init_linear,
    init_module,
    init_output_bias,
    logit_prior,
)
from src.utils.seed import set_seed

B, L, D = 4, 7, 16
PREVALENCIA = 0.13


@pytest.fixture(autouse=True)
def semilla():
    set_seed(0)


@pytest.fixture()
def h():
    return torch.randn(B, L, D)


@pytest.fixture()
def mask():
    m = torch.ones(B, L, dtype=torch.bool)
    m[:, -2:] = False
    return m


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #

def test_el_mean_pooling_ignora_el_padding(h, mask):
    obtenido = masked_mean(h, mask)
    esperado = h[:, :-2].mean(dim=1)
    assert torch.allclose(obtenido, esperado, atol=1e-6)


def test_el_mean_pooling_sin_mascara_es_el_promedio_llano(h):
    assert torch.allclose(masked_mean(h, None), h.mean(dim=1), atol=1e-6)


def test_el_padding_no_diluye_el_promedio(h, mask):
    # Si el pooling contara los [PAD], cambiar su contenido cambiaria la salida.
    alterado = h.clone()
    alterado[:, -2:] = 100.0
    assert torch.allclose(masked_mean(h, mask), masked_mean(alterado, mask), atol=1e-6)


def test_un_ejemplo_sin_tokens_reales_no_da_nan(h):
    vacia = torch.zeros(B, L, dtype=torch.bool)
    assert torch.isfinite(masked_mean(h, vacia)).all()


def test_el_cls_pooling_toma_la_posicion_cero(h, mask):
    assert torch.equal(CLSPooling()(h, mask), h[:, 0])


def test_la_fabrica_devuelve_cada_pooling():
    assert isinstance(make_pooling("cls"), CLSPooling)
    assert isinstance(make_pooling("mean"), MeanPooling)
    with pytest.raises(ValueError, match="pooling desconocido"):
        make_pooling("max")


def test_los_dos_poolings_dan_vectores_distintos(h, mask):
    assert not torch.allclose(CLSPooling()(h, mask), MeanPooling()(h, mask), atol=1e-3)


# --------------------------------------------------------------------------- #
# Cabeza de clasificacion
# --------------------------------------------------------------------------- #

def test_la_cabeza_devuelve_un_logit_por_ejemplo():
    cabeza = ClassifierHead(D, prevalence=PREVALENCIA).eval()
    salida = cabeza(torch.randn(B, D))
    assert salida.shape == (B,)
    assert salida.dtype == torch.float32


def test_la_cabeza_devuelve_logits_y_no_probabilidades():
    # La sigmoide la aplica BCEWithLogitsLoss, que es numericamente estable.
    cabeza = ClassifierHead(D, prevalence=PREVALENCIA).eval()
    salida = cabeza(torch.randn(64, D) * 10)
    assert (salida < 0).any() or (salida > 1).any()


def test_el_bias_de_salida_arranca_en_la_tasa_base():
    cabeza = ClassifierHead(D, prevalence=PREVALENCIA)
    bias = cabeza.salida.bias.detach().item()
    assert bias == pytest.approx(math.log(0.13 / 0.87), abs=1e-6)
    assert torch.sigmoid(torch.tensor(bias)).item() == pytest.approx(PREVALENCIA, abs=1e-6)


def test_sin_prevalencia_el_bias_arranca_en_cero():
    assert ClassifierHead(D).salida.bias.detach().item() == 0.0


def test_set_prior_reajusta_el_bias():
    cabeza = ClassifierHead(D, prevalence=PREVALENCIA)
    cabeza.set_prior(0.5)
    assert cabeza.salida.bias.detach().item() == pytest.approx(0.0, abs=1e-7)


def test_la_cabeza_propaga_gradiente():
    cabeza = ClassifierHead(D, prevalence=PREVALENCIA)
    perdida = nn.BCEWithLogitsLoss()(cabeza(torch.randn(B, D)), torch.zeros(B))
    perdida.backward()
    assert torch.isfinite(cabeza.salida.weight.grad).all()


# --------------------------------------------------------------------------- #
# Inicializacion
# --------------------------------------------------------------------------- #

def test_logit_prior():
    assert logit_prior(0.5) == pytest.approx(0.0)
    assert logit_prior(0.13) == pytest.approx(-1.9010, abs=1e-4)
    for invalida in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="prevalencia"):
            logit_prior(invalida)


def test_los_embeddings_arrancan_en_002_con_padding_en_cero():
    emb = nn.Embedding(500, 64, padding_idx=0)
    init_embedding(emb, pad_idx=0)
    assert torch.count_nonzero(emb.weight[0]) == 0
    assert emb.weight[1:].std().item() == pytest.approx(EMBEDDING_STD, rel=0.1)


def test_las_lineales_arrancan_con_xavier_y_bias_cero():
    lineal = nn.Linear(64, 64)
    init_linear(lineal)
    assert torch.count_nonzero(lineal.bias) == 0
    limite = math.sqrt(6.0 / (64 + 64))          # Xavier uniforme
    assert lineal.weight.abs().max().item() <= limite + 1e-6


def test_init_module_se_aplica_a_toda_la_red():
    red = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))
    red.apply(init_module)
    assert all(torch.count_nonzero(m.bias) == 0 for m in red if isinstance(m, nn.Linear))


def test_init_output_bias_exige_una_sola_unidad():
    with pytest.raises(ValueError, match="se esperaba 1"):
        init_output_bias(nn.Linear(8, 3), PREVALENCIA)
