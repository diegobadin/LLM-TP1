"""Inicializacion de pesos (Fase 7, critico por la decision A3).

Sin pesos pre-entrenados, la inicializacion es responsabilidad del equipo y
afecta directamente la convergencia. Los criterios adoptados:

* **Embeddings de token**: ``N(0, 0.02)``, el mismo criterio que usa BERT. La
  fila de ``[PAD]`` se pone en cero y se excluye del gradiente: es relleno, no
  tiene por que aprender nada, y dejarla suelta agrega ruido al unico embedding
  que aparece en casi todas las secuencias.
* **Proyecciones lineales**: Xavier/Glorot uniforme, que mantiene la varianza de
  las activaciones estable a traves de las capas cuando la activacion es
  aproximadamente lineal alrededor del origen.
* **Bias**: cero.
* **Bias de la capa de salida**: ``log(p / (1 - p))`` con ``p`` la prevalencia de
  train. El modelo arranca prediciendo la tasa base en vez de gastar las
  primeras epocas corrigiendo el intercepto. Es barato y estabiliza el comienzo.
"""

from __future__ import annotations

import math

import torch
from torch import nn

EMBEDDING_STD = 0.02


def logit_prior(prevalence: float) -> float:
    """``log(p / (1 - p))``. Con p = 0.13 da -1.90."""
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"la prevalencia tiene que estar en (0, 1), llego {prevalence}")
    return math.log(prevalence / (1.0 - prevalence))


def init_embedding(emb: nn.Embedding, std: float = EMBEDDING_STD, pad_idx: int = 0) -> None:
    """``N(0, std)`` y la fila de padding en cero."""
    nn.init.normal_(emb.weight, mean=0.0, std=std)
    if pad_idx is not None and 0 <= pad_idx < emb.num_embeddings:
        with torch.no_grad():
            emb.weight[pad_idx].zero_()


def init_linear(linear: nn.Linear) -> None:
    """Xavier uniforme en los pesos, cero en el bias."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_module(module: nn.Module) -> None:
    """Para usar con ``model.apply(init_module)``.

    ``nn.LayerNorm`` se deja como viene de PyTorch (ganancia 1, sesgo 0), que ya
    es lo que corresponde.
    """
    if isinstance(module, nn.Linear):
        init_linear(module)
    elif isinstance(module, nn.Embedding):
        init_embedding(module, pad_idx=getattr(module, "padding_idx", None))


def init_output_bias(linear: nn.Linear, prevalence: float) -> None:
    """Deja la capa de salida prediciendo la tasa base antes de entrenar."""
    if linear.out_features != 1:
        raise ValueError(
            f"la capa de salida tiene {linear.out_features} unidades; se esperaba 1"
        )
    with torch.no_grad():
        linear.bias.fill_(logit_prior(prevalence))
