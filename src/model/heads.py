"""Pooling y cabeza de clasificacion (Fase 7).

El encoder devuelve un vector por token; la prediccion es una sola por
impresion. El pooling es el puente, y cual conviene es una ablacion de la
Fase 9:

* ``cls``  toma el estado del token ``[CLS]``, que no corresponde a ninguna
  palabra y por eso puede especializarse en resumir la secuencia entera. Es lo
  que hace BERT.
* ``mean`` promedia los tokens reales. **Con la mascara**: promediar tambien los
  ``[PAD]`` diluye el vector segun cuanto relleno tenga cada ejemplo, que es un
  bug que no rompe nada y solo se ve como metricas peores.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.init import init_linear, init_output_bias

POOLINGS: tuple[str, ...] = ("cls", "mean")


def masked_mean(h: Tensor, padding_mask: Tensor | None) -> Tensor:
    """Promedio sobre los tokens reales. ``h``: ``(B, L, D)``, mascara ``(B, L)``."""
    if padding_mask is None:
        return h.mean(dim=1)
    mascara = padding_mask.unsqueeze(-1).to(h.dtype)
    total = (h * mascara).sum(dim=1)
    # clamp para no dividir por cero si un ejemplo quedara sin tokens reales.
    return total / mascara.sum(dim=1).clamp(min=1.0)


class CLSPooling(nn.Module):
    """El estado de la posicion 0, donde el tokenizador pone ``[CLS]``."""

    def forward(self, h: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        return h[:, 0]


class MeanPooling(nn.Module):
    """Promedio enmascarado de los estados."""

    def forward(self, h: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        return masked_mean(h, padding_mask)


def make_pooling(kind: str) -> nn.Module:
    if kind == "cls":
        return CLSPooling()
    if kind == "mean":
        return MeanPooling()
    raise ValueError(f"pooling desconocido: {kind!r}. Opciones: {POOLINGS}")


class ClassifierHead(nn.Module):
    """MLP de una capa oculta que termina en un unico logit.

    Devuelve **logits**, no probabilidades: la sigmoide la aplica
    ``BCEWithLogitsLoss`` (que es numericamente estable) o la evaluacion. Las
    metricas de la Fase 10 si se calculan sobre probabilidades.

    El bias de salida se inicializa en ``log(p / (1 - p))``, asi que antes de
    entrenar el modelo ya predice la tasa base.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int | None = None,
        dropout: float = 0.1,
        prevalence: float | None = None,
    ) -> None:
        super().__init__()
        d_hidden = d_hidden if d_hidden is not None else d_in
        self.oculta = nn.Linear(d_in, d_hidden)
        self.activacion = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.salida = nn.Linear(d_hidden, 1)

        init_linear(self.oculta)
        init_linear(self.salida)
        if prevalence is not None:
            init_output_bias(self.salida, prevalence)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, d_in)`` -> ``(B,)`` de logits."""
        return self.salida(self.dropout(self.activacion(self.oculta(x)))).squeeze(-1)

    @torch.no_grad()
    def set_prior(self, prevalence: float) -> None:
        """Reajusta el bias de salida a una prevalencia dada."""
        init_output_bias(self.salida, prevalence)
