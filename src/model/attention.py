"""Atencion multi-cabeza escrita a mano (Fase 7).

Politica de la fase: de PyTorch se usan ``nn.Linear``, ``nn.Dropout`` y
``F.softmax``; el producto punto escalado, el reparto en cabezas, la mascara de
padding y la proyeccion de salida se escriben aca.

El motivo no es solo pedagogico. La Fase 10 necesita los mapas de atencion **por
cabeza**: ``F.scaled_dot_product_attention`` usa kernels fusionados y no devuelve
la matriz de pesos, ``nn.MultiheadAttention`` la devuelve promediada entre
cabezas salvo que se desactive el camino rapido, y ``nn.TransformerEncoderLayer``
directamente no la expone. Con la implementacion propia, guardarla es una linea.

El modulo se expone con tres implementaciones intercambiables que **comparten los
mismos parametros**, asi que se pueden comparar entre si:

===============  ==========================================================
``scratch``      Default. La implementacion propia; es la que se reporta y
                 de la que salen los mapas de atencion.
``torch_sdpa``   ``F.scaled_dot_product_attention``. Camino rapido, sin pesos
                 observables.
``torch_mha``    ``nn.MultiheadAttention``. Solo para el test de equivalencia.
===============  ==========================================================

Convencion de mascara del proyecto: ``padding_mask`` es booleana de forma
``(batch, seq)`` con **True en los tokens reales**. ``nn.MultiheadAttention`` usa
la inversa (True = posicion a ignorar), y esa conversion vive en un solo lugar,
``to_torch_mha``, para que la mitad de los bugs de convencion no exista.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

#: Valor que se suma a las posiciones invalidas antes del softmax.
#:
#: No se usa ``-inf``: si una fila queda enteramente enmascarada (un ejemplo sin
#: ningun token real, que puede aparecer en un batch degenerado), ``softmax``
#: de una fila de ``-inf`` es NaN y el NaN se propaga a todo el batch por los
#: gradientes. Con un finito grande, esa fila da una distribucion uniforme sobre
#: valores que despues la mascara de pooling descarta.
MASK_VALUE = -1e9

IMPLEMENTACIONES: tuple[str, ...] = ("scratch", "torch_sdpa", "torch_mha")


def additive_mask(padding_mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    """De mascara booleana ``(B, L)`` a mascara aditiva ``(B, 1, 1, L)``.

    Se suma **antes** del softmax, no despues: si se aplicara despues, las
    posiciones de padding ya habrian participado en la normalizacion y el resto
    de los pesos quedarian mal escalados.
    """
    if padding_mask is None:
        return None
    if padding_mask.dim() != 2:
        raise ValueError(f"padding_mask debe ser (batch, seq), llego {tuple(padding_mask.shape)}")
    invalidas = ~padding_mask.bool()
    return invalidas[:, None, None, :].to(dtype) * MASK_VALUE


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[Tensor, Tensor]:
    """``softmax(Q Kt / sqrt(d_head) + mask) V``, escrita a mano.

    Parametros de forma ``(batch, heads, seq, d_head)``; ``mask`` es la aditiva
    de ``additive_mask``, con broadcasting sobre cabezas y consultas.

    Se divide por ``sqrt(d_head)``, **no** por ``sqrt(d_model)``: el producto
    punto se hace dentro de cada cabeza, sobre vectores de dimension ``d_head``.
    Dividir por la dimension equivocada no rompe nada visible, solo cambia la
    temperatura del softmax y degrada el entrenamiento en silencio.
    """
    d_head = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    if mask is not None:
        scores = scores + mask
    # dim=-1: se normaliza sobre las *claves*, que es lo que hace que los pesos
    # de cada consulta sumen 1. Sobre cualquier otro eje el modelo entrena igual
    # y da metricas plausiblemente malas.
    weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, v), weights


class MultiHeadSelfAttention(nn.Module):
    """Atencion multi-cabeza sobre una sola secuencia (self-attention).

    Parametros
    ----------
    d_model
        Dimension del modelo. Debe ser divisible por ``n_heads``.
    n_heads
        Cantidad de cabezas. ``d_head = d_model // n_heads``, asi que el costo
        total no depende de cuantas sean: la ablacion de la Fase 9 aisla el
        efecto de la multi-cabeza del efecto de la capacidad.
    dropout
        Dropout sobre los pesos de atencion.
    attn_impl
        Ver el docstring del modulo.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attn_impl: str = "scratch",
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) tiene que ser divisible por n_heads "
                f"({n_heads}): cada cabeza trabaja con d_model / n_heads = "
                f"{d_model / n_heads} dimensiones, que tiene que ser entero."
            )
        if attn_impl not in IMPLEMENTACIONES:
            raise ValueError(
                f"attn_impl desconocido: {attn_impl!r}. Opciones: {IMPLEMENTACIONES}"
            )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.attn_impl = attn_impl

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

        #: Ultimos pesos de atencion, ``(batch, heads, seq, seq)``, desprendidos
        #: del grafo. Los llena solo la implementacion propia; es la materia
        #: prima de los mapas de atencion de la Fase 10.
        self.last_attn_weights: Tensor | None = None

    # -- reparto en cabezas -------------------------------------------------- #

    def _split_heads(self, x: Tensor) -> Tensor:
        """``(B, L, d_model)`` -> ``(B, H, L, d_head)``."""
        b, l, _ = x.shape
        return x.view(b, l, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """``(B, H, L, d_head)`` -> ``(B, L, d_model)``."""
        b, _, l, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, l, self.d_model)

    # -- forward ------------------------------------------------------------- #

    def forward(
        self,
        x: Tensor,
        padding_mask: Tensor | None = None,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        """``x``: ``(batch, seq, d_model)``. ``padding_mask``: ``(batch, seq)``, True = real.

        ``attn_mask`` es una mascara booleana opcional por par (consulta, clave),
        de forma ``(batch, seq, seq)`` o ``(1, seq, seq)``, tambien con True =
        permitido. La usa la atencion cross-item de la Fase 9, donde lo que
        restringe no es el padding sino a que query pertenece cada item.
        """
        if self.attn_impl == "torch_mha":
            if attn_mask is not None:
                raise ValueError("el modo torch_mha no acepta attn_mask; es solo para tests")
            return self._forward_torch_mha(x, padding_mask)

        q = self._split_heads(self.w_q(x))
        k = self._split_heads(self.w_k(x))
        v = self._split_heads(self.w_v(x))
        permitido = self._combinar_mascaras(x, padding_mask, attn_mask)

        if self.attn_impl == "scratch":
            aditiva = None if permitido is None else (~permitido).to(x.dtype) * MASK_VALUE
            salida, pesos = scaled_dot_product_attention(
                q, k, v, aditiva, self.attn_dropout
            )
            self.last_attn_weights = pesos.detach()
        else:  # torch_sdpa
            # La mascara booleana de torch tiene la MISMA convencion que la
            # nuestra: True = la posicion participa.
            salida = F.scaled_dot_product_attention(
                q, k, v, attn_mask=permitido,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            self.last_attn_weights = None

        return self.w_o(self._merge_heads(salida))

    @staticmethod
    def _combinar_mascaras(
        x: Tensor, padding_mask: Tensor | None, attn_mask: Tensor | None
    ) -> Tensor | None:
        """Una sola mascara booleana ``(B, 1, Lq, Lk)`` con True = permitido."""
        partes = []
        if padding_mask is not None:
            if padding_mask.dim() != 2:
                raise ValueError(
                    f"padding_mask debe ser (batch, seq), llego {tuple(padding_mask.shape)}"
                )
            partes.append(padding_mask.bool()[:, None, None, :])
        if attn_mask is not None:
            if attn_mask.dim() != 3:
                raise ValueError(
                    f"attn_mask debe ser (batch, seq, seq), llego {tuple(attn_mask.shape)}"
                )
            partes.append(attn_mask.bool().unsqueeze(1))
        if not partes:
            return None
        combinada = partes[0]
        for parte in partes[1:]:
            combinada = combinada & parte
        return combinada

    def _forward_torch_mha(self, x: Tensor, padding_mask: Tensor | None) -> Tensor:
        """Camino de verificacion: delega en ``nn.MultiheadAttention``.

        Reconstruye el modulo de la libreria en cada llamada a partir de los
        parametros propios. Es ineficiente a proposito: este modo existe solo
        para el test de equivalencia, no para entrenar.
        """
        mha = self.to_torch_mha()
        key_padding_mask = None if padding_mask is None else ~padding_mask.bool()
        salida, pesos = mha(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = pesos.detach()
        return salida

    # -- puente con la libreria ---------------------------------------------- #

    def to_torch_mha(self) -> nn.MultiheadAttention:
        """Una ``nn.MultiheadAttention`` equivalente, con **estos** pesos.

        Dos convenciones que hay que respetar y que explican la mitad de los
        fallos del test de equivalencia:

        1. ``nn.MultiheadAttention`` empaqueta W_Q, W_K y W_V en un unico tensor
           ``in_proj_weight`` de forma ``(3 * d_model, d_model)``, en ese orden.
        2. Su default es ``batch_first=False``. Aca se pide ``True`` para que las
           formas coincidan con las del resto del proyecto.
        """
        mha = nn.MultiheadAttention(
            self.d_model, self.n_heads, dropout=0.0, bias=True, batch_first=True
        ).to(self.w_q.weight.device, self.w_q.weight.dtype)
        with torch.no_grad():
            mha.in_proj_weight.copy_(
                torch.cat([self.w_q.weight, self.w_k.weight, self.w_v.weight], dim=0)
            )
            mha.in_proj_bias.copy_(
                torch.cat([self.w_q.bias, self.w_k.bias, self.w_v.bias], dim=0)
            )
            mha.out_proj.weight.copy_(self.w_o.weight)
            mha.out_proj.bias.copy_(self.w_o.bias)
        return mha

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_heads={self.n_heads}, "
            f"d_head={self.d_head}, attn_impl={self.attn_impl!r}"
        )
