"""Embeddings, positional encoding y bloque encoder (Fase 7).

Se escribe a mano lo que el enunciado pide afianzar: los residuales, el orden de
las normalizaciones, el feed-forward y el positional encoding. **No** se usa
``nn.TransformerEncoderLayer``: encapsula pre-LN, dropout, residuales y FFN en
una sola llamada, que es justamente lo que hay que construir, y ademas bloquea
el acceso a los pesos de atencion que necesita la Fase 10.

Pre-LN por defecto (``x = x + Sublayer(LN(x))``) en lugar de post-LN: converge de
forma mas estable en modelos chicos y con warmup corto. Post-LN queda disponible
como ablacion.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from src.model.attention import MultiHeadSelfAttention
from src.model.init import init_embedding

POS_ENCODINGS: tuple[str, ...] = ("sinusoidal", "learned", "none")
NORMS: tuple[str, ...] = ("pre_ln", "post_ln")


# --------------------------------------------------------------------------- #
# Positional encoding
# --------------------------------------------------------------------------- #

class SinusoidalPositionalEncoding(nn.Module):
    """Codificacion sinusoidal fija del paper original.

    ``PE(pos, 2i) = sin(pos / 10000^(2i/d))`` y coseno en las impares. No tiene
    parametros: se guarda como buffer para que viaje con el modelo entre
    dispositivos y entre en el ``state_dict`` sin recibir gradiente.
    """

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        posicion = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(posicion * divisor)
        pe[:, 1::2] = torch.cos(posicion * divisor)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.shape[1]].unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    """Una tabla de embeddings por posicion, aprendida desde cero."""

    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        init_embedding(self.pe, pad_idx=None)

    def forward(self, x: Tensor) -> Tensor:
        posiciones = torch.arange(x.shape[1], device=x.device)
        return x + self.pe(posiciones).unsqueeze(0)


class NoPositionalEncoding(nn.Module):
    """Sin informacion de posicion. Es una ablacion, no un descuido.

    Sin positional encoding la atencion es **invariante a permutaciones**: el
    modelo ve un conjunto de tokens, no una secuencia. Como la senal de este
    dataset es un marcador que esta presente sin importar donde, es plausible
    que casi no cambie el resultado, y eso seria un hallazgo interesante.
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


def make_positional_encoding(kind: str, d_model: int, max_len: int) -> nn.Module:
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len)
    if kind == "learned":
        return LearnedPositionalEncoding(d_model, max_len)
    if kind == "none":
        return NoPositionalEncoding()
    raise ValueError(f"pos_encoding desconocido: {kind!r}. Opciones: {POS_ENCODINGS}")


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #

class TokenEmbedding(nn.Module):
    """Embedding de token + positional encoding + LayerNorm + dropout.

    El ``LayerNorm`` posterior a la suma no es decorativo: los embeddings se
    inicializan en ``N(0, 0.02)`` y la codificacion sinusoidal tiene amplitud 1,
    dos escalas que difieren en un factor 50. Sin normalizar, la variante
    sinusoidal arrancaria dominada por la posicion y la comparacion contra la
    aprendida (que si nace en 0.02) no mediria el efecto del encoding sino el de
    la escala.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int,
        pos_encoding: str = "sinusoidal",
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        init_embedding(self.tokens, pad_idx=pad_id)
        self.pos = make_positional_encoding(pos_encoding, d_model, max_len)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_encoding = pos_encoding

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.dropout(self.norm(self.pos(self.tokens(input_ids))))


# --------------------------------------------------------------------------- #
# Bloque encoder
# --------------------------------------------------------------------------- #

class FeedForward(nn.Module):
    """``Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)``."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.entrada = nn.Linear(d_model, d_ff)
        self.activacion = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.salida = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.salida(self.dropout(self.activacion(self.entrada(x))))


class EncoderBlock(nn.Module):
    """Un bloque: atencion multi-cabeza + feed-forward, con residuales.

    pre-LN:  ``x = x + Drop(MHSA(LN(x)))`` y ``x = x + Drop(FFN(LN(x)))``
    post-LN: ``x = LN(x + Drop(MHSA(x)))`` y ``x = LN(x + Drop(FFN(x)))``
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        norm: str = "pre_ln",
        attn_impl: str = "scratch",
    ) -> None:
        super().__init__()
        if norm not in NORMS:
            raise ValueError(f"norm desconocida: {norm!r}. Opciones: {NORMS}")
        self.norm_kind = norm
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout, attn_impl)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        if self.norm_kind == "pre_ln":
            x = x + self.dropout(self.attn(self.norm_attn(x), padding_mask))
            return x + self.dropout(self.ff(self.norm_ff(x)))
        x = self.norm_attn(x + self.dropout(self.attn(x, padding_mask)))
        return self.norm_ff(x + self.dropout(self.ff(x)))

    @property
    def attn_weights(self) -> Tensor | None:
        return self.attn.last_attn_weights


class TransformerEncoder(nn.Module):
    """Embeddings + N bloques encoder. Devuelve los estados por token.

    El pooling y el clasificador viven en ``heads.py``, y la fusion con la rama
    tabular en ``model.py`` (Fase 8): este modulo es solo el encoder de texto.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_len: int = 56,
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",
        norm: str = "pre_ln",
        attn_impl: str = "scratch",
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embeddings = TokenEmbedding(
            vocab_size, d_model, max_len, pos_encoding, dropout, pad_id
        )
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout, norm, attn_impl)
            for _ in range(n_layers)
        ])
        # Con pre-LN la salida del ultimo bloque no esta normalizada (el residual
        # se suma despues de la ultima LN interna): hace falta una LN final para
        # que el pooling reciba activaciones de escala controlada.
        self.norm_final = nn.LayerNorm(d_model) if norm == "pre_ln" else nn.Identity()

    def forward(self, input_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
        x = self.embeddings(input_ids)
        for bloque in self.blocks:
            x = bloque(x, padding_mask)
        return self.norm_final(x)

    def attention_maps(self) -> list[Tensor]:
        """Pesos de atencion de la ultima pasada, una entrada por capa.

        Cada tensor es ``(batch, heads, seq, seq)``. Vacio si la implementacion
        activa no los expone (``torch_sdpa``).
        """
        return [b.attn_weights for b in self.blocks if b.attn_weights is not None]

    @property
    def n_layers(self) -> int:
        return len(self.blocks)
