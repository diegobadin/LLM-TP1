"""Modelo completo: encoder de texto + rama tabular + fusion (Fase 8).

::

    text ---> tokenizer ---> emb + PE ---> N x EncoderBlock ---> pooling ---> h_txt
                                                                               |
    categoricas ---> nn.Embedding por columna ---+                             |
    numericas   ---> escaladas ------------------+--> MLP ---> h_tab ----------+
                                                                               v
                                             concat ---> MLP ---> logit ---> sigmoide

**Por que fusionar.** La rama de texto sola llega a ROC ~ 0.958 (baseline
``tfidf_texto``) y la tabular sola a 0.598 (``gbdt_tabular``), pero el baseline
que combina el marcador con las tabulares llega a 0.974 con PR-AUC 0.808. Esos
12 puntos de PR-AUC vienen de ``allergens`` y ``category`` modulando la compra
dentro del nivel ALTO: la fusion no es decorativa, es donde esta el margen.

Todo se activa por configuracion (``use_text``, ``use_tabular``, ``pooling``,
``pos_encoding``, ``use_cross_item``, ...) para que cada fila de la tabla de
ablacion de la Fase 9 sea un ``ModelConfig`` distinto y no una edicion del
codigo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from src.data.dataset import Batch
from src.model.attention import MultiHeadSelfAttention
from src.model.encoder import TransformerEncoder
from src.model.heads import ClassifierHead, make_pooling
from src.model.init import init_linear


@dataclass
class ModelConfig:
    """Todo lo que define una arquitectura. Una fila de ablacion = un cambio."""

    # -- texto -------------------------------------------------------------- #
    vocab_size: int = 1355
    max_len: int = 56
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    pos_encoding: str = "sinusoidal"      # sinusoidal | learned | none
    pooling: str = "cls"                  # cls | mean
    norm: str = "pre_ln"                  # pre_ln | post_ln
    attn_impl: str = "scratch"            # scratch | torch_sdpa | torch_mha
    pad_id: int = 0

    # -- tabular ------------------------------------------------------------ #
    cat_cardinalities: tuple[int, ...] = ()
    d_num: int = 0
    cat_emb_dim: int = 12
    d_tab: int = 32

    # -- fusion y cabeza ----------------------------------------------------- #
    dropout: float = 0.1
    d_head_hidden: int | None = None
    prevalence: float = 0.13

    # -- que ramas estan activas --------------------------------------------- #
    use_text: bool = True
    use_tabular: bool = True
    use_cross_item: bool = False

    def __post_init__(self) -> None:
        if not (self.use_text or self.use_tabular):
            raise ValueError("hay que dejar prendida al menos una rama")
        if self.use_tabular and not self.cat_cardinalities and not self.d_num:
            raise ValueError("use_tabular=True pero no hay columnas tabulares")

    @property
    def d_fusion(self) -> int:
        return (self.d_model if self.use_text else 0) + (self.d_tab if self.use_tabular else 0)


class TabularEncoder(nn.Module):
    """Un ``nn.Embedding`` por columna categorica + las numericas, y un MLP.

    Embeddings en vez de one-hot porque el modelo es una red: 87 columnas
    dispersas con casi todo en cero desperdician capacidad, y ademas el
    embedding puede compartir estructura entre niveles parecidos (por ejemplo,
    los alergenos marinos).
    """

    def __init__(
        self,
        cat_cardinalities: tuple[int, ...],
        d_num: int,
        emb_dim: int = 12,
        d_out: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Cardinalidades chicas no necesitan 12 dimensiones: la regla habitual
        # es min(emb_dim, ceil(k/2)), que para storage_type (4) da 2.
        self.dims = tuple(min(emb_dim, max(2, (k + 1) // 2)) for k in cat_cardinalities)
        self.embeddings = nn.ModuleList([
            nn.Embedding(k, d) for k, d in zip(cat_cardinalities, self.dims)
        ])
        for emb in self.embeddings:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

        d_in = sum(self.dims) + d_num
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_out, d_out)
        )
        for capa in self.mlp:
            if isinstance(capa, nn.Linear):
                init_linear(capa)
        self.d_out = d_out

    def forward(self, num: Tensor, cat: Tensor) -> Tensor:
        partes = [emb(cat[:, j]) for j, emb in enumerate(self.embeddings)]
        if num.shape[1]:
            partes.append(num)
        return self.mlp(torch.cat(partes, dim=-1))


class CrossItemAttention(nn.Module):
    """Deja que los items de una misma query se miren entre si (decision A4).

    Se aplica sobre los vectores fusionados, tratando el batch como una
    secuencia de items y enmascarando todo par que no comparta ``query_id``.
    Requiere que el batch traiga las queries completas, que es lo que garantiza
    ``QueryBatchSampler``.

    La evidencia del EDA anticipa ganancia nula: la tasa de compra de un item de
    nivel ALTO es la misma haya 1, 2, 3 o 4 items ALTO compitiendo en la query
    (0.643 / 0.645 / 0.669 / 0.630). Se corre igual y se reporta como resultado
    negativo, que es tan valido como uno positivo.
    """

    def __init__(self, d_model: int, n_heads: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: Tensor, group: Tensor | None) -> Tensor:
        if group is None:
            return h
        misma_query = (group[:, None] == group[None, :]).unsqueeze(0)   # (1, B, B)
        x = self.norm(h).unsqueeze(0)                                   # (1, B, d)
        return h + self.dropout(self.attn(x, attn_mask=misma_query).squeeze(0))

    @property
    def attn_weights(self) -> Tensor | None:
        return self.attn.last_attn_weights


class BTRModel(nn.Module):
    """El modelo completo. Devuelve **logits**, uno por impresion."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_text:
            self.encoder = TransformerEncoder(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_heads=config.n_heads,
                n_layers=config.n_layers,
                d_ff=config.d_ff,
                max_len=config.max_len,
                dropout=config.dropout,
                pos_encoding=config.pos_encoding,
                norm=config.norm,
                attn_impl=config.attn_impl,
                pad_id=config.pad_id,
            )
            self.pooling = make_pooling(config.pooling)

        if config.use_tabular:
            self.tabular = TabularEncoder(
                config.cat_cardinalities, config.d_num, config.cat_emb_dim,
                config.d_tab, config.dropout,
            )

        self.cross_item = (
            CrossItemAttention(config.d_fusion, n_heads=2, dropout=config.dropout)
            if config.use_cross_item else None
        )

        self.head = ClassifierHead(
            d_in=config.d_fusion,
            d_hidden=config.d_head_hidden,
            dropout=config.dropout,
            prevalence=config.prevalence,
        )

    # -- forward ------------------------------------------------------------- #

    def representar(self, batch: Batch) -> Tensor:
        """El vector fusionado de cada impresion, antes de la cabeza."""
        partes = []
        if self.config.use_text:
            if batch.input_ids is None:
                raise ValueError(
                    "el modelo usa texto pero el batch no trae input_ids: el "
                    "dataset se construyo sin la rama de texto"
                )
            estados = self.encoder(batch.input_ids, batch.attention_mask)
            partes.append(self.pooling(estados, batch.attention_mask))
        if self.config.use_tabular:
            partes.append(self.tabular(batch.num, batch.cat))

        h = torch.cat(partes, dim=-1) if len(partes) > 1 else partes[0]
        if self.cross_item is not None:
            h = self.cross_item(h, batch.group)
        return h

    def forward(self, batch: Batch) -> Tensor:
        return self.head(self.representar(batch))

    @torch.no_grad()
    def predict_proba(self, batch: Batch) -> Tensor:
        """Probabilidades. El BTR predicho *es* esta salida."""
        self.eval()
        return torch.sigmoid(self(batch))

    # -- introspeccion -------------------------------------------------------- #

    def attention_maps(self) -> list[Tensor]:
        """Pesos de atencion del texto en la ultima pasada, uno por capa."""
        return self.encoder.attention_maps() if self.config.use_text else []

    def n_parameters(self, entrenables: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not entrenables)

    def resumen_parametros(self) -> dict[str, int]:
        """Parametros por bloque. Util para justificar el tamano del vocabulario."""
        contar = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
        resumen: dict[str, int] = {}
        if self.config.use_text:
            resumen["embeddings de token"] = contar(self.encoder.embeddings.tokens)
            resumen["positional encoding"] = contar(self.encoder.embeddings.pos)
            resumen["bloques encoder"] = contar(self.encoder.blocks)
            # Las dos LayerNorm sueltas del encoder (la de los embeddings y la
            # final de pre-LN) van aparte para que la suma cierre con el total.
            resumen["layernorm de entrada y salida"] = (
                contar(self.encoder)
                - resumen["embeddings de token"]
                - resumen["positional encoding"]
                - resumen["bloques encoder"]
            )
        if self.config.use_tabular:
            resumen["rama tabular"] = contar(self.tabular)
        if self.cross_item is not None:
            resumen["atencion cross-item"] = contar(self.cross_item)
        resumen["cabeza"] = contar(self.head)
        resumen["total"] = sum(resumen.values())
        return resumen


def build_model(config: ModelConfig) -> BTRModel:
    """Constructor con nombre explicito, para leer mejor en los scripts."""
    return BTRModel(config)
