"""``Dataset``, ``collate`` y ``DataLoader`` (Fase 4).

El dataset entero son 10.000 filas de ~90 tokens: entra holgadamente en memoria.
Por eso todo se precomputa **una vez** como tensores (la tokenizacion incluida,
que la produce la Fase 6) y ``__getitem__`` solo indexa. Tokenizar adentro de
``__getitem__`` es el error clasico que hace que el tiempo por epoca lo domine el
preprocesamiento en vez de la GPU.

El movimiento a ``DEVICE`` ocurre en **un solo lugar**, el ``collate_fn`` que
arma ``make_dataloader``. No hay ninguna llamada a ``.cuda()`` dispersa.

La rama de texto es opcional (``input_ids=None``): las ablaciones "solo tabular"
y "solo texto" de la Fase 9 se expresan construyendo el dataset sin una de las
dos partes, no con banderas adentro del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.features import Transformed
from src.utils.seed import DEVICE, SEED, make_generator, seed_worker


@dataclass
class Batch:
    """Un batch ya apilado. ``input_ids`` y ``attention_mask`` pueden ser None."""

    num: torch.Tensor                        # (B, d_num)  float32
    cat: torch.Tensor                        # (B, d_cat)  int64
    y: torch.Tensor                          # (B,)        float32
    input_ids: torch.Tensor | None = None    # (B, L)      int64
    attention_mask: torch.Tensor | None = None  # (B, L)   bool, True = token real

    def __len__(self) -> int:
        return int(self.num.shape[0])

    def to(self, device: torch.device) -> "Batch":
        """Mueve el batch entero. Es el unico punto que toca el dispositivo."""
        mover = lambda t: None if t is None else t.to(device, non_blocking=True)  # noqa: E731
        return Batch(
            num=mover(self.num),
            cat=mover(self.cat),
            y=mover(self.y),
            input_ids=mover(self.input_ids),
            attention_mask=mover(self.attention_mask),
        )

    @property
    def tiene_texto(self) -> bool:
        return self.input_ids is not None


class ImpressionsDataset(Dataset):
    """Una impresion por fila: texto tokenizado + tabulares + target.

    Parametros
    ----------
    num, cat
        Salidas de ``FeaturePipeline.transform``: numericas escaladas y codigos
        categoricos.
    y
        Target binario. Se guarda como float32 porque ``BCEWithLogitsLoss`` lo
        pide asi.
    input_ids, attention_mask
        Tokenizacion precomputada (Fase 6). ``None`` para las corridas solo
        tabulares. La mascara es booleana con ``True`` en los tokens reales, que
        es la convencion que usa la atencion propia de la Fase 7 (``nn.Multihead
        Attention`` usa la inversa, y eso se maneja solo en el test de
        equivalencia).
    """

    def __init__(
        self,
        num: np.ndarray,
        cat: np.ndarray,
        y: np.ndarray,
        input_ids: np.ndarray | None = None,
        attention_mask: np.ndarray | None = None,
    ) -> None:
        n = len(num)
        for nombre, arr in [("cat", cat), ("y", y), ("input_ids", input_ids),
                            ("attention_mask", attention_mask)]:
            if arr is not None and len(arr) != n:
                raise ValueError(f"{nombre} tiene {len(arr)} filas y num tiene {n}")
        if (input_ids is None) != (attention_mask is None):
            raise ValueError("input_ids y attention_mask van juntos o no van")

        self.num = torch.as_tensor(np.asarray(num), dtype=torch.float32)
        self.cat = torch.as_tensor(np.asarray(cat), dtype=torch.long)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.input_ids = (
            None if input_ids is None else torch.as_tensor(np.asarray(input_ids), dtype=torch.long)
        )
        self.attention_mask = (
            None
            if attention_mask is None
            else torch.as_tensor(np.asarray(attention_mask), dtype=torch.bool)
        )

    @classmethod
    def from_transformed(
        cls,
        X: Transformed,
        input_ids: np.ndarray | None = None,
        attention_mask: np.ndarray | None = None,
    ) -> "ImpressionsDataset":
        """Constructor desde la salida del ``FeaturePipeline``."""
        if X.y is None:
            raise ValueError("El dataframe transformado no trae target")
        return cls(X.num, X.cat, X.y, input_ids, attention_mask)

    def __len__(self) -> int:
        return int(self.num.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = {"num": self.num[i], "cat": self.cat[i], "y": self.y[i]}
        if self.input_ids is not None:
            item["input_ids"] = self.input_ids[i]
            item["attention_mask"] = self.attention_mask[i]
        return item

    @property
    def tiene_texto(self) -> bool:
        return self.input_ids is not None

    @property
    def prevalencia(self) -> float:
        return float(self.y.mean())

    @property
    def pos_weight(self) -> float:
        """``(negativos / positivos)``, para la ablacion de desbalance.

        Con prevalencia 0.13 da ~6.7. Es una opcion, no una necesidad: desplaza
        las probabilidades hacia arriba y arruina la calibracion, asi que si se
        usa hay que recalibrar antes de reportar BTR absolutos.
        """
        positivos = float(self.y.sum())
        return (len(self) - positivos) / positivos if positivos else float("nan")


def collate(items: list[dict[str, torch.Tensor]], device: torch.device | None = None) -> Batch:
    """Apila una lista de items en un ``Batch``, opcionalmente ya en ``device``."""
    tiene_texto = "input_ids" in items[0]
    batch = Batch(
        num=torch.stack([it["num"] for it in items]),
        cat=torch.stack([it["cat"] for it in items]),
        y=torch.stack([it["y"] for it in items]),
        input_ids=torch.stack([it["input_ids"] for it in items]) if tiene_texto else None,
        attention_mask=(
            torch.stack([it["attention_mask"] for it in items]) if tiene_texto else None
        ),
    )
    return batch if device is None else batch.to(device)


def make_dataloader(
    dataset: ImpressionsDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = SEED,
    num_workers: int = 0,
    device: torch.device | None = DEVICE,
    drop_last: bool = False,
) -> DataLoader:
    """``DataLoader`` reproducible.

    Sin ``generator`` y ``worker_init_fn`` el orden de los batches depende del
    estado global de torch y de la semilla que hereda cada worker, y dos
    corridas de la misma configuracion dejan de ser comparables.

    El batch se mueve a ``device`` dentro del ``collate_fn``, que es el unico
    punto del proyecto que lo hace. Con ``num_workers > 0`` eso no es posible:
    el ``collate`` corre en el proceso worker y no puede inicializar CUDA, asi
    que en ese caso se devuelve el batch en CPU y hay que llamar a
    ``batch.to(DEVICE)`` en el step. Con el dataset entero en memoria,
    ``num_workers = 0`` es lo razonable y es el default.
    """
    if num_workers > 0 and device is not None and device.type == "cuda":
        raise ValueError(
            "collate no puede mover a CUDA con num_workers > 0 (el collate corre "
            "en el worker). Usar num_workers=0, o device=None y mover con "
            "batch.to(DEVICE) en el step."
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        collate_fn=lambda items: collate(items, device=device),
        generator=make_generator(seed),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
