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
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

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
    group: torch.Tensor | None = None        # (B,)        int64, query de cada fila

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
            group=mover(self.group),
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
    group
        Codigo entero de la query de cada fila. Solo lo necesita la ablacion de
        atencion cross-item, que deja que los items de una misma busqueda se
        miren entre si; para el resto de las corridas es ``None``.
    """

    def __init__(
        self,
        num: np.ndarray,
        cat: np.ndarray,
        y: np.ndarray,
        input_ids: np.ndarray | None = None,
        attention_mask: np.ndarray | None = None,
        group: np.ndarray | None = None,
    ) -> None:
        n = len(num)
        for nombre, arr in [("cat", cat), ("y", y), ("input_ids", input_ids),
                            ("attention_mask", attention_mask), ("group", group)]:
            if arr is not None and len(arr) != n:
                raise ValueError(f"{nombre} tiene {len(arr)} filas y num tiene {n}")
        if (input_ids is None) != (attention_mask is None):
            raise ValueError("input_ids y attention_mask van juntos o no van")

        self.num = torch.tensor(np.asarray(num), dtype=torch.float32)
        self.cat = torch.tensor(np.asarray(cat), dtype=torch.long)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)
        self.input_ids = (
            None if input_ids is None else torch.tensor(np.asarray(input_ids), dtype=torch.long)
        )
        self.attention_mask = (
            None
            if attention_mask is None
            else torch.tensor(np.asarray(attention_mask), dtype=torch.bool)
        )
        self.group = (
            None if group is None else torch.tensor(np.asarray(group), dtype=torch.long)
        )

    @classmethod
    def from_transformed(
        cls,
        X: Transformed,
        input_ids: np.ndarray | None = None,
        attention_mask: np.ndarray | None = None,
        group: np.ndarray | None = None,
    ) -> "ImpressionsDataset":
        """Constructor desde la salida del ``FeaturePipeline``."""
        if X.y is None:
            raise ValueError("El dataframe transformado no trae target")
        return cls(X.num, X.cat, X.y, input_ids, attention_mask, group)

    def __len__(self) -> int:
        return int(self.num.shape[0])

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        """Una fila, o **un lote entero** si ``i`` es una secuencia de indices.

        El segundo caso es el camino rapido: apilar 64 filas con
        ``torch.stack`` sobre 64 tensores sueltos cuesta mas tiempo de Python
        que el forward completo de un modelo de este tamano. Indexando de una
        sola vez, la epoca pasa de segundos a decimas.
        """
        if isinstance(i, (list, tuple, np.ndarray, torch.Tensor)):
            idx = torch.as_tensor(i, dtype=torch.long, device=self.num.device)
            return self.take(idx)
        item = {"num": self.num[i], "cat": self.cat[i], "y": self.y[i]}
        if self.input_ids is not None:
            item["input_ids"] = self.input_ids[i]
            item["attention_mask"] = self.attention_mask[i]
        if self.group is not None:
            item["group"] = self.group[i]
        return item

    def take(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Las filas ``idx``, ya apiladas."""
        lote = {"num": self.num[idx], "cat": self.cat[idx], "y": self.y[idx]}
        if self.input_ids is not None:
            lote["input_ids"] = self.input_ids[idx]
            lote["attention_mask"] = self.attention_mask[idx]
        if self.group is not None:
            lote["group"] = self.group[idx]
        return lote

    def to(self, device: torch.device) -> "ImpressionsDataset":
        """Deja el dataset entero en ``device``. Son pocos MB y evita copias.

        Con 10.000 filas de 56 tokens el dataset completo ocupa unos 5 MB: cabe
        de sobra en la GPU, y tenerlo ahi elimina una transferencia por batch.
        """
        for nombre in ("num", "cat", "y", "input_ids", "attention_mask", "group"):
            tensor = getattr(self, nombre)
            if tensor is not None:
                setattr(self, nombre, tensor.to(device))
        return self

    @property
    def tiene_texto(self) -> bool:
        return self.input_ids is not None

    @property
    def tiene_grupos(self) -> bool:
        return self.group is not None

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
    tiene_grupos = "group" in items[0]
    batch = Batch(
        num=torch.stack([it["num"] for it in items]),
        cat=torch.stack([it["cat"] for it in items]),
        y=torch.stack([it["y"] for it in items]),
        input_ids=torch.stack([it["input_ids"] for it in items]) if tiene_texto else None,
        attention_mask=(
            torch.stack([it["attention_mask"] for it in items]) if tiene_texto else None
        ),
        group=torch.stack([it["group"] for it in items]) if tiene_grupos else None,
    )
    return batch if device is None else batch.to(device)


class QueryBatchSampler(Sampler):
    """Arma batches sin partir queries: todos los items de una van juntos.

    Lo necesita la ablacion de atencion cross-item: si las filas de una busqueda
    caen en batches distintos, los items no pueden mirarse entre si y la
    ablacion mediria otra cosa. Como las queries tienen entre 1 y 8 items, los
    batches quedan de tamano variable alrededor de ``batch_size``.

    Para el resto de las corridas no cambiaria nada del modelo, solo el orden en
    que se recorren las filas, asi que se activa unicamente cuando hace falta.
    """

    def __init__(
        self,
        groups: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
        seed: int = SEED,
        drop_last: bool = False,
    ) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        indices_por_query: dict[int, list[int]] = {}
        for i, g in enumerate(groups.tolist()):
            indices_por_query.setdefault(g, []).append(i)
        self.queries = list(indices_por_query.values())
        self.epoca = 0

    def _armar(self, orden):
        batch: list[int] = []
        for q in orden:
            if batch and len(batch) + len(self.queries[q]) > self.batch_size:
                yield batch
                batch = []
            batch.extend(self.queries[q])
        if batch and not self.drop_last:
            yield batch

    def __iter__(self):
        orden = list(range(len(self.queries)))
        if self.shuffle:
            # La semilla avanza por epoca: dos epocas no ven el mismo orden, y
            # dos corridas con la misma semilla si ven la misma secuencia.
            generador = torch.Generator().manual_seed(self.seed + self.epoca)
            orden = torch.randperm(len(self.queries), generator=generador).tolist()
            self.epoca += 1
        return iter(list(self._armar(orden)))

    def __len__(self) -> int:
        return len(list(self._armar(range(len(self.queries)))))


def make_dataloader(
    dataset: ImpressionsDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    seed: int = SEED,
    num_workers: int = 0,
    device: torch.device | None = DEVICE,
    drop_last: bool = False,
    by_query: bool = False,
    fast: bool = True,
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

    if fast and num_workers == 0:
        return _fast_dataloader(
            dataset, batch_size, shuffle, seed, device, drop_last, by_query
        )

    comunes = dict(
        num_workers=num_workers,
        collate_fn=lambda items: collate(items, device=device),
        generator=make_generator(seed),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    if by_query:
        if dataset.group is None:
            raise ValueError(
                "by_query=True necesita un dataset construido con group=<codigo de query>"
            )
        sampler = QueryBatchSampler(dataset.group, batch_size, shuffle, seed, drop_last)
        return DataLoader(dataset, batch_sampler=sampler, **comunes)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **comunes,
    )


def _fast_dataloader(
    dataset: ImpressionsDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device | None,
    drop_last: bool,
    by_query: bool,
) -> DataLoader:
    """``DataLoader`` que entrega los indices de todo el lote de una vez.

    Mismas garantias de reproducibilidad que el camino general: el orden lo fija
    un ``torch.Generator`` con semilla propia. Lo que cambia es que el fetch
    indexa el tensor completo en lugar de recorrer fila por fila en Python.
    """
    if by_query:
        if dataset.group is None:
            raise ValueError(
                "by_query=True necesita un dataset construido con group=<codigo de query>"
            )
        muestreador = QueryBatchSampler(dataset.group, batch_size, shuffle, seed, drop_last)
    else:
        base = (
            RandomSampler(dataset, generator=make_generator(seed))
            if shuffle
            else SequentialSampler(dataset)
        )
        muestreador = BatchSampler(base, batch_size, drop_last)

    def armar(lote: dict[str, torch.Tensor]) -> Batch:
        batch = Batch(**lote)
        return batch if device is None else batch.to(device)

    # batch_size=None desactiva el colacionado automatico: cada "muestra" que
    # pide el DataLoader es ya la lista de indices del lote.
    return DataLoader(dataset, sampler=muestreador, batch_size=None, collate_fn=armar)
