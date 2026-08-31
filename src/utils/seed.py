"""Semillas y dispositivo.

Punto unico de control de la reproducibilidad del proyecto. Todo el codigo
importa ``DEVICE`` desde aca; no hay ninguna llamada a ``.cuda()`` dispersa.

``CUBLAS_WORKSPACE_CONFIG`` se fija a nivel de modulo, antes de importar torch,
porque cuBLAS lee la variable cuando inicializa su workspace y despues de eso
ignora los cambios. Sin esa variable, ``torch.use_deterministic_algorithms``
levanta un error en la primera GEMM sobre GPU.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad del proceso.

    Se llama una sola vez, al inicio del proceso, nunca dentro del loop de
    entrenamiento: reseedear por epoca hace que todas las epocas vean el mismo
    orden de batches.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # benchmark=True elige el algoritmo de convolucion/GEMM por autotuning en
    # cada corrida, y el resultado deja de ser bit a bit reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` para ``DataLoader``.

    Con ``num_workers > 0`` cada worker es un proceso propio: hereda la semilla
    de torch pero no las de ``random`` ni ``numpy``, asi que cualquier
    transformacion que las use difiere entre corridas si no se resiembra aca.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = SEED) -> torch.Generator:
    """Generador para el ``shuffle`` del ``DataLoader``.

    Se pasa como ``generator=`` para que el orden de los batches dependa de esta
    semilla y no del estado global de torch, que otras llamadas pueden consumir.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def describe_environment() -> dict[str, str]:
    """Datos del entorno que hay que anotar junto a cada corrida.

    La version de CUDA y el nombre de la GPU cambian el resultado bit a bit
    entre maquinas aunque la semilla sea la misma.
    """
    info = {
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_version": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "device": str(DEVICE),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
