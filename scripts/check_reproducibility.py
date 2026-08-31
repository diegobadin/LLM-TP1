"""Criterio de aceptacion de la Fase 0.

Entrena 20 steps dos veces con la misma semilla y verifica que la secuencia de
losses sea identica bit a bit. Corre en el dispositivo que reporte
``src.utils.seed.DEVICE``, asi que sobre una maquina con GPU valida el camino
CUDA, que es donde ``cudnn.benchmark`` y los kernels no deterministas rompen la
reproducibilidad.

    python scripts/check_reproducibility.py
    python scripts/check_reproducibility.py --num-workers 2

Se ejecuta dos veces dentro del mismo proceso y ademas imprime el hash de la
secuencia de losses, para poder comparar entre dos invocaciones distintas del
script (que es la forma literal en que el plan enuncia el criterio).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.utils.seed import DEVICE, SEED, make_generator, seed_worker, set_seed  # noqa: E402


def run_once(seed: int, steps: int, num_workers: int) -> list[float]:
    """Entrena un MLP minimo sobre datos sinteticos y devuelve las losses."""
    set_seed(seed)

    features = torch.randn(512, 32)
    labels = (torch.randn(512, 1) > 1.0).float()
    loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=32,
        shuffle=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
        num_workers=num_workers,
        drop_last=True,
    )

    model = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 1)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    losses: list[float] = []
    batches = iter(loader)
    for _ in range(steps):
        try:
            x, y = next(batches)
        except StopIteration:
            batches = iter(loader)
            x, y = next(batches)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return losses


def fingerprint(losses: list[float]) -> str:
    payload = "|".join(f"{value!r}" for value in losses).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    print(f"device={DEVICE}  torch={torch.__version__}  cuda={torch.version.cuda}")
    if DEVICE.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    first = run_once(args.seed, args.steps, args.num_workers)
    second = run_once(args.seed, args.steps, args.num_workers)

    print(f"loss inicial : {first[0]!r}")
    print(f"loss final   : {first[-1]!r}")
    print(f"fingerprint  : {fingerprint(first)}")

    if first != second:
        mismatches = [i for i, (a, b) in enumerate(zip(first, second)) if a != b]
        print(f"FALLA: las dos corridas difieren en los steps {mismatches[:5]}")
        print(f"  corrida A step {mismatches[0]}: {first[mismatches[0]]!r}")
        print(f"  corrida B step {mismatches[0]}: {second[mismatches[0]]!r}")
        return 1

    print(f"OK: {args.steps} steps identicos bit a bit en dos corridas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
