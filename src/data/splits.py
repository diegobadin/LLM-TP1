"""Particion agrupada por ``query_id`` (Fase 3).

Los splits se generan **una sola vez** y quedan congelados en disco. Ningun
notebook ni script los recalcula: todos llaman a ``load_splits()``. Recalcular
la particion en cada notebook es la via mas comun a que dos experimentos se
comparen contra particiones distintas sin que nadie lo note.

Estructura:

* ``test`` (20% de las queries) queda intacto hasta la Fase 10. Se toca **una
  sola vez**.
* ``dev`` (80%) se usa para todo lo demas. Sobre el se define:
  * una CV agrupada de 5 folds, que es contra lo que se decide todo (seleccion
    de modelo y ablaciones);
  * un split fijo ``train`` / ``valid`` para el desarrollo rapido y el early
    stopping mientras se construye el modelo.

Todos los indices son **posicionales sobre el dataframe que devuelve
``load_raw()``**, que preserva el orden del CSV.

Los ``.npy`` viven en ``data/``, que esta gitignoreado, asi que no se versionan:
se regeneran con una linea. Lo que si esta versionado es
``EXPECTED_FINGERPRINT``, un hash de los indices que permite detectar que una
regeneracion dio algo distinto de lo que produjo los resultados reportados.

Uso:

    python -m src.data.splits            # genera, guarda e informa
    python -m src.data.splits --check    # verifica lo que hay en disco
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from src.data.load import GROUP, TARGET, PROJECT_ROOT, load_raw

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

SEED = 42
TEST_SIZE = 0.20
VALID_SIZE = 0.20
N_FOLDS = 5

#: Hash de los indices producidos por ``make_splits`` con los parametros por
#: defecto. Congelado tras la primera generacion: si una regeneracion no lo
#: reproduce, cambio el CSV, la semilla o la version de scikit-learn, y los
#: resultados ya reportados dejan de ser comparables.
EXPECTED_FINGERPRINT = "6a7a655a7caf4ae4"

#: Tolerancia del criterio de aceptacion: la prevalencia de cada particion no
#: puede desviarse mas de 1.5 puntos porcentuales de la global.
PREVALENCE_TOL = 0.015


@dataclass(frozen=True)
class Splits:
    """Indices posicionales sobre el dataframe completo."""

    dev: np.ndarray
    test: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def make_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    valid_size: float = VALID_SIZE,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> Splits:
    """Construye la particion agrupada por ``query_id``.

    No se estratifica por ``bought`` a nivel fila: agrupando por query, la
    estratificacion a nivel fila no tiene sentido (las filas de una query van
    juntas si o si). Se verifica a posteriori con ``describe``.
    """
    groups = df[GROUP].to_numpy()
    posiciones = np.arange(len(df))

    # Split externo: dev / test.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_local, test_idx = next(gss.split(posiciones, groups=groups))
    dev_idx = posiciones[dev_local]
    test_idx = posiciones[test_idx]

    # Split interno de desarrollo: train / valid dentro de dev.
    gss_interno = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    tr_local, va_local = next(gss_interno.split(dev_idx, groups=groups[dev_idx]))
    train_idx = dev_idx[tr_local]
    valid_idx = dev_idx[va_local]

    # CV agrupada sobre dev. GroupKFold sin shuffle es determinista por si solo.
    gkf = GroupKFold(n_splits=n_folds)
    folds = tuple(
        (dev_idx[tr], dev_idx[va])
        for tr, va in gkf.split(dev_idx, groups=groups[dev_idx])
    )

    return Splits(
        dev=np.sort(dev_idx),
        test=np.sort(test_idx),
        train=np.sort(train_idx),
        valid=np.sort(valid_idx),
        folds=tuple((np.sort(tr), np.sort(va)) for tr, va in folds),
    )


def fingerprint(splits: Splits) -> str:
    """Hash estable de todos los indices de la particion."""
    h = hashlib.sha256()
    for nombre, arr in [("dev", splits.dev), ("test", splits.test),
                        ("train", splits.train), ("valid", splits.valid)]:
        h.update(nombre.encode())
        h.update(np.ascontiguousarray(arr, dtype=np.int64).tobytes())
    for k, (tr, va) in enumerate(splits.folds):
        h.update(f"fold{k}".encode())
        h.update(np.ascontiguousarray(tr, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(va, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def save_splits(splits: Splits, directory: Path = SPLITS_DIR) -> None:
    """Escribe los ``.npy`` y un manifiesto con los parametros de generacion."""
    directory.mkdir(parents=True, exist_ok=True)
    for nombre in ("dev", "test", "train", "valid"):
        np.save(directory / f"{nombre}_idx.npy", getattr(splits, nombre))
    for k, (tr, va) in enumerate(splits.folds):
        np.save(directory / f"fold{k}_train_idx.npy", tr)
        np.save(directory / f"fold{k}_valid_idx.npy", va)

    manifiesto = {
        "seed": SEED,
        "test_size": TEST_SIZE,
        "valid_size": VALID_SIZE,
        "n_folds": splits.n_folds,
        "group": GROUP,
        "fingerprint": fingerprint(splits),
        "sizes": {
            "dev": int(splits.dev.size),
            "test": int(splits.test.size),
            "train": int(splits.train.size),
            "valid": int(splits.valid.size),
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")


def load_splits(directory: Path = SPLITS_DIR) -> Splits:
    """Lee los splits congelados. Falla si no estan generados."""
    manifiesto_path = directory / "manifest.json"
    if not manifiesto_path.exists():
        raise FileNotFoundError(
            f"No hay splits en {directory}. Generarlos con:\n"
            f"    python -m src.data.splits"
        )
    manifiesto = json.loads(manifiesto_path.read_text(encoding="utf-8"))
    n_folds = int(manifiesto["n_folds"])

    cargar = lambda nombre: np.load(directory / f"{nombre}_idx.npy")  # noqa: E731
    splits = Splits(
        dev=cargar("dev"),
        test=cargar("test"),
        train=cargar("train"),
        valid=cargar("valid"),
        folds=tuple(
            (cargar(f"fold{k}_train"), cargar(f"fold{k}_valid")) for k in range(n_folds)
        ),
    )

    obtenido = fingerprint(splits)
    if obtenido != manifiesto["fingerprint"]:
        raise ValueError(
            f"Los .npy no coinciden con su manifiesto ({obtenido} != "
            f"{manifiesto['fingerprint']}). Regenerar con --force."
        )
    return splits


def describe(df: pd.DataFrame, splits: Splits) -> pd.DataFrame:
    """Tamano, queries y prevalencia de cada particion."""
    y = df[TARGET].astype(int)
    grupos = df[GROUP]
    global_prev = y.mean()

    particiones: list[tuple[str, np.ndarray]] = [
        ("dev", splits.dev),
        ("test", splits.test),
        ("  train (dentro de dev)", splits.train),
        ("  valid (dentro de dev)", splits.valid),
    ]
    for k, (tr, va) in enumerate(splits.folds):
        particiones.append((f"  fold {k} train", tr))
        particiones.append((f"  fold {k} valid", va))

    filas = []
    for nombre, idx in particiones:
        prev = y.iloc[idx].mean()
        filas.append({
            "particion": nombre,
            "filas": len(idx),
            "% del total": round(len(idx) / len(df) * 100, 1),
            "queries": grupos.iloc[idx].nunique(),
            "positivos": int(y.iloc[idx].sum()),
            "prevalencia": round(prev, 4),
            "delta vs global (pp)": round((prev - global_prev) * 100, 2),
        })
    return pd.DataFrame(filas)


def group_overlap(df: pd.DataFrame, splits: Splits) -> pd.DataFrame:
    """Solapamiento de ``query_id`` entre particiones que deben ser disjuntas."""
    grupos = df[GROUP]

    def qs(idx: np.ndarray) -> set:
        return set(grupos.iloc[idx].unique())

    pares = [
        ("dev", splits.dev, "test", splits.test),
        ("train", splits.train, "valid", splits.valid),
        ("train", splits.train, "test", splits.test),
        ("valid", splits.valid, "test", splits.test),
    ]
    for k, (tr, va) in enumerate(splits.folds):
        pares.append((f"fold{k} train", tr, f"fold{k} valid", va))
        pares.append((f"fold{k} valid", va, "test", splits.test))

    return pd.DataFrame([
        {"a": na, "b": nb, "queries compartidas": len(qs(ia) & qs(ib))}
        for na, ia, nb, ib in pares
    ])


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verifica los splits en disco en vez de generarlos")
    parser.add_argument("--force", action="store_true",
                        help="regenera y sobreescribe aunque ya existan")
    args = parser.parse_args()

    pd.set_option("display.width", 160)
    df = load_raw()

    if args.check:
        splits = load_splits()
        print(f"Splits leidos de {SPLITS_DIR.relative_to(PROJECT_ROOT)}")
    else:
        if (SPLITS_DIR / "manifest.json").exists() and not args.force:
            print(f"Ya existen splits en {SPLITS_DIR.relative_to(PROJECT_ROOT)}. "
                  f"Usar --force para regenerarlos o --check para verificarlos.")
            return 0
        splits = make_splits(df)
        save_splits(splits)
        print(f"Splits generados en {SPLITS_DIR.relative_to(PROJECT_ROOT)}")

    huella = fingerprint(splits)
    print(f"fingerprint: {huella}  (esperado: {EXPECTED_FINGERPRINT})")

    print("\n" + "=" * 110)
    print("PARTICIONES")
    print("=" * 110)
    print(describe(df, splits).to_string(index=False))

    print("\n" + "=" * 110)
    print("SOLAPAMIENTO DE query_id (todo debe dar 0)")
    print("=" * 110)
    solape = group_overlap(df, splits)
    print(solape.to_string(index=False))

    problemas = []
    if huella != EXPECTED_FINGERPRINT:
        problemas.append(
            f"el fingerprint {huella} no coincide con el congelado "
            f"{EXPECTED_FINGERPRINT}"
        )
    if solape["queries compartidas"].any():
        problemas.append("hay query_id en dos particiones a la vez")

    resumen = describe(df, splits)
    fuera = resumen.loc[resumen["delta vs global (pp)"].abs() > PREVALENCE_TOL * 100]
    if not fuera.empty:
        problemas.append(
            f"prevalencia fuera de +-{PREVALENCE_TOL * 100:.1f} pp en: "
            f"{', '.join(fuera['particion'].str.strip())}"
        )

    if problemas:
        print("\nFALLA:")
        for p in problemas:
            print(f"  - {p}")
        return 1

    print("\nOK: sin solapamiento de queries y prevalencias dentro de "
          f"+-{PREVALENCE_TOL * 100:.1f} pp de la global.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
