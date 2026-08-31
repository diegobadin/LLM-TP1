"""Estudio de ablacion (Fase 9).

Ocho configuraciones, cada una con **5 semillas sobre los 5 folds agrupados de
dev**: 25 entrenamientos por fila, 200 en total. Con ~200 positivos por fold de
validacion, diferencias de PR-AUC de pocos puntos son indistinguibles del ruido,
y sin desvio la tabla no permite concluir nada.

**Sobre cambiar mas de una cosa por fila.** Las cuatro filas de las que salen las
conclusiones del trabajo (el marcador, las dos ramas y la fuga) cambian **una
sola** clave, porque de ellas se afirma algo atribuible. Las dos filas de
arquitectura mueven tres claves a la vez a proposito: no se quiere saber si
importa la profundidad *o* el ancho *o* la cantidad de cabezas, sino si mover la
capacidad en bloque cambia algo, y para eso una receta vale por tres filas. Cada
configuracion **declara** que claves toca (``CAMBIOS_DECLARADOS``) y un test
verifica que la declaracion coincida exactamente con el diff contra la base: la
regla deja de ser implicita, pero la tabla sigue siendo auditable.

El conjunto de test **no se toca** en ninguna corrida de esta fase.

Uso:

    python -m src.ablations --listar               # que se va a correr
    python -m src.ablations                        # todo, secuencial
    python -m src.ablations --shard 0 --shards 4   # una parte (para paralelizar)
    python -m src.ablations --merge                # junta los shards en results.csv
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import pandas as pd

from src.data.features import CATEGORICAS
from src.data.load import PROJECT_ROOT, load_raw
from src.data.splits import load_splits
from src.evaluate import RESULTS_CSV, append_results, format_summary, git_commit, summarize
from src.train import (
    BASE,
    FoldData,
    RunConfig,
    model_config_de,
    prepare_fold,
    train_model,
)

SHARDS_DIR = PROJECT_ROOT / "experiments" / "shards"

#: Las cinco semillas del protocolo. 5 semillas x 5 folds = 25 corridas por fila.
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

_sin = lambda col: tuple(c for c in CATEGORICAS if c != col)  # noqa: E731


#: Que claves toca cada configuracion respecto de la base. Lo llena el propio
#: constructor, asi que no puede quedar desactualizado; el test compara esta
#: declaracion contra el diff real campo por campo.
CAMBIOS_DECLARADOS: dict[str, frozenset[str]] = {}


def _ablacion(name: str, nota: str, **cambios) -> tuple[RunConfig, str]:
    CAMBIOS_DECLARADOS[name] = frozenset(cambios)
    return replace(BASE, name=name, **cambios), nota


#: La grilla. El segundo elemento de cada par es que demuestra la fila.
#:
#: Ocho configuraciones: cuatro que sostienen las conclusiones (una clave cada
#: una), dos recetas de capacidad (tres claves), el tokenizador y la fuga.
GRILLA: tuple[tuple[RunConfig, str], ...] = (
    _ablacion("base",
              "Configuracion de referencia: texto + tabular, marcador de reputacion, "
              "2 capas, d_model 64, 4 cabezas, pre-LN y pooling [CLS]."),

    # -- las cuatro filas de las que salen las conclusiones ------------------ #
    _ablacion("sin_marcador",
              "Sin sufijo del titulo Y sin oracion de reputacion: las dos "
              "codificaciones del nivel. Es el hallazgo central del trabajo, y se "
              "esperan metricas de nivel tabular. Toca dos claves porque apagar una "
              "sola no saca la senal del input.",
              keep_suffix=False, keep_reputation_sentence=False),
    _ablacion("solo_texto",
              "Apaga la rama tabular. Mide el margen de la fusion: allergens y "
              "category modulan la compra dentro del nivel ALTO y ahi estan los ~10 "
              "puntos de PR-AUC que el texto solo no alcanza.",
              use_tabular=False),
    _ablacion("solo_tabular",
              "Apaga la rama de texto. Sin el texto el modelo no puede ver el "
              "marcador: es el techo de ROC 0.58 que justifica que el modulo "
              "principal sea un encoder de texto y no una arquitectura tabular.",
              use_text=False),
    _ablacion("con_cart",
              "Agrega cart, que es fuga: P(bought | cart=False) = 0 exacto y su valor "
              "es posterior al momento de la prediccion. Se corre para mostrar con "
              "numeros que la trampa ni siquiera paga.",
              include_cart=True),

    # -- capacidad, en bloque -------------------------------------------------- #
    _ablacion("arq_minima",
              "Receta minima: 1 capa, d_model 32 y 1 cabeza. La pregunta no es cual "
              "de los tres importa sino si hace falta algo de todo esto para leer un "
              "marcador de 4 tokens.",
              n_layers=1, d_model=32, n_heads=1),
    _ablacion("arq_grande",
              "Receta grande: 4 capas, d_model 96 (el maximo que permite el "
              "enunciado) y 8 cabezas. Mide si escalar la capacidad ayuda o "
              "sobreajusta con 6.300 ejemplos de entrenamiento.",
              n_layers=4, d_model=96, n_heads=8),

    # -- tokenizador -------------------------------------------------------------- #
    _ablacion("vocab_256",
              "Vocabulario BPE de 256 en lugar de los 1.355 que satura el corpus. El "
              "marcador pasa de 4 tokens a 7.26: el modelo tiene que recomponerlo "
              "antes de poder usarlo. Es el contenido de la Clase 2 puesto a prueba.",
              vocab_size=256),
)


#: Claves de las que depende la preparacion de datos. Dos corridas que solo
#: difieren en la semilla o en la arquitectura comparten los mismos tensores, y
#: preparar el fold cuesta mas que varias epocas de entrenamiento.
CLAVES_DE_DATOS: tuple[str, ...] = (
    "use_text", "use_tabular", "keep_suffix", "keep_reputation_sentence",
    "text_fields", "categoricas", "include_cart", "vocab_size", "use_cross_item",
)


class _CacheDeFolds:
    """Memoriza los **datos** preparados por (fold, claves de datos).

    Lo que se reutiliza son los tensores: el pipeline ajustado, el tokenizador y
    los datasets. La **arquitectura se rearma en cada corrida** con
    ``model_config_de``: si se reutilizara la del fold cacheado, una ablacion de
    profundidad o de cabezas entrenaria la arquitectura de la corrida anterior y
    su fila saldria identica a ella, sin ningun error visible.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple, FoldData] = {}

    def get(self, df, train_idx, eval_idx, fold_id: int, run: RunConfig) -> FoldData:
        clave = (fold_id,) + tuple(getattr(run, k) for k in CLAVES_DE_DATOS)
        if clave not in self._cache:
            self._cache[clave] = prepare_fold(df, train_idx, eval_idx, run)
        cacheado = self._cache[clave]
        return replace(cacheado, model_config=model_config_de(run, cacheado))


def run_ablations(
    grilla: tuple[tuple[RunConfig, str], ...] = GRILLA,
    seeds: tuple[int, ...] = SEEDS,
    verbose: bool = True,
    al_terminar_config=None,
) -> list[dict]:
    """Corre la grilla completa: por configuracion, semilla y fold."""
    df = load_raw()
    splits = load_splits()
    commit = git_commit()
    cache = _CacheDeFolds()

    filas: list[dict] = []
    for config, nota in grilla:
        t_config = time.perf_counter()
        for seed in seeds:
            run = replace(config, seed=seed)
            for k, (tr_idx, ev_idx) in enumerate(splits.folds):
                fold = cache.get(df, tr_idx, ev_idx, k, run)
                resultado = train_model(fold, run)
                filas.append({
                    "run_name": run.name,
                    "config": "ablacion",
                    "features": run.features_str,
                    "seed": seed,
                    "fold": k,
                    "split": "cv_dev",
                    "n_train": len(fold.train),
                    "n_eval": len(fold.eval),
                    "prevalence": round(resultado.best["prevalence"], 6),
                    "roc_auc": round(resultado.best["roc_auc"], 6),
                    "pr_auc": round(resultado.best["pr_auc"], 6),
                    "log_loss": round(resultado.best["log_loss"], 6),
                    "brier": round(resultado.best["brier"], 6),
                    "epochs_ran": resultado.epochs_ran,
                    "seconds": round(resultado.seconds, 2),
                    "commit": commit,
                    "notes": nota,
                })
        if al_terminar_config is not None:
            al_terminar_config(filas)
        if verbose:
            ultimas = pd.DataFrame(filas).query("run_name == @config.name")
            print(
                f"{config.name:>20}  PR {ultimas['pr_auc'].mean():.4f} "
                f"+- {ultimas['pr_auc'].std():.4f}   "
                f"ROC {ultimas['roc_auc'].mean():.4f}   "
                f"({time.perf_counter() - t_config:.0f}s, {len(ultimas)} corridas)",
                flush=True,
            )
    return filas


def _guardar_shard(filas: list[dict], shard: int) -> None:
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(SHARDS_DIR / f"ablaciones_shard{shard}.csv", index=False)


def merge_shards() -> pd.DataFrame:
    """Junta los shards en ``results.csv``.

    Cada shard escribe su propio archivo: si todos escribieran en ``results.csv``
    a la vez, el ultimo en cerrar se llevaria puesto el trabajo de los demas.
    """
    archivos = sorted(SHARDS_DIR.glob("ablaciones_shard*.csv"))
    if not archivos:
        raise FileNotFoundError(f"no hay shards en {SHARDS_DIR}")
    filas = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
    todas = append_results(filas.to_dict("records"))
    print(f"{len(filas)} corridas de {len(archivos)} shards -> "
          f"{RESULTS_CSV.relative_to(PROJECT_ROOT)} ({len(todas)} filas en total)")
    return todas


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listar", action="store_true", help="muestra la grilla y sale")
    parser.add_argument("--merge", action="store_true", help="junta los shards y sale")
    parser.add_argument("--shard", type=int, default=None, help="indice de este shard")
    parser.add_argument("--shards", type=int, default=1, help="cantidad de shards")
    parser.add_argument("--solo", nargs="*", default=None, help="nombres a correr")
    parser.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 90)

    if args.merge:
        merge_shards()
        return 0

    grilla = GRILLA
    if args.solo:
        desconocidos = set(args.solo) - {c.name for c, _ in GRILLA}
        if desconocidos:
            raise SystemExit(f"configuraciones desconocidas: {sorted(desconocidos)}")
        grilla = tuple((c, n) for c, n in GRILLA if c.name in args.solo)

    if args.listar:
        print(pd.DataFrame([
            {"configuracion": c.name, "que demuestra": n} for c, n in GRILLA
        ]).to_string(index=False))
        print(f"\n{len(GRILLA)} configuraciones x {len(SEEDS)} semillas x 5 folds = "
              f"{len(GRILLA) * len(SEEDS) * 5} entrenamientos")
        return 0

    if args.shard is not None:
        grilla = tuple(g for i, g in enumerate(grilla) if i % args.shards == args.shard)
        if not grilla:
            raise SystemExit(
                f"el shard {args.shard} de {args.shards} quedo vacio: hay menos "
                f"configuraciones que shards"
            )
        print(f"shard {args.shard}/{args.shards}: "
              f"{', '.join(c.name for c, _ in grilla)}", flush=True)

    # Se guarda despues de cada configuracion: una corrida de horas no puede
    # perderse entera porque el proceso se caiga en la ultima fila.
    guardar = (lambda filas: _guardar_shard(filas, args.shard)) if args.shard is not None else None

    inicio = time.perf_counter()
    filas = run_ablations(grilla, seeds=tuple(args.seeds), al_terminar_config=guardar)
    print(f"\n{len(filas)} corridas en {(time.perf_counter() - inicio) / 60:.1f} min")

    if args.shard is not None:
        _guardar_shard(filas, args.shard)
        print(f"shard guardado. Al terminar todos: python -m src.ablations --merge")
    else:
        append_results(filas)
        print(f"escrito en {RESULTS_CSV.relative_to(PROJECT_ROOT)}")

    if filas:
        print("\n" + "=" * 120)
        print("ABLACIONES  |  media +- desvio sobre 5 semillas x 5 folds")
        print("=" * 120)
        print(format_summary(summarize(pd.DataFrame(filas))).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
