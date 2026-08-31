"""Evaluacion final e interpretabilidad (Fase 10).

Protocolo, en este orden y sin volver atras:

1. La configuracion final se elige **solo con los resultados de CV** sobre
   ``dev``. Regla explicita: la de mayor PR-AUC medio, salvo que su ventaja
   sobre la base no supere un error estandar, en cuyo caso gana la base por
   simplicidad. La regla se fija antes de mirar los numeros.
2. Se reentrena sobre ``dev`` **completo**. Como ya no queda particion para
   early stopping, la cantidad de epocas se fija en la mediana de la epoca
   optima de la CV de esa misma configuracion: es informacion de dev, no de
   test.
3. Se evalua **una sola vez** sobre ``test``.

Lo que se reporta: prevalencia del test (piso del PR-AUC), PR-AUC y lift,
ROC-AUC, log loss, Brier, intervalos por bootstrap agrupado por ``query_id``,
curva de calibracion, metricas de ranking por query y los mapas de atencion.

Uso:

    python -m src.final_eval --dry-run    # elige la configuracion, no toca test
    python -m src.final_eval              # entrena en dev y evalua test UNA vez
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from src.data.load import GROUP, PROJECT_ROOT, TARGET, load_raw
from src.data.splits import load_splits
from src.evaluate import append_results, compute_metrics, git_commit, load_results, summarize
from src.train import BASE, RunConfig, prepare_fold, predict, train_model
from src.utils.seed import DEVICE, set_seed
from src.ablations import GRILLA, SEEDS

N_BOOTSTRAP = 1000
DECILES = 10


# --------------------------------------------------------------------------- #
# 1. Eleccion de la configuracion, solo con CV
# --------------------------------------------------------------------------- #

def elegir_configuracion(
    resultados: pd.DataFrame | None = None, verbose: bool = True
) -> tuple[RunConfig, pd.DataFrame]:
    """La configuracion final, decidida contra la CV agrupada de ``dev``.

    Se descartan de entrada las que no pueden competir por razones que no son de
    metrica: ``con_cart`` usa una columna posterior al momento de la prediccion,
    y ``sin_marcador`` / ``solo_tabular`` son ablaciones diagnosticas (miden que
    pasa cuando se le saca al modelo lo que necesita), no candidatas. El nombre
    ``sin_sufijo`` se deja en la lista por si vuelve a la grilla: descalificar
    algo que no existe no cuesta nada, y olvidarse de descalificarlo si.
    """
    resultados = load_results() if resultados is None else resultados
    ablaciones = resultados.loc[resultados["config"] == "ablacion"]
    if ablaciones.empty:
        raise RuntimeError(
            "No hay ablaciones en results.csv. Correr primero:\n"
            "    python -m src.ablations"
        )

    descalificadas = {"con_cart", "sin_sufijo", "sin_marcador", "solo_tabular"}
    resumen = summarize(ablaciones).set_index("run_name")
    candidatas = resumen.drop(index=[d for d in descalificadas if d in resumen.index])

    mejor = candidatas["pr_auc"].idxmax()
    n = int(candidatas.loc[mejor, "corridas"])
    error_estandar = float(candidatas.loc[mejor, "pr_auc_sd"]) / np.sqrt(n)
    ventaja = float(candidatas.loc[mejor, "pr_auc"] - candidatas.loc["base", "pr_auc"])

    elegida = mejor if ventaja > error_estandar else "base"
    if verbose:
        print(f"mejor por PR-AUC de CV : {mejor} ({candidatas.loc[mejor, 'pr_auc']:.4f})")
        print(f"base                   : {candidatas.loc['base', 'pr_auc']:.4f}")
        print(f"ventaja {ventaja:+.4f} contra un error estandar de {error_estandar:.4f}")
        print(f"=> configuracion final : {elegida}")

    config = next(c for c, _ in GRILLA if c.name == elegida)
    return config, candidatas.reset_index()


def epocas_de_la_cv(config: RunConfig, seed: int = BASE.seed, verbose: bool = True) -> int:
    """Mediana de la epoca optima sobre los 5 folds de ``dev``.

    Reentrenar sobre dev completo deja sin particion al early stopping. Fijar
    las epocas con la CV de la misma configuracion usa informacion de dev y
    ninguna de test, que es la condicion que importa.
    """
    df, splits = load_raw(), load_splits()
    epocas = []
    for k, (tr_idx, ev_idx) in enumerate(splits.folds):
        run = replace(config, seed=seed)
        resultado = train_model(prepare_fold(df, tr_idx, ev_idx, run), run)
        epocas.append(resultado.best_epoch)
        if verbose:
            print(f"  fold {k}: mejor epoca {resultado.best_epoch} "
                  f"(PR {resultado.best['pr_auc']:.4f})")
    return int(np.median(epocas))


# --------------------------------------------------------------------------- #
# 2 y 3. Reentrenar sobre dev y evaluar test una sola vez
# --------------------------------------------------------------------------- #

def entrenar_y_evaluar(
    config: RunConfig, epocas: int, verbose: bool = True
) -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """Entrena sobre ``dev`` completo y predice sobre ``test``.

    Devuelve ``(modelo, y_true, y_prob, query_ids)``.
    """
    df, splits = load_raw(), load_splits()
    # patience >= epochs desactiva el early stopping: la cantidad de epocas ya
    # quedo fijada por la CV.
    run = replace(config, name=f"{config.name}_final", epochs=epocas, patience=epocas + 1)

    fold = prepare_fold(df, splits.dev, splits.test, run)
    # select_best=False es lo que hace legitima esta corrida: se entrena la
    # cantidad de epocas que fijo la CV y se reporta el estado final. Con
    # select_best=True se elegiria la epoca mirando la metrica de test, que es
    # exactamente la forma sutil de espiarlo.
    resultado = train_model(fold, run, verbose=verbose, select_best=False)
    grupos = df[GROUP].to_numpy()[splits.test]
    return resultado.model, resultado.y_true, resultado.y_prob, grupos


# --------------------------------------------------------------------------- #
# Intervalos, calibracion y ranking
# --------------------------------------------------------------------------- #

def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> pd.DataFrame:
    """Intervalos del 95% por bootstrap **agrupado por query**.

    Remuestrear filas sueltas trataria las impresiones de una misma busqueda
    como independientes y daria intervalos demasiado angostos: comparten
    filtros, contexto y, sobre todo, el sorteo del marcador.
    """
    rng = np.random.default_rng(seed)
    queries = np.unique(groups)
    indices_por_query = {q: np.flatnonzero(groups == q) for q in queries}

    muestras: list[dict[str, float]] = []
    for _ in range(n_boot):
        elegidas = rng.choice(queries, size=len(queries), replace=True)
        idx = np.concatenate([indices_por_query[q] for q in elegidas])
        if y_true[idx].sum() == 0:
            continue
        muestras.append(compute_metrics(y_true[idx], y_prob[idx]))

    tabla = pd.DataFrame(muestras)
    return pd.DataFrame({
        "metrica": tabla.columns,
        "media": tabla.mean().to_numpy(),
        "IC 2.5%": tabla.quantile(0.025).to_numpy(),
        "IC 97.5%": tabla.quantile(0.975).to_numpy(),
    })


def curva_calibracion(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = DECILES
) -> tuple[pd.DataFrame, float]:
    """Probabilidad media predicha contra frecuencia observada, por decil.

    Importa porque la salida del modelo *es* un BTR: una probabilidad con
    significado propio, no solo un puntaje para ordenar. Devuelve tambien el ECE
    (error de calibracion esperado), el promedio ponderado de las brechas.
    """
    bordes = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bordes[0], bordes[-1] = -np.inf, np.inf
    bin_id = np.digitize(y_prob, bordes[1:-1])

    filas = []
    for b in range(n_bins):
        m = bin_id == b
        if not m.any():
            continue
        filas.append({
            "decil": b + 1,
            "n": int(m.sum()),
            "prob. media predicha": float(y_prob[m].mean()),
            "frecuencia observada": float(y_true[m].mean()),
            "brecha": float(y_prob[m].mean() - y_true[m].mean()),
        })
    tabla = pd.DataFrame(filas)
    ece = float((tabla["n"] / tabla["n"].sum() * tabla["brecha"].abs()).sum())
    return tabla, ece


def metricas_de_ranking(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    ks: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Precision@k y Recall@k **por query**, mas el mAP por query.

    Es la forma de leer el modelo en su caso de uso: dentro de una busqueda, si
    hay que promocionar k productos, cuantos de los comprados aparecen ahi.
    """
    orden_global = np.argsort(-y_prob)
    filas = []
    for k in ks:
        por_query_p, por_query_r = [], []
        for q in np.unique(groups):
            m = np.flatnonzero(groups == q)
            if y_true[m].sum() == 0:
                continue
            top = m[np.argsort(-y_prob[m])[:k]]
            por_query_p.append(y_true[top].sum() / min(k, len(m)))
            por_query_r.append(y_true[top].sum() / y_true[m].sum())
        # Global: las k*n_queries impresiones con mayor BTR predicho de todo el test.
        n_global = k * len(np.unique(groups))
        top_global = orden_global[:n_global]
        filas.append({
            "k": k,
            "precision@k (por query)": float(np.mean(por_query_p)),
            "recall@k (por query)": float(np.mean(por_query_r)),
            "precision@k (global)": float(y_true[top_global].mean()),
            "recall@k (global)": float(y_true[top_global].sum() / y_true.sum()),
        })
    return pd.DataFrame(filas)


def map_por_query(y_true: np.ndarray, y_prob: np.ndarray, groups: np.ndarray) -> float:
    """``mAP``: average precision dentro de cada query, promediado.

    Solo sobre las queries que tienen al menos una compra: en las demas el AP no
    esta definido.
    """
    from sklearn.metrics import average_precision_score

    aps = []
    for q in np.unique(groups):
        m = np.flatnonzero(groups == q)
        if 0 < y_true[m].sum() < len(m):
            aps.append(average_precision_score(y_true[m], y_prob[m]))
        elif y_true[m].sum() == len(m):
            aps.append(1.0)
    return float(np.mean(aps))


# --------------------------------------------------------------------------- #
# Interpretabilidad: mapas de atencion
# --------------------------------------------------------------------------- #

@torch.no_grad()
def atencion_del_cls(
    model, fold, batch_size: int = 256, capa: int = -1
) -> tuple[np.ndarray, np.ndarray]:
    """Peso que el token ``[CLS]`` le da a cada posicion, promediado por cabeza.

    Devuelve ``(pesos, mascara)`` de forma ``(n, seq)``: el mapa promediado sobre
    las cabezas de la capa pedida, y la mascara de tokens reales.

    Es el punto 7 de la Fase 10: la hipotesis a verificar es que el modelo
    concentra la atencion sobre los tokens del marcador de reputacion.
    """
    from src.data.dataset import make_dataloader

    model.eval()
    fold.eval.to(DEVICE)
    loader = make_dataloader(fold.eval, batch_size=batch_size, shuffle=False, device=None)
    pesos, mascaras = [], []
    for batch in loader:
        model(batch)
        mapa = model.attention_maps()[capa]            # (B, H, L, L)
        pesos.append(mapa[:, :, 0, :].mean(dim=1).cpu().numpy())   # consulta = [CLS]
        mascaras.append(batch.attention_mask.cpu().numpy())
    return np.concatenate(pesos), np.concatenate(mascaras)


def atencion_por_token(
    pesos: np.ndarray, mascara: np.ndarray, tokens: list[list[str]], top: int = 25
) -> pd.DataFrame:
    """Agrega la atencion del ``[CLS]`` por *token*, no por posicion.

    Promediar por posicion mezcla tokens distintos; lo que interesa es a que
    palabras mira el modelo. Se reporta el peso medio que recibe cada token
    cuando aparece, y en cuantos ejemplos aparece.
    """
    acumulado: dict[str, list[float]] = {}
    for fila_pesos, fila_mascara, fila_tokens in zip(pesos, mascara, tokens):
        for j, token in enumerate(fila_tokens):
            if j < len(fila_mascara) and fila_mascara[j]:
                acumulado.setdefault(token, []).append(float(fila_pesos[j]))
    tabla = pd.DataFrame([
        {"token": t, "apariciones": len(v), "atencion media": float(np.mean(v))}
        for t, v in acumulado.items()
    ])
    return tabla.sort_values("atencion media", ascending=False).head(top).reset_index(drop=True)


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="elige la configuracion y calcula las epocas, sin tocar test")
    parser.add_argument("--epocas", type=int, default=None,
                        help="saltea la CV de epocas y usa este valor")
    args = parser.parse_args()

    pd.set_option("display.width", 200)

    print("=" * 100)
    print("1. ELECCION DE LA CONFIGURACION FINAL  (solo con CV sobre dev)")
    print("=" * 100)
    config, candidatas = elegir_configuracion()

    print("\n" + "=" * 100)
    print("2. CANTIDAD DE EPOCAS  (mediana de la epoca optima en la CV)")
    print("=" * 100)
    epocas = args.epocas or epocas_de_la_cv(config)
    print(f"=> se reentrena sobre dev completo por {epocas} epocas")

    if args.dry_run:
        print("\n--dry-run: no se toca el test.")
        return 0

    print("\n" + "=" * 100)
    print("3. TEST  (se evalua UNA sola vez)")
    print("=" * 100)
    modelo, y_true, y_prob, grupos = entrenar_y_evaluar(config, epocas, verbose=False)
    metricas = compute_metrics(y_true, y_prob)
    prevalencia = metricas["prevalence"]

    print(f"  prevalencia del test : {prevalencia:.4f}  (piso del PR-AUC)")
    print(f"  PR-AUC               : {metricas['pr_auc']:.4f}  "
          f"(lift {metricas['pr_auc'] / prevalencia:.2f}x)")
    print(f"  ROC-AUC              : {metricas['roc_auc']:.4f}")
    print(f"  log loss / Brier     : {metricas['log_loss']:.4f} / {metricas['brier']:.4f}")

    print("\nIntervalos del 95% por bootstrap agrupado por query:")
    print(bootstrap_ci(y_true, y_prob, grupos).to_string(index=False))

    print("\nCalibracion por decil:")
    tabla, ece = curva_calibracion(y_true, y_prob)
    print(tabla.to_string(index=False))
    print(f"ECE = {ece:.4f}")

    print("\nRanking:")
    print(metricas_de_ranking(y_true, y_prob, grupos).to_string(index=False))
    print(f"mAP por query = {map_por_query(y_true, y_prob, grupos):.4f}")

    append_results([{
        "run_name": f"{config.name}_final",
        "config": "final",
        "features": config.features_str,
        "seed": config.seed,
        "fold": -1,
        "split": "test",
        "n_train": len(load_splits().dev),
        "n_eval": len(y_true),
        "prevalence": round(prevalencia, 6),
        "roc_auc": round(metricas["roc_auc"], 6),
        "pr_auc": round(metricas["pr_auc"], 6),
        "log_loss": round(metricas["log_loss"], 6),
        "brier": round(metricas["brier"], 6),
        "epochs_ran": epocas,
        "seconds": None,
        "commit": git_commit(),
        "notes": "Evaluacion final sobre test. Se corre una sola vez (Fase 10).",
    }])
    print("\nresultado final anotado en experiments/results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
