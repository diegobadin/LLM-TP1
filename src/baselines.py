"""Baselines (Fase 5).

Sirven para dos cosas, y las dos son imprescindibles antes de escribir una sola
linea del Transformer:

1. **Validar el pipeline.** Si el baseline tabular no reproduce ROC ~ 0.58 y el
   de TF-IDF sobre el titulo no reproduce ~ 0.956, hay un bug en la Fase 4 y no
   en el modelo que todavia no existe.
2. **Fijar el objetivo.** El GBDT con el sufijo explicito llega a ROC ~ 0.970 y
   PR-AUC ~ 0.775. Si el Transformer da mucho mas, hay fuga; si da mucho menos,
   hay un bug. Es previsible y aceptable que el GBDT gane: la senal es un token
   categorico y no requiere composicionalidad.

Todos se evaluan con la **CV agrupada de 5 folds sobre dev** congelada en la
Fase 3. El test no se toca. Cada fold ajusta su propio ``FeaturePipeline`` y su
propio ``TfidfVectorizer`` con las filas de train del fold y nada mas.

Uso:

    python -m src.baselines                 # corre todo y escribe results.csv
    python -m src.baselines --solo prior tfidf_title
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from src.data.features import CATEGORICAS, CATEGORICAS_OPCIONALES, FeaturePipeline
from src.data.load import TARGET, load_raw
from src.data.splits import load_splits
from src.evaluate import append_results, compute_metrics, format_summary, git_commit, summarize

SEED = 42

#: Resultado de un baseline sobre un fold: probabilidades y epocas efectivas.
Prediccion = tuple[np.ndarray, int | None]


@dataclass(frozen=True)
class Baseline:
    """Un baseline: como se ajusta, con que features y que se espera de el."""

    nombre: str
    features: str
    notas: str
    fn: Callable[[pd.DataFrame, pd.DataFrame, int], Prediccion]


# --------------------------------------------------------------------------- #
# Los seis baselines
# --------------------------------------------------------------------------- #

def _tabular(train: pd.DataFrame, evaluar: pd.DataFrame, con_sufijo: bool = False):
    """Numericas escaladas + categoricas, ajustado solo con ``train``."""
    categoricas = CATEGORICAS + (CATEGORICAS_OPCIONALES if con_sufijo else ())
    pipe = FeaturePipeline(categoricas=categoricas).fit(train)
    return pipe, pipe.transform(train), pipe.transform(evaluar)


def prior(train: pd.DataFrame, evaluar: pd.DataFrame, seed: int) -> Prediccion:
    """Predice siempre la prevalencia de train. Es el piso del PR-AUC.

    ROC-AUC 0.5 por construccion y PR-AUC ~ prevalencia. Su log loss y su Brier
    son la referencia contra la que se mide si un modelo esta calibrado o solo
    ordena bien.
    """
    p = float(train[TARGET].astype(int).mean())
    return np.full(len(evaluar), p), None


def logreg_tabular(train: pd.DataFrame, evaluar: pd.DataFrame, seed: int) -> Prediccion:
    """Regresion logistica sobre numericas escaladas + one-hot."""
    _, Xtr, Xev = _tabular(train, evaluar)
    modelo = LogisticRegression(max_iter=2000, random_state=seed)
    modelo.fit(Xtr.tabular, Xtr.y)
    return modelo.predict_proba(Xev.tabular)[:, 1], None


def gbdt_tabular(train: pd.DataFrame, evaluar: pd.DataFrame, seed: int) -> Prediccion:
    """GBDT sobre las mismas features, con las categoricas nativas.

    Techo conocido: ROC ~ 0.58. Es la evidencia de que una arquitectura tabular
    pura no es una opcion viable para el modulo principal: sin el texto, el
    marcador de reputacion no existe.
    """
    _, Xtr, Xev = _tabular(train, evaluar)
    modelo = _hist_gbdt(Xtr, seed)
    modelo.fit(np.hstack([Xtr.num, Xtr.cat]), Xtr.y)
    return modelo.predict_proba(np.hstack([Xev.num, Xev.cat]))[:, 1], modelo.n_iter_


def gbdt_sufijo(train: pd.DataFrame, evaluar: pd.DataFrame, seed: int) -> Prediccion:
    """GBDT con el sufijo del titulo como una categorica mas.

    El baseline fuerte de referencia. El sufijo se extrae del texto con una
    regex, sin mirar el target: es la misma informacion que el encoder deberia
    aprender a leer, servida en bandeja.
    """
    _, Xtr, Xev = _tabular(train, evaluar, con_sufijo=True)
    modelo = _hist_gbdt(Xtr, seed)
    modelo.fit(np.hstack([Xtr.num, Xtr.cat]), Xtr.y)
    return modelo.predict_proba(np.hstack([Xev.num, Xev.cat]))[:, 1], modelo.n_iter_


def _hist_gbdt(Xtr, seed: int) -> HistGradientBoostingClassifier:
    """GBDT con la mascara de categoricas armada segun las columnas de ``Xtr``."""
    mascara = [False] * Xtr.num.shape[1] + [True] * Xtr.cat.shape[1]
    return HistGradientBoostingClassifier(
        categorical_features=mascara,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )


def _tfidf_logreg(
    campo: str, keep_suffix: bool = True, keep_reputation_sentence: bool = True
) -> Callable[[pd.DataFrame, pd.DataFrame, int], Prediccion]:
    """Fabrica de baselines de texto: TF-IDF (1,2)-gramas + regresion logistica.

    El vectorizador se ajusta **por fold, solo con train**. Ajustarlo sobre todo
    el dataset filtra el vocabulario y las frecuencias de documento del test
    hacia train.
    """

    def baseline(train: pd.DataFrame, evaluar: pd.DataFrame, seed: int) -> Prediccion:
        pipe = FeaturePipeline(
            keep_suffix=keep_suffix, keep_reputation_sentence=keep_reputation_sentence
        ).fit(train)
        Xtr, Xev = pipe.transform(train), pipe.transform(evaluar)
        texto_tr, texto_ev = _texto(Xtr, campo), _texto(Xev, campo)

        vect = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
        Vtr = vect.fit_transform(texto_tr)
        modelo = LogisticRegression(max_iter=2000, random_state=seed)
        modelo.fit(Vtr, Xtr.y)
        return modelo.predict_proba(vect.transform(texto_ev))[:, 1], None

    return baseline


def _texto(X, campo: str) -> np.ndarray:
    if campo == "title":
        return X.text_a
    if campo == "description":
        return X.text_b
    if campo == "both":
        return X.text
    raise ValueError(f"campo de texto desconocido: {campo!r}")


def mlp_texto_tabular(train: pd.DataFrame, evaluar: pd.DataFrame, seed: int) -> Prediccion:
    """MLP sobre tabulares + TF-IDF reducido con SVD.

    Aisla el efecto "es una red neuronal" del efecto "tiene atencion": si el
    Transformer no le gana a esto, lo que aporta no es el mecanismo de atencion.
    """
    pipe = FeaturePipeline().fit(train)
    Xtr, Xev = pipe.transform(train), pipe.transform(evaluar)

    vect = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    Vtr = vect.fit_transform(Xtr.text)
    svd = TruncatedSVD(n_components=100, random_state=seed)
    Ztr = svd.fit_transform(Vtr)
    Zev = svd.transform(vect.transform(Xev.text))

    modelo = MLPClassifier(
        hidden_layer_sizes=(64,),
        alpha=1e-3,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=seed,
    )
    modelo.fit(np.hstack([Xtr.tabular, Ztr]), Xtr.y)
    return modelo.predict_proba(np.hstack([Xev.tabular, Zev]))[:, 1], int(modelo.n_iter_)


BASELINES: tuple[Baseline, ...] = (
    Baseline(
        "prior", "ninguna",
        "Predice la prevalencia de train. Piso del PR-AUC.", prior,
    ),
    Baseline(
        "logreg_tabular", "num(11) + onehot(87)",
        "Sin texto: no puede ver el marcador de reputacion.", logreg_tabular,
    ),
    Baseline(
        "gbdt_tabular", "num(11) + cat(7)",
        "Sin texto. Techo esperado ROC ~ 0.58.", gbdt_tabular,
    ),
    Baseline(
        "tfidf_title", "tfidf(1,2) de title",
        "Solo el titulo, con su marcador. Esperado ROC ~ 0.956.",
        _tfidf_logreg("title"),
    ),
    Baseline(
        "tfidf_description", "tfidf(1,2) de description",
        "Solo la descripcion: su ultima oracion codifica el mismo nivel.",
        _tfidf_logreg("description"),
    ),
    Baseline(
        "tfidf_texto", "tfidf(1,2) de title + description",
        "Los dos campos. Esperado: sin diferencia, son redundantes.",
        _tfidf_logreg("both"),
    ),
    Baseline(
        "tfidf_sin_marcador", "tfidf(1,2) de texto sin marcador",
        "Sin sufijo y sin oracion de reputacion: el problema se vuelve "
        "irresoluble. Anticipa la ablacion central de la Fase 9.",
        _tfidf_logreg("both", keep_suffix=False, keep_reputation_sentence=False),
    ),
    Baseline(
        "gbdt_sufijo", "num(11) + cat(7) + sufijo",
        "Baseline fuerte de referencia. Esperado ROC ~ 0.970, PR-AUC ~ 0.775.",
        gbdt_sufijo,
    ),
    Baseline(
        "mlp_texto_tabular", "num(11) + onehot(87) + svd100(tfidf)",
        "Red neuronal sin atencion: aisla el efecto de la arquitectura.",
        mlp_texto_tabular,
    ),
)


# --------------------------------------------------------------------------- #
# Corrida
# --------------------------------------------------------------------------- #

def run_cv(
    baselines: tuple[Baseline, ...] = BASELINES,
    seed: int = SEED,
    verbose: bool = True,
) -> list[dict]:
    """Corre cada baseline sobre los 5 folds agrupados de ``dev``."""
    df = load_raw()
    splits = load_splits()
    commit = git_commit()

    filas: list[dict] = []
    for base in baselines:
        for k, (tr_idx, ev_idx) in enumerate(splits.folds):
            train, evaluar = df.iloc[tr_idx], df.iloc[ev_idx]
            t0 = time.perf_counter()
            y_prob, epocas = base.fn(train, evaluar, seed)
            segundos = time.perf_counter() - t0

            metricas = compute_metrics(evaluar[TARGET].astype(int).to_numpy(), y_prob)
            filas.append({
                "run_name": base.nombre,
                "config": "baseline",
                "features": base.features,
                "seed": seed,
                "fold": k,
                "split": "cv_dev",
                "n_train": len(train),
                "n_eval": len(evaluar),
                "prevalence": round(metricas.pop("prevalence"), 6),
                **{m: round(v, 6) for m, v in metricas.items()},
                "epochs_ran": epocas,
                "seconds": round(segundos, 2),
                "commit": commit,
                "notes": base.notas,
            })
            if verbose:
                ultima = filas[-1]
                print(
                    f"{base.nombre:>20}  fold {k}  "
                    f"ROC {ultima['roc_auc']:.4f}  PR {ultima['pr_auc']:.4f}  "
                    f"({segundos:.1f}s)"
                )
    return filas


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solo", nargs="*", default=None, help="nombres de baselines a correr")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--no-guardar", action="store_true", help="no escribe results.csv")
    args = parser.parse_args()

    elegidos = BASELINES
    if args.solo:
        desconocidos = set(args.solo) - {b.nombre for b in BASELINES}
        if desconocidos:
            raise SystemExit(f"baselines desconocidos: {sorted(desconocidos)}")
        elegidos = tuple(b for b in BASELINES if b.nombre in args.solo)

    pd.set_option("display.width", 200)
    filas = run_cv(elegidos, seed=args.seed)

    resumen = summarize(pd.DataFrame(filas))
    print("\n" + "=" * 120)
    print("BASELINES  |  media +- desvio sobre los 5 folds agrupados de dev")
    print("=" * 120)
    print(format_summary(resumen).to_string(index=False))

    if not args.no_guardar:
        todas = append_results(filas)
        print(f"\nresults.csv: {len(todas)} filas")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
