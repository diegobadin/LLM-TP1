"""Entrenamiento del modelo completo (Fase 8).

Una corrida es un ``RunConfig``: features, tokenizador, arquitectura y
optimizacion en un solo objeto. Cada fila de la tabla de ablacion de la Fase 9
es ``replace(BASE, ...)`` cambiando **una** clave, nunca una edicion del codigo.

Todo lo que depende de los datos se ajusta **por fold y solo con train**: el
``FeaturePipeline`` (escalador y vocabularios) y el tokenizador BPE. Entrenar el
tokenizador una vez sobre ``dev`` seria un sesgo chico pero real, y reentrenarlo
cuesta menos de un segundo.

Uso:

    python -m src.train --overfit     # test de sobreajuste sobre 64 ejemplos
    python -m src.train               # entrena en train, valida en valid
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field, replace

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from src.data.dataset import ImpressionsDataset, make_dataloader
from src.data.features import CATEGORICAS, FeaturePipeline
from src.data.load import GROUP, PROJECT_ROOT, TARGET, load_raw
from src.data.splits import load_splits
from src.evaluate import compute_metrics
from src.model.model import BTRModel, ModelConfig, build_model
from src.tokenizer.bpe import BPETokenizer
from src.utils.seed import DEVICE, set_seed

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CURVES_DIR = PROJECT_ROOT / "experiments" / "curves"

LOSSES: tuple[str, ...] = ("bce", "bce_pos_weight", "focal")
TEXT_FIELDS: tuple[str, ...] = ("title", "description", "both")


# --------------------------------------------------------------------------- #
# Configuracion de una corrida
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RunConfig:
    """Una corrida completa. Los defaults son la configuracion base del plan."""

    name: str = "base"

    # -- features ------------------------------------------------------------ #
    use_text: bool = True
    use_tabular: bool = True
    keep_suffix: bool = True
    keep_reputation_sentence: bool = True
    text_fields: str = "both"                      # title | description | both
    categoricas: tuple[str, ...] = CATEGORICAS
    include_cart: bool = False                     # ablacion de fuga (Fase 9)

    # -- tokenizador ---------------------------------------------------------- #
    vocab_size: int = 2000

    # -- arquitectura --------------------------------------------------------- #
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    pos_encoding: str = "sinusoidal"
    pooling: str = "cls"
    norm: str = "pre_ln"
    attn_impl: str = "scratch"
    use_cross_item: bool = False
    cat_emb_dim: int = 12
    d_tab: int = 32

    # -- optimizacion --------------------------------------------------------- #
    epochs: int = 50
    # El plan arranca en 64. Se sube a 256 porque a esta escala cada step cuesta
    # ~20 ms fijos de lanzamiento de kernels y casi nada de computo: con 256 la
    # epoca pasa de 2.4 s a 0.7 s sin mover la metrica. El lr se escala
    # linealmente con el batch (3e-4 x 4), que es la regla estandar; se probaron
    # 6e-4, 1e-3 y 2e-3 contra valid y las tres quedan dentro del desvio entre
    # folds, asi que se toma el valor que justifica la regla y no la busqueda.
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.10
    grad_clip: float = 1.0
    loss: str = "bce"
    focal_gamma: float = 2.0
    patience: int = 10
    metric: str = "pr_auc"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.text_fields not in TEXT_FIELDS:
            raise ValueError(f"text_fields desconocido: {self.text_fields!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss desconocida: {self.loss!r}. Opciones: {LOSSES}")

    @property
    def features_str(self) -> str:
        """Descripcion corta de las features, para ``results.csv``."""
        partes = []
        if self.use_text:
            marcador = "con marcador" if self.keep_suffix else "sin sufijo"
            if not self.keep_reputation_sentence:
                marcador += " ni oracion"
            partes.append(f"texto[{self.text_fields}, {marcador}]")
        if self.use_tabular:
            tab = f"tabular[{len(self.categoricas)} cat]"
            if self.include_cart:
                tab += " + CART (fuga)"
            partes.append(tab)
        return " + ".join(partes)


BASE = RunConfig()


# --------------------------------------------------------------------------- #
# Preparacion de datos de un fold
# --------------------------------------------------------------------------- #

@dataclass
class FoldData:
    """Lo que hace falta para entrenar y evaluar un fold."""

    train: ImpressionsDataset
    eval: ImpressionsDataset
    model_config: ModelConfig
    tokenizer: BPETokenizer | None
    pipeline: FeaturePipeline
    eval_index: np.ndarray = field(default_factory=lambda: np.array([]))


def prepare_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    run: RunConfig = BASE,
) -> FoldData:
    """Ajusta pipeline y tokenizador **con train** y arma los dos datasets."""
    categoricas = tuple(run.categoricas) + (("cart",) if run.include_cart else ())
    pipe = FeaturePipeline(
        categoricas=categoricas,
        keep_suffix=run.keep_suffix,
        keep_reputation_sentence=run.keep_reputation_sentence,
    ).fit(df.iloc[train_idx])
    Xtr, Xev = pipe.transform(df.iloc[train_idx]), pipe.transform(df.iloc[eval_idx])

    tokenizer, tokens_tr, tokens_ev = None, (None, None), (None, None)
    if run.use_text:
        tokenizer = BPETokenizer.train_from_pairs(
            Xtr.text_a, Xtr.text_b, vocab_size=run.vocab_size
        )
        tokens_tr = _encode(tokenizer, Xtr, run.text_fields)
        tokens_ev = _encode(tokenizer, Xev, run.text_fields)

    # Codigo entero de query, solo necesario para la atencion cross-item.
    grupos = None
    if run.use_cross_item:
        codigos = pd.factorize(df[GROUP])[0]
        grupos = (codigos[train_idx], codigos[eval_idx])

    train_ds = ImpressionsDataset.from_transformed(
        Xtr, *tokens_tr, group=None if grupos is None else grupos[0]
    )
    eval_ds = ImpressionsDataset.from_transformed(
        Xev, *tokens_ev, group=None if grupos is None else grupos[1]
    )

    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size if tokenizer else 0,
        max_len=tokenizer.max_len if tokenizer else 0,
        d_model=run.d_model,
        n_heads=run.n_heads,
        n_layers=run.n_layers,
        d_ff=run.d_ff,
        pos_encoding=run.pos_encoding,
        pooling=run.pooling,
        norm=run.norm,
        attn_impl=run.attn_impl,
        cat_cardinalities=pipe.cardinalidades,
        d_num=Xtr.num.shape[1],
        cat_emb_dim=run.cat_emb_dim,
        d_tab=run.d_tab,
        dropout=run.dropout,
        prevalence=float(Xtr.y.mean()),
        use_text=run.use_text,
        use_tabular=run.use_tabular,
        use_cross_item=run.use_cross_item,
    )
    return FoldData(train_ds, eval_ds, model_config, tokenizer, pipe, np.asarray(eval_idx))


def _encode(tokenizer: BPETokenizer, X, campo: str):
    if campo == "both":
        return tokenizer.encode(X.text_a, X.text_b)
    return tokenizer.encode(X.text_a if campo == "title" else X.text_b)


# --------------------------------------------------------------------------- #
# Perdida, optimizador y scheduler
# --------------------------------------------------------------------------- #

def make_loss(run: RunConfig, prevalence: float) -> nn.Module:
    """La perdida segun la ablacion de desbalance.

    Con prevalencia 0.13 el desbalance es moderado: ``pos_weight`` es una opcion,
    no una necesidad. Advertencia que vale para las tres variantes menos la
    primera: pesar los positivos desplaza las probabilidades hacia arriba y
    arruina la calibracion, asi que si se usan hay que recalibrar antes de
    reportar BTR absolutos.
    """
    if run.loss == "bce":
        return nn.BCEWithLogitsLoss()
    if run.loss == "bce_pos_weight":
        peso = torch.tensor((1 - prevalence) / prevalence, device=DEVICE)
        return nn.BCEWithLogitsLoss(pos_weight=peso)
    return FocalLoss(gamma=run.focal_gamma)


class FocalLoss(nn.Module):
    """Focal loss binaria: baja el peso de los ejemplos ya bien clasificados."""

    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)                      # probabilidad de la clase correcta
        return ((1 - p_t) ** self.gamma * bce).mean()


def make_optimizer(model: nn.Module, run: RunConfig) -> torch.optim.AdamW:
    """AdamW con weight decay solo donde corresponde.

    Los bias y las ganancias de ``LayerNorm`` se excluyen del decay: son
    parametros de escala y desplazamiento, y penalizarlos hacia cero sesga las
    activaciones sin ningun beneficio de regularizacion.
    """
    con_decay, sin_decay = [], []
    for nombre, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (sin_decay if p.ndim <= 1 or nombre.endswith(".bias") else con_decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": con_decay, "weight_decay": run.weight_decay},
            {"params": sin_decay, "weight_decay": 0.0},
        ],
        lr=run.lr,
    )


def make_scheduler(optimizer, total_steps: int, run: RunConfig):
    """Warmup lineal del 10% + decaimiento coseno, por step."""
    warmup = max(1, int(total_steps * run.warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        avance = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, avance)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# --------------------------------------------------------------------------- #
# Loop
# --------------------------------------------------------------------------- #

@dataclass
class TrainResult:
    """Lo que devuelve un entrenamiento."""

    model: BTRModel
    history: pd.DataFrame
    best: dict[str, float]
    best_epoch: int
    epochs_ran: int
    seconds: float
    y_true: np.ndarray
    y_prob: np.ndarray


@torch.no_grad()
def predict(model: BTRModel, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Probabilidades sobre un loader. ``model.eval()`` y sin gradientes."""
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        y_prob.append(torch.sigmoid(model(batch)).float().cpu().numpy())
        y_true.append(batch.y.cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_prob)


def train_model(
    fold: FoldData,
    run: RunConfig = BASE,
    verbose: bool = False,
) -> TrainResult:
    """Entrena con early stopping sobre la metrica de validacion del ``run``."""
    set_seed(run.seed)

    model = build_model(fold.model_config).to(DEVICE)
    criterio = make_loss(run, fold.model_config.prevalence)
    optimizador = make_optimizer(model, run)

    # El dataset entero (unos 5 MB) vive en el dispositivo: los batches se
    # indexan ahi mismo y no hay una transferencia por batch. Por eso los
    # loaders reciben device=None: el movimiento ya ocurrio, una sola vez.
    fold.train.to(DEVICE)
    fold.eval.to(DEVICE)
    train_loader = make_dataloader(
        fold.train, batch_size=run.batch_size, shuffle=True, seed=run.seed,
        by_query=run.use_cross_item, device=None,
    )
    eval_loader = make_dataloader(
        fold.eval, batch_size=max(256, run.batch_size), shuffle=False, seed=run.seed,
        by_query=run.use_cross_item, device=None,
    )
    scheduler = make_scheduler(optimizador, run.epochs * len(train_loader), run)

    mejor, mejor_epoca, sin_mejora = -np.inf, 0, 0
    mejor_estado, mejor_pred = None, None
    historia: list[dict] = []
    inicio = time.perf_counter()

    for epoca in range(1, run.epochs + 1):
        model.train()
        perdida_acumulada, vistos = 0.0, 0
        for batch in train_loader:
            optimizador.zero_grad(set_to_none=True)
            perdida = criterio(model(batch), batch.y)
            perdida.backward()
            nn.utils.clip_grad_norm_(model.parameters(), run.grad_clip)
            optimizador.step()
            scheduler.step()
            perdida_acumulada += perdida.detach().item() * len(batch)
            vistos += len(batch)

        y_true, y_prob = predict(model, eval_loader)
        metricas = compute_metrics(y_true, y_prob)
        historia.append({
            "epoca": epoca,
            "train_loss": perdida_acumulada / vistos,
            "lr": scheduler.get_last_lr()[0],
            **{k: v for k, v in metricas.items() if k != "prevalence"},
        })
        if verbose:
            ultima = historia[-1]
            print(f"  epoca {epoca:>3}  loss {ultima['train_loss']:.4f}  "
                  f"ROC {ultima['roc_auc']:.4f}  PR {ultima['pr_auc']:.4f}")

        actual = metricas[run.metric]
        if actual > mejor:
            mejor, mejor_epoca, sin_mejora = actual, epoca, 0
            # El mejor estado se guarda en CPU para no duplicar memoria de GPU.
            mejor_estado = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            mejor_pred = (y_true, y_prob)
        else:
            sin_mejora += 1
            if sin_mejora >= run.patience:
                break

    segundos = time.perf_counter() - inicio
    if mejor_estado is not None:
        model.load_state_dict(mejor_estado)
        model.to(DEVICE)

    y_true, y_prob = mejor_pred
    return TrainResult(
        model=model,
        history=pd.DataFrame(historia),
        best=compute_metrics(y_true, y_prob),
        best_epoch=mejor_epoca,
        epochs_ran=len(historia),
        seconds=segundos,
        y_true=y_true,
        y_prob=y_prob,
    )


def run_fold(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    run: RunConfig = BASE,
    verbose: bool = False,
) -> TrainResult:
    """Prepara el fold y entrena. Es la unidad que repite la Fase 9."""
    return train_model(prepare_fold(df, train_idx, eval_idx, run), run, verbose)


# --------------------------------------------------------------------------- #
# Test de sobreajuste
# --------------------------------------------------------------------------- #

def overfit_test(n: int = 64, epochs: int = 300, verbose: bool = True) -> float:
    """Entrena sobre ``n`` ejemplos sin regularizacion hasta memorizarlos.

    Es el diagnostico mas barato que existe: si el modelo no puede sobreajustar
    64 ejemplos, hay un bug en la mascara, en el reparto de cabezas o en el
    cableado de las ramas, y no tiene sentido entrenar sobre el dataset entero.
    Criterio de aceptacion de la Fase 8: loss < 0.01.
    """
    df = load_raw()
    splits = load_splits()
    # Se toman ejemplos con positivos: 64 filas al azar podrian tener 8 y el
    # sobreajuste seria trivial de otra manera.
    idx = splits.train[:n]
    run = replace(BASE, name="overfit", dropout=0.0, weight_decay=0.0, lr=1e-3,
                  epochs=epochs, patience=epochs, batch_size=n, warmup_ratio=0.05)

    fold = prepare_fold(df, idx, idx, run)
    set_seed(run.seed)
    model = build_model(fold.model_config).to(DEVICE)
    criterio = nn.BCEWithLogitsLoss()
    optimizador = make_optimizer(model, run)
    fold.train.to(DEVICE)
    loader = make_dataloader(fold.train, batch_size=n, shuffle=True, seed=run.seed, device=None)
    scheduler = make_scheduler(optimizador, epochs * len(loader), run)

    perdida = float("nan")
    for epoca in range(1, epochs + 1):
        model.train()
        for batch in loader:
            optimizador.zero_grad(set_to_none=True)
            perdida_tensor = criterio(model(batch), batch.y)
            perdida_tensor.backward()
            nn.utils.clip_grad_norm_(model.parameters(), run.grad_clip)
            optimizador.step()
            scheduler.step()
            perdida = perdida_tensor.detach().item()
        if verbose and (epoca % 50 == 0 or perdida < 0.01):
            print(f"  epoca {epoca:>3}: loss {perdida:.6f}")
        if perdida < 0.005:
            break
    return perdida


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def save_checkpoint(resultado: TrainResult, run: RunConfig, path=None) -> None:
    """Guarda pesos, configuracion y curvas."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    path = path or CHECKPOINT_DIR / f"{run.name}.pt"
    torch.save(
        {
            "state_dict": resultado.model.state_dict(),
            "model_config": asdict(resultado.model.config),
            "run_config": asdict(run),
            "best": resultado.best,
            "best_epoch": resultado.best_epoch,
        },
        path,
    )
    resultado.history.to_csv(CURVES_DIR / f"{run.name}.csv", index=False)
    (CHECKPOINT_DIR / f"{run.name}.json").write_text(
        json.dumps({"run": asdict(run), "best": resultado.best}, indent=2, default=str),
        encoding="utf-8",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overfit", action="store_true", help="test de sobreajuste")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=BASE.seed)
    parser.add_argument("--name", default="base")
    args = parser.parse_args()

    pd.set_option("display.width", 160)

    if args.overfit:
        print(f"TEST DE SOBREAJUSTE sobre 64 ejemplos  (device={DEVICE})")
        perdida = overfit_test()
        print(f"\nloss final: {perdida:.6f}  (criterio: < 0.01)")
        if perdida >= 0.01:
            print("FALLA: el modelo no puede memorizar 64 ejemplos. Hay un bug.")
            return 1
        print("OK: el modelo sobreajusta 64 ejemplos.")
        return 0

    run = replace(BASE, name=args.name, seed=args.seed,
                  **({"epochs": args.epochs} if args.epochs else {}))
    df = load_raw()
    splits = load_splits()

    print(f"device={DEVICE}   configuracion={run.name}   semilla={run.seed}")
    fold = prepare_fold(df, splits.train, splits.valid, run)
    print(f"vocabulario {fold.model_config.vocab_size}, max_len {fold.model_config.max_len}, "
          f"cardinalidades {fold.model_config.cat_cardinalities}")

    resultado = train_model(fold, run, verbose=True)
    parametros = resultado.model.resumen_parametros()

    print("\n" + "=" * 90)
    print("PARAMETROS")
    print("=" * 90)
    for bloque, cantidad in parametros.items():
        print(f"  {bloque:>24}: {cantidad:>8,}")

    print("\n" + "=" * 90)
    print("RESULTADO SOBRE VALID (mejor epoca por PR-AUC)")
    print("=" * 90)
    print(f"  mejor epoca      : {resultado.best_epoch} de {resultado.epochs_ran} corridas")
    print(f"  ROC-AUC          : {resultado.best['roc_auc']:.4f}")
    print(f"  PR-AUC           : {resultado.best['pr_auc']:.4f}  "
          f"(prevalencia {resultado.best['prevalence']:.4f}, "
          f"lift {resultado.best['pr_auc'] / resultado.best['prevalence']:.2f}x)")
    print(f"  log loss / Brier : {resultado.best['log_loss']:.4f} / {resultado.best['brier']:.4f}")
    print(f"  tiempo           : {resultado.seconds:.1f}s "
          f"({resultado.seconds / resultado.epochs_ran:.2f}s por epoca)")

    save_checkpoint(resultado, run)
    print(f"\ncheckpoint en checkpoints/{run.name}.pt y curvas en "
          f"experiments/curves/{run.name}.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
