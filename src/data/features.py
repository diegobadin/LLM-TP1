"""Features y preprocesamiento (Fase 4).

Un unico objeto, ``FeaturePipeline``, que se ajusta **una sola vez y solo con
train** y despues transforma cualquier particion. Ajustar el escalador o el
vocabulario de las categoricas con el dataset completo es contaminacion entre
particiones: la media del test participaria en la normalizacion de train. Es la
fuga mas frecuente y la mas facil de evitar con disciplina de pipeline, asi que
aca se vuelve imposible por construccion: ``fit`` falla si se lo llama dos veces
sobre el mismo objeto (para la CV se construye un pipeline nuevo por fold).

El pipeline produce las representaciones que necesitan las fases siguientes,
todas a partir del mismo ajuste:

* ``num``      numericas escaladas              -> baselines y rama tabular
* ``onehot``   categoricas en one-hot           -> baselines lineales y GBDT
* ``cat``      categoricas como codigos enteros -> ``nn.Embedding`` (Fase 8)
* ``text_a`` / ``text_b``  titulo y descripcion -> tokenizador (Fase 6)

Uso:

    pipe = FeaturePipeline().fit(df.iloc[train_idx])
    Xtr = pipe.transform(df.iloc[train_idx])
    Xva = pipe.transform(df.iloc[valid_idx])

    python -m src.data.features        # informa que construye y que descarta
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.data.load import (
    FEATURES_ADMITIDAS,
    TARGET,
    extract_suffix,
    load_raw,
    strip_suffix,
)

# --------------------------------------------------------------------------- #
# Que entra y que no
# --------------------------------------------------------------------------- #

#: Numericas que vienen del CSV tal cual.
NUMERICAS_BASE: tuple[str, ...] = (
    "price",
    "net_weight_oz",
    "nutrition_score",
    "filter_price_min",
    "filter_price_max",
)

#: Numericas construidas. ``price_pos`` es la unica feature de contexto de
#: filtro que no es constante: las tres condiciones de coherencia del filtro se
#: cumplen en el 100% de las filas (seccion 0.5 del plan), asi que "el precio
#: cae dentro del rango" no informa nada, pero *donde* cae si (r = 0.105 dentro
#: del nivel ALTO contra 0.039 marginal).
NUMERICAS_DERIVADAS: tuple[str, ...] = (
    "price_pos",
    "price_per_oz",
    "dim_largo",
    "dim_ancho",
    "dim_alto",
    "dim_volumen",
)

#: Categoricas admitidas. Todas de cardinalidad baja o media (3 a 27 valores),
#: asi que no hace falta bucket ``[RARE]``.
CATEGORICAS: tuple[str, ...] = (
    "category",
    "brand",
    "storage_type",
    "country_of_origin",
    "unit_of_measure",
    "package_size",
    "allergens",
)

#: Categorica opcional derivada del texto: el marcador de reputacion del titulo.
#: No es una columna del CSV sino una extraccion mecanica. La usa el baseline
#: fuerte de la Fase 5 (GBDT sufijo + tabulares); ninguna corrida del
#: Transformer la recibe, porque el encoder la lee del texto.
CATEGORICAS_OPCIONALES: tuple[str, ...] = ("sufijo",)

#: Campos de texto. El orden importa: ``text_a`` va antes del primer ``[SEP]``.
TEXTO: tuple[str, ...] = ("title", "description")

#: Columnas admitidas por la auditoria de la Fase 1 que **no** entran al
#: pipeline, con el motivo escrito. El criterio de aceptacion de la fase exige
#: que no quede ninguna admitida afuera sin justificacion, y hay un test que
#: compara este diccionario contra ``FEATURES_ADMITIDAS``.
COLUMNAS_DESCARTADAS: dict[str, str] = {
    "ingredients": (
        "Redundante con category. Normalizando cada lista a su conjunto ordenado "
        "quedan 12 conjuntos para 12 categorias, en correspondencia biunivoca "
        "(Produce <-> {Whole produce}, Seafood <-> {Salt, Seafood}, ...). Las 190 "
        "combinaciones crudas son permutaciones del mismo conjunto: Bakery sola "
        "tiene 120. Incluirla agrega 190 niveles de ruido de orden sobre una "
        "senal que category ya trae limpia."
    ),
    "timestamp": (
        "Sin efecto medible. El EDA (fig_08_temporal_vs_ruido) muestra que la "
        "variacion de la tasa por hora, dia de semana y mes dentro del nivel ALTO "
        "esta en el orden del ruido de muestreo (razon observado/azar ~ 1). Las "
        "features ciclicas seno/coseno estan implementadas en "
        "add_temporal_features y se activan con use_temporal=True, para que el "
        "descarte sea reversible y auditable."
    ),
}

#: Categoria propia para ``allergens`` nulo. El 44.55% de las filas no declara
#: alergeno y su tasa de compra es 0.1385, entre Soy (0.134) y Milk (0.146): el
#: nulo es informativo. Imputar con la moda lo mezclaria con Wheat (0.155).
ALLERGENS_NONE = "NONE"

#: Codigo 0 de toda categorica: valor no visto en train. Reservarlo es lo que
#: permite transformar una particion nueva sin recalcular el vocabulario.
UNK = "[UNK]"

#: Formato de ``dimensions_in``: 3.3 x 4.0 x 4.1 seguido del simbolo de pulgada.
DIMENSIONS_PATTERN = re.compile(r"^\s*([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)")

#: La descripcion tiene dos oraciones estructurales (que producto es y bajo que
#: categoria/almacenamiento esta listado) y, en 9.541 de las 10.000 filas, una
#: tercera que codifica el nivel de reputacion igual que el sufijo del titulo.
SEPARADOR_ORACION = re.compile(r"(?<=\.)\s+")

#: Segunda oracion, siempre presente: "Listed under frozen and intended for
#: frozen storage.". Sirve para distinguir las 459 descripciones que **no**
#: traen oracion de reputacion (todas de nivel CERO) de las que si: en esas 459
#: la ultima oracion es esta, y borrarla no seria la ablacion del marcador sino
#: la perdida de category y storage_type dentro del texto.
ORACION_ESTRUCTURAL = re.compile(r"^Listed under .* storage\.$")


# --------------------------------------------------------------------------- #
# Derivaciones sobre el dataframe crudo
# --------------------------------------------------------------------------- #

def parse_dimensions(dimensions: pd.Series) -> pd.DataFrame:
    """Largo, ancho y alto a partir de ``dimensions_in``.

    El formato es fijo en el 100% de las filas. Si alguna dejara de matchear, el
    resultado queda en NaN y el escalador lo hace explotar en vez de que se
    impute en silencio.
    """
    partes = dimensions.str.extract(DIMENSIONS_PATTERN).astype(float)
    partes.columns = ["dim_largo", "dim_ancho", "dim_alto"]
    return partes


def extract_final_sentence(descriptions: pd.Series) -> pd.Series:
    """Ultima oracion de la descripcion, sea cual sea.

    Hay 36 valores distintos y ninguno aparece en mas de un nivel de reputacion:
    19 son oraciones de reputacion y las otras 17 son la estructural, que queda
    ultima en las 459 filas sin oracion de reputacion (todas de nivel CERO).
    """
    return descriptions.map(lambda t: SEPARADOR_ORACION.split(t.strip())[-1].strip())


def extract_reputation_sentence(descriptions: pd.Series) -> pd.Series:
    """La oracion de reputacion, o cadena vacia si la descripcion no la trae.

    Es la segunda codificacion del nivel latente, independiente del sufijo del
    titulo: cada sufijo se reparte entre las oraciones de su nivel a ~27% cada
    una, no hay correspondencia uno a uno.
    """
    final = extract_final_sentence(descriptions)
    return final.where(~final.str.match(ORACION_ESTRUCTURAL), "")


def strip_reputation_sentence(descriptions: pd.Series) -> pd.Series:
    """Descripcion sin la oracion de reputacion. Insumo de la ablacion.

    Solo saca la ultima oracion cuando **es** de reputacion. Borrar a ciegas la
    ultima oracion mutilaria las 459 descripciones que no la tienen, y esa
    diferencia sistematica entre las dos ramas de la ablacion se confundiria con
    el efecto que se quiere medir.
    """
    def _sacar(texto: str) -> str:
        oraciones = SEPARADOR_ORACION.split(texto.strip())
        if len(oraciones) > 1 and not ORACION_ESTRUCTURAL.match(oraciones[-1].strip()):
            oraciones = oraciones[:-1]
        return " ".join(oraciones).strip()

    return descriptions.map(_sacar)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hora, dia de semana y mes con codificacion ciclica seno/coseno.

    Descartadas por defecto (ver ``COLUMNAS_DESCARTADAS['timestamp']``). Se
    dejan implementadas para que la decision sea reversible: activarlas es
    ``FeaturePipeline(use_temporal=True)`` y volver a medir.
    """
    ts = df["timestamp"].dt
    ciclos = {"hora": (ts.hour, 24), "dow": (ts.dayofweek, 7), "mes": (ts.month - 1, 12)}
    salida = {}
    for nombre, (valores, periodo) in ciclos.items():
        angulo = 2 * np.pi * valores.to_numpy() / periodo
        salida[f"{nombre}_sin"] = np.sin(angulo)
        salida[f"{nombre}_cos"] = np.cos(angulo)
    return pd.DataFrame(salida, index=df.index)


def add_derived_columns(df: pd.DataFrame, use_temporal: bool = False) -> pd.DataFrame:
    """Copia del dataframe con las columnas derivadas agregadas.

    Todas las derivaciones son deterministas y fila a fila: no miran el target
    ni ningun estadistico del conjunto, asi que pueden calcularse antes de
    partir en train/valid sin fugar nada. Lo que si depende de train (escalado y
    vocabulario de las categoricas) vive en ``FeaturePipeline.fit``.
    """
    out = df.copy()

    rango_filtro = out["filter_price_max"] - out["filter_price_min"]
    out["price_pos"] = (out["price"] - out["filter_price_min"]) / rango_filtro
    out["price_per_oz"] = out["price"] / out["net_weight_oz"]

    dims = parse_dimensions(out["dimensions_in"])
    for col in dims.columns:
        out[col] = dims[col]
    out["dim_volumen"] = dims["dim_largo"] * dims["dim_ancho"] * dims["dim_alto"]

    out["sufijo"] = extract_suffix(out["title"])
    out["allergens"] = out["allergens"].fillna(ALLERGENS_NONE)

    if use_temporal:
        temporales = add_temporal_features(out)
        for col in temporales.columns:
            out[col] = temporales[col]

    return out


def build_text(
    df: pd.DataFrame,
    keep_suffix: bool = True,
    keep_reputation_sentence: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Los dos segmentos de texto: titulo y descripcion.

    Se devuelven separados, no concatenados, porque el tokenizador de la Fase 6
    los arma como ``[CLS] titulo [SEP] descripcion [SEP]`` con su propio
    template. ``Transformed.text`` los une con ``" [SEP] "`` para los baselines
    de TF-IDF, que no tienen template.

    Las dos banderas son la ablacion central de la Fase 9, y van separadas a
    proposito: el sufijo del titulo y la ultima oracion de la descripcion son
    **dos codificaciones independientes del mismo nivel latente**, asi que
    borrar solo el sufijo no borra la senal. La ablacion honesta de "sin
    marcador de reputacion" apaga las dos.
    """
    titulo = df["title"] if keep_suffix else strip_suffix(df["title"])
    desc = (
        df["description"]
        if keep_reputation_sentence
        else strip_reputation_sentence(df["description"])
    )
    return titulo.reset_index(drop=True), desc.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Resultado de la transformacion
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Transformed:
    """Las representaciones de una particion, todas alineadas por fila."""

    num: np.ndarray          # (n, d_num)  float32, escaladas
    cat: np.ndarray          # (n, d_cat)  int64, codigos para nn.Embedding
    onehot: np.ndarray       # (n, d_oh)   float32, para los baselines
    text_a: np.ndarray       # (n,)        titulo
    text_b: np.ndarray       # (n,)        descripcion
    y: np.ndarray | None     # (n,)        int64, None si el df no trae target
    nombres_num: tuple[str, ...]
    nombres_cat: tuple[str, ...]
    nombres_onehot: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.num)

    @property
    def text(self) -> np.ndarray:
        """Titulo y descripcion en un solo string, para TF-IDF."""
        return np.array(
            [f"{a} [SEP] {b}" for a, b in zip(self.text_a, self.text_b)], dtype=object
        )

    @property
    def tabular(self) -> np.ndarray:
        """Numericas escaladas + categoricas en one-hot, listo para sklearn."""
        return np.hstack([self.num, self.onehot]).astype(np.float32)

    @property
    def nombres_tabular(self) -> tuple[str, ...]:
        return self.nombres_num + self.nombres_onehot


# --------------------------------------------------------------------------- #
# El pipeline
# --------------------------------------------------------------------------- #

@dataclass
class FeaturePipeline:
    """Preprocesamiento ajustado exclusivamente con train.

    Parametros
    ----------
    numericas, categoricas
        Columnas a usar. Los defaults son las admitidas por la Fase 1 menos los
        descartes documentados en ``COLUMNAS_DESCARTADAS``. Las ablaciones de la
        Fase 9 ("sin allergens", "sin category") se expresan sacando la columna
        de esta lista, no editando el modulo.
    scaler
        ``"standard"`` o ``"quantile"``. Las cinco numericas base tienen
        asimetria moderada; ``price_per_oz`` y ``dim_volumen`` tienen cola larga.
        Queda elegible y se decide con evidencia en la Fase 5.
    keep_suffix, keep_reputation_sentence
        Ver ``build_text``.
    use_temporal
        Agrega las seis features ciclicas de ``timestamp``. Descartadas por
        defecto porque el EDA no les encontro efecto.
    """

    numericas: Sequence[str] = NUMERICAS_BASE + NUMERICAS_DERIVADAS
    categoricas: Sequence[str] = CATEGORICAS
    scaler: str = "standard"
    keep_suffix: bool = True
    keep_reputation_sentence: bool = True
    use_temporal: bool = False

    _scaler: StandardScaler | QuantileTransformer | None = field(
        default=None, init=False, repr=False
    )
    _vocabularios: dict[str, dict[str, int]] = field(
        default_factory=dict, init=False, repr=False
    )
    _nombres_onehot: tuple[str, ...] = field(default=(), init=False, repr=False)
    _n_fits: int = field(default=0, init=False, repr=False)

    # -- ajuste ------------------------------------------------------------- #

    def fit(self, df_train: pd.DataFrame) -> "FeaturePipeline":
        """Ajusta escalador y vocabularios con **train y nada mas**.

        Falla si el objeto ya fue ajustado. Reajustar el mismo pipeline sobre
        otra particion es el camino corto a que el escalador termine viendo el
        test; para la CV se construye uno nuevo por fold, que ademas es lo unico
        correcto.
        """
        if self._n_fits:
            raise RuntimeError(
                "Este pipeline ya fue ajustado. Construir uno nuevo por fold: "
                "reajustar el mismo objeto es como termina el test adentro del "
                "escalador."
            )

        derivado = add_derived_columns(df_train, use_temporal=self.use_temporal)
        columnas_numericas = self._columnas_numericas()

        self._scaler = self._nuevo_scaler()
        self._scaler.fit(derivado[list(columnas_numericas)].to_numpy(dtype=np.float64))

        nombres_onehot: list[str] = []
        for col in self.categoricas:
            valores = sorted(derivado[col].astype(str).unique())
            # El codigo 0 queda reservado para [UNK]: un valor que aparezca solo
            # en valid o test no rompe el transform ni corre los demas codigos.
            self._vocabularios[col] = {UNK: 0, **{v: i + 1 for i, v in enumerate(valores)}}
            nombres_onehot.extend(f"{col}={v}" for v in [UNK, *valores])
        self._nombres_onehot = tuple(nombres_onehot)

        self._n_fits = 1
        return self

    # -- transformacion ----------------------------------------------------- #

    def transform(self, df: pd.DataFrame) -> Transformed:
        """Aplica el ajuste de train a cualquier particion."""
        if not self._n_fits:
            raise RuntimeError("Pipeline sin ajustar: llamar a fit(train) primero.")

        derivado = add_derived_columns(df, use_temporal=self.use_temporal)
        columnas_numericas = self._columnas_numericas()

        num = self._scaler.transform(
            derivado[list(columnas_numericas)].to_numpy(dtype=np.float64)
        ).astype(np.float32)

        codigos = np.zeros((len(derivado), len(self.categoricas)), dtype=np.int64)
        for j, col in enumerate(self.categoricas):
            vocab = self._vocabularios[col]
            codigos[:, j] = (
                derivado[col].astype(str).map(vocab).fillna(0).to_numpy(dtype=np.int64)
            )

        text_a, text_b = build_text(
            derivado,
            keep_suffix=self.keep_suffix,
            keep_reputation_sentence=self.keep_reputation_sentence,
        )

        y = derivado[TARGET].astype(int).to_numpy() if TARGET in derivado.columns else None

        return Transformed(
            num=num,
            cat=codigos,
            onehot=self._onehot(codigos),
            text_a=text_a.to_numpy(dtype=object),
            text_b=text_b.to_numpy(dtype=object),
            y=y,
            nombres_num=tuple(columnas_numericas),
            nombres_cat=tuple(self.categoricas),
            nombres_onehot=self._nombres_onehot,
        )

    # -- metadatos ---------------------------------------------------------- #

    @property
    def cardinalidades(self) -> tuple[int, ...]:
        """Tamano de cada tabla de ``nn.Embedding``: valores de train + [UNK]."""
        return tuple(len(self._vocabularios[col]) for col in self.categoricas)

    @property
    def esta_ajustado(self) -> bool:
        return bool(self._n_fits)

    def resumen(self) -> pd.DataFrame:
        """Una fila por columna categorica: cardinalidad y ejemplos."""
        return pd.DataFrame(
            [
                {
                    "columna": col,
                    "niveles (con [UNK])": len(self._vocabularios[col]),
                    "ejemplos": ", ".join(list(self._vocabularios[col])[1:4]),
                }
                for col in self.categoricas
            ]
        )

    # -- internos ----------------------------------------------------------- #

    def _columnas_numericas(self) -> tuple[str, ...]:
        temporales = ("hora_sin", "hora_cos", "dow_sin", "dow_cos", "mes_sin", "mes_cos")
        return tuple(self.numericas) + (temporales if self.use_temporal else ())

    def _nuevo_scaler(self):
        if self.scaler == "standard":
            return StandardScaler()
        if self.scaler == "quantile":
            # n_quantiles acotado por el tamano de train (6.323 filas en el split
            # de desarrollo, ~6.327 por fold de la CV).
            return QuantileTransformer(
                n_quantiles=1000,
                output_distribution="normal",
                random_state=42,
                subsample=None,
            )
        raise ValueError(f"scaler desconocido: {self.scaler!r} (usar 'standard' o 'quantile')")

    def _onehot(self, codigos: np.ndarray) -> np.ndarray:
        """One-hot derivado de los codigos: una sola fuente de verdad.

        La columna ``[UNK]`` de cada categorica queda en cero sobre train (por
        construccion no hay valores no vistos ahi) y se mantiene igual, para que
        la matriz tenga exactamente las mismas columnas en las tres particiones.
        """
        bloques = []
        for j, col in enumerate(self.categoricas):
            k = len(self._vocabularios[col])
            bloque = np.zeros((len(codigos), k), dtype=np.float32)
            bloque[np.arange(len(codigos)), codigos[:, j]] = 1.0
            bloques.append(bloque)
        if not bloques:
            return np.zeros((len(codigos), 0), dtype=np.float32)
        return np.hstack(bloques)


# --------------------------------------------------------------------------- #
# Informe
# --------------------------------------------------------------------------- #

def cobertura_de_admitidas() -> pd.DataFrame:
    """Cada columna admitida por la Fase 1, con el uso que le da la Fase 4."""
    usadas = {
        **{c: "numerica (escalada)" for c in NUMERICAS_BASE},
        **{c: "categorica (one-hot / embedding)" for c in CATEGORICAS},
        **{c: "texto (encoder)" for c in TEXTO},
        "dimensions_in": "derivada -> dim_largo, dim_ancho, dim_alto, dim_volumen",
    }
    filas = []
    for col in FEATURES_ADMITIDAS:
        if col in usadas:
            filas.append({"columna": col, "uso": usadas[col], "motivo": ""})
        else:
            filas.append(
                {"columna": col, "uso": "DESCARTADA", "motivo": COLUMNAS_DESCARTADAS[col]}
            )
    return pd.DataFrame(filas)


def _main() -> int:
    from src.data.splits import load_splits

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 95)

    df = load_raw()
    splits = load_splits()

    pipe = FeaturePipeline().fit(df.iloc[splits.train])
    Xtr = pipe.transform(df.iloc[splits.train])
    Xva = pipe.transform(df.iloc[splits.valid])

    print("=" * 110)
    print("COBERTURA DE LAS COLUMNAS ADMITIDAS EN LA FASE 1")
    print("=" * 110)
    cobertura = cobertura_de_admitidas()
    print(cobertura.drop(columns=["motivo"]).to_string(index=False))
    print("\nDescartes, con su justificacion:")
    for col, motivo in COLUMNAS_DESCARTADAS.items():
        print(f"\n  {col}:")
        print(textwrap.fill(motivo, width=100, initial_indent="    ", subsequent_indent="    "))

    print("\n" + "=" * 110)
    print("CATEGORICAS (vocabulario ajustado con train)")
    print("=" * 110)
    print(pipe.resumen().to_string(index=False))
    print(f"\ncardinalidades para nn.Embedding: {pipe.cardinalidades}")

    print("\n" + "=" * 110)
    print("MATRICES")
    print("=" * 110)
    for nombre, X in [("train", Xtr), ("valid", Xva)]:
        print(
            f"{nombre:>6}: num {X.num.shape}  cat {X.cat.shape}  onehot {X.onehot.shape}  "
            f"tabular {X.tabular.shape}  prevalencia {X.y.mean():.4f}"
        )
    print("\nnumericas: " + ", ".join(Xtr.nombres_num))
    print(
        f"\nmedia/desvio de train tras escalar: {Xtr.num.mean():+.2e} / {Xtr.num.std():.4f}"
        "  (tiene que dar 0 y 1)"
    )
    print(
        f"media/desvio de valid tras escalar : {Xva.num.mean():+.4f} / {Xva.num.std():.4f}"
        "  (NO tiene por que dar 0 y 1: el escalador no vio valid)"
    )

    print("\nejemplo de texto:")
    print(f"  text_a: {Xtr.text_a[0]}")
    print(f"  text_b: {Xtr.text_b[0]}")
    sin_marcador = FeaturePipeline(keep_suffix=False, keep_reputation_sentence=False)
    Xsin = sin_marcador.fit(df.iloc[splits.train]).transform(df.iloc[splits.train])
    print("\nmismo ejemplo sin marcador de reputacion (ablacion de la Fase 9):")
    print(f"  text_a: {Xsin.text_a[0]}")
    print(f"  text_b: {Xsin.text_b[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
