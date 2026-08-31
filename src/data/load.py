"""Carga del dataset y auditoria de fugas (Fase 1).

Este modulo es la unica puerta de entrada al CSV. Congela dos cosas:

* ``PREDICTION_MOMENT``: el instante contra el cual se juzga si una columna es
  utilizable. Sin un momento definido por escrito, "fuga" es una opinion.
* ``COLUMN_AUDIT``: una fila por cada una de las 22 columnas del archivo, con
  el momento en que su valor queda determinado y el veredicto.

El leakage no se detecta con validacion cruzada: si una columna posterior al
momento de la prediccion entra al pipeline, las tres particiones quedan
igualmente contaminadas y las curvas se ven sanas. La unica defensa es esta
tabla, escrita antes de entrenar nada.

Uso:

    python -m src.data.load          # imprime la auditoria y verifica la seccion 0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "supermarket_products.csv"

TARGET = "bought"
GROUP = "query_id"

PREDICTION_MOMENT = (
    "El momento de la prediccion es el instante previo a mostrar la pagina de "
    "resultados, cuando el sistema decide que productos promocionar."
)

#: Regex del marcador de reputacion al final del titulo: "... (Best Seller)".
SUFFIX_PATTERN = r"\(([^)]*)\)$"

#: Marca de las filas sin marcador. Son 511 sobre 10.000, y su tasa de compra es
#: 0: "sin sufijo" es uno mas de los valores del nivel CERO, no un faltante.
SUFFIX_NONE = "[SIN SUFIJO]"


@dataclass(frozen=True)
class ColumnAudit:
    """Una fila de la tabla de auditoria de la Fase 1."""

    columna: str
    momento: str
    disponible: bool
    veredicto: str  # ADMITIDA | EXCLUIDA | TARGET | GRUPO
    rol: str  # texto | numerica | categorica | derivada | -
    motivo: str


_ANTES = "Antes de la busqueda (atributo del catalogo)"
_QUERY = "Al construirse la query (filtros elegidos por el usuario)"
_IMPRESION = "Al generarse la impresion"
_POSTERIOR = "DESPUES de mostrar los resultados (conducta del usuario)"

#: Auditoria completa. Las 22 columnas del CSV, sin excepcion.
COLUMN_AUDIT: tuple[ColumnAudit, ...] = (
    ColumnAudit(
        "title", _ANTES, True, "ADMITIDA", "texto",
        "Entrada principal del encoder. Contiene marca, tamano y el marcador de "
        "reputacion entre parentesis, que es donde esta la senal.",
    ),
    ColumnAudit(
        "description", _ANTES, True, "ADMITIDA", "texto",
        "Texto plantillado; su ultima oracion codifica el mismo nivel de "
        "reputacion que el sufijo del titulo. Redundante pero disponible.",
    ),
    ColumnAudit(
        "price", _ANTES, True, "ADMITIDA", "numerica",
        "Atributo del producto. Correlacion marginal con bought ~0, pero entra "
        "en price_pos, que si tiene efecto dentro del nivel ALTO.",
    ),
    ColumnAudit(
        "category", _ANTES, True, "ADMITIDA", "categorica",
        "12 valores. Modula la compra dentro del nivel ALTO (Seafood 0.30 vs "
        "Bakery 0.74).",
    ),
    ColumnAudit(
        "timestamp", _IMPRESION, True, "ADMITIDA", "derivada",
        "Conocido al servir la pagina. Se usa solo para features ciclicas "
        "(hora, dia, mes) y unicamente si el EDA muestra efecto. NO ordena las "
        "queries: el rango intra-query tiene mediana de 488 dias.",
    ),
    ColumnAudit(
        "query_id", _QUERY, True, "GRUPO", "-",
        "Clave de agrupamiento para la particion. NO es feature: 2.012 valores "
        "unicos, codificarla solo memoriza.",
    ),
    ColumnAudit(
        "filter_category", _QUERY, True, "EXCLUIDA", "-",
        "Identica a category en el 100% de las filas. Incluir ambas duplica la "
        "senal sin agregar informacion.",
    ),
    ColumnAudit(
        "filter_price_min", _QUERY, True, "ADMITIDA", "numerica",
        "Contexto de la query. Solo util combinada: price_pos = (price - min) / "
        "(max - min). La condicion min <= price <= max se cumple al 100%, asi "
        "que la feature de coherencia seria constante.",
    ),
    ColumnAudit(
        "filter_price_max", _QUERY, True, "ADMITIDA", "numerica",
        "Idem filter_price_min.",
    ),
    ColumnAudit(
        "filter_storage_type", _QUERY, True, "EXCLUIDA", "-",
        "Identica a storage_type en el 100% de las filas.",
    ),
    ColumnAudit(
        "cart", _POSTERIOR, False, "EXCLUIDA", "-",
        "FUGA CONFIRMADA. P(bought=1 | cart=False) = 0.0000 exacto: agregar al "
        "carrito es condicion necesaria de la compra. Ademas es posterior al "
        "momento de la prediccion, que es la razon causal por la que se "
        "excluye, independientemente de cuanta metrica aporte.",
    ),
    ColumnAudit(
        "bought", _POSTERIOR, False, "TARGET", "-",
        "Variable a predecir. y = bought.astype(int); la probabilidad predicha "
        "es el BTR.",
    ),
    ColumnAudit(
        "brand", _ANTES, True, "ADMITIDA", "categorica",
        "15 valores. Aparece tambien dentro de title en el 100% de las filas.",
    ),
    ColumnAudit(
        "package_size", _ANTES, True, "ADMITIDA", "categorica",
        "27 valores. Aparece dentro de title y description.",
    ),
    ColumnAudit(
        "unit_of_measure", _ANTES, True, "ADMITIDA", "categorica",
        "5 valores. Redundante con package_size, del que es el sufijo; se "
        "mantiene por ser barata y porque desambigua el peso.",
    ),
    ColumnAudit(
        "net_weight_oz", _ANTES, True, "ADMITIDA", "numerica",
        "Peso neto. Entra ademas en price_per_oz.",
    ),
    ColumnAudit(
        "dimensions_in", _ANTES, True, "ADMITIDA", "derivada",
        "Formato '3.3 x 4.0 x 4.1\"'. Se parsea a largo, ancho, alto y volumen.",
    ),
    ColumnAudit(
        "storage_type", _ANTES, True, "ADMITIDA", "categorica",
        "3 valores. Se queda esta y se descarta filter_storage_type, que es su "
        "copia exacta.",
    ),
    ColumnAudit(
        "ingredients", _ANTES, True, "ADMITIDA", "categorica",
        "190 combinaciones. No es multi-etiqueta real en este dataset; se trata "
        "como categorica de cardinalidad media. Se descarta si el EDA no le "
        "encuentra efecto.",
    ),
    ColumnAudit(
        "allergens", _ANTES, True, "ADMITIDA", "categorica",
        "7 valores + 44.55% de nulos. El nulo es informativo (tasa 0.1385): se "
        "codifica como categoria propia, no se imputa. Es la variable con mayor "
        "efecto dentro del nivel ALTO (Fish 0.29 vs Milk 0.72).",
    ),
    ColumnAudit(
        "nutrition_score", _ANTES, True, "ADMITIDA", "numerica",
        "Atributo del producto.",
    ),
    ColumnAudit(
        "country_of_origin", _ANTES, True, "ADMITIDA", "categorica",
        "10 valores. Se admite y se mide su efecto en el EDA; es candidata a "
        "descarte si resulta plana dentro del nivel ALTO.",
    ),
)

#: Lista congelada de columnas que pueden entrar al pipeline de features.
FEATURES_ADMITIDAS: tuple[str, ...] = tuple(
    entry.columna for entry in COLUMN_AUDIT if entry.veredicto == "ADMITIDA"
)

#: Columnas explicitamente vetadas, con su motivo a mano para los asserts.
FEATURES_EXCLUIDAS: dict[str, str] = {
    entry.columna: entry.motivo for entry in COLUMN_AUDIT if entry.veredicto == "EXCLUIDA"
}

#: Orden canonico de las columnas del CSV. Se valida en cada carga.
EXPECTED_COLUMNS: tuple[str, ...] = tuple(entry.columna for entry in COLUMN_AUDIT)

#: Numeros de la seccion 0 del plan. La carga los re-mide y compara.
EXPECTED_FACTS: dict[str, float] = {
    "n_filas": 10_000,
    "n_columnas": 22,
    "duplicados_exactos": 0,
    "duplicados_query_title": 1,
    "queries_unicas": 2_012,
    "prevalencia_bought": 0.1301,
    "prevalencia_cart": 0.3007,
    "p_bought_dado_cart_true": 0.4327,
    "p_bought_dado_cart_false": 0.0000,
    "nulos_allergens_pct": 44.55,
}


def load_raw(path: str | Path = CSV_PATH) -> pd.DataFrame:
    """Carga el CSV crudo con ``timestamp`` parseado a datetime UTC.

    No filtra ni transforma nada mas: la seleccion de columnas es
    responsabilidad de ``src.data.features`` y se hace contra
    ``FEATURES_ADMITIDAS``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en {path}. El CSV no se versiona "
            f"(esta en .gitignore); copiarlo a data/raw/ antes de correr nada."
        )

    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    extra = set(df.columns) - set(EXPECTED_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"El CSV no coincide con la auditoria. Faltan: {sorted(missing)}. "
            f"Sobran: {sorted(extra)}. Actualizar COLUMN_AUDIT antes de seguir."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    for col in ("cart", "bought"):
        df[col] = df[col].astype(bool)
    return df.loc[:, list(EXPECTED_COLUMNS)]


def audit_table() -> pd.DataFrame:
    """La tabla de auditoria de la Fase 1, como DataFrame para el notebook."""
    return pd.DataFrame(
        [
            {
                "columna": e.columna,
                "momento en que queda determinado": e.momento,
                "disponible al predecir": "si" if e.disponible else "NO",
                "veredicto": e.veredicto,
                "rol": e.rol,
                "motivo": e.motivo,
            }
            for e in COLUMN_AUDIT
        ]
    )


def extract_suffix(titles: pd.Series) -> pd.Series:
    """Marcador de reputacion al final del titulo, sin parentesis.

    Operacion puramente mecanica sobre el texto: no mira el target. El
    agrupamiento de los sufijos en niveles ALTO/MEDIO/CERO se deriva del target
    y por eso vive en el notebook de EDA, no aca.
    """
    return titles.str.extract(SUFFIX_PATTERN, expand=False).fillna(SUFFIX_NONE)


def strip_suffix(titles: pd.Series) -> pd.Series:
    """Titulo sin el marcador final. Insumo de la ablacion del sufijo."""
    return titles.str.replace(SUFFIX_PATTERN, "", regex=True).str.strip()


def structural_facts(df: pd.DataFrame) -> dict[str, float]:
    """Re-mide los hechos de la seccion 0 del plan sobre el dataframe cargado."""
    n_rows, n_cols = df.shape
    cart = df["cart"]
    bought = df[TARGET]
    return {
        "n_filas": n_rows,
        "n_columnas": n_cols,
        "duplicados_exactos": int(df.duplicated().sum()),
        "duplicados_query_title": int(df.duplicated(subset=[GROUP, "title"]).sum()),
        "queries_unicas": int(df[GROUP].nunique()),
        "prevalencia_bought": float(bought.mean()),
        "prevalencia_cart": float(cart.mean()),
        "p_bought_dado_cart_true": float(bought[cart].mean()),
        "p_bought_dado_cart_false": float(bought[~cart].mean()),
        "nulos_allergens_pct": float(df["allergens"].isna().mean() * 100),
    }


def verify_expected_facts(df: pd.DataFrame, tol: float = 5e-4) -> pd.DataFrame:
    """Compara lo medido contra los numeros declarados en el plan.

    Criterio de aceptacion de la Fase 2: esta tabla no debe tener ningun ``ok``
    en falso. Si alguno falla, el CSV cambio y hay que revisar el plan entero,
    no ajustar la tolerancia.
    """
    measured = structural_facts(df)
    rows = []
    for key, expected in EXPECTED_FACTS.items():
        value = measured[key]
        delta = abs(value - expected)
        # Los conteos se comparan exacto; las proporciones, con tolerancia de
        # redondeo al cuarto decimal (los porcentajes, al segundo).
        limit = 0.0 if float(expected).is_integer() and expected > 1 else tol
        if key.endswith("_pct"):
            limit = 5e-3
        rows.append(
            {"hecho": key, "plan": expected, "medido": round(value, 6), "delta": round(delta, 6), "ok": delta <= limit}
        )
    return pd.DataFrame(rows)


def leakage_contingency(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de contingencia ``cart`` x ``bought`` con las condicionales.

    El resultado esperado es ``P(bought=1 | cart=False) == 0`` exacto, es decir
    una celda con cero absoluto. Eso no es una correlacion alta: es una
    implicacion logica, y por eso ``cart`` se excluye.
    """
    table = pd.crosstab(df["cart"], df[TARGET], margins=True, margins_name="total")
    table["P(bought=1 | cart)"] = pd.crosstab(
        df["cart"], df[TARGET], normalize="index", margins=True, margins_name="total"
    )[True]
    return table


def filter_coherence(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica las tres condiciones de la seccion 0.5.

    Si las tres dan 100%, las features de "coherencia con el filtro" serian
    columnas constantes y no hay que construirlas.
    """
    checks = {
        "category == filter_category": df["category"] == df["filter_category"],
        "storage_type == filter_storage_type": df["storage_type"] == df["filter_storage_type"],
        "filter_price_min <= price <= filter_price_max": (
            (df["filter_price_min"] <= df["price"]) & (df["price"] <= df["filter_price_max"])
        ),
    }
    return pd.DataFrame(
        [
            {
                "condicion": name,
                "se cumple en": f"{mask.mean() * 100:.2f}%",
                "filas que fallan": int((~mask).sum()),
                "feature constante": bool(mask.all()),
            }
            for name, mask in checks.items()
        ]
    )


def column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Perfil por columna: dtype, nulos, cardinalidad y un ejemplo."""
    return pd.DataFrame(
        [
            {
                "columna": col,
                "dtype": str(df[col].dtype),
                "nulos": int(df[col].isna().sum()),
                "nulos %": round(df[col].isna().mean() * 100, 2),
                "valores unicos": int(df[col].nunique(dropna=True)),
                "ejemplo": str(df[col].dropna().iloc[0])[:60],
            }
            for col in df.columns
        ]
    )


def _main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 80)

    df = load_raw()
    print(f"\n{PREDICTION_MOMENT}\n")

    print("=" * 100)
    print("PERFIL POR COLUMNA")
    print("=" * 100)
    print(column_profile(df).to_string(index=False))

    print("\n" + "=" * 100)
    print("VERIFICACION DE LOS HECHOS DE LA SECCION 0 DEL PLAN")
    print("=" * 100)
    facts = verify_expected_facts(df)
    print(facts.to_string(index=False))

    print("\n" + "=" * 100)
    print("CONTINGENCIA cart x bought (fuga)")
    print("=" * 100)
    print(leakage_contingency(df).to_string())

    print("\n" + "=" * 100)
    print("COHERENCIA CON LOS FILTROS (seccion 0.5)")
    print("=" * 100)
    print(filter_coherence(df).to_string(index=False))

    print("\n" + "=" * 100)
    print("AUDITORIA DE COLUMNAS")
    print("=" * 100)
    audit = audit_table()
    print(audit.drop(columns=["motivo"]).to_string(index=False))
    print(f"\nAdmitidas ({len(FEATURES_ADMITIDAS)}): {', '.join(FEATURES_ADMITIDAS)}")
    print(f"Excluidas ({len(FEATURES_EXCLUIDAS)}): {', '.join(FEATURES_EXCLUIDAS)}")
    print(f"Target: {TARGET}   Grupo: {GROUP}")

    if not facts["ok"].all():
        print("\nFALLA: hay hechos del plan que no se reproducen.")
        return 1
    print("\nOK: los hechos de la seccion 0 se reproducen sobre el CSV.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
