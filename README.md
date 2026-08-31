# TP1 — Transformers · Predicción de Buy Through Rate

**73.69 Large Language Models — ITBA, 2026**

Predicción del *Buy Through Rate* (BTR) sobre impresiones de productos de un
supermercado online, con un encoder Transformer implementado desde cero.

El plan de trabajo completo, con las fases, los criterios de aceptación y las
decisiones de diseño justificadas, está en [plan.md](plan.md).

---

## Estado

| Fase | Contenido | Estado |
|---|---|---|
| 0 | Setup, semillas, estructura | hecho |
| 1 | Carga y auditoría de fugas | hecho |
| 2 | EDA formal (Ejercicio 1) | hecho |
| 3 | Target y partición | pendiente |
| 4–12 | Features, baselines, tokenizer, Transformer, ablaciones | pendiente |

---

## El problema

Cada fila del dataset es una **impresión**: un producto mostrado dentro de los
resultados de una búsqueda con filtros (`query_id`). El target es `bought`, y la
probabilidad predicha *es* el Buy Through Rate del producto en ese contexto.

Formulación elegida: **clasificación binaria a nivel impresión**. No se usa una
formulación *listwise* con softmax sobre la query porque hay queries con
múltiples compras (670 queries con 1 compra, 226 con 2, 53 con 3, 5 con 4): el
problema no es de elección única.

### Momento de la predicción

> El momento de la predicción es el instante previo a mostrar la página de
> resultados, cuando el sistema decide qué productos promocionar.

Esta frase es el criterio contra el cual se juzga cada columna. Sin ella,
"fuga" es una opinión. Está congelada en
[`src/data/load.py`](src/data/load.py) como `PREDICTION_MOMENT`.

### Auditoría de fugas

Las 22 columnas del CSV están auditadas una por una en `COLUMN_AUDIT`, con el
momento en que su valor queda determinado, si está disponible al momento de
predecir, el veredicto y el motivo escrito. La lista resultante,
`FEATURES_ADMITIDAS`, es la única fuente de verdad sobre qué puede entrar al
pipeline.

```bash
python -m src.data.load      # imprime la auditoría y verifica los hechos del plan
```

**Columnas excluidas (3):**

| Columna | Motivo |
|---|---|
| `cart` | **Fuga.** `P(bought = 1 \| cart = False) = 0.0000` exacto sobre 6.993 filas: agregar al carrito es condición necesaria de la compra. Además su valor es posterior al momento de la predicción, que es la razón causal por la que se excluye — independientemente de cuánta métrica aporte. |
| `filter_category` | Idéntica a `category` en el 100% de las filas. Duplica la señal sin agregar información. |
| `filter_storage_type` | Idéntica a `storage_type` en el 100% de las filas. |

`query_id` no es feature: es la **clave de agrupamiento** de la partición.
Tiene 2.012 valores únicos y codificarla solo memoriza. `bought` es el target.

Quedan **17 columnas admitidas**.

Las features de "coherencia con el filtro" (`category == filter_category`,
`storage_type == filter_storage_type`, `filter_price_min <= price <=
filter_price_max`) **no se construyen**: las tres condiciones se cumplen en el
100% de las filas, así que serían constantes. Sí es útil `price_pos`, que mide
dónde cae el precio dentro del rango del filtro y no es constante.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows
# source .venv/bin/activate            # Linux / macOS

pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

El dataset **no se versiona**. Copiar `supermarket_products.csv` a
`data/raw/` antes de correr nada.

---

## Reproducibilidad

Toda la aleatoriedad del proyecto se controla desde un único lugar,
[src/utils/seed.py](src/utils/seed.py):

- `set_seed(seed)` siembra `random`, `numpy` y `torch` (CPU y CUDA), fija
  `cudnn.deterministic = True`, `cudnn.benchmark = False` y activa
  `torch.use_deterministic_algorithms`.
- `CUBLAS_WORKSPACE_CONFIG` se fija a nivel de módulo, **antes** de importar
  torch: cuBLAS lee esa variable al inicializar su workspace y después ignora
  los cambios.
- `seed_worker` y `make_generator` se pasan a cada `DataLoader`. Sin ellos, con
  `num_workers > 0` el orden de los batches difiere entre corridas.
- `DEVICE` se define una sola vez y se importa desde ahí. No hay ninguna llamada
  a `.cuda()` dispersa en el código.

Verificación del criterio de aceptación de la Fase 0:

```bash
python scripts/check_reproducibility.py
python scripts/check_reproducibility.py --num-workers 2
```

Entrena 20 steps dos veces con la misma semilla y compara las losses bit a bit.
Además imprime un *fingerprint* de la secuencia completa, para poder comparar
entre dos invocaciones distintas del script.

Resultado en el entorno de referencia (RTX 4070, CUDA 12.8, torch 2.11.0):

```
device=cuda  torch=2.11.0+cu128  cuda=12.8
loss final   : 0.6940507888793945
fingerprint  : 22e69c6f75c5622a
OK: 20 steps identicos bit a bit en dos corridas.
```

El mismo *fingerprint* se obtuvo en corridas separadas del proceso y con
`--num-workers 2`.

---

## EDA — Ejercicio 1

[`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb). Se versiona **sin outputs**, para que
los diffs sean legibles; corre de punta a punta sin errores y escribe sus figuras en
[`report/figures/`](report/figures/), que sí están versionadas.

```bash
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
# o, interactivo:
jupyter lab notebooks/01_eda.ipynb
```

El notebook incluye un `assert` que falla si alguno de los hechos declarados en el plan
deja de reproducirse sobre el CSV.

### Hallazgos que condicionan el diseño

**La señal está en un marcador de reputación insertado en el título.** Los 20 sufijos
(`(Best Seller)`, `(Clearance Listing)`, …, incluido *sin sufijo*) se separan en tres
niveles sin solapamiento:

| Nivel | Sufijos | Filas | Tasa de compra | % de las compras |
|---|---|---|---|---|
| ALTO | `#1 Pick`, `Best Seller`, `Customer Favorite`, `Top Rated` | 1.931 | **0.6468** | 96.0% |
| MEDIO | `Highly Rated`, `Popular Choice`, `Shopper Favorite`, `Well Reviewed` | 1.973 | 0.0264 | 4.0% |
| CERO | los 12 restantes | 6.096 | **0.0000** | 0.0% |

Por eso el módulo Transformer tiene que ser un **encoder de texto**: una arquitectura
tabular pura no puede ver el marcador y tiene techo en ROC ≈ 0.58.

**La última oración de `description` codifica el mismo nivel.** Las 36 oraciones finales
particionan limpiamente en los tres niveles: ninguna aparece en más de uno. El sufijo y la
oración final son **dos codificaciones independientes del mismo nivel latente** — cada
sufijo se reparte entre las 4 oraciones de su nivel a ~27% cada una, no hay
correspondencia uno a uno. Predice que la ablación *solo título / solo descripción / ambos*
no va a mostrar diferencia.

**El texto es plantillado.** `title` contiene `brand` en el 100% de las filas;
`description` contiene `package_size` y `category` en el 100%. Lo único que el texto aporta
y las columnas tabulares no es el marcador de reputación.

**El análisis marginal esconde los efectos reales.** Condicionando al nivel ALTO
(tasa base 0.647), `allergens` va de Fish 0.291 a Milk 0.721 y `category` de Seafood 0.302
a Bakery 0.742. `price_pos` pasa de r = 0.039 marginal a 0.105 dentro del nivel. Ahí está
el margen de la fusión texto + tabular.

**No hay split temporal.** El dataset cubre 730 días, pero el rango temporal *dentro de una
misma query* tiene mediana de **488 días**. El `timestamp` no ordena las queries. Ver
`fig_03_coherencia_temporal.png`.

**No hay efecto temporal.** La variación de la tasa por hora, día de semana y mes dentro
del nivel ALTO está en el orden del ruido de muestreo (razón observado/azar ≈ 1). Las
features cíclicas de `timestamp` se descartan.

**Las impresiones de una query no compiten entre sí.** La tasa de un ítem ALTO es la misma
haya 1, 2, 3 o 4 ítems ALTO en la query (0.643 / 0.645 / 0.669 / 0.630). Anticipa que la
atención *cross-item* va a dar ganancia nula.

**Seafood y Fish/Shellfish son la misma señal.** Correspondencia perfecta en ambos
sentidos: no se puede separar el efecto "es pescado" del efecto "tiene alérgeno marino".

### Figuras

| Archivo | Qué muestra |
|---|---|
| `fig_01_numericas.png` | Distribución y asimetría de las 5 numéricas |
| `fig_02_estructura_queries.png` | Ítems por query y compras por query |
| `fig_03_coherencia_temporal.png` | **Justifica descartar el split temporal (D4)** |
| `fig_04_tasa_por_sufijo.png` | **Tasa de compra por sufijo — el hallazgo central** |
| `fig_05_longitud_texto.png` | Longitud del texto, para dimensionar `max_len` |
| `fig_06_bivariado_marginal.png` | Tasa marginal por categórica |
| `fig_07_condicionado_alto.png` | `allergens` y `category` dentro del nivel ALTO |
| `fig_08_temporal_vs_ruido.png` | Efecto horario contra la banda de ruido |

---

## Tests

```bash
pytest                      # todo
pytest -m "not data"        # solo lo que no necesita el CSV
```

Los criterios de aceptación de cada fase están escritos como tests, no como
prosa: que la auditoría cubra las 22 columnas, que `cart` esté excluida, que
`query_id` sea grupo y no feature, que ninguna columna posterior al momento de
la predicción entre a `FEATURES_ADMITIDAS`, y que los hechos de la sección 0
del plan se reproduzcan sobre el CSV.

Los tests marcados con `@pytest.mark.data` necesitan
`data/raw/supermarket_products.csv` y se saltean solos si no está.

---

## Estructura del repositorio

```
LLM-TP1/
├── README.md
├── requirements.txt            # versiones exactas + versión de CUDA
├── plan.md                     # plan de trabajo por fases
├── configs/
│   ├── base.yaml               # configuración de referencia
│   └── ablations/              # una config por fila de la tabla de ablación
├── data/                       # gitignored por completo
│   ├── raw/                    # supermarket_products.csv
│   └── splits/                 # índices congelados (.npy)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_resultados.ipynb
├── scripts/
│   └── check_reproducibility.py
├── src/
│   ├── data/                   # load, features, splits, dataset
│   ├── tokenizer/              # BPE entrenado sobre el corpus del TP
│   ├── model/                  # attention, encoder, heads, model
│   ├── train.py
│   ├── evaluate.py
│   └── utils/seed.py
├── tests/
├── experiments/
│   └── results.csv             # una fila por corrida
└── report/
```

`src/` separado de `notebooks/` para que las ablaciones se corran por script con
configuración declarativa, en lugar de editar celdas a mano.

---

## Restricciones asumidas

- **Ningún peso pre-entrenado**, en ninguna corrida: ni BERT, ni MiniLM, ni
  GloVe, ni word2vec. Los embeddings de token, el positional encoding y los
  pesos de atención se aprenden desde inicialización aleatoria. La biblioteca
  `tokenizers` se usa solo para *entrenar* un BPE sobre el corpus del TP.
- La atención se implementa a mano. `nn.MultiheadAttention` se usa únicamente
  como oráculo en los tests de equivalencia numérica.
