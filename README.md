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
| 3 | Target y partición | hecho |
| 4 | Features y preprocesamiento | hecho |
| 5 | Baselines | hecho |
| 6 | Tokenizador BPE | hecho |
| 7–12 | Transformer, entrenamiento, ablaciones, presentación | pendiente |

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

## Partición

```bash
python -m src.data.splits            # genera, guarda e informa
python -m src.data.splits --check    # verifica lo que hay en disco
```

Toda la partición está **agrupada por `query_id`**: las filas de una misma búsqueda
comparten filtros y contexto, así que van siempre juntas. Un split a nivel fila dejaría
impresiones de la misma query en train y en test.

| Partición | Filas | Queries | Prevalencia | Δ vs global |
|---|---|---|---|---|
| `dev` | 7.909 | 1.609 | 0.1292 | −0.09 pp |
| `test` | 2.091 | 403 | **0.1334** | +0.33 pp |
| ↳ `train` (dentro de dev) | 6.323 | 1.287 | 0.1298 | −0.03 pp |
| ↳ `valid` (dentro de dev) | 1.586 | 322 | 0.1267 | −0.34 pp |
| ↳ 5 folds de `GroupKFold` sobre dev | ~6.327 / ~1.582 | ~1.287 / ~322 | 0.1220 – 0.1403 | máx +1.02 pp |

- **`test` se toca una sola vez**, en la Fase 10. Todas las decisiones — selección de
  modelo y ablaciones — se toman contra la CV agrupada de 5 folds sobre `dev`. Esto protege
  contra el *leakage del experimentador*: con 1.301 positivos totales, elegir contra un
  único split de test es sobreajustarlo.
- **`train` / `valid`** es un split fijo dentro de `dev`, para desarrollo rápido y early
  stopping mientras se construye el modelo.
- **No se estratifica** por `bought` a nivel fila: agrupando por query eso no tiene sentido.
  Se verifica a posteriori que ninguna partición se desvíe más de **±1.5 pp** de la
  prevalencia global. La desviación máxima es +1.02 pp (fold 3 valid).

La prevalencia de `test` (0.1334) es el **piso del PR-AUC** y hay que reportarla al lado de
cualquier PR-AUC.

### Los splits no se versionan, pero están congelados

Los `.npy` viven en `data/splits/`, que está gitignoreado. Lo que sí está versionado es
`EXPECTED_FINGERPRINT` en [`src/data/splits.py`](src/data/splits.py): un hash de todos los
índices, verificado por un test. Si una regeneración no lo reproduce —porque cambió el CSV,
la semilla o la versión de scikit-learn— el test falla y los resultados ya reportados dejan
de ser comparables. Ningún notebook ni script recalcula la partición: todos llaman a
`load_splits()`.

---

## Features y preprocesamiento

```bash
python -m src.data.features    # informa qué construye, qué descarta y por qué
```

Todo el preprocesamiento vive en un objeto,
[`FeaturePipeline`](src/data/features.py), que **se ajusta una sola vez y solo con
train**:

```python
pipe = FeaturePipeline().fit(df.iloc[splits.train])   # escalador y vocabularios
Xtr, Xva = pipe.transform(df.iloc[splits.train]), pipe.transform(df.iloc[splits.valid])
```

Ajustar el escalador con el dataset completo hace que la media del test participe en la
normalización de train. Es la fuga más frecuente, así que acá es imposible por
construcción: `fit` **falla si se lo llama dos veces** sobre el mismo objeto. Para la CV se
construye un pipeline nuevo por fold, que además es lo único correcto.

Del mismo ajuste salen las cuatro representaciones que usan las fases siguientes:

| Salida | Forma (train) | Para qué |
|---|---|---|
| `num` | 6.323 × 11 | numéricas escaladas — baselines y rama tabular |
| `onehot` | 6.323 × 87 | categóricas en one-hot — baselines lineales y GBDT |
| `cat` | 6.323 × 7 | códigos enteros — `nn.Embedding` (Fase 8) |
| `text_a` / `text_b` | 6.323 | título y descripción — tokenizador (Fase 6) |

**Numéricas (11).** Las 5 del CSV más `price_pos`, `price_per_oz` y las cuatro
dimensiones parseadas de `dimensions_in` (largo, ancho, alto, volumen). `price_pos` es la
única feature de contexto de filtro que no es constante: que el precio caiga dentro del
rango se cumple en el 100% de las filas, pero *dónde* cae correlaciona 0.105 con la compra
dentro del nivel ALTO.

**Categóricas (7 → 87 columnas one-hot).** Cardinalidad de 3 a 27, así que no hace falta
bucket `[RARE]`. El código 0 de cada columna queda reservado para `[UNK]`: un valor que
aparezca solo en valid o test no rompe el `transform`. El one-hot se deriva de los mismos
códigos que consume `nn.Embedding`, así que las dos ramas no pueden desincronizarse.

`allergens` nulo (44.55%) se codifica como categoría propia `NONE`, **no se imputa**: su
tasa de compra es 0.1385, entre Soy (0.134) y Milk (0.146); imputarlo con la moda lo
mezclaría con Wheat (0.155).

**Descartadas, con motivo escrito** (`COLUMNAS_DESCARTADAS`, verificado por un test contra
`FEATURES_ADMITIDAS`):

| Columna | Motivo |
|---|---|
| `ingredients` | **Redundante con `category`.** Normalizando cada lista a su conjunto ordenado quedan **12 conjuntos para 12 categorías, en correspondencia biunívoca** (Produce ↔ {Whole produce}, Seafood ↔ {Salt, Seafood}, …). Las 190 combinaciones crudas son permutaciones del mismo conjunto: Bakery sola tiene 120. |
| `timestamp` | Sin efecto medible (`fig_08`). Las cíclicas seno/coseno están implementadas y se activan con `use_temporal=True`, para que el descarte sea reversible. |

### La ablación del marcador de reputación tiene dos interruptores

El nivel está codificado **dos veces y de forma independiente**: en el sufijo del título y
en la última oración de la descripción. Por eso son dos banderas separadas
(`keep_suffix`, `keep_reputation_sentence`) y la ablación honesta de "sin marcador" apaga
las dos: borrar solo el sufijo no borra la señal.

`strip_reputation_sentence` saca la última oración **solo cuando es de reputación**. 459 de
las 10.000 descripciones no la tienen (todas de nivel CERO) y terminan en la oración
estructural `Listed under … storage.`; borrárselas las dejaría sistemáticamente más cortas
que al resto y esa diferencia se confundiría con el efecto que la ablación quiere medir.

### Dataset y DataLoader

[`src/data/dataset.py`](src/data/dataset.py). La tokenización se precomputa una sola vez
como tensores y `__getitem__` solo indexa. El batch se mueve a `DEVICE` en **un único
lugar**, el `collate_fn` que arma `make_dataloader`, que además recibe `generator` y
`worker_init_fn` para que el orden de los batches sea reproducible. La rama de texto es
opcional (`input_ids=None`): las ablaciones "solo texto" y "solo tabular" se expresan
construyendo el dataset sin una de las dos partes, no con banderas dentro del modelo.

---

## Baselines

```bash
python -m src.baselines                      # corre los 9 y escribe results.csv
python -m src.baselines --solo prior tfidf_title
```

Todos se evalúan con la **CV agrupada de 5 folds sobre `dev`** congelada en la Fase 3. El
test no se toca. Cada fold ajusta su propio `FeaturePipeline` y su propio
`TfidfVectorizer` **solo con las filas de train del fold**: ajustar el vectorizador sobre
todo el dataset filtraría el vocabulario y las frecuencias de documento del test hacia
train.

| Baseline | Features | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|---|
| `prior` | ninguna | 0.5000 | 0.1292 ± 0.0067 | 0.1125 |
| `logreg_tabular` | num(11) + onehot(87) | 0.5641 ± 0.0210 | 0.1513 ± 0.0083 | 0.1126 |
| `gbdt_tabular` | num(11) + cat(7) | 0.5975 ± 0.0101 | 0.1731 ± 0.0112 | 0.1119 |
| `tfidf_title` | TF-IDF (1,2) de `title` | 0.9568 ± 0.0051 | 0.6760 ± 0.0351 | 0.0522 |
| `tfidf_description` | TF-IDF (1,2) de `description` | 0.9591 ± 0.0057 | 0.6911 ± 0.0418 | 0.0493 |
| `tfidf_texto` | TF-IDF (1,2) de ambos | 0.9579 ± 0.0059 | 0.6851 ± 0.0497 | 0.0510 |
| `tfidf_sin_marcador` | ídem, sin sufijo ni oración de reputación | **0.5244 ± 0.0221** | **0.1375 ± 0.0127** | 0.1143 |
| **`gbdt_sufijo`** | num(11) + cat(7) + sufijo | **0.9742 ± 0.0025** | **0.8079 ± 0.0274** | 0.0417 |
| `mlp_texto_tabular` | num + onehot + SVD-100 de TF-IDF | 0.9644 ± 0.0071 | 0.7574 ± 0.0478 | 0.0489 |

Media ± desvío sobre los 5 folds. Prevalencia de `dev` = **0.1292**, que es el piso del
PR-AUC: `gbdt_sufijo` rinde 6.25× el azar.

Los tres números del plan se reproducen: solo-tabular 0.5975 contra ≈0.58 esperado,
TF-IDF de título 0.9568 contra 0.9562, `gbdt_sufijo` 0.9742 contra 0.9695. El notebook
[`02_baselines.ipynb`](notebooks/02_baselines.ipynb) tiene esa verificación como `assert`.

### Qué fijan estos números

- **El objetivo realista del Transformer es PR-AUC ≈ 0.81 / ROC ≈ 0.974.** Superarlo por
  mucho sería señal de fuga; quedarse muy por debajo, de un bug.
- **Es previsible que el GBDT gane.** La señal es un token categórico de 20 valores: no
  requiere composicionalidad, que es lo que un encoder aporta. Se presenta con el análisis
  de por qué, y vale más que un número inflado.
- **El margen está en la fusión, no en el texto solo.** `tfidf_texto` 0.685 contra
  `gbdt_sufijo` 0.808 son 12 puntos de PR-AUC que aportan `allergens` y `category`
  modulando la compra dentro del nivel ALTO.
- **`mlp_texto_tabular` es el control de arquitectura**: ya combina texto y tabular sin
  atención. Si el Transformer no le gana, lo que aporte no será el mecanismo de atención.
- **`tfidf_sin_marcador` (ROC 0.524) es la ablación central adelantada.** Sin el marcador,
  el problema es irresoluble: el modelo no aprende qué productos se compran, aprende a leer
  un marcador de reputación insertado sintéticamente.

Cada corrida queda registrada en [`experiments/results.csv`](experiments/results.csv), una
fila por baseline y fold, con ROC-AUC, PR-AUC, log loss, Brier, segundos y el hash del
commit. Las medias y los desvíos se calculan al leer, nunca se guardan agregados.

| Figura | Qué muestra |
|---|---|
| `fig_09_baselines.png` | ROC-AUC y PR-AUC por baseline, con el piso de cada métrica |
| `fig_10_curvas_baselines.png` | Curvas ROC y PR *out-of-fold* sobre `dev` |

---

## Tokenizador

```bash
python -m src.tokenizer.bpe              # entrena, mide, verifica y guarda
python -m src.tokenizer.bpe --ablacion   # tabla de tamaños de vocabulario
```

BPE entrenado **desde cero con los textos del TP** (decisión A3: la biblioteca
`tokenizers` aporta el algoritmo, no pesos). Se entrena **solo con el texto de train**:
el vocabulario BPE sale de las frecuencias del corpus, así que construirlo sobre el
dataset completo filtraría información de test hacia train.

Formato de entrada: `[CLS] título [SEP] descripción [SEP]`, con `[PAD]`=0, `[UNK]`=1,
`[CLS]`=2, `[SEP]`=3. Que `[PAD]` sea 0 hace que la máscara sea `ids != 0` y que el
embedding de padding sea la fila 0, que se inicializa en cero y se excluye del gradiente.

| Criterio de aceptación | Resultado |
|---|---|
| `max_len` **medido**, no elegido a ojo | p99 = 52 sobre train → **56** (múltiplo de 8 siguiente). 0% de ejemplos truncados en train y en valid |
| Tasa de `[UNK]` en validación < 1% | **0.0000%** |
| Round-trip salvo espaciado | exacto sobre 4.000 textos |
| Marcador de reputación estable | **19/19** marcadores se tokenizan igual en todo contexto, en 4 tokens: `( Best Seller )` |

Se usa un pre-tokenizador `Whitespace` en lugar de ByteLevel a propósito: los tokens
quedan legibles (`Seller`, no `ĠSeller`), lo que importa para los mapas de atención de la
Fase 10, y la tasa de `[UNK]` pasa a ser una métrica con sentido en vez de cero por
construcción.

### El vocabulario satura en 1.355 tokens

El corpus de train tiene solo **622 tipos de palabra**: el texto es plantillado. El BPE
llega a 1.355 tokens (alfabeto + subpalabras + las 622 palabras enteras + los 4 especiales)
y ahí **no quedan pares adyacentes que fusionar**. Verificado con `min_frequency` 1, 2 y 3:
el techo es el mismo.

Consecuencia: la grilla de ablación que propone el plan ({1.000, 2.000, 4.000, 8.000})
**degenera** — de 2.000 en adelante es literalmente el mismo tokenizador. La ablación útil
va hacia abajo:

| Vocab pedido | Vocab real | `max_len` | Tokens/ejemplo | Tokens del marcador |
|---|---|---|---|---|
| 256 | 256 | 104 | 79.3 | 7.26 |
| 512 | 512 | 80 | 60.9 | 4.89 |
| 1.000 | 1.000 | 64 | 47.7 | 4.00 |
| **2.000** | **1.355** | **56** | **44.9** | **4.00** |

Con 256 tokens el marcador se fragmenta y el modelo tiene que recomponerlo antes de poder
usarlo; con 1.000 ya entra entero. Ese es el efecto que la ablación de la Fase 9 va a medir
sobre la métrica final.

El tokenizador serializado va a `data/tokenizer/` (gitignoreado, regenerable con una línea).
Lo versionado es `EXPECTED_FINGERPRINT`, un hash del vocabulario y los merges, verificado
por un test. Para la CV **se reentrena por fold**: cuesta menos de un segundo, así que no
hace falta ni siquiera aceptar el sesgo de entrenarlo una vez sobre `dev`.

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
