# Plan de implementación — TP1 Transformers

**73.69 Large Language Models — 2026**
Predicción de *Buy Through Rate* (BTR) sobre `supermarket_products.csv`

---

## Cómo usar este documento

Cada fase tiene: **objetivo**, **entregable**, **pasos**, **justificación**, **criterio de aceptación** y **errores a evitar**. No se avanza a la fase siguiente sin cumplir el criterio de aceptación. El orden está pensado para que cada fase produzca algo verificable y para que un error se detecte cerca de donde se cometió.

Todos los números citados en la sección 1 fueron medidos sobre el CSV provisto y deben ser reproducidos por el equipo en la Fase 2 antes de darlos por válidos.

---

## 0. Hallazgos del EDA preliminar

Estos hechos condicionan todo el diseño. Están verificados sobre las 10.000 filas del archivo.

### 0.1 Estructura

| Hecho | Valor |
|---|---|
| Filas × columnas | 10.000 × 22 |
| Filas duplicadas exactas | 0 |
| Duplicados `(query_id, title)` | 1 |
| Nulos | Solo en `allergens` (44.55%) |
| Queries únicas (`query_id`) | 2.012 |
| Ítems por query | mín 1, máx 8, media 4.97 |
| Productos únicos (`brand` + título sin sufijo) | 8.578 |
| Rango de `timestamp` | 2024-07-08 a 2026-07-08 |
| Rango temporal **dentro** de una misma query | mediana 488 días, máx 728 |

### 0.2 Target

| Hecho | Valor |
|---|---|
| Prevalencia de `bought` | **0.1301** (1.301 positivos) |
| Prevalencia de `cart` | 0.3007 |
| `P(bought = 1 \| cart = True)` | 0.4327 |
| `P(bought = 1 \| cart = False)` | **0.0000** |
| Compras por query | 0 → 1.058 queries · 1 → 670 · 2 → 226 · 3 → 53 · 4 → 5 |

`cart` es condición necesaria de `bought` con certeza absoluta. Es fuga confirmada, no sospecha.

Hay queries con múltiples compras. El problema **no** es de elección única, así que no corresponde una formulación *listwise* con softmax sobre la query.

### 0.3 La señal está en el sufijo del título

Todos los títulos terminan en un marcador entre paréntesis. Hay 20 valores (incluido "sin sufijo"), que se agrupan en tres niveles perfectamente separados:

| Nivel | Sufijos | Filas | Tasa de compra |
|---|---|---|---|
| ALTO | `#1 Pick`, `Best Seller`, `Customer Favorite`, `Top Rated` | 1.931 | **0.6468** |
| MEDIO | `Highly Rated`, `Popular Choice`, `Shopper Favorite`, `Well Reviewed` | 1.973 | 0.0264 |
| CERO | los 12 restantes (`Clearance Listing`, `New Listing`, `Unrated Listing`, …) | 6.096 | **0.0000** |

El nivel ALTO concentra 1.249 de las 1.301 compras (96%).

La última oración de `description` codifica el mismo nivel de forma redundante: el mapeo sufijo → frase final es casi determinístico.

### 0.4 Qué modula la compra dentro del nivel ALTO

Dentro de las 1.931 filas de nivel ALTO (tasa base 0.647):

| Variable | Efecto medido |
|---|---|
| `allergens` | Fish 0.291 · Shellfish 0.314 · Tree nuts 0.337 · Peanuts 0.394 **vs** Wheat 0.688 · sin alérgeno 0.704 · Soy 0.705 · Milk 0.721 |
| `category` | Seafood 0.302 **vs** Bakery 0.742 · Baby 0.741 · Dairy 0.733 |
| `price_pos` (posición del precio dentro del rango del filtro) | correlación 0.105 |
| `price`, `net_weight_oz`, `nutrition_score` | correlación < 0.04, sin efecto |
| Cantidad de ítems ALTO compitiendo en la misma query | 1 → 0.643 · 2 → 0.645 · 3 → 0.670 · 4 → 0.630 |

Seafood y Fish/Shellfish están confundidos entre sí; hay que decidir si se modelan ambos o uno.

### 0.5 Features que NO existen

Verificado sobre el 100% de las filas:

- `category == filter_category` siempre.
- `storage_type == filter_storage_type` siempre.
- `filter_price_min <= price <= filter_price_max` siempre.

Las features de coherencia con el filtro son **constantes**. No hay que construirlas. Sí es útil `price_pos`, que mide *dónde* cae el precio dentro del rango y no es constante.

### 0.6 Techo de performance conocido

Baselines medidos con `GroupShuffleSplit(test_size=0.2, random_state=42)` agrupando por `query_id`. Prevalencia del test = 0.1334, que es el piso del PR-AUC.

| Modelo | ROC-AUC | PR-AUC |
|---|---|---|
| GBDT solo tabular (sin texto) | 0.5789 | 0.1645 |
| LogReg sobre título **sin** sufijo | 0.4991 | 0.1291 |
| LogReg TF-IDF solo `description` | 0.9556 | 0.6552 |
| LogReg TF-IDF `title` | 0.9562 | 0.6564 |
| LogReg TF-IDF `title` + `description` | 0.9554 | 0.6523 |
| GBDT sufijo (categórica) + tabulares | **0.9695** | **0.7750** |
| GBDT tabular + `cart` (con fuga) | 0.9197 | 0.5046 |

Lecturas obligatorias de esta tabla:

1. **El Transformer tiene que ser un encoder de texto.** Una arquitectura tabular pura (FT-Transformer, TabTransformer) tiene techo en ROC 0.58. No es una opción viable para el módulo principal.
2. El objetivo realista del modelo propio es **acercarse a ROC ≈ 0.97 / PR-AUC ≈ 0.78**. Superar eso significativamente es señal de fuga.
3. `cart` da *peor* resultado que el sufijo. Es fuga igual: la razón por la que se excluye es causal y temporal, no que la métrica sea alta.

---

## 1. Decisiones de diseño

### 1.1 Cerradas (justificadas por el EDA)

| # | Decisión | Justificación |
|---|---|---|
| D1 | Formulación: clasificación binaria a nivel impresión, `y = bought` | Es lo que habilita PR-AUC/ROC-AUC que pide el enunciado; la probabilidad predicha *es* el BTR |
| D2 | Excluir `cart` del set de features | `P(bought \| cart=False) = 0`; posterior al momento de la predicción |
| D3 | Partición con `GroupShuffleSplit` / `GroupKFold` por `query_id` | Filas de la misma query comparten filtros y contexto |
| D4 | **No** usar split temporal | Rango intra-query mediano de 488 días: el timestamp no ordena las queries |
| D5 | El módulo Transformer opera sobre texto (`title`, opcionalmente `description`) | Rama tabular sola: ROC 0.579 |
| D6 | No construir features de match con filtros | Son constantes (100%) |
| D7 | Mantener el sufijo en el input del modelo principal | Sin él, el problema es irresoluble (ROC 0.499). Se ablaciona, no se elimina |
| D8 | Métrica principal PR-AUC (Average Precision), secundaria ROC-AUC | Enunciado + desbalance 13/87 |
| D9 | Selección de modelo por CV agrupada sobre train+valid; test tocado una sola vez | 1.301 positivos totales: un único split tiene demasiada varianza |

### 1.2 Confirmadas por el equipo

| # | Decisión | Valor | Consecuencia sobre el plan |
|---|---|---|---|
| A1 | Framework | **PyTorch** | Fases 7 y 8 se escriben en PyTorch |
| A2 | Cómputo | **GPU local / de laboratorio** | El costo computacional no es limitante: el protocolo de ablación pasa a 5 semillas × 5 folds |
| A3 | Alcance del encoder | **Todo desde cero, sin pre-entrenados** | Se elimina la rama de transfer learning de la Fase 9. El vocabulario, los embeddings de token, el positional encoding y los pesos de atención se aprenden desde inicialización aleatoria |
| A4 | Atención cross-item entre productos de una query | **Se implementa como ablación de baja prioridad** | Con GPU el costo marginal es despreciable. La evidencia de la sección 0.4 anticipa ganancia nula (tasas 0.643 / 0.645 / 0.670 / 0.630 según la cantidad de ítems ALTO compitiendo). Se corre y se reporta como resultado negativo, que es tan válido como uno positivo |

**Nota sobre A3.** "Desde cero" se refiere a los pesos del modelo, no a las bibliotecas. Usar `tokenizers` de HuggingFace para *entrenar* un BPE sobre el corpus propio no viola la restricción: el tokenizador se ajusta con los datos del TP, no descarga nada pre-entrenado. Lo que queda excluido es cargar pesos de BERT, MiniLM, GloVe, word2vec o cualquier checkpoint público. Si la cátedra interpreta la restricción de forma más estricta, la alternativa es implementar el algoritmo BPE a mano (ver Fase 6, paso 0).

---

## 2. Fuera de alcance

Para no dispersar esfuerzo, explícitamente **no** se hace:

- **Cualquier peso pre-entrenado**: BERT, DistilBERT, MiniLM, sentence-transformers, GloVe, word2vec, fastText. Ni congelados, ni como baseline, ni como inicialización. Decisión A3.
- Búsqueda masiva de hiperparámetros (Optuna, random search de cientos de trials). Tener GPU no cambia esto: el enunciado evalúa la justificación de las decisiones, no el barrido exhaustivo.
- Ingeniería de despliegue, API, contenedores, MLOps.
- Modelos de secuencia de usuario (el dataset no tiene `user_id`; eso pertenece al Ejercicio 3 y es teórico).
- Definición de umbral de decisión (el enunciado lo excluye explícitamente).
- Calibración avanzada más allá de reportar Brier y la curva de calibración.

---

## 3. Estructura del repositorio

```
tp1-transformers/
├── README.md
├── requirements.txt
├── plan.md
├── configs/
│   ├── base.yaml
│   └── ablations/
├── data/                      # gitignored
│   ├── raw/supermarket_products.csv
│   └── splits/                # índices congelados
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baselines.ipynb
│   └── 03_resultados.ipynb
├── src/
│   ├── data/
│   │   ├── load.py            # carga + auditoría
│   │   ├── features.py        # preprocesamiento
│   │   ├── splits.py          # partición agrupada
│   │   └── dataset.py         # Dataset / collate
│   ├── tokenizer/
│   │   └── bpe.py
│   ├── model/
│   │   ├── attention.py       # MHSA desde cero
│   │   ├── encoder.py         # bloque encoder
│   │   ├── heads.py           # pooling + clasificador
│   │   └── model.py           # arquitectura completa
│   ├── train.py
│   ├── evaluate.py
│   └── utils/seed.py
├── experiments/
│   └── results.csv            # una fila por corrida
└── report/
    └── presentacion.pdf
```

Justificación: el enunciado exige repositorio con README y hash de commit. Separar `src/` de `notebooks/` permite que las ablaciones se corran por script con configuración declarativa, en lugar de editar celdas a mano, que es la fuente número uno de resultados irreproducibles.

---

# FASES

---

## Fase 0 — Setup

**Objetivo.** Entorno reproducible y esqueleto del repo.

**Entregable.** Repo inicializado, `requirements.txt` con versiones fijadas, `src/utils/seed.py`.

**Pasos.**

1. Crear el repo con la estructura de la sección 3.
2. `requirements.txt` con versiones exactas (`==`), no rangos. Incluir `torch`, `numpy`, `pandas`, `scikit-learn`, `tokenizers`, `pyyaml`, `matplotlib`. Anotar también la versión de CUDA del entorno, porque afecta la reproducibilidad entre máquinas.
3. Escribir `set_seed(seed)` en `src/utils/seed.py`:

   ```python
   import os, random, numpy as np, torch

   def set_seed(seed: int) -> None:
       os.environ["PYTHONHASHSEED"] = str(seed)
       random.seed(seed)
       np.random.seed(seed)
       torch.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False
       torch.use_deterministic_algorithms(True, warn_only=True)
   ```

   Además, los `DataLoader` necesitan `worker_init_fn` y un `torch.Generator` con semilla propia, o el shuffling difiere entre corridas cuando `num_workers > 0`.
4. Definir `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")` en un único lugar e importarlo desde ahí. Nunca hardcodear `.cuda()` disperso por el código.
5. `.gitignore` que excluya `data/`, checkpoints y `__pycache__`.
6. Primer commit.

**Justificación.** El enunciado pide entregar el hash del commit. Sin semillas fijadas, dos corridas de la misma configuración dan métricas distintas y el estudio de ablación pierde sentido. Con `cudnn.benchmark = True` (el default en muchos setups) los algoritmos de convolución y GEMM se eligen dinámicamente y el resultado deja de ser bit a bit reproducible.

**Criterio de aceptación.** Dos ejecuciones consecutivas de un script que entrene 20 steps sobre GPU producen exactamente la misma loss final.

**Errores a evitar.**
- Fijar la semilla dentro del loop de entrenamiento en vez de al inicio del proceso.
- Dejar `cudnn.benchmark = True`.
- Olvidar `worker_init_fn` en los `DataLoader`.
- Mezclar tensores en CPU y GPU. Con el dataset en memoria (10.000 filas) conviene mover todo el batch a `DEVICE` en el `collate_fn` o al inicio del step, en un solo punto.

---

## Fase 1 — Carga y auditoría de fugas

**Objetivo.** Confirmar de forma independiente los hechos de la sección 0 y decidir el set de columnas admisibles.

**Entregable.** `src/data/load.py` y una tabla de auditoría en el notebook de EDA.

**Pasos.**

1. Cargar el CSV. Parsear `timestamp` a datetime con `utc=True`.
2. Reproducir: forma, nulos por columna, cardinalidades, duplicados exactos y por `(query_id, title)`.
3. Calcular la prevalencia de `bought`.
4. Construir la tabla de contingencia `cart × bought` y calcular `P(bought | cart)` en ambos sentidos.
5. Escribir una tabla explícita de auditoría con una fila por columna y estas cuatro entradas: *nombre*, *momento en que su valor queda determinado*, *¿está disponible al momento de decidir qué promocionar?*, *veredicto*.
6. Definir en el README la frase: "El momento de la predicción es el instante previo a mostrar la página de resultados, cuando el sistema decide qué productos promocionar."
7. Congelar la lista `FEATURES_ADMITIDAS` en `load.py` como constante.

**Justificación.** El leakage no se detecta con validación: las tres particiones quedan igualmente contaminadas y las curvas se ven sanas. La única defensa es la auditoría manual contra un momento de predicción definido explícitamente. Hacerlo antes de escribir cualquier modelo evita rehacer todo el diseño.

**Criterio de aceptación.**
- La tabla de auditoría cubre las 22 columnas sin excepción.
- `cart` está marcada como excluida con justificación escrita.
- `query_id` está marcada como clave de agrupamiento, no como feature.
- `bought` está marcada como target.

**Errores a evitar.**
- Dejar `cart` "por si acaso" para probar después. Si entra al pipeline, contamina la selección de features y de arquitectura.
- Codificar `query_id` como categórica. Tiene 2.012 valores y solo memoriza.
- Construir features agregadas por producto o marca a partir de `bought` sobre todo el dataset. Si más adelante se quiere una feature de este tipo, tiene que calcularse solo con train y solo con filas de `timestamp` anterior.

---

## Fase 2 — EDA formal (Ejercicio 1)

**Objetivo.** Producir el material que responde los cuatro puntos del Ejercicio 1.

**Entregable.** `notebooks/01_eda.ipynb` y las figuras para la presentación.

**Pasos.**

1. **Univariado.** Distribución de cada numérica (`price`, `net_weight_oz`, `nutrition_score`, `filter_price_min`, `filter_price_max`). Histograma y estadísticos. Verificar asimetría.
2. **Cardinalidad de categóricas.** `category` 12, `brand` 15, `storage_type` 3, `country_of_origin` 10, `unit_of_measure` 5, `package_size` 27, `ingredients` 190, `allergens` 7 + nulo. Todas son de cardinalidad baja o media: no hace falta bucket `[RARE]`.
3. **Estructura de queries.** Histograma de ítems por query. Distribución de compras por query.
4. **Coherencia temporal.** Calcular el rango de `timestamp` dentro de cada `query_id` y mostrar la mediana. Este gráfico es la justificación del descarte del split temporal (decisión D4) y hay que llevarlo a la presentación.
5. **Coherencia con filtros.** Verificar y documentar que las tres condiciones de la sección 0.5 se cumplen al 100%. Esto justifica no construir esas features.
6. **Análisis del texto.**
   - Extraer el sufijo con `title.str.extract(r'\(([^)]*)\)$')`. Tabular sufijo × tasa de compra.
   - Definir los tres niveles y reportar la tabla de la sección 0.3.
   - Extraer la última oración de `description` y cruzarla con el sufijo para mostrar la redundancia.
   - Verificar que `description` contiene `package_size` y `category` (ambos al 100%) y que `title` contiene `brand` (100%). Esto documenta que el texto es plantillado y que la parte descriptiva no aporta información nueva.
   - Medir longitud en palabras: título 6–14 (media 10.5), descripción 19–32 (media 26.2). Sirve para dimensionar `max_len` en la Fase 6.
7. **Bivariado con el target.** Tasa de compra por `category`, `brand`, `storage_type`, `country_of_origin`, `allergens`, `unit_of_measure`, y por decil de cada numérica.
8. **Análisis condicionado al nivel ALTO.** Repetir el bivariado solo sobre las filas de nivel ALTO. Es donde aparecen los efectos reales de `allergens` y `category` (sección 0.4). Sin condicionar, esos efectos quedan diluidos por el nivel.
9. **Confusión Seafood / Fish-Shellfish.** Tabla cruzada para documentar que son la misma señal.
10. Redactar en el notebook las cuatro respuestas del Ejercicio 1: qué se predice, características de la información, qué features se usan, qué preprocesamiento tiene cada una.

**Justificación.** El punto 8 es el que distingue un EDA bueno de uno mecánico. La estructura del dataset es jerárquica: primero el nivel de reputación decide si la compra es posible, después las características del producto modulan la probabilidad dentro del nivel. Un análisis marginal no lo muestra.

**Criterio de aceptación.**
- Están reproducidos todos los números de la sección 0.
- Existe una figura que justifica el descarte del split temporal.
- Existe una figura que muestra la tasa de compra por sufijo.
- Están escritas las cuatro respuestas del Ejercicio 1.

**Errores a evitar.**
- Reportar la correlación de `price` con `bought` como si fuera evidencia de algo. Es 0.003.
- Concluir que `category` importa mirando solo la tabla marginal (rango 0.057–0.191, mayormente confundido con el nivel).

---

## Fase 3 — Target y partición

**Objetivo.** Splits congelados en disco, usados por todos los experimentos.

**Entregable.** `src/data/splits.py` + `data/splits/{train,valid,test}_idx.npy`.

**Pasos.**

1. Definir `y = df['bought'].astype(int)`.
2. Split externo: `GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)` agrupando por `query_id`. Produce `dev` (80%) y `test` (20%).
3. Sobre `dev`, definir `GroupKFold(n_splits=5)` agrupado por `query_id` para la selección de modelo y las ablaciones.
4. Adicionalmente, materializar un split fijo `train` / `valid` dentro de `dev` (80/20 agrupado) para el desarrollo rápido y el early stopping durante la construcción del modelo.
5. Guardar todos los índices en `.npy`. Ningún script recalcula splits.
6. Escribir un test que verifique: ningún `query_id` aparece en dos particiones, y la prevalencia de cada partición está dentro de ±1.5 puntos de la global.

**Justificación.** Con 1.301 positivos en total, un único split de test deja ~260 positivos y el PR-AUC tiene varianza alta. La CV agrupada de 5 folds sobre `dev` reduce el ruido en la comparación de ablaciones; el `test` externo queda intacto para el número final. Esto también protege contra el leakage del experimentador: las decisiones se toman contra CV, no contra test.

**Criterio de aceptación.** El test de solapamiento de `query_id` pasa. Los archivos de índices existen y están commiteados o son reproducibles desde la semilla.

**Errores a evitar.**
- `train_test_split` sin `groups`.
- Recalcular el split en cada notebook.
- Estratificar por `bought` a nivel fila cuando se agrupa por query: la estratificación tiene que ser a nivel grupo (por ejemplo, por la cantidad de compras de la query) o simplemente verificarse a posteriori.

---

## Fase 4 — Features y preprocesamiento

**Objetivo.** Un pipeline de transformación que se ajusta solo con train.

**Entregable.** `src/data/features.py`, `src/data/dataset.py`.

**Pasos.**

1. **Numéricas.** `price`, `net_weight_oz`, `nutrition_score`, `filter_price_min`, `filter_price_max`.
   - Derivadas: `price_pos = (price - filter_price_min) / (filter_price_max - filter_price_min)`, `price_per_oz = price / net_weight_oz`, y las dimensiones parseadas de `dimensions_in` (largo, ancho, alto, volumen).
   - Parseo de `dimensions_in`: el formato es `"3.3 x 4.0 x 4.1\""`. Usar regex sobre los tres primeros floats.
   - Escalado con `StandardScaler` o `QuantileTransformer` ajustado **solo con train**.
2. **Categóricas.** `category`, `brand`, `storage_type`, `country_of_origin`, `unit_of_measure`, `package_size`, `allergens`.
   - `allergens` tiene 44.55% de nulos. El nulo es informativo (tasa 0.1385, entre Milk y Soy): codificarlo como categoría propia `"NONE"`, no imputar.
   - `ingredients` tiene 190 valores; tratarlo como categórica de baja cardinalidad o descartarlo, según lo que muestre el EDA. No es un campo multi-etiqueta real en este dataset.
   - Para los baselines: one-hot. Para el Transformer: `nn.Embedding` de dimensión 8–16 por columna.
3. **Temporales.** Derivar hora UTC, día de semana, mes de `timestamp`, con codificación cíclica seno/coseno. Incluir solo si el EDA muestra efecto; caso contrario, documentar el descarte.
4. **Texto.** Campo `text = title + " [SEP] " + description`. Mantener el sufijo. Preparar además la variante `text_sin_sufijo` para la ablación.
5. **Excluidas.** `cart` (fuga), `query_id` (clave de agrupamiento), `filter_category` y `filter_storage_type` (duplicados exactos de `category` y `storage_type`).
6. Implementar todo dentro de un objeto con `fit(train)` / `transform(X)`, o un `ColumnTransformer` de scikit-learn.

**Justificación.** Ajustar el escalador con todo el dataset es contaminación entre particiones: la media del test participa en la normalización de train. Es el tipo de fuga más frecuente y el más fácil de evitar con disciplina de pipeline.

Sobre el punto 5: `filter_category` y `filter_storage_type` son idénticas a `category` y `storage_type` en el 100% de las filas. Incluir ambas duplica la señal sin agregar información.

**Criterio de aceptación.**
- `fit` se llama exactamente una vez, sobre train.
- Un test verifica que transformar el mismo dataframe dos veces da el mismo resultado.
- Ninguna columna de `FEATURES_ADMITIDAS` quedó fuera sin justificación escrita.

**Errores a evitar.**
- `fit_transform` sobre el dataframe completo.
- Imputar `allergens` con la moda.
- Construir features de match con los filtros (constantes).

---

## Fase 5 — Baselines

**Objetivo.** Tener contra qué comparar y validar que el pipeline funciona antes de escribir el Transformer.

**Entregable.** `notebooks/02_baselines.ipynb`, primeras filas de `experiments/results.csv`.

**Pasos.** Entrenar y evaluar por CV agrupada de 5 folds sobre `dev`:

1. **Prior.** Predecir siempre la prevalencia. Da el piso del PR-AUC.
2. **Regresión logística** sobre features tabulares one-hot + numéricas escaladas.
3. **GBDT** (`HistGradientBoostingClassifier` o LightGBM) sobre las mismas features.
4. **LogReg sobre TF-IDF** de `text`, n-gramas (1,2), `min_df=3`.
5. **GBDT sobre sufijo + tabulares**, tomando el sufijo como variable categórica. Este es el baseline fuerte de referencia: ROC ≈ 0.970, PR-AUC ≈ 0.775.
6. **MLP** sobre features tabulares + TF-IDF reducido, para aislar el efecto "es una red neuronal" del efecto "tiene atención".

Registrar cada corrida en `results.csv` con: nombre, features usadas, semilla, fold, ROC-AUC, PR-AUC, log loss, Brier.

**Justificación.** El enunciado evalúa la comparación de alternativas. Sin baselines, el número del Transformer no significa nada. Además, el baseline 5 fija el objetivo realista: si el Transformer da mucho más, hay fuga; si da mucho menos, hay un bug.

Es previsible y aceptable que el GBDT con el sufijo explícito iguale o supere al Transformer. Ese resultado, presentado con el análisis de por qué (el dataset es sintético, la señal es un token categórico y no requiere composicionalidad), vale más que un número inflado.

**Criterio de aceptación.**
- Los seis baselines corridos, con media y desvío sobre los 5 folds.
- El baseline solo-tabular reproduce aproximadamente ROC 0.58.
- El baseline TF-IDF de título reproduce aproximadamente ROC 0.956.

**Errores a evitar.**
- Ajustar el `TfidfVectorizer` sobre todo el dataset. Debe ajustarse por fold, solo con train.
- Reportar accuracy.
- Usar `sklearn.metrics.auc(recall, precision)` en lugar de `average_precision_score`.

---

## Fase 6 — Tokenizador

**Objetivo.** Convertir `text` en secuencias de IDs.

**Entregable.** `src/tokenizer/bpe.py` + tokenizador serializado.

**Pasos.**

0. **Definir el nivel de "desde cero" del tokenizador.** Dos opciones, ambas compatibles con la decisión A3:
   - *(a)* Entrenar BPE con la biblioteca `tokenizers` sobre el corpus del TP. Es lo recomendado: rápido, robusto, y el vocabulario sale de los datos propios.
   - *(b)* Implementar BPE a mano: contar pares de símbolos adyacentes, fusionar iterativamente el par más frecuente, guardar la tabla de merges. Con un corpus de 10.000 textos cortos corre en segundos. Toma media jornada de trabajo extra y demuestra comprensión directa del contenido de la Clase 2.

   Elegir *(b)* solo si la cátedra exige explícitamente no usar bibliotecas de tokenización. Si se elige *(b)*, validar contra *(a)*: entrenar ambos con el mismo vocab objetivo y verificar que las tablas de merges coinciden en los primeros cientos de merges.
1. Entrenar el tokenizador BPE **solo sobre el texto de train**.
2. Vocabulario objetivo: 2.000–4.000. Justificación: el corpus tiene 8.578 productos únicos con texto plantillado y un vocabulario natural pequeño; un vocab de 30k dejaría casi todos los embeddings sin entrenar. Como todos los embeddings se entrenan desde cero (A3), el tamaño del vocabulario define directamente cuántos parámetros hay que aprender con solo ~7.900 ejemplos: es un hiperparámetro con consecuencias reales, no un detalle.
3. Tokens especiales: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`.
4. Formato de entrada: `[CLS] title [SEP] description [SEP]`.
5. Medir la distribución de longitudes en tokens sobre train y fijar `max_len` en el percentil 99, redondeado hacia arriba. Con títulos de 6–14 palabras y descripciones de 19–32, se espera algo del orden de 80–110 tokens BPE; **medirlo, no asumirlo**.
6. Verificar que el sufijo se tokeniza de forma estable (por ejemplo, que `Best Seller` no se fragmente distinto según el contexto). Si el sufijo se parte en sub-tokens inconsistentes, el modelo tiene que reconstruirlo y el problema se vuelve innecesariamente difícil. Es un chequeo barato y decisivo.
7. Guardar el tokenizador entrenado. Para la CV, reentrenarlo por fold o aceptar y documentar el sesgo mínimo de entrenarlo una vez sobre `dev`.
8. Ablación de tamaño de vocabulario: {1.000, 2.000, 4.000, 8.000}. Con GPU es barata y conecta directamente con el contenido de la Clase 2.

**Justificación.** El vocabulario BPE se construye a partir de frecuencias del corpus. Construirlo sobre el dataset completo filtra información de test a train. Es una fuga sutil y de impacto bajo en este caso, pero mencionarla y manejarla explícitamente es exactamente el tipo de rigor que el enunciado evalúa.

**Criterio de aceptación.**
- `max_len` está medido, no elegido a ojo.
- La tasa de `[UNK]` en validación es menor al 1%.
- Round-trip: decodificar la codificación de un texto de train devuelve el texto original salvo espaciado.

**Errores a evitar.**
- Entrenar el tokenizador con todo el CSV.
- Fijar `max_len` en 512 por costumbre. Con secuencias de ~90 tokens, eso multiplica por 30 el costo de la atención sin ganancia.
- Olvidar la máscara de padding aguas abajo.

---

## Fase 7 — Transformer desde cero

**Objetivo.** Implementar y testear los bloques de la arquitectura antes de armar el modelo completo.

**Entregable.** `src/model/attention.py`, `encoder.py`, `heads.py` + tests unitarios.

### 7.0 Qué se toma de PyTorch y qué se escribe a mano

La decisión A1 fija PyTorch como framework. La pregunta separada es hasta qué nivel se usan sus módulos de alto nivel. Política adoptada:

| Se usa de la librería, sin discusión | Se escribe a mano |
|---|---|
| `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` | Scaled dot-product attention |
| `nn.GELU`, `F.softmax` | Multi-head attention (split, atención por cabeza, concat, `W_O`) |
| `nn.BCEWithLogitsLoss` | Bloque encoder (residuales + pre-LN + FFN) |
| `torch.optim.AdamW`, schedulers de `torch.optim.lr_scheduler` | Positional encoding (sinusoidal y aprendido) |
| `DataLoader`, `Dataset`, `nn.utils.clip_grad_norm_` | Construcción y aplicación de la máscara de padding |
| `nn.MultiheadAttention` **solo como oráculo en tests** | Pooling (`[CLS]` y mean con máscara) |

Nadie espera que se reimplemente `Linear` o AdamW. Lo que el enunciado pide afianzar es la arquitectura, y eso es la columna derecha.

**Motivo operativo, además del pedagógico.** La Fase 10 requiere los mapas de atención por cabeza. `F.scaled_dot_product_attention` usa kernels fusionados y no devuelve la matriz de pesos. `nn.MultiheadAttention` la devuelve solo con `need_weights=True` y promediada entre cabezas salvo que además se pase `average_attn_weights=False`, lo que desactiva el camino rápido. `nn.TransformerEncoderLayer` no la expone. Con la implementación propia, guardar `attn_weights` es una línea.

**Configuración.** El módulo de atención se expone con un flag `attn_impl`:

| Valor | Uso |
|---|---|
| `"scratch"` | **Default.** Implementación propia. Es el modelo que se reporta y del que salen los mapas de atención |
| `"torch_sdpa"` | `F.scaled_dot_product_attention` como camino rápido. Mismos parámetros, sin pesos de atención observables. Útil si algún experimento resulta pesado |
| `"torch_mha"` | `nn.MultiheadAttention`. Se usa únicamente en el test de equivalencia |

Los tres modos comparten las mismas matrices de pesos, así que se pueden cargar entre sí y comparar salidas. Eso convierte la librería en un verificador de la implementación propia en lugar de un reemplazo.

### 7.1 Pasos

1. **Scaled dot-product attention.**
   `Attention(Q,K,V) = softmax(QKᵀ / √d_k + mask) V`
   La máscara de padding se suma como `-inf` (en la práctica, `-1e9`) en las posiciones inválidas, **antes** del softmax.
2. **Multi-head attention.** Proyecciones `W_Q`, `W_K`, `W_V` de `d_model → d_model`, reshape a `(batch, heads, seq, d_head)` con `d_head = d_model / n_heads`, atención por cabeza, concatenación, proyección de salida `W_O`.
3. **Feed-forward.** `Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)`.
4. **Bloque encoder con pre-LayerNorm.**
   `x = x + Dropout(MHSA(LN(x)))`
   `x = x + Dropout(FFN(LN(x)))`
   Pre-LN en lugar de post-LN porque converge de forma más estable en modelos chicos y con warmup corto.
5. **Embeddings.** Tabla de tokens + positional encoding. Implementar las dos variantes (sinusoidal fija y aprendida) para poder ablacionarlas.
6. **Pooling.** Implementar `[CLS]` y mean pooling con máscara. Ambas son ablación.

**Tests unitarios obligatorios.**

| Test | Qué verifica |
|---|---|
| **Equivalencia con `nn.MultiheadAttention`**: copiar los pesos de la implementación propia a una `nn.MultiheadAttention` con la misma configuración y comparar salidas con `torch.allclose(atol=1e-5)`, con y sin máscara de padding | Correctitud numérica de la implementación propia. Es el test más valioso de la lista |
| Forma de salida = forma de entrada en el bloque encoder | Errores de reshape |
| Los pesos de atención suman 1 sobre la última dimensión | Softmax en el eje correcto |
| Con máscara de padding, el peso hacia posiciones `[PAD]` es exactamente 0 | Máscara aplicada antes del softmax |
| Cambiar el valor de un token `[PAD]` no cambia la salida en las posiciones válidas | Máscara efectiva de punta a punta |
| Permutar dos tokens **sin** positional encoding no cambia la salida del mean pooling | Confirma la invariancia a permutación de la atención |
| Con positional encoding, la misma permutación **sí** cambia la salida | Confirma que el PE está conectado |
| `d_model % n_heads == 0` levanta una excepción explícita si no se cumple | Error de configuración detectado temprano |
| Un batch de tamaño 1 y uno de tamaño 8 dan la misma salida para el mismo ejemplo | Sin filtración entre elementos del batch |
| `attn_impl="scratch"` y `attn_impl="torch_sdpa"` dan la misma salida con los mismos pesos | Consistencia entre modos |

**Cuidado con el test de equivalencia.** `nn.MultiheadAttention` empaqueta `W_Q`, `W_K` y `W_V` en un único tensor `in_proj_weight` de forma `(3·d_model, d_model)`, en ese orden, y usa `batch_first=False` por defecto. Copiar los pesos requiere respetar ambas convenciones. Además, su `key_padding_mask` es booleana con `True` en las posiciones **a ignorar**, que es la convención inversa a la que suele escribirse a mano. La mitad de los fallos de este test son de convención, no de implementación.

**Justificación.** Los bugs de máscara y de reshape en la atención no producen errores: producen un modelo que entrena y da métricas plausiblemente malas. Se descubren tarde y son carísimos. Los tests de la tabla se escriben en menos de una hora y cubren los modos de falla conocidos.

Los dos tests de permutación tienen además valor expositivo: son la demostración empírica de por qué el positional encoding hace falta en texto, y se pueden mostrar en la presentación.

**Criterio de aceptación.** Los diez tests pasan, incluido el de equivalencia contra `nn.MultiheadAttention`. El bloque encoder corre en forward y backward sin NaN sobre un batch aleatorio en GPU.

**Errores a evitar.**
- Aplicar el softmax sobre el eje equivocado.
- Sumar la máscara después del softmax.
- Dividir por `√d_model` en lugar de `√d_head`.
- Usar `nn.TransformerEncoderLayer` como bloque principal. Encapsula pre-LN, dropout, residuales y FFN en una sola llamada: es justamente lo que el enunciado pide construir, y además bloquea el acceso a los pesos de atención que necesita la Fase 10.
- Usar `-float('inf')` en la máscara. Si una fila queda enteramente enmascarada, el softmax produce NaN. Usar `-1e9` (o `torch.finfo(dtype).min`).
- Confundir la convención booleana de `key_padding_mask` al escribir el test de equivalencia.

---

## Fase 8 — Modelo completo y entrenamiento

**Objetivo.** Primer end-to-end funcional, después escalado.

**Entregable.** `src/model/model.py`, `src/train.py`, primer checkpoint.

### 8.1 Arquitectura

```
text ──► tokenizer ──► embeddings + PE ──► N × EncoderBlock ──► pooling ──► h_txt (d_model)
                                                                                │
categóricas ──► nn.Embedding por columna ──┐                                    │
numéricas   ──► escaladas ─────────────────┴──► MLP ──► h_tab (d_tab) ──────────┤
                                                                                ▼
                                                          concat ──► MLP ──► logit ──► sigmoid
```

**Justificación de la fusión.** La rama de texto sola llega a ROC ≈ 0.956. La tabular sola llega a 0.579. Pero el baseline que combina sufijo y tabulares llega a 0.970 con PR-AUC 0.775, muy por encima de cualquiera de las dos ramas. Esa ganancia viene de `allergens` y `category`, que modulan la compra dentro del nivel ALTO (sección 0.4). La fusión no es decorativa: es donde está el margen.

### 8.2 Configuración inicial

Siguiendo la sugerencia del enunciado de empezar con `d_model < 100`:

| Hiperparámetro | Valor inicial |
|---|---|
| `d_model` | 64 |
| `n_heads` | 4 (`d_head` = 16) |
| `n_layers` | 2 |
| `d_ff` | 128 |
| `dropout` | 0.1 |
| Activación | GELU |
| Normalización | pre-LN |
| Optimizador | AdamW, `lr` 3e-4, `weight_decay` 1e-2 |
| Scheduler | warmup lineal 10% + cosine decay |
| Batch size | 64 |
| Loss | `BCEWithLogitsLoss` |
| Gradient clipping | 1.0 |
| Early stopping | PR-AUC de validación, `patience` 10 |

### 8.3 Inicialización (crítico por A3)

Como no hay pesos pre-entrenados, la inicialización es responsabilidad del equipo y afecta la convergencia:

- Embeddings de token: `N(0, 0.02)`, el mismo criterio que usa BERT. Inicializar el embedding de `[PAD]` en cero y congelarlo, o excluirlo del gradiente.
- Proyecciones lineales: inicialización de Xavier/Glorot uniforme.
- Bias: cero.
- Bias de la capa de salida: inicializarlo en `log(p / (1 - p))` con `p = 0.13`, la prevalencia de train. Esto hace que el modelo arranque prediciendo la tasa base y evita las primeras épocas gastadas en corregir el intercepto. Es barato y estabiliza el comienzo del entrenamiento.

### 8.4 Pasos

1. Implementar el modelo con todos los módulos activables por configuración (`use_text`, `use_tabular`, `pooling`, `pos_encoding`, `use_cross_item`).
2. Correr un **overfit test**: entrenar sobre 64 ejemplos sin dropout ni weight decay hasta llegar a loss ≈ 0. Si no lo logra, hay un bug.
3. Recién entonces entrenar sobre train completo con validación.
4. Registrar curvas de train y validación por época.
5. Guardar el mejor checkpoint por PR-AUC de validación.

### 8.5 Consideraciones de GPU

El modelo es chico (`d_model` 64, 2 capas, ~90 tokens) y el dataset tiene 10.000 filas. Una época entera cabe holgadamente en memoria de cualquier GPU moderna.

- **No usar mixed precision (AMP).** A esta escala no acelera nada y agrega una fuente de no-determinismo y de posibles NaN. Entrenar en fp32.
- **No usar `DataParallel` ni `DistributedDataParallel`.** El overhead supera cualquier ganancia.
- Precomputar la tokenización una sola vez y guardarla como tensor de enteros. No tokenizar dentro del `__getitem__`.
- Con el dataset entero en memoria, `num_workers = 0` o 2 alcanza. Más workers agregan overhead de arranque por época.
- El batch de 64 puede subirse a 256 sin problema. Si se sube, escalar `lr` proporcionalmente o revalidar el learning rate.
- Medir el tiempo por época en la primera corrida y anotarlo. Es el dato que permite dimensionar el presupuesto total de la Fase 9.

**Sobre el desbalance.** La prevalencia es 13%, que es un desbalance moderado. `pos_weight = 6.7` es una opción, no una necesidad. Tratarlo como ablación (sin peso / `pos_weight` / focal loss) y decidir con evidencia. Advertencia: usar `pos_weight` desplaza las probabilidades hacia arriba y arruina la calibración; si se usa, hay que recalibrar antes de reportar BTR absolutos.

**Criterio de aceptación.**
- El overfit test sobre 64 ejemplos llega a loss < 0.01.
- El modelo con texto supera al baseline solo-tabular (ROC 0.58) por un margen amplio.
- Las curvas de train y validación se registran y son inspeccionables.

**Errores a evitar.**
- Saltear el overfit test. Es el diagnóstico más barato que existe.
- Escalar la arquitectura antes de tener el end-to-end funcionando.
- Calcular la métrica sobre logits en lugar de probabilidades. Para AUC da lo mismo porque es monótono, pero para log loss y Brier no.
- Olvidar `model.eval()` y `torch.no_grad()` en la evaluación.

---

## Fase 9 — Experimentación y ablación

**Objetivo.** Producir la tabla de ablación que el enunciado exige.

**Entregable.** `experiments/results.csv` completo + `notebooks/03_resultados.ipynb`.

**Protocolo.** Cada configuración se corre con **5 semillas** sobre la CV agrupada de 5 folds de `dev`, es decir 25 entrenamientos por fila. Se reporta media ± desvío. Cada fila cambia **una sola cosa** respecto de la configuración base.

Con GPU (decisión A2) y un modelo de este tamaño, cada entrenamiento es cuestión de segundos a un par de minutos. Las 25 corridas por fila son perfectamente asumibles y reducen sustancialmente el ruido en la comparación, que es el principal problema con 1.301 positivos. Estimar el presupuesto total tras medir el tiempo por época en la Fase 8.

| Grupo | Variantes | Qué demuestra |
|---|---|---|
| Módulos | solo texto / solo tabular / texto + tabular | Cuánto aporta cada rama. Resultado esperado: la fusión gana |
| **Sufijo** | con sufijo / sin sufijo en el input | El resultado esperado es un colapso a ROC ≈ 0.50. Es el hallazgo central del trabajo |
| Campo de texto | solo `title` / solo `description` / ambos | Esperado: sin diferencia, porque son redundantes |
| Positional encoding | sinusoidal / aprendido / **ninguno** | Con "ninguno" el modelo pierde el orden. Dado que la señal es un marcador presente sin importar la posición, es plausible que casi no cambie: sería un resultado interesante y honesto |
| Pooling | `[CLS]` / mean pooling | |
| Profundidad | `n_layers` ∈ {1, 2, 4} | |
| Ancho | `d_model` ∈ {32, 64, 96} | Respeta el límite `< 100` del enunciado |
| Cabezas | `n_heads` ∈ {1, 2, 4, 8} con `d_model` = 64 fijo | Aísla el efecto de la multi-cabeza del efecto de la capacidad |
| Normalización | pre-LN / post-LN | |
| Desbalance | sin peso / `pos_weight` / focal loss | |
| Tabulares | con / sin `allergens`; con / sin `category` | Verifica el efecto medido en la sección 0.4 |
| **Fuga** | con / sin `cart` | Convierte la trampa en un resultado presentable |
| Vocabulario BPE | {1.000, 2.000, 4.000, 8.000} | Contenido de la Clase 2. Con embeddings desde cero, el tamaño del vocabulario controla directamente cuántos parámetros hay que aprender |
| Atención cross-item | con / sin | Decisión A4. Resultado esperado: ganancia nula. Se reporta igual |

No hay fila de transfer learning: la decisión A3 excluye los pesos pre-entrenados. Si en la defensa preguntan por la comparación, la respuesta es que fue una restricción de alcance asumida deliberadamente, y que el contenido de Clase 3 se cubre en el Ejercicio 3.

**Justificación del protocolo de semillas.** Con ~260 positivos por fold, diferencias de PR-AUC menores a unos pocos puntos son indistinguibles del ruido. Sin desvío, la tabla de ablación no permite concluir nada. Cinco semillas es el punto donde el error estándar de la media baja lo suficiente como para que las comparaciones sean interpretables, y con GPU el costo es trivial.

**Criterio de aceptación.**
- Cada fila tiene media y desvío sobre semillas y folds.
- La ablación del sufijo está corrida y documentada.
- Ninguna corrida usó el conjunto de test.
- Ninguna corrida cargó pesos externos.

**Errores a evitar.**
- Cambiar dos cosas a la vez.
- Comparar contra el test.
- Descartar una variante por una sola corrida.

---

## Fase 10 — Evaluación final e interpretabilidad

**Objetivo.** El número que va a la presentación, más el análisis que lo sostiene.

**Entregable.** Sección de resultados del notebook 03 + figuras.

**Pasos.**

1. Elegir la configuración final **usando solo los resultados de CV**. Congelarla.
2. Reentrenar sobre `dev` completo. Evaluar una única vez sobre `test`.
3. Reportar:
   - Prevalencia del test (piso del PR-AUC).
   - PR-AUC (Average Precision) y lift = PR-AUC / prevalencia.
   - ROC-AUC.
   - Log loss y Brier score.
   - Intervalo de confianza por bootstrap (1.000 remuestreos, agrupando por `query_id`).
4. **Curva de calibración.** Agrupar predicciones en deciles y graficar probabilidad media predicha contra frecuencia observada. Relevante porque el output es un BTR, una probabilidad con significado propio.
5. **Curvas ROC y PR** del modelo final junto a los baselines, en un mismo gráfico.
6. **Diagnóstico de over/underfitting.** Curvas de train vs. validación. Learning curve: performance contra fracción de train usada. Si sigue subiendo, el límite son los datos y no la arquitectura, y eso justifica no escalar más el modelo.
7. **Mapas de atención.** Promediar sobre el test los pesos de atención del token `[CLS]` (o del pooling) hacia cada token de entrada. La hipótesis a verificar es que el modelo concentra la atención sobre los tokens del sufijo. Contrastar con la importancia de features del GBDT.
8. **Métricas de ranking.** Precision@k y Recall@k para el caso de uso de promoción, y `mAP` calculado por `query_id`.

**Justificación.** El punto 7 es la mejor pieza de la presentación: es evidencia visual directa de que el mecanismo de atención aprendió a localizar la señal, y conecta la arquitectura con el hallazgo del EDA. El punto 3 evita el error de reportar un PR-AUC sin contexto de prevalencia.

**Criterio de aceptación.** El test se evaluó exactamente una vez. Todas las métricas de la lista están reportadas.

**Errores a evitar.**
- Volver a tocar hiperparámetros después de ver el test.
- Reportar PR-AUC sin la prevalencia al lado.
- Presentar mapas de atención de una sola muestra elegida a mano.

---

## Fase 11 — Ejercicio 3: personalización

**Objetivo.** Una diapositiva, menos de cinco minutos. Es teórico.

**Contenido.**

1. **Punto de partida.** El dataset no tiene `user_id`. Es el primer requisito.
2. **Reformulación.** De `P(buy | item, query)` a `P(buy | item, query, user)`.
3. **Arquitectura propuesta.** Two-tower: torre de ítem (la actual) + torre de usuario. El embedding de usuario se obtiene de un Transformer sobre la secuencia de interacciones históricas del usuario, aplicando self-attention sobre el eje temporal en lugar del eje de tokens.
4. **Fusión.** Cross-attention entre el historial del usuario (como K, V) y el ítem candidato (como Q), o producto punto entre las dos torres.
5. **Cold start.** Token `[NEW_USER]` aprendido, o embedding de segmento a partir de atributos demográficos.
6. **Consideraciones.** Privacidad y datos personales. Costo de inferencia: el BTR deja de ser precomputable por producto. Bucle de retroalimentación y pérdida de diversidad. Evaluación por usuario en lugar de global.

**Criterio de aceptación.** Cabe en una diapositiva y se explica en menos de cinco minutos.

---

## Fase 12 — Presentación y entrega

**Objetivo.** Cumplir el formato pedido: 25–30 minutos.

**Estructura sugerida.**

| Bloque | Minutos | Contenido |
|---|---|---|
| Problema | 3 | Qué es el BTR, por qué importa, formulación elegida y la alternativa descartada |
| EDA | 6 | Estructura, la tabla de sufijos, la evidencia del leakage de `cart`, la incoherencia temporal intra-query |
| Diseño | 6 | Por qué encoder de texto y no tabular (mostrar ROC 0.58 vs 0.96), arquitectura, decisiones de partición |
| Resultados | 6 | Baselines vs. modelo, tabla de ablación, calibración, mapas de atención |
| Desafíos | 3 | Leakage, dataset sintético, varianza con pocos positivos |
| Personalización | 3 | Ejercicio 3 |
| Conclusiones | 3 | |

**Entrega.** Repositorio con `README.md`, hash del commit y presentación por Campus. Congelar el commit antes de armar los slides y anotar el hash.

El `README.md` debe contener: definición del problema, momento de la predicción, features excluidas y por qué, cómo reproducir (comandos), tabla de resultados principales, y la estructura del repo.

---

## Preguntas frecuentes de defensa

Preparar respuesta para cada una:

1. ¿Por qué sacaron `cart`? ¿Y si el caso de uso fuera otro?
2. ¿Por qué no hicieron split temporal?
3. ¿Su PR-AUC de X es bueno? (Sin la prevalencia no se puede responder)
4. ¿Por qué un Transformer y no un GBDT, si el GBDT gana?
5. ¿Por qué implementaron la atención a mano si PyTorch ya la trae? (Respuesta: para poder extraer los mapas de atención por cabeza, y se validó por equivalencia numérica contra `nn.MultiheadAttention`)
6. ¿Qué representa el token `[CLS]` en su arquitectura?
7. ¿Por qué usaron positional encoding acá y no lo usarían en un modelo tabular?
8. ¿Qué pasa si `d_model` no es divisible por `n_heads`?
9. ¿Cómo manejaron el padding?
10. ¿Su modelo está calibrado? ¿Importa para el BTR?
11. ¿Qué aprendió realmente el modelo? (Respuesta honesta: a leer un marcador de reputación insertado sintéticamente)
12. ¿Cómo evitaron que la misma query esté en train y test?
13. ¿Cuántas veces miraron el test?

---

## Riesgos y mitigaciones

| Riesgo | Señal temprana | Mitigación |
|---|---|---|
| Fuga no detectada | ROC-AUC > 0.98 | Tabla de auditoría de la Fase 1; ablación por feature |
| Bug de máscara en la atención | Métricas planas o sensibilidad al padding | Tests unitarios de la Fase 7 |
| Ablaciones no concluyentes | Desvíos mayores que las diferencias | 3 semillas × 5 folds; si aun así no separa, reportarlo como tal |
| El GBDT gana al Transformer | — | Presentarlo con el análisis de por qué. Es un resultado válido |
| Sobreajustar el test | Muchas corridas contra test | Selección solo por CV; test una vez |
| Alcance desbordado | Semanas en hyperparameter search | Respetar la sección 2 (fuera de alcance) |

---

## Checklist final

- [ ] Momento de la predicción definido por escrito en el README
- [ ] Tabla de auditoría de las 22 columnas
- [ ] `cart` excluida y justificada
- [ ] Split agrupado por `query_id`, congelado en disco, con test de solapamiento
- [ ] Preprocesamiento ajustado solo con train
- [ ] Tokenizador BPE entrenado solo con train
- [ ] `cudnn.benchmark = False` y semillas fijadas, con reproducibilidad verificada en GPU
- [ ] Ningún peso pre-entrenado cargado en ninguna corrida
- [ ] Bias de salida inicializado en `log(0.13 / 0.87)`
- [ ] Los diez tests unitarios de la Fase 7 pasan, incluido el de equivalencia con `nn.MultiheadAttention`
- [ ] Overfit test sobre 64 ejemplos superado
- [ ] Los seis baselines corridos
- [ ] Tabla de ablación con media y desvío sobre 5 semillas × 5 folds
- [ ] Ablación del sufijo documentada
- [ ] Test evaluado una sola vez
- [ ] PR-AUC reportado junto a la prevalencia
- [ ] Curva de calibración
- [ ] Mapas de atención
- [ ] Ejercicio 3 en una diapositiva
- [ ] README con hash del commit
