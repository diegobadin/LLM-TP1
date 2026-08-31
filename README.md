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
| 1 | Carga y auditoría de fugas | pendiente |
| 2 | EDA formal (Ejercicio 1) | pendiente |
| 3 | Target y partición | pendiente |
| 4–12 | Features, baselines, tokenizer, Transformer, ablaciones | pendiente |

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
