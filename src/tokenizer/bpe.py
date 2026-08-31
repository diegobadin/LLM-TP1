"""Tokenizador BPE entrenado sobre el corpus del TP (Fase 6).

Se entrena un BPE **desde cero con los textos del TP**, no se descarga ninguno
pre-entrenado (decision A3). La biblioteca ``tokenizers`` se usa solo como
implementacion del algoritmo: el vocabulario y la tabla de merges salen de este
corpus y de ningun otro. Es la opcion (a) del paso 0 de la Fase 6; la (b),
implementar BPE a mano, solo corresponde si la catedra prohibe explicitamente
las bibliotecas de tokenizacion.

Dos decisiones que condicionan todo lo que sigue:

* **Se entrena solo con el texto de train.** El vocabulario BPE se construye a
  partir de frecuencias del corpus: construirlo sobre el dataset completo filtra
  informacion de test hacia train. El impacto es chico en este caso, pero es una
  fuga real y manejarla explicitamente cuesta una linea.
* **El vocabulario es chico, y mas chico de lo que el plan anticipaba.** El
  corpus de train tiene solo **622 tipos de palabra** distintos: el texto es
  plantillado y las variaciones son pocas. El BPE **satura en 1.355 tokens**
  (alfabeto + subpalabras intermedias + las 622 palabras enteras + los 4
  especiales): a partir de ahi no quedan pares adyacentes que fusionar, y pedir
  2.000, 4.000 u 8.000 devuelve exactamente el mismo tokenizador. Se verifico con
  ``min_frequency`` 1, 2 y 3, que dan el mismo techo.

  Consecuencia practica: la grilla de ablacion que propone el plan
  ({1.000, 2.000, 4.000, 8.000}) **degenera en dos tokenizadores distintos**. La
  ablacion util se corre hacia abajo, con vocabularios que si obligan a partir
  palabras: {256, 512, 1.000, 2.000}. Ahi el efecto es visible y es exactamente el
  contenido de la Clase 2: con 256 tokens el marcador de reputacion se fragmenta
  y el modelo tiene que recomponerlo.

  Que el vocabulario sea chico importa por otra razon: como todos los embeddings
  se aprenden desde inicializacion aleatoria (A3), su tamano define cuantos
  parametros hay que entrenar con ~6.300 ejemplos. Con 30k, casi todos quedarian
  sin entrenar.

Formato de entrada al encoder: ``[CLS] titulo [SEP] descripcion [SEP]``.

Uso:

    python -m src.tokenizer.bpe              # entrena, mide, verifica y guarda
    python -m src.tokenizer.bpe --ablacion   # tabla de tamanos de vocabulario
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

from src.data.load import PROJECT_ROOT

TOKENIZER_DIR = PROJECT_ROOT / "data" / "tokenizer"

#: El orden fija los ids: PAD=0, UNK=1, CLS=2, SEP=3. Que ``[PAD]`` sea 0 es
#: comodo aguas abajo: la mascara de padding es ``ids != 0``, y el embedding de
#: padding es la fila 0, que se inicializa en cero y se excluye del gradiente.
ESPECIALES: tuple[str, ...] = ("[PAD]", "[UNK]", "[CLS]", "[SEP]")
PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 1, 2, 3

#: Vocabulario por defecto. Cualquier valor >= VOCAB_SATURACION da el mismo
#: tokenizador; se pide 2.000 (dentro del rango que sugiere el plan) y se
#: obtienen 1.355.
VOCAB_SIZE = 2000

#: Techo medido del BPE sobre este corpus: no hay mas pares que fusionar.
VOCAB_SATURACION = 1355

#: Tamanos para la ablacion de la Fase 9, reescalados hacia abajo: la grilla
#: original del plan no produce tokenizadores distintos (ver el docstring).
VOCABS_ABLACION: tuple[int, ...] = (256, 512, 1000, 2000)

#: ``max_len`` **medido**: percentil 99 de la longitud del par sobre train (52),
#: redondeado al multiplo de 8 siguiente. Ningun ejemplo de train ni de valid
#: queda truncado. Lo recalcula ``train_from_pairs`` salvo que se lo pasen
#: explicito; la constante esta para documentarlo y para los tests.
MAX_LEN = 56

#: Hash del vocabulario y los merges del tokenizador por defecto (pedido 2.000,
#: entrenado sobre el split ``train``). Si cambia, cambio el corpus o la version
#: de ``tokenizers``, y los resultados ya reportados dejan de ser comparables.
EXPECTED_FINGERPRINT = "e3265b3bd0be67f8"


@dataclass
class BPETokenizer:
    """Envoltorio delgado sobre ``tokenizers.Tokenizer``.

    Agrega lo que el resto del proyecto necesita y la biblioteca no da hecho:
    ``max_len`` medido, codificacion por lotes a arreglos de numpy con su
    mascara, tasa de ``[UNK]`` y verificacion de estabilidad del sufijo.
    """

    tokenizer: Tokenizer
    max_len: int

    # -- entrenamiento ------------------------------------------------------ #

    @classmethod
    def train(
        cls,
        textos: Sequence[str],
        vocab_size: int = VOCAB_SIZE,
        max_len: int | None = None,
        percentil: float = 99.0,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Entrena BPE sobre ``textos`` (que deben ser **solo de train**).

        El pre-tokenizador ``Whitespace`` corta por espacios y por puntuacion, y
        el BPE recompone subpalabras dentro de cada pedazo. Se elige sobre
        ByteLevel para que los tokens sean legibles: los mapas de atencion de la
        Fase 10 se leen mucho mejor con ``Seller`` que con ``ĠSeller``, y la tasa
        de ``[UNK]`` pasa a ser una metrica con sentido en vez de cero por
        construccion.
        """
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()

        entrenador = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=list(ESPECIALES),
            min_frequency=min_frequency,
            show_progress=False,
        )
        tokenizer.train_from_iterator(list(textos), trainer=entrenador)

        tokenizer.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", CLS_ID), ("[SEP]", SEP_ID)],
        )

        tok = cls(tokenizer=tokenizer, max_len=max_len or 0)
        if max_len is None:
            tok.max_len = tok.medir_max_len(textos, percentil=percentil)
        tok._configurar_padding()
        return tok

    @classmethod
    def train_from_pairs(
        cls,
        text_a: Sequence[str],
        text_b: Sequence[str],
        vocab_size: int = VOCAB_SIZE,
        max_len: int | None = None,
        percentil: float = 99.0,
    ) -> "BPETokenizer":
        """Entrena con titulo y descripcion como documentos separados.

        Son dos registros distintos (uno nominal y uno de oraciones completas) y
        el BPE aprende mejor los merges de cada uno si no se los concatena.
        ``max_len`` si se mide sobre el par, que es lo que ve el modelo.
        """
        tok = cls.train(list(text_a) + list(text_b), vocab_size=vocab_size, max_len=1)
        tok.max_len = max_len or tok.medir_max_len(text_a, text_b, percentil=percentil)
        tok._configurar_padding()
        return tok

    def _configurar_padding(self) -> None:
        self.tokenizer.enable_truncation(max_length=self.max_len)
        self.tokenizer.enable_padding(
            pad_id=PAD_ID, pad_token="[PAD]", length=self.max_len
        )

    # -- medicion ----------------------------------------------------------- #

    @contextmanager
    def _sin_padding(self):
        """Desactiva padding y truncamiento mientras dure el bloque.

        Lo necesitan las mediciones (longitud real de cada ejemplo) y la
        inspeccion de tokens: con padding activo, la cola de la secuencia son
        ``[PAD]`` y cualquier chequeo sobre los ultimos tokens mira relleno.
        """
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        try:
            yield
        finally:
            if self.max_len:
                self._configurar_padding()

    def lengths(
        self, text_a: Sequence[str], text_b: Sequence[str] | None = None
    ) -> np.ndarray:
        """Longitud en tokens de cada ejemplo, **sin** padding ni truncamiento."""
        with self._sin_padding():
            codificados = self._encode_raw(text_a, text_b)
            return np.array([len(e.ids) for e in codificados], dtype=int)

    def medir_max_len(
        self,
        text_a: Sequence[str],
        text_b: Sequence[str] | None = None,
        percentil: float = 99.0,
        multiplo: int = 8,
    ) -> int:
        """``max_len`` = percentil 99 de las longitudes, redondeado hacia arriba.

        Fijarlo en 512 por costumbre multiplicaria por ~30 el costo de la
        atencion, que es cuadratico en la longitud, sin ganar nada.
        """
        largos = self.lengths(text_a, text_b)
        objetivo = int(np.ceil(np.percentile(largos, percentil)))
        return int(np.ceil(objetivo / multiplo) * multiplo)

    def unk_rate(
        self, text_a: Sequence[str], text_b: Sequence[str] | None = None
    ) -> float:
        """Fraccion de tokens que caen en ``[UNK]``. Criterio de aceptacion: < 1%."""
        ids, mask = self.encode(text_a, text_b)
        reales = int(mask.sum())
        return float(((ids == UNK_ID) & mask).sum() / reales) if reales else 0.0

    def truncation_rate(
        self, text_a: Sequence[str], text_b: Sequence[str] | None = None
    ) -> float:
        """Fraccion de ejemplos mas largos que ``max_len``."""
        return float((self.lengths(text_a, text_b) > self.max_len).mean())

    # -- codificacion ------------------------------------------------------- #

    def _encode_raw(self, text_a: Sequence[str], text_b: Sequence[str] | None):
        if text_b is None:
            return self.tokenizer.encode_batch([str(t) for t in text_a])
        if len(text_a) != len(text_b):
            raise ValueError(f"text_a tiene {len(text_a)} y text_b {len(text_b)}")
        return self.tokenizer.encode_batch(
            [(str(a), str(b)) for a, b in zip(text_a, text_b)]
        )

    def encode(
        self, text_a: Sequence[str], text_b: Sequence[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Codifica un lote a ``(ids, mask)`` de forma ``(n, max_len)``.

        ``mask`` es booleana con **True en los tokens reales**, que es la
        convencion de la atencion propia de la Fase 7.
        """
        codificados = self._encode_raw(text_a, text_b)
        ids = np.array([e.ids for e in codificados], dtype=np.int64)
        mask = np.array([e.attention_mask for e in codificados], dtype=bool)
        return ids, mask

    def encode_one(self, text_a: str, text_b: str | None = None) -> list[str]:
        """Los tokens de un ejemplo, como strings y **sin padding**.

        Es la vista para inspeccion, chequeos y figuras: con relleno, comparar
        las ultimas posiciones de dos secuencias solo compara ``[PAD]``.
        """
        with self._sin_padding():
            codificado = self._encode_raw([text_a], None if text_b is None else [text_b])[0]
            return list(codificado.tokens)

    def decode(self, ids: Sequence[int]) -> str:
        """Texto reconstruido, sin los tokens especiales."""
        return self.tokenizer.decode([int(i) for i in ids], skip_special_tokens=True)

    # -- metadatos ---------------------------------------------------------- #

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def fingerprint(self) -> str:
        """Hash del vocabulario y los merges."""
        estado = json.loads(self.tokenizer.to_str())["model"]
        h = hashlib.sha256()
        h.update(json.dumps(estado["vocab"], sort_keys=True).encode())
        h.update(json.dumps(estado["merges"]).encode())
        return h.hexdigest()[:16]

    # -- persistencia ------------------------------------------------------- #

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {
                    "max_len": self.max_len,
                    "vocab_size": self.vocab_size,
                    "fingerprint": self.fingerprint(),
                    "especiales": list(ESPECIALES),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No hay tokenizador en {path}. Entrenarlo con:\n"
                f"    python -m src.tokenizer.bpe"
            )
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        tok = cls(tokenizer=Tokenizer.from_file(str(path)), max_len=int(meta["max_len"]))
        tok._configurar_padding()
        return tok


# --------------------------------------------------------------------------- #
# Verificaciones
# --------------------------------------------------------------------------- #

def suffix_stability(tok: BPETokenizer, titulos: Sequence[str]) -> pd.DataFrame:
    """Verifica que cada marcador de reputacion se tokenice siempre igual.

    Si ``Best Seller`` se parte en sub-tokens distintos segun el contexto, el
    modelo tiene que reconstruir el marcador antes de poder usarlo y el problema
    se vuelve innecesariamente dificil. Es un chequeo barato y decisivo, porque
    ahi esta el 96% de la senal.
    """
    from src.data.load import SUFFIX_NONE, extract_suffix

    sufijos = extract_suffix(pd.Series(list(titulos)))
    filas = []
    for marcador in sorted(s for s in sufijos.unique() if s != SUFFIX_NONE):
        con_marcador = [t for t, s in zip(titulos, sufijos) if s == marcador]
        # Tokens del marcador tokenizado por si solo, sin [CLS] ni [SEP].
        aislado = tuple(tok.encode_one(f"({marcador})")[1:-1])
        # Y los de la ventana final de cada titulo que lo lleva, del mismo largo.
        colas = {tuple(tok.encode_one(t)[-1 - len(aislado):-1]) for t in con_marcador}
        filas.append({
            "marcador": marcador,
            "n": len(con_marcador),
            "tokens": " ".join(aislado),
            "n_tokens": len(aislado),
            "tokenizaciones distintas": len(colas),
            "estable": colas == {aislado},
        })
    return pd.DataFrame(filas)


def round_trip_ok(tok: BPETokenizer, textos: Sequence[str]) -> tuple[bool, list[str]]:
    """Decodificar la codificacion devuelve el texto original salvo espaciado.

    El pre-tokenizador separa la puntuacion, asi que ``orders.`` vuelve como
    ``orders .``. La comparacion ignora todo el espaciado, que es exactamente lo
    que el criterio de aceptacion permite.
    """
    normalizar = lambda t: "".join(str(t).split())  # noqa: E731
    ids, _ = tok.encode(list(textos))
    fallan = [
        t for t, fila in zip(textos, ids) if normalizar(tok.decode(fila)) != normalizar(t)
    ]
    return not fallan, fallan[:5]


def ablacion_vocabulario(
    text_a: Sequence[str],
    text_b: Sequence[str],
    eval_a: Sequence[str],
    eval_b: Sequence[str],
    vocabs: Sequence[int] = VOCABS_ABLACION,
) -> pd.DataFrame:
    """Tabla de la ablacion de tamano de vocabulario (paso 8 de la Fase 6).

    Mide el efecto sobre la tokenizacion: cuantos tokens hace falta por ejemplo,
    cuantos embeddings hay que aprender y si el marcador sigue siendo estable. El
    efecto sobre la metrica final es una fila de la Fase 9.
    """
    filas = []
    for vocab in vocabs:
        tok = BPETokenizer.train_from_pairs(text_a, text_b, vocab_size=vocab)
        largos = tok.lengths(eval_a, eval_b)
        estabilidad = suffix_stability(tok, list(eval_a))
        filas.append({
            "vocab pedido": vocab,
            "vocab real": tok.vocab_size,
            "max_len (p99)": tok.max_len,
            "tokens/ejemplo (media)": round(float(largos.mean()), 1),
            "p99": int(np.percentile(largos, 99)),
            "maximo": int(largos.max()),
            "UNK %": round(100 * tok.unk_rate(eval_a, eval_b), 4),
            "tokens del marcador": round(float(estabilidad["n_tokens"].mean()), 2),
            "marcadores estables": f"{int(estabilidad['estable'].sum())}/{len(estabilidad)}",
        })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------- #
# Entrenamiento del tokenizador por defecto
# --------------------------------------------------------------------------- #

def default_path(vocab_size: int = VOCAB_SIZE) -> Path:
    return TOKENIZER_DIR / f"bpe_{vocab_size}.json"


def train_default(vocab_size: int = VOCAB_SIZE, guardar: bool = True) -> BPETokenizer:
    """Entrena el tokenizador de desarrollo con el split ``train``.

    Para la CV de la Fase 9 el tokenizador se reentrena **por fold** (cuesta
    menos de un segundo), asi que ni siquiera hace falta aceptar el sesgo de
    entrenarlo una vez sobre ``dev``.
    """
    from src.data.features import FeaturePipeline
    from src.data.load import load_raw
    from src.data.splits import load_splits

    df = load_raw()
    splits = load_splits()
    pipe = FeaturePipeline().fit(df.iloc[splits.train])
    Xtr = pipe.transform(df.iloc[splits.train])

    tok = BPETokenizer.train_from_pairs(Xtr.text_a, Xtr.text_b, vocab_size=vocab_size)
    if guardar:
        tok.save(default_path(vocab_size))
    return tok


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab", type=int, default=VOCAB_SIZE)
    parser.add_argument("--ablacion", action="store_true",
                        help="corre la tabla de tamanos de vocabulario")
    parser.add_argument("--no-guardar", action="store_true")
    args = parser.parse_args()

    from src.data.features import FeaturePipeline
    from src.data.load import load_raw
    from src.data.splits import load_splits

    pd.set_option("display.width", 200)
    df = load_raw()
    splits = load_splits()
    pipe = FeaturePipeline().fit(df.iloc[splits.train])
    Xtr, Xva = pipe.transform(df.iloc[splits.train]), pipe.transform(df.iloc[splits.valid])

    tok = BPETokenizer.train_from_pairs(Xtr.text_a, Xtr.text_b, vocab_size=args.vocab)
    if not args.no_guardar:
        tok.save(default_path(args.vocab))
        print(f"guardado en {default_path(args.vocab).relative_to(PROJECT_ROOT)}")

    print(f"\nvocabulario : {tok.vocab_size} (pedido {args.vocab})")
    print(f"fingerprint : {tok.fingerprint()}  (esperado {EXPECTED_FINGERPRINT})")

    largos_tr = tok.lengths(Xtr.text_a, Xtr.text_b)
    largos_va = tok.lengths(Xva.text_a, Xva.text_b)
    print("\n" + "=" * 100)
    print("LONGITUD EN TOKENS DEL PAR [CLS] titulo [SEP] descripcion [SEP]")
    print("=" * 100)
    print(pd.DataFrame({
        "particion": ["train", "valid"],
        "media": [largos_tr.mean().round(1), largos_va.mean().round(1)],
        "p50": [np.percentile(largos_tr, 50), np.percentile(largos_va, 50)],
        "p99": [np.percentile(largos_tr, 99), np.percentile(largos_va, 99)],
        "maximo": [largos_tr.max(), largos_va.max()],
    }).to_string(index=False))
    print(f"\nmax_len medido (p99 de train, al multiplo de 8) = {tok.max_len}")
    print(f"ejemplos truncados: train {tok.truncation_rate(Xtr.text_a, Xtr.text_b):.2%}  "
          f"valid {tok.truncation_rate(Xva.text_a, Xva.text_b):.2%}")

    print("\n" + "=" * 100)
    print("TASA DE [UNK]  (criterio de aceptacion: < 1% en validacion)")
    print("=" * 100)
    print(f"train {tok.unk_rate(Xtr.text_a, Xtr.text_b):.4%}   "
          f"valid {tok.unk_rate(Xva.text_a, Xva.text_b):.4%}")

    print("\n" + "=" * 100)
    print("ESTABILIDAD DEL MARCADOR DE REPUTACION")
    print("=" * 100)
    estabilidad = suffix_stability(tok, list(Xva.text_a))
    print(estabilidad.to_string(index=False))

    print("\n" + "=" * 100)
    print("ROUND-TRIP")
    print("=" * 100)
    ok, fallan = round_trip_ok(tok, list(Xtr.text_a[:2000]) + list(Xtr.text_b[:2000]))
    print(f"decodificar la codificacion devuelve el original salvo espaciado: {ok}")
    for t in fallan:
        print(f"  falla: {t[:90]}")

    print("\nejemplo de codificacion:")
    tokens = tok.encode_one(Xva.text_a[0], Xva.text_b[0])
    print("  " + " | ".join(tokens[:28]) + " ...")

    if args.ablacion:
        print("\n" + "=" * 100)
        print("ABLACION DE TAMANO DE VOCABULARIO")
        print("=" * 100)
        print(ablacion_vocabulario(Xtr.text_a, Xtr.text_b, Xva.text_a, Xva.text_b)
              .to_string(index=False))

    problemas = []
    if tok.unk_rate(Xva.text_a, Xva.text_b) >= 0.01:
        problemas.append("la tasa de [UNK] en validacion supera el 1%")
    if not estabilidad["estable"].all():
        problemas.append("algun marcador se tokeniza distinto segun el contexto")
    if not ok:
        problemas.append("el round-trip no reconstruye el texto")
    if problemas:
        print("\nFALLA:")
        for p in problemas:
            print(f"  - {p}")
        return 1
    print("\nOK: max_len medido, [UNK] < 1%, marcadores estables y round-trip exacto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
