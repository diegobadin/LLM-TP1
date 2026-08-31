"""Criterio de aceptacion de la Fase 6, como tests.

Los tres que pide el plan:

* ``max_len`` esta **medido** (percentil 99 sobre train), no elegido a ojo;
* la tasa de ``[UNK]`` en validacion es menor al 1%;
* el round-trip devuelve el texto original salvo espaciado.

Mas los dos que evitan bugs caros aguas abajo: que el marcador de reputacion se
tokenice siempre igual (ahi esta el 96% de la senal) y que el padding quede
marcado en la mascara, no escondido entre los tokens reales.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data import load as L
from src.tokenizer import bpe as T

CORPUS = [
    "Cedar House Steamable Pepperoni Pizza - 10 oz (Best Seller)",
    "Blue Cart Lemon Scent Toilet Paper - 6 ct (New Listing)",
    "Green Fork Organic Baby Carrots - 1 lb (Top Rated)",
    "Harvest Lane Whole Wheat Bread - 20 oz (Clearance Listing)",
] * 20
DESCRIPCIONES = [
    "Steamable pepperoni pizza in a 10 oz package for online grocery orders. "
    "Listed under frozen and intended for frozen storage. "
    "Frequently reordered by returning customers.",
    "Lemon scent toilet paper in a 6 ct package for online grocery orders. "
    "Listed under household and intended for ambient storage. "
    "Recently added to the online catalog.",
    "Organic baby carrots in a 1 lb package for online grocery orders. "
    "Listed under produce and intended for refrigerated storage. "
    "Consistently praised in customer feedback.",
    "Whole wheat bread in a 20 oz package for online grocery orders. "
    "Listed under bakery and intended for ambient storage. "
    "Stocked as part of the regular lineup.",
] * 20


@pytest.fixture(scope="module")
def tok():
    """Tokenizador chico entrenado con el corpus sintetico de arriba."""
    return T.BPETokenizer.train_from_pairs(CORPUS, DESCRIPCIONES, vocab_size=300)


# --------------------------------------------------------------------------- #
# Tokens especiales y formato de entrada
# --------------------------------------------------------------------------- #

def test_los_ids_especiales_estan_fijos(tok):
    vocab = tok.tokenizer.get_vocab()
    assert vocab["[PAD]"] == T.PAD_ID == 0
    assert vocab["[UNK]"] == T.UNK_ID == 1
    assert vocab["[CLS]"] == T.CLS_ID == 2
    assert vocab["[SEP]"] == T.SEP_ID == 3


def test_el_formato_es_cls_titulo_sep_descripcion_sep(tok):
    tokens = tok.encode_one(CORPUS[0], DESCRIPCIONES[0])
    assert tokens[0] == "[CLS]"
    assert tokens[-1] == "[SEP]"
    assert tokens.count("[SEP]") == 2, "falta el separador entre titulo y descripcion"


def test_un_solo_campo_usa_el_template_simple(tok):
    tokens = tok.encode_one(CORPUS[0])
    assert tokens[0] == "[CLS]" and tokens[-1] == "[SEP]"
    assert tokens.count("[SEP]") == 1


# --------------------------------------------------------------------------- #
# max_len, padding y mascara
# --------------------------------------------------------------------------- #

def test_max_len_es_el_percentil_99_redondeado(tok):
    largos = tok.lengths(CORPUS, DESCRIPCIONES)
    p99 = np.percentile(largos, 99)
    assert tok.max_len >= p99
    assert tok.max_len % 8 == 0, "se redondea al multiplo de 8 siguiente"
    assert tok.max_len < 8 + p99 + 8, "quedo mucho mas grande que el p99"


def test_max_len_no_es_512_por_costumbre(tok):
    # La atencion es cuadratica en la longitud: 512 sobre secuencias de ~50
    # tokens multiplicaria el costo por 100 sin ganar nada.
    assert tok.max_len <= 128


def test_encode_devuelve_ids_y_mascara_de_la_forma_esperada(tok):
    ids, mask = tok.encode(CORPUS, DESCRIPCIONES)
    assert ids.shape == (len(CORPUS), tok.max_len) == mask.shape
    assert ids.dtype == np.int64 and mask.dtype == bool


def test_la_mascara_marca_exactamente_los_tokens_reales(tok):
    ids, mask = tok.encode(CORPUS, DESCRIPCIONES)
    # Convencion propia: True = token real.
    assert np.array_equal(mask, ids != T.PAD_ID)
    largos = tok.lengths(CORPUS, DESCRIPCIONES)
    assert np.array_equal(mask.sum(axis=1), np.minimum(largos, tok.max_len))


def test_el_padding_va_al_final(tok):
    _, mask = tok.encode(CORPUS[:1], DESCRIPCIONES[:1])
    fila = mask[0]
    # Una vez que empieza el padding, no vuelve a haber tokens reales.
    assert not np.any(fila[np.argmin(fila):]) or fila.all()


def test_un_texto_larguisimo_se_trunca_a_max_len(tok):
    largo = " ".join(CORPUS) * 5
    ids, mask = tok.encode([largo])
    assert ids.shape[1] == tok.max_len
    assert mask.all(), "un texto mas largo que max_len no deberia tener padding"


# --------------------------------------------------------------------------- #
# [UNK] y round-trip
# --------------------------------------------------------------------------- #

def test_round_trip_salvo_espaciado(tok):
    ok, fallan = T.round_trip_ok(tok, CORPUS[:4] + DESCRIPCIONES[:4])
    assert ok, f"no reconstruye: {fallan}"


def test_un_caracter_no_visto_cae_en_unk(tok):
    ids, _ = tok.encode(["Cedar House 中文 (Best Seller)"])
    assert (ids == T.UNK_ID).any()


def test_la_tasa_de_unk_del_corpus_conocido_es_cero(tok):
    assert tok.unk_rate(CORPUS, DESCRIPCIONES) == 0.0


# --------------------------------------------------------------------------- #
# Estabilidad del marcador
# --------------------------------------------------------------------------- #

def test_el_marcador_se_tokeniza_igual_en_todo_contexto(tok):
    estabilidad = T.suffix_stability(tok, CORPUS)
    inestables = estabilidad.loc[~estabilidad["estable"]]
    assert inestables.empty, f"marcadores inestables:\n{inestables.to_string(index=False)}"


# --------------------------------------------------------------------------- #
# Entrenamiento y persistencia
# --------------------------------------------------------------------------- #

def test_el_entrenamiento_es_determinista():
    a = T.BPETokenizer.train_from_pairs(CORPUS, DESCRIPCIONES, vocab_size=300)
    b = T.BPETokenizer.train_from_pairs(CORPUS, DESCRIPCIONES, vocab_size=300)
    assert a.fingerprint() == b.fingerprint()
    assert a.max_len == b.max_len


def test_el_vocabulario_sale_solo_del_corpus_que_se_le_pasa():
    tok = T.BPETokenizer.train(["aaa bbb", "aaa ccc"], vocab_size=300, max_len=8)
    vocab = tok.tokenizer.get_vocab()
    assert "Seller" not in vocab
    # Solo los 4 especiales, las letras vistas y los merges posibles.
    assert tok.vocab_size < 20


def test_pedir_mas_vocabulario_del_que_el_corpus_permite_satura():
    # El corpus de train tiene 622 tipos de palabra: el BPE no puede pasar de
    # 1.355 tokens, y por eso la grilla de ablacion se reescalo hacia abajo.
    chico = T.BPETokenizer.train_from_pairs(CORPUS, DESCRIPCIONES, vocab_size=200)
    grande = T.BPETokenizer.train_from_pairs(CORPUS, DESCRIPCIONES, vocab_size=100_000)
    assert chico.vocab_size == 200
    assert grande.vocab_size < 1_000


def test_la_grilla_de_ablacion_no_repite_tokenizadores():
    # Todo pedido >= VOCAB_SATURACION devuelve el mismo tokenizador, asi que la
    # grilla puede tener a lo sumo un punto en la zona saturada: si tuviera dos,
    # dos filas de la tabla de ablacion serian la misma corrida con otro nombre.
    saturados = [v for v in T.VOCABS_ABLACION if v >= T.VOCAB_SATURACION]
    assert len(saturados) <= 1, f"puntos redundantes en la grilla: {saturados}"
    assert len(set(T.VOCABS_ABLACION)) == len(T.VOCABS_ABLACION)


def test_save_y_load_preservan_todo(tok, tmp_path):
    ruta = tmp_path / "bpe.json"
    tok.save(ruta)
    leido = T.BPETokenizer.load(ruta)
    assert leido.max_len == tok.max_len
    assert leido.fingerprint() == tok.fingerprint()
    a, _ = tok.encode(CORPUS[:3], DESCRIPCIONES[:3])
    b, _ = leido.encode(CORPUS[:3], DESCRIPCIONES[:3])
    assert np.array_equal(a, b)


def test_load_falla_con_mensaje_util_si_no_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m src.tokenizer.bpe"):
        T.BPETokenizer.load(tmp_path / "no-existe.json")


def test_encode_rechaza_pares_desalineados(tok):
    with pytest.raises(ValueError, match="text_a tiene"):
        tok.encode(CORPUS[:3], DESCRIPCIONES[:2])


# --------------------------------------------------------------------------- #
# Sobre el dataset real
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def real():
    """Tokenizador por defecto entrenado con el split ``train``, y valid."""
    if not L.CSV_PATH.exists():
        pytest.skip(f"falta {L.CSV_PATH} (el dataset no se versiona)")
    from src.data.features import FeaturePipeline
    from src.data.splits import load_splits

    df = L.load_raw()
    splits = load_splits()
    pipe = FeaturePipeline().fit(df.iloc[splits.train])
    Xtr, Xva = pipe.transform(df.iloc[splits.train]), pipe.transform(df.iloc[splits.valid])
    tok = T.BPETokenizer.train_from_pairs(Xtr.text_a, Xtr.text_b)
    return tok, Xtr, Xva


@pytest.mark.data
def test_el_tokenizador_por_defecto_reproduce_su_fingerprint(real):
    tok, _, _ = real
    assert tok.fingerprint() == T.EXPECTED_FINGERPRINT
    assert tok.max_len == T.MAX_LEN
    assert tok.vocab_size == T.VOCAB_SATURACION


@pytest.mark.data
def test_la_tasa_de_unk_en_validacion_es_menor_al_uno_por_ciento(real):
    tok, _, Xva = real
    assert tok.unk_rate(Xva.text_a, Xva.text_b) < 0.01


@pytest.mark.data
def test_ningun_ejemplo_real_queda_truncado(real):
    tok, Xtr, Xva = real
    assert tok.truncation_rate(Xtr.text_a, Xtr.text_b) == 0.0
    assert tok.truncation_rate(Xva.text_a, Xva.text_b) == 0.0


@pytest.mark.data
def test_los_veinte_marcadores_reales_son_estables(real):
    tok, _, Xva = real
    estabilidad = T.suffix_stability(tok, list(Xva.text_a))
    assert len(estabilidad) == 19, "faltan marcadores (19 + el caso sin marcador)"
    assert estabilidad["estable"].all()


@pytest.mark.data
def test_el_round_trip_funciona_sobre_el_texto_real(real):
    tok, Xtr, _ = real
    ok, fallan = T.round_trip_ok(tok, list(Xtr.text_a[:500]) + list(Xtr.text_b[:500]))
    assert ok, f"no reconstruye: {fallan}"


@pytest.mark.data
def test_el_tokenizador_no_se_entreno_con_valid(real):
    # Comprobacion indirecta pero concreta: entrenarlo tambien con valid cambia
    # el vocabulario, asi que los dos fingerprints tienen que diferir.
    tok, Xtr, Xva = real
    con_valid = T.BPETokenizer.train_from_pairs(
        list(Xtr.text_a) + list(Xva.text_a), list(Xtr.text_b) + list(Xva.text_b)
    )
    assert con_valid.fingerprint() != tok.fingerprint()
