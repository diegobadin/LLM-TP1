"""Criterio de aceptacion de la Fase 4, como tests.

Los tres que importan de verdad:

* ``fit`` se llama exactamente una vez y solo con train (el pipeline se niega a
  reajustarse, asi que la fuga del escalador no puede ocurrir por descuido);
* transformar dos veces el mismo dataframe da exactamente el mismo resultado;
* ninguna columna admitida en la Fase 1 quedo afuera sin justificacion escrita.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import features as F
from src.data import load as L


@pytest.fixture(scope="module")
def df():
    if not L.CSV_PATH.exists():
        pytest.skip(f"falta {L.CSV_PATH} (el dataset no se versiona)")
    return L.load_raw()


@pytest.fixture(scope="module")
def train_valid(df):
    from src.data import splits as S

    sp = S.make_splits(df)
    return df.iloc[sp.train].copy(), df.iloc[sp.valid].copy()


@pytest.fixture(scope="module")
def ajustado(train_valid):
    train, _ = train_valid
    return F.FeaturePipeline().fit(train)


# --------------------------------------------------------------------------- #
# Cobertura de las columnas: no depende del CSV
# --------------------------------------------------------------------------- #

def test_toda_columna_admitida_se_usa_o_tiene_descarte_escrito():
    usadas = set(F.NUMERICAS_BASE) | set(F.CATEGORICAS) | set(F.TEXTO) | {"dimensions_in"}
    descartadas = set(F.COLUMNAS_DESCARTADAS)
    cubiertas = usadas | descartadas
    faltan = set(L.FEATURES_ADMITIDAS) - cubiertas
    sobran = cubiertas - set(L.FEATURES_ADMITIDAS)
    assert not faltan, f"admitidas sin uso ni justificacion: {sorted(faltan)}"
    assert not sobran, f"el pipeline usa columnas que la Fase 1 no admitio: {sorted(sobran)}"


def test_cada_descarte_tiene_motivo_escrito():
    for col, motivo in F.COLUMNAS_DESCARTADAS.items():
        assert len(motivo) >= 80, f"{col}: la justificacion es demasiado corta"


def test_ninguna_columna_excluida_por_la_fase_1_entra_al_pipeline():
    usadas = set(F.NUMERICAS_BASE) | set(F.CATEGORICAS) | set(F.TEXTO) | {"dimensions_in"}
    assert not (usadas & set(L.FEATURES_EXCLUIDAS)), "entro una columna vetada"
    assert L.GROUP not in usadas and L.TARGET not in usadas


def test_las_derivadas_no_reconstruyen_una_columna_excluida():
    # filter_category y filter_storage_type son duplicados exactos de category y
    # storage_type; price_pos usa los limites del filtro, que si son admitidos.
    for nombre in F.NUMERICAS_DERIVADAS:
        assert "filter_category" not in nombre and "filter_storage" not in nombre


# --------------------------------------------------------------------------- #
# Disciplina de ajuste
# --------------------------------------------------------------------------- #

@pytest.mark.data
def test_fit_se_puede_llamar_una_sola_vez(train_valid):
    train, valid = train_valid
    pipe = F.FeaturePipeline().fit(train)
    with pytest.raises(RuntimeError, match="ya fue ajustado"):
        pipe.fit(valid)


@pytest.mark.data
def test_transform_sin_fit_falla(train_valid):
    train, _ = train_valid
    with pytest.raises(RuntimeError, match="sin ajustar"):
        F.FeaturePipeline().transform(train)


@pytest.mark.data
def test_el_escalador_se_ajusta_solo_con_train(train_valid, ajustado):
    train, valid = train_valid
    Xtr = ajustado.transform(train)
    Xva = ajustado.transform(valid)

    # Sobre train, media 0 y desvio 1 por definicion de StandardScaler.
    assert np.allclose(Xtr.num.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(Xtr.num.std(axis=0), 1, atol=1e-4)

    # Sobre valid NO: si diera exactamente 0 y 1, el escalador habria visto
    # valid, que es justamente la fuga que este pipeline evita.
    assert not np.allclose(Xva.num.mean(axis=0), 0, atol=1e-6)


@pytest.mark.data
def test_los_parametros_del_escalador_son_los_de_train(train_valid, ajustado):
    train, _ = train_valid
    derivado = F.add_derived_columns(train)
    esperado = derivado[list(ajustado.transform(train).nombres_num)].to_numpy(dtype=float)
    assert np.allclose(ajustado._scaler.mean_, esperado.mean(axis=0))
    assert np.allclose(ajustado._scaler.scale_, esperado.std(axis=0), rtol=1e-6)


@pytest.mark.data
def test_ver_otra_particion_no_cambia_el_ajuste(train_valid):
    train, valid = train_valid
    a = F.FeaturePipeline().fit(train)
    primero = a.transform(train).num.copy()
    a.transform(valid)                      # transformar valid no puede mover nada
    assert np.array_equal(a.transform(train).num, primero)


@pytest.mark.data
def test_transformar_dos_veces_da_lo_mismo(train_valid, ajustado):
    _, valid = train_valid
    A, B = ajustado.transform(valid), ajustado.transform(valid)
    assert np.array_equal(A.num, B.num)
    assert np.array_equal(A.cat, B.cat)
    assert np.array_equal(A.onehot, B.onehot)
    assert (A.text_a == B.text_a).all() and (A.text_b == B.text_b).all()
    assert np.array_equal(A.y, B.y)


@pytest.mark.data
def test_dos_pipelines_iguales_dan_lo_mismo(train_valid):
    train, valid = train_valid
    a = F.FeaturePipeline().fit(train).transform(valid)
    b = F.FeaturePipeline().fit(train).transform(valid)
    assert np.array_equal(a.num, b.num) and np.array_equal(a.onehot, b.onehot)


# --------------------------------------------------------------------------- #
# Categoricas
# --------------------------------------------------------------------------- #

@pytest.mark.data
def test_allergens_nulo_es_categoria_propia_y_no_se_imputa(train_valid, ajustado):
    train, _ = train_valid
    assert train["allergens"].isna().any(), "el fixture ya no tiene nulos que probar"
    vocab = ajustado._vocabularios["allergens"]
    assert F.ALLERGENS_NONE in vocab

    X = ajustado.transform(train)
    j = X.nombres_cat.index("allergens")
    era_nulo = train["allergens"].isna().to_numpy()
    moda = train["allergens"].mode()[0]
    assert (X.cat[era_nulo, j] == vocab[F.ALLERGENS_NONE]).all()
    assert (X.cat[era_nulo, j] != vocab[moda]).all(), "se imputo con la moda"


@pytest.mark.data
def test_un_valor_no_visto_en_train_cae_en_unk(train_valid, ajustado):
    _, valid = train_valid
    raro = valid.copy()
    raro.iloc[0, raro.columns.get_loc("brand")] = "Marca Inexistente SRL"
    X = ajustado.transform(raro)
    j = X.nombres_cat.index("brand")
    assert X.cat[0, j] == 0
    assert X.onehot[0, X.nombres_onehot.index(f"brand={F.UNK}")] == 1.0


@pytest.mark.data
def test_el_onehot_es_consistente_con_los_codigos(train_valid, ajustado):
    _, valid = train_valid
    X = ajustado.transform(valid)
    assert X.onehot.shape[1] == sum(ajustado.cardinalidades)
    # Exactamente un 1 por columna categorica y por fila.
    assert (X.onehot.sum(axis=1) == len(X.nombres_cat)).all()
    base = 0
    for j, col in enumerate(X.nombres_cat):
        k = ajustado.cardinalidades[j]
        bloque = X.onehot[:, base:base + k]
        assert np.array_equal(bloque.argmax(axis=1), X.cat[:, j])
        base += k


@pytest.mark.data
def test_las_particiones_comparten_las_columnas_del_onehot(train_valid, ajustado):
    train, valid = train_valid
    a, b = ajustado.transform(train), ajustado.transform(valid)
    assert a.nombres_onehot == b.nombres_onehot
    assert a.onehot.shape[1] == b.onehot.shape[1]


@pytest.mark.data
def test_las_cardinalidades_incluyen_el_unk(ajustado, train_valid):
    train, _ = train_valid
    for col, k in zip(ajustado.categoricas, ajustado.cardinalidades):
        esperado = train[col].fillna(F.ALLERGENS_NONE).nunique() + 1
        assert k == esperado, f"{col}: {k} != {esperado}"


# --------------------------------------------------------------------------- #
# Numericas derivadas
# --------------------------------------------------------------------------- #

@pytest.mark.data
def test_price_pos_cae_en_cero_uno(df):
    derivado = F.add_derived_columns(df)
    # Consecuencia de que filter_price_min <= price <= filter_price_max se
    # cumpla en el 100% de las filas (seccion 0.5).
    assert derivado["price_pos"].between(0, 1).all()


@pytest.mark.data
def test_las_dimensiones_se_parsean_en_todas_las_filas(df):
    dims = F.parse_dimensions(df["dimensions_in"])
    assert not dims.isna().any().any()
    assert (dims > 0).all().all()
    derivado = F.add_derived_columns(df)
    assert np.allclose(
        derivado["dim_volumen"], dims["dim_largo"] * dims["dim_ancho"] * dims["dim_alto"]
    )


@pytest.mark.data
def test_las_derivadas_no_traen_nan_ni_infinitos(df, ajustado):
    X = ajustado.transform(df)
    assert np.isfinite(X.num).all()


@pytest.mark.data
def test_las_features_de_coherencia_con_el_filtro_no_se_construyen():
    # Serian constantes (100% de cumplimiento). No hay ninguna feature cuyo
    # nombre las sugiera.
    prohibidas = {"price_en_rango", "match_category", "match_storage_type"}
    assert not (prohibidas & set(F.NUMERICAS_BASE + F.NUMERICAS_DERIVADAS))


@pytest.mark.data
def test_las_ciclicas_temporales_estan_apagadas_por_defecto(train_valid):
    train, _ = train_valid
    X = F.FeaturePipeline().fit(train).transform(train)
    assert not any(n.endswith(("_sin", "_cos")) for n in X.nombres_num)


@pytest.mark.data
def test_use_temporal_agrega_seis_ciclicas_acotadas(train_valid):
    train, _ = train_valid
    X = F.FeaturePipeline(use_temporal=True).fit(train).transform(train)
    ciclicas = [n for n in X.nombres_num if n.endswith(("_sin", "_cos"))]
    assert len(ciclicas) == 6
    crudas = F.add_temporal_features(train).to_numpy()
    assert crudas.min() >= -1.0 and crudas.max() <= 1.0


@pytest.mark.data
def test_el_scaler_quantile_es_una_alternativa_valida(train_valid):
    train, _ = train_valid
    X = F.FeaturePipeline(scaler="quantile").fit(train).transform(train)
    assert np.isfinite(X.num).all()
    with pytest.raises(ValueError, match="scaler desconocido"):
        F.FeaturePipeline(scaler="minmax").fit(train)


# --------------------------------------------------------------------------- #
# Texto y ablacion del marcador de reputacion
# --------------------------------------------------------------------------- #

@pytest.mark.data
def test_el_texto_por_defecto_conserva_el_marcador(train_valid, ajustado):
    train, _ = train_valid
    X = ajustado.transform(train)
    assert (X.text_a == train["title"].to_numpy()).all()
    assert (X.text_b == train["description"].to_numpy()).all()
    assert " [SEP] " in X.text[0]


@pytest.mark.data
def test_sin_sufijo_desaparece_el_marcador_del_titulo(train_valid):
    train, _ = train_valid
    X = F.FeaturePipeline(keep_suffix=False).fit(train).transform(train)
    assert not pd.Series(X.text_a).str.endswith(")").any()
    # La descripcion sigue intacta: son dos ablaciones distintas.
    assert (X.text_b == train["description"].to_numpy()).all()


@pytest.mark.data
def test_sin_oracion_de_reputacion_desaparece_la_segunda_codificacion(train_valid):
    train, _ = train_valid
    reputacion = {s for s in F.extract_reputation_sentence(train["description"]) if s}
    assert len(reputacion) >= 15, "el fixture no tiene oraciones de reputacion que borrar"
    X = F.FeaturePipeline(keep_reputation_sentence=False).fit(train).transform(train)
    for texto in X.text_b:
        assert not any(texto.endswith(f) for f in reputacion)


@pytest.mark.data
def test_la_ablacion_no_mutila_las_descripciones_sin_oracion_de_reputacion(train_valid):
    # 459 descripciones de las 10.000 (todas de nivel CERO) terminan en la
    # oracion estructural "Listed under ... storage.". Borrarles la ultima
    # oracion las dejaria sistematicamente mas cortas que al resto, y esa
    # diferencia se confundiria con el efecto que la ablacion quiere medir.
    train, _ = train_valid
    sin_reputacion = F.extract_reputation_sentence(train["description"]) == ""
    assert sin_reputacion.any()
    X = F.FeaturePipeline(keep_reputation_sentence=False).fit(train).transform(train)
    intactas = train.loc[sin_reputacion.to_numpy(), "description"].to_numpy()
    assert (X.text_b[sin_reputacion.to_numpy()] == intactas).all()


@pytest.mark.data
def test_la_oracion_estructural_sobrevive_a_la_ablacion(train_valid):
    # category y storage_type siguen estando en el texto: lo unico que se saca
    # es el marcador de reputacion.
    train, _ = train_valid
    X = F.FeaturePipeline(keep_reputation_sentence=False).fit(train).transform(train)
    assert pd.Series(X.text_b).str.contains("Listed under").all()


@pytest.mark.data
def test_la_ablacion_completa_borra_las_dos_codificaciones_del_nivel(train_valid):
    train, _ = train_valid
    pipe = F.FeaturePipeline(keep_suffix=False, keep_reputation_sentence=False)
    X = pipe.fit(train).transform(train)
    marcadores = [s for s in L.extract_suffix(train["title"]).unique() if s != L.SUFFIX_NONE]
    junto = " || ".join(X.text[:500])
    for marcador in marcadores:
        assert marcador not in junto, f"quedo el marcador {marcador!r} en el texto"


@pytest.mark.data
def test_la_ablacion_del_marcador_no_toca_las_tabulares(train_valid):
    train, _ = train_valid
    con = F.FeaturePipeline().fit(train).transform(train)
    sin = F.FeaturePipeline(keep_suffix=False, keep_reputation_sentence=False).fit(train).transform(train)
    assert np.array_equal(con.num, sin.num)
    assert np.array_equal(con.cat, sin.cat)


# --------------------------------------------------------------------------- #
# Target y alineacion
# --------------------------------------------------------------------------- #

@pytest.mark.data
def test_el_target_queda_alineado_con_las_filas(train_valid, ajustado):
    _, valid = train_valid
    X = ajustado.transform(valid)
    assert np.array_equal(X.y, valid[L.TARGET].astype(int).to_numpy())
    assert len(X) == len(valid) == len(X.text_a) == len(X.cat)


@pytest.mark.data
def test_transformar_sin_target_no_falla(train_valid, ajustado):
    _, valid = train_valid
    X = ajustado.transform(valid.drop(columns=[L.TARGET]))
    assert X.y is None
    assert len(X) == len(valid)


@pytest.mark.data
def test_el_sufijo_esta_disponible_como_categorica_opcional(train_valid):
    train, _ = train_valid
    pipe = F.FeaturePipeline(categoricas=F.CATEGORICAS + F.CATEGORICAS_OPCIONALES)
    X = pipe.fit(train).transform(train)
    assert "sufijo" in X.nombres_cat
    # 20 marcadores (incluido "sin sufijo") + [UNK].
    assert pipe.cardinalidades[X.nombres_cat.index("sufijo")] == 21
