"""Criterio de aceptacion de la Fase 1, como tests.

La auditoria de fugas no sirve de nada si queda como prosa en un notebook y el
codigo despues hace otra cosa. Estos tests la vuelven ejecutable.

Los tests marcados con ``@pytest.mark.data`` necesitan el CSV, que no se
versiona; se saltean solos si no esta.
"""

from __future__ import annotations

import pytest

from src.data import load as L


# --------------------------------------------------------------------------- #
# Auditoria: no dependen del CSV
# --------------------------------------------------------------------------- #

def test_la_auditoria_cubre_las_22_columnas():
    assert len(L.COLUMN_AUDIT) == 22
    nombres = [e.columna for e in L.COLUMN_AUDIT]
    assert len(set(nombres)) == 22, "hay columnas repetidas en COLUMN_AUDIT"


def test_todos_los_veredictos_son_validos():
    validos = {"ADMITIDA", "EXCLUIDA", "TARGET", "GRUPO"}
    for e in L.COLUMN_AUDIT:
        assert e.veredicto in validos, f"{e.columna}: veredicto {e.veredicto!r}"


def test_toda_columna_tiene_motivo_escrito():
    for e in L.COLUMN_AUDIT:
        assert len(e.motivo) >= 20, f"{e.columna} no tiene justificacion escrita"
        assert e.momento, f"{e.columna} no declara cuando queda determinado su valor"


def test_cart_esta_excluida_con_justificacion():
    cart = next(e for e in L.COLUMN_AUDIT if e.columna == "cart")
    assert cart.veredicto == "EXCLUIDA"
    assert cart.disponible is False
    assert "FUGA" in cart.motivo.upper()
    assert "cart" not in L.FEATURES_ADMITIDAS


def test_query_id_es_clave_de_agrupamiento_no_feature():
    grupo = next(e for e in L.COLUMN_AUDIT if e.columna == L.GROUP)
    assert grupo.veredicto == "GRUPO"
    assert L.GROUP not in L.FEATURES_ADMITIDAS


def test_bought_es_el_target():
    target = next(e for e in L.COLUMN_AUDIT if e.columna == L.TARGET)
    assert target.veredicto == "TARGET"
    assert L.TARGET not in L.FEATURES_ADMITIDAS


def test_ninguna_excluida_entro_en_features_admitidas():
    assert not (set(L.FEATURES_EXCLUIDAS) & set(L.FEATURES_ADMITIDAS))


def test_las_columnas_duplicadas_del_filtro_estan_excluidas():
    # filter_category y filter_storage_type son identicas a category y
    # storage_type en el 100% de las filas (seccion 0.5 del plan).
    for col in ("filter_category", "filter_storage_type"):
        assert col in L.FEATURES_EXCLUIDAS


def test_el_momento_de_la_prediccion_esta_definido():
    assert "antes de mostrar" in L.PREDICTION_MOMENT.lower() or "previo a mostrar" in L.PREDICTION_MOMENT.lower()


def test_toda_columna_no_disponible_esta_fuera_de_las_features():
    # La regla de oro: si su valor no existe al momento de la prediccion,
    # no puede ser feature. Da igual cuanta metrica aporte.
    for e in L.COLUMN_AUDIT:
        if not e.disponible:
            assert e.veredicto != "ADMITIDA", f"{e.columna} es posterior y quedo admitida"


# --------------------------------------------------------------------------- #
# Sobre el CSV
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def df():
    if not L.CSV_PATH.exists():
        pytest.skip(f"falta {L.CSV_PATH} (el dataset no se versiona)")
    return L.load_raw()


@pytest.mark.data
def test_la_auditoria_coincide_con_el_csv(df):
    assert list(df.columns) == list(L.EXPECTED_COLUMNS)


@pytest.mark.data
def test_timestamp_es_datetime_utc(df):
    assert str(df["timestamp"].dtype).endswith("UTC]")


@pytest.mark.data
def test_se_reproducen_los_hechos_de_la_seccion_0(df):
    check = L.verify_expected_facts(df)
    fallan = check.loc[~check["ok"]]
    assert fallan.empty, f"no se reproducen:\n{fallan.to_string(index=False)}"


@pytest.mark.data
def test_cart_es_condicion_necesaria_de_bought(df):
    # P(bought = 1 | cart = False) == 0 exacto. No es correlacion alta: es una
    # implicacion logica, y es la evidencia dura de la fuga.
    assert df.loc[~df["cart"], L.TARGET].sum() == 0


@pytest.mark.data
def test_las_features_de_coherencia_con_el_filtro_serian_constantes(df):
    coherencia = L.filter_coherence(df)
    assert coherencia["feature constante"].all(), (
        "alguna condicion del filtro dejo de cumplirse al 100%; revisar la "
        "decision D6 antes de seguir"
    )


@pytest.mark.data
def test_solo_allergens_tiene_nulos(df):
    con_nulos = df.columns[df.isna().any()].tolist()
    assert con_nulos == ["allergens"]


@pytest.mark.data
def test_hay_20_valores_de_sufijo_incluido_el_vacio(df):
    sufijos = L.extract_suffix(df["title"])
    assert sufijos.nunique() == 20
    # 511 titulos no traen marcador. No es un faltante: su tasa de compra es 0,
    # igual que la de los otros 11 sufijos del nivel CERO.
    sin_marcador = sufijos == L.SUFFIX_NONE
    assert int(sin_marcador.sum()) == 511
    assert df.loc[sin_marcador, L.TARGET].sum() == 0


@pytest.mark.data
def test_strip_suffix_no_deja_el_marcador(df):
    sin_sufijo = L.strip_suffix(df["title"])
    assert not sin_sufijo.str.endswith(")").any()
    # Solo se acorta el titulo de las filas que tenian marcador; el resto queda
    # intacto. Es lo que hace comparable la ablacion "con / sin sufijo".
    tenia_sufijo = L.extract_suffix(df["title"]) != L.SUFFIX_NONE
    assert (sin_sufijo[tenia_sufijo].str.len() < df.loc[tenia_sufijo, "title"].str.len()).all()
    assert (sin_sufijo[~tenia_sufijo] == df.loc[~tenia_sufijo, "title"]).all()
