"""Los diez tests obligatorios de la Fase 7, mas los que cubren NaN y post-LN.

Los bugs de mascara y de reshape en la atencion **no producen errores**:
producen un modelo que entrena y da metricas plausiblemente malas. Se descubren
tarde y son carisimos. Estos tests cubren los modos de falla conocidos y se
escriben antes de entrenar nada.

El mas valioso es el de equivalencia: copia los pesos de la implementacion
propia a una ``nn.MultiheadAttention`` y compara salidas. Convierte a la
libreria en un verificador en lugar de un reemplazo.
"""

from __future__ import annotations

import pytest
import torch

from src.model.attention import MASK_VALUE, MultiHeadSelfAttention, additive_mask
from src.model.encoder import EncoderBlock, TransformerEncoder
from src.model.heads import masked_mean
from src.utils.seed import set_seed

B, L, D, H = 4, 9, 32, 4
VOCAB, MAX_LEN = 60, L


@pytest.fixture(autouse=True)
def semilla():
    set_seed(0)


@pytest.fixture()
def x():
    return torch.randn(B, L, D)


@pytest.fixture()
def mask():
    """Mascara de padding: True = token real. Las 3 ultimas posiciones son [PAD]."""
    m = torch.ones(B, L, dtype=torch.bool)
    m[:, -3:] = False
    return m


@pytest.fixture()
def attn():
    return MultiHeadSelfAttention(D, H).eval()


def _encoder(**kwargs) -> TransformerEncoder:
    defaults = dict(vocab_size=VOCAB, d_model=D, n_heads=H, n_layers=2, d_ff=2 * D,
                    max_len=MAX_LEN, dropout=0.0)
    return TransformerEncoder(**(defaults | kwargs)).eval()


# --------------------------------------------------------------------------- #
# 1. Equivalencia con nn.MultiheadAttention  (el test mas valioso de la lista)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("con_mascara", [False, True], ids=["sin_mascara", "con_mascara"])
def test_equivalencia_con_nn_multihead_attention(attn, x, mask, con_mascara):
    padding = mask if con_mascara else None
    propia = attn(x, padding)

    mha = attn.to_torch_mha().eval()
    # key_padding_mask de torch: True = posicion A IGNORAR, la inversa de la
    # convencion del proyecto. La mitad de los fallos de este test son esto.
    referencia, _ = mha(
        x, x, x,
        key_padding_mask=None if padding is None else ~padding,
        need_weights=False,
    )
    assert torch.allclose(propia, referencia, atol=1e-5), (
        f"maxima diferencia {(propia - referencia).abs().max().item():.2e}"
    )


def test_el_modo_torch_mha_coincide_con_scratch(x, mask):
    propia = MultiHeadSelfAttention(D, H, attn_impl="scratch").eval()
    libreria = MultiHeadSelfAttention(D, H, attn_impl="torch_mha").eval()
    libreria.load_state_dict(propia.state_dict())
    assert torch.allclose(propia(x, mask), libreria(x, mask), atol=1e-5)


# --------------------------------------------------------------------------- #
# 2. Formas
# --------------------------------------------------------------------------- #

def test_el_bloque_encoder_preserva_la_forma(x, mask):
    bloque = EncoderBlock(D, H, 2 * D, dropout=0.0).eval()
    assert bloque(x, mask).shape == x.shape


def test_las_cabezas_reparten_d_model(attn, x):
    attn(x)
    assert attn.d_head * attn.n_heads == D
    assert attn.last_attn_weights.shape == (B, H, L, L)


# --------------------------------------------------------------------------- #
# 3 y 4. Softmax y mascara
# --------------------------------------------------------------------------- #

def test_los_pesos_de_atencion_suman_uno(attn, x, mask):
    attn(x, mask)
    sumas = attn.last_attn_weights.sum(dim=-1)
    # dim=-1 son las claves: si el softmax estuviera en otro eje, esto no da 1.
    assert torch.allclose(sumas, torch.ones_like(sumas), atol=1e-6)


def test_el_peso_hacia_el_padding_es_exactamente_cero(attn, x, mask):
    attn(x, mask)
    pesos = attn.last_attn_weights                     # (B, H, Lq, Lk)
    hacia_padding = pesos[..., ~mask[0]]
    assert torch.count_nonzero(hacia_padding) == 0, (
        "la mascara se esta sumando despues del softmax, o no se esta sumando"
    )


def test_cambiar_un_token_de_padding_no_cambia_las_posiciones_validas(mask):
    encoder = _encoder()
    ids = torch.randint(1, VOCAB, (B, L))
    ids[:, -3:] = 0                                    # [PAD]
    salida = encoder(ids, mask)

    alterado = ids.clone()
    alterado[:, -3:] = 7                               # cualquier token real
    salida_alterada = encoder(alterado, mask)

    validas = mask[0]
    assert torch.allclose(salida[:, validas], salida_alterada[:, validas], atol=1e-6)


def test_una_fila_enteramente_enmascarada_no_produce_nan(attn, x):
    vacia = torch.zeros(B, L, dtype=torch.bool)
    salida = attn(x, vacia)
    # Con -inf en toda la fila, softmax daria NaN y el NaN se propagaria por los
    # gradientes a todo el batch. Con un finito grande, da uniforme.
    assert torch.isfinite(salida).all()
    assert MASK_VALUE < 0 and MASK_VALUE > float("-inf")


def test_additive_mask_valida_la_forma():
    with pytest.raises(ValueError, match="batch, seq"):
        additive_mask(torch.ones(B, L, 3, dtype=torch.bool), torch.float32)


# --------------------------------------------------------------------------- #
# 5 y 6. Invariancia a permutaciones y positional encoding
# --------------------------------------------------------------------------- #

def _mean_pool_sin_pe(encoder, ids):
    todo_valido = torch.ones_like(ids, dtype=torch.bool)
    return masked_mean(encoder(ids, todo_valido), todo_valido)


def test_sin_positional_encoding_permutar_no_cambia_el_mean_pooling():
    encoder = _encoder(pos_encoding="none")
    ids = torch.randint(1, VOCAB, (B, L))
    permutado = ids.clone()
    permutado[:, [2, 5]] = permutado[:, [5, 2]]
    # La atencion es equivariante a permutaciones y el promedio es invariante:
    # sin PE el modelo ve un conjunto de tokens, no una secuencia.
    assert torch.allclose(
        _mean_pool_sin_pe(encoder, ids), _mean_pool_sin_pe(encoder, permutado), atol=1e-5
    )


@pytest.mark.parametrize("pe", ["sinusoidal", "learned"])
def test_con_positional_encoding_la_misma_permutacion_si_cambia(pe):
    encoder = _encoder(pos_encoding=pe)
    ids = torch.randint(1, VOCAB, (B, L))
    permutado = ids.clone()
    permutado[:, [2, 5]] = permutado[:, [5, 2]]
    assert not torch.allclose(
        _mean_pool_sin_pe(encoder, ids), _mean_pool_sin_pe(encoder, permutado), atol=1e-4
    )


# --------------------------------------------------------------------------- #
# 7. Configuracion invalida
# --------------------------------------------------------------------------- #

def test_d_model_no_divisible_por_n_heads_falla_explicito():
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(d_model=64, n_heads=5)


def test_una_implementacion_inexistente_falla_explicito():
    with pytest.raises(ValueError, match="attn_impl desconocido"):
        MultiHeadSelfAttention(D, H, attn_impl="flash")


# --------------------------------------------------------------------------- #
# 8. Sin filtracion entre elementos del batch
# --------------------------------------------------------------------------- #

def test_batch_de_uno_y_batch_de_ocho_dan_lo_mismo(mask):
    encoder = _encoder()
    ids = torch.randint(1, VOCAB, (8, L))
    ids[:, -3:] = 0
    m = torch.ones(8, L, dtype=torch.bool)
    m[:, -3:] = False

    solo = encoder(ids[3:4], m[3:4])
    junto = encoder(ids, m)[3:4]
    assert torch.allclose(solo, junto, atol=1e-6), "hay filtracion entre elementos del batch"


# --------------------------------------------------------------------------- #
# 9. Consistencia entre implementaciones
# --------------------------------------------------------------------------- #

def test_scratch_y_torch_sdpa_dan_la_misma_salida(x, mask):
    propia = MultiHeadSelfAttention(D, H, attn_impl="scratch").eval()
    rapida = MultiHeadSelfAttention(D, H, attn_impl="torch_sdpa").eval()
    rapida.load_state_dict(propia.state_dict())
    assert torch.allclose(propia(x, mask), rapida(x, mask), atol=1e-5)
    # El camino rapido no expone los pesos: por eso el modelo que se reporta y
    # del que salen los mapas de atencion es el propio.
    assert rapida.last_attn_weights is None
    assert propia.last_attn_weights is not None


# --------------------------------------------------------------------------- #
# 10. Forward y backward sanos
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("norm", ["pre_ln", "post_ln"])
def test_forward_y_backward_sin_nan(norm):
    from src.utils.seed import DEVICE

    encoder = TransformerEncoder(
        vocab_size=VOCAB, d_model=D, n_heads=H, n_layers=2, d_ff=2 * D,
        max_len=MAX_LEN, dropout=0.1, norm=norm,
    ).to(DEVICE)
    ids = torch.randint(1, VOCAB, (B, L), device=DEVICE)
    ids[:, -3:] = 0
    m = torch.ones(B, L, dtype=torch.bool, device=DEVICE)
    m[:, -3:] = False

    salida = encoder(ids, m)
    assert torch.isfinite(salida).all()
    salida.sum().backward()
    for nombre, p in encoder.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"gradiente no finito en {nombre}"


def test_el_embedding_de_padding_arranca_en_cero_y_sin_gradiente():
    encoder = _encoder()
    tabla = encoder.embeddings.tokens
    assert torch.count_nonzero(tabla.weight[0]) == 0
    ids = torch.randint(1, VOCAB, (B, L))
    ids[:, -2:] = 0
    encoder.train()
    encoder(ids).sum().backward()
    # padding_idx=0 hace que la fila de [PAD] no reciba gradiente.
    assert torch.count_nonzero(tabla.weight.grad[0]) == 0


def test_los_mapas_de_atencion_salen_por_capa():
    encoder = _encoder(n_layers=3)
    ids = torch.randint(1, VOCAB, (B, L))
    encoder(ids)
    mapas = encoder.attention_maps()
    assert len(mapas) == 3
    assert all(m.shape == (B, H, L, L) for m in mapas)
    assert all(not m.requires_grad for m in mapas), "los mapas deben venir desprendidos"


def test_las_configuraciones_invalidas_del_bloque_fallan():
    with pytest.raises(ValueError, match="norm desconocida"):
        EncoderBlock(D, H, 2 * D, norm="batch_norm")
    with pytest.raises(ValueError, match="pos_encoding desconocido"):
        _encoder(pos_encoding="rope")
