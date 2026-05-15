"""
HKDF-SHA256 utilities to derive hierarchical keys from a master key K.

Provides helpers to:
- derive per-layer keys K_i via HKDF expand with context info
- split K_i into attention and ffn subkeys
- map subkeys to Arnold key parameters and permutation seeds
"""

import hashlib
import hmac
from typing import Tuple


def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()


def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    t = b""
    okm = b""
    i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]


def derive_layer_key(master_key: bytes, layer_index: int, salt: bytes = b"PUF-OWL-ViT") -> bytes:
    prk = hkdf_extract(salt, master_key)
    info = b"layer:" + layer_index.to_bytes(2, "big")
    return hkdf_expand(prk, info, 32)


def split_attn_ffn(layer_key: bytes) -> Tuple[bytes, bytes]:
    # Simple split: first 16 bytes for attention, last 16 for ffn
    return layer_key[:16], layer_key[16:32]


def subkey_to_arnold_params(
    subkey: bytes,
    matrix_size: int,
    n_override: int = None,
) -> Tuple[int, int, int, int, int]:
    # Use canonical invertible form a=1, b=p, c=q, d=p*q+1 (mod matrix_size).
    # ``n_override`` pins N to a fixed iteration count (used by the demo CLIs
    # to make the secure-inference slowdown controllable). When None, sample
    # in [3..7] from the subkey for reasonable scrambling.
    if n_override is not None:
        N = int(n_override)
    else:
        N = 3 + (subkey[0] % 5)
    # Derive p, q from subkey bytes for good spread
    p_seed = int.from_bytes(subkey[1:5], "big", signed=False)
    q_seed = int.from_bytes(subkey[5:9], "big", signed=False)
    p = p_seed % matrix_size
    q = q_seed % matrix_size
    if p == 0:
        p = 1
    if q == 0:
        q = 1
    a = 1 % matrix_size
    b = p
    c = q
    d = (p * q + 1) % matrix_size
    return (N, a, b, c, d)


def subkey_to_perm_password(subkey: bytes) -> str:
    # Derive a deterministic string password from subkey
    return hashlib.sha256(subkey).hexdigest()
