"""
Matematiksel morfoloji — yapısal elemanlar, genişletme, aşındırma, açma, kapama.
Tüm algoritmalar sıfırdan yalnızca NumPy kullanılarak uygulanmıştır — scipy.ndimage yok.

Genel API
---------
create_structuring_element(shape, size) -> np.ndarray   (ikili YE)
dilate(img, kernel, ...)                -> np.ndarray
erode(img, kernel, ...)                 -> np.ndarray
morphological_open(img, kernel, ...)    -> np.ndarray   (önce aşındır, sonra genişlet)
morphological_close(img, kernel, ...)   -> np.ndarray   (önce genişlet, sonra aşındır)
gradient(img, kernel_size, shape)       -> np.ndarray   (genişlet - aşındır)

app.py için geriye dönük uyumluluk takma adları
------------------------------------------------
opening(img, kernel_size, shape)
closing(img, kernel_size, shape)
"""
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# 4. create_structuring_element
# ─────────────────────────────────────────────────────────────────────────

def create_structuring_element(shape: str = 'rect', size: int = 3) -> np.ndarray:
    """
    İstenen geometriye sahip ikili (0/1) yapısal eleman oluşturur.

    Şekiller
    --------
    'rect'    Tüm girişler 1.
    'cross'   Orta satır ve orta sütun 1, geri kalanlar 0 (artı işareti).
    'ellipse' Birim elipsin içindeyse piksel (r,c) = 1:
                  ((r-cy)/(size/2))^2 + ((c-cx)/(size/2))^2 <= 1

    Parametreler
    ------------
    shape : str  'rect', 'cross', 'ellipse' değerlerinden biri. Varsayılan 'rect'.
    size  : int  Kenar uzunluğu (tek sayıya zorlanır, min 3).

    Döndürür
    --------
    np.ndarray  (size, size) şeklinde ikili uint8 dizisi.
    """
    size = max(3, size | 1)
    se   = np.zeros((size, size), dtype=np.uint8)
    cy = cx = size // 2

    if shape == 'rect':
        se[:] = 1

    elif shape == 'cross':
        se[cy, :] = 1    # yatay kol
        se[:, cx] = 1    # dikey kol

    elif shape == 'ellipse':
        rows = np.arange(size, dtype=np.float64)
        cols = np.arange(size, dtype=np.float64)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        half   = size / 2.0
        inside = ((rr - cy) / half) ** 2 + ((cc - cx) / half) ** 2 <= 1.0
        se[inside] = 1
    else:
        se[:] = 1   # bilinmeyen şekil → dikdörtgen

    return se


# ─────────────────────────────────────────────────────────────────────────
# Dahili kanal bazlı işlemler
# ─────────────────────────────────────────────────────────────────────────

def _dilate_plane(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Tek bir uint8 kanalın gri tonlamalı genişlemesi.

    Tanım:
        (f ⊕ B)[y,x] = max_{b in B}  f[y-by, x-bx]

    Uygulama: etkin YE konumlarını sıfır dolgulu görüntü üzerinde kaydır,
    tüm kaymalar arasında eleman bazlı maksimum al.
    Sıfır dolgu (siyah kenar) — genişleme dışarıdan içeriye sızmaz.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(
        plane.astype(np.int16),
        ((ph, ph), (pw, pw)),
        mode='constant', constant_values=0,
    )
    h, w   = plane.shape
    out    = np.zeros((h, w), dtype=np.int16)
    active = np.argwhere(kernel == 1)

    for dr, dc in active:
        out = np.maximum(out, padded[dr:dr + h, dc:dc + w])

    return np.clip(out, 0, 255).astype(np.uint8)


def _erode_plane(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Tek bir uint8 kanalın gri tonlamalı aşındırması.

    Tanım:
        (f ⊖ B)[y,x] = min_{b in B}  f[y+by, x+bx]

    Uygulama: etkin YE konumlarını 255 dolgulu görüntü üzerinde kaydır,
    tüm kaymalar arasında eleman bazlı minimum al.
    255 dolgu (beyaz kenar) — aşındırma kenarlarda küçülmez.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(
        plane.astype(np.int16),
        ((ph, ph), (pw, pw)),
        mode='constant', constant_values=255,
    )
    h, w   = plane.shape
    out    = np.full((h, w), 255, dtype=np.int16)
    active = np.argwhere(kernel == 1)

    for dr, dc in active:
        out = np.minimum(out, padded[dr:dr + h, dc:dc + w])

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply(img: np.ndarray, func, kernel: np.ndarray) -> np.ndarray:
    """func'u kanal bazında uygula; hem 2 boyutlu hem 3 boyutlu girişleri işle."""
    if img.ndim == 2:
        return func(img, kernel)
    return np.stack([func(img[:, :, c], kernel)
                     for c in range(img.shape[2])], axis=-1)


def _resolve_kernel(kernel, kernel_size: int, shape: str) -> np.ndarray:
    """Önceden oluşturulmuş çekirdeği döndür veya boyut/şekilden yeni oluştur."""
    if kernel is not None:
        return np.asarray(kernel, dtype=np.uint8)
    return create_structuring_element(shape, kernel_size)


# ─────────────────────────────────────────────────────────────────────────
# 5. dilate
# ─────────────────────────────────────────────────────────────────────────

def dilate(img: np.ndarray,
           kernel=None,
           kernel_size: int = 3,
           shape: str = 'rect') -> np.ndarray:
    """
    Morfolojik genişleme — parlak bölgeleri büyütür.

    YE o konuma ortalandığında herhangi bir etkin YE konumu parlak kaynak
    pikselle örtüşüyorsa piksel parlak olur.

    Parametreler
    ------------
    img         : (H,W) veya (H,W,3) uint8 dizisi.
    kernel      : Önceden oluşturulmuş ikili YE dizisi. None ise
                  kernel_size ve shape'ten oluşturulur.
    kernel_size : kernel None iken YE kenar uzunluğu (varsayılan 3).
    shape       : kernel None iken YE şekli — 'rect'|'cross'|'ellipse'.
    """
    se = _resolve_kernel(kernel, kernel_size, shape)
    return _apply(img, _dilate_plane, se)


# ─────────────────────────────────────────────────────────────────────────
# 6. erode
# ─────────────────────────────────────────────────────────────────────────

def erode(img: np.ndarray,
          kernel=None,
          kernel_size: int = 3,
          shape: str = 'rect') -> np.ndarray:
    """
    Morfolojik aşındırma — parlak bölgeleri küçültür.

    YE o konuma ortalandığında tüm etkin YE konumları parlak kaynak
    piksellerle örtüşüyorsa piksel parlak kalır.

    Parametreler dilate() ile aynıdır.
    """
    se = _resolve_kernel(kernel, kernel_size, shape)
    return _apply(img, _erode_plane, se)


# ─────────────────────────────────────────────────────────────────────────
# 7. morphological_open  (önce aşındır, sonra genişlet)
# ─────────────────────────────────────────────────────────────────────────

def morphological_open(img: np.ndarray,
                        kernel=None,
                        kernel_size: int = 3,
                        shape: str = 'rect') -> np.ndarray:
    """
    Morfolojik açma: aynı YE ile önce aşındırma, sonra genişletme.

        f ∘ B = (f ⊖ B) ⊕ B

    Etki: daha büyük parlak bölgelerin şeklini korurken küçük parlak
    noktaları/ince çıkıntıları kaldırır.
    İşlem kendi tersidir: iki kez uygulamak bir kez uygulamakla eşdeğerdir.
    """
    se      = _resolve_kernel(kernel, kernel_size, shape)
    eroded  = _apply(img, _erode_plane,  se)
    return   _apply(eroded, _dilate_plane, se)


# ─────────────────────────────────────────────────────────────────────────
# 8. morphological_close  (önce genişlet, sonra aşındır)
# ─────────────────────────────────────────────────────────────────────────

def morphological_close(img: np.ndarray,
                         kernel=None,
                         kernel_size: int = 3,
                         shape: str = 'rect') -> np.ndarray:
    """
    Morfolojik kapama: aynı YE ile önce genişletme, sonra aşındırma.

        f • B = (f ⊕ B) ⊖ B

    Etki: genel şekli korurken parlak bölgelerin içindeki küçük koyu
    delikleri/boşlukları doldurur.
    """
    se      = _resolve_kernel(kernel, kernel_size, shape)
    dilated = _apply(img, _dilate_plane, se)
    return   _apply(dilated, _erode_plane, se)


# ─────────────────────────────────────────────────────────────────────────
# Morfolojik gradyan  (genişleme − aşındırma)
# ─────────────────────────────────────────────────────────────────────────

def gradient(img: np.ndarray,
             kernel_size: int = 3,
             shape: str = 'rect') -> np.ndarray:
    """
    Morfolojik gradyan: (f ⊕ B) − (f ⊖ B).
    Tüm yönlerdeki kenarları vurgular. Sonuç [0,255] ile kırpılır.
    """
    se = create_structuring_element(shape, kernel_size)
    d  = _apply(img, _dilate_plane, se).astype(np.int16)
    e  = _apply(img, _erode_plane,  se).astype(np.int16)
    return np.clip(d - e, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────
# Geriye dönük uyumluluk takma adları  (app.py opening() ve closing() çağırır)
# ─────────────────────────────────────────────────────────────────────────

def opening(img: np.ndarray,
            kernel_size: int = 3,
            shape: str = 'rect') -> np.ndarray:
    """morphological_open için takma ad (geriye dönük uyumluluk)."""
    return morphological_open(img, kernel_size=kernel_size, shape=shape)


def closing(img: np.ndarray,
            kernel_size: int = 3,
            shape: str = 'rect') -> np.ndarray:
    """morphological_close için takma ad (geriye dönük uyumluluk)."""
    return morphological_close(img, kernel_size=kernel_size, shape=shape)
