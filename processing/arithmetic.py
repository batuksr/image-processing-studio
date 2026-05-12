"""
İki görüntü arasındaki aritmetik işlemler.
Tüm algoritmalar sıfırdan yalnızca NumPy kullanılarak uygulanmıştır.

Genel API
---------
add_images(img1, img2, alpha)   → np.ndarray
multiply_images(img1, img2)     → np.ndarray
subtract_images(img1, img2)     → np.ndarray  (app.py tarafından kullanılır)

Her iki fonksiyon da img2'yi img1'in uzamsal boyutlarına sığdırmaz ise
en yakın komşu örnekleme ile yeniden boyutlandırır (PIL resize kullanılmaz).
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Ortak yardımcı fonksiyonlar
# ═══════════════════════════════════════════════════════════════════════════

def _resize_nn(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    ``img``'yi (target_h, target_w) boyutuna en yakın komşu örnekleme ile
    yeniden boyutlandırır.

    **Algoritma (geriye dönük eşleme):**

        src_row = floor( (dst_row + 0.5) / scale_y )
        src_col = floor( (dst_col + 0.5) / scale_x )

    Merkez hizalı eşleme sistematik yarım piksel kaymasını önler ve
    tamamen NumPy süslü indeksleme ile uygulanır — PIL veya scipy yok.

    Hem 2 boyutlu (H, W) hem de 3 boyutlu (H, W, C) diziler için çalışır.
    """
    h, w = img.shape[:2]
    if (h, w) == (target_h, target_w):
        return img

    scale_y = target_h / h
    scale_x = target_w / w

    dst_rows = np.arange(target_h)
    dst_cols = np.arange(target_w)

    src_rows = np.clip(
        np.floor((dst_rows + 0.5) / scale_y).astype(np.int32), 0, h - 1
    )
    src_cols = np.clip(
        np.floor((dst_cols + 0.5) / scale_x).astype(np.int32), 0, w - 1
    )

    # Süslü indeksleme: önce satırları, sonra sütunları seç
    return img[src_rows][:, src_cols]


def _prepare(img1: np.ndarray,
             img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    img2'yi img1'in uzamsal boyutlarına göre yeniden boyutlandırır, ardından
    her ikisini float32'ye yükseltir ve aynı kanal sayısına yayar.
    """
    h1, w1 = img1.shape[:2]

    # Adım 1 – uzamsal hizalama (img2'yi img1'in boyutuna göre yeniden boyutlandır)
    img2 = _resize_nn(img2, h1, w1)

    # Adım 2 – kanal hizalama
    if img1.ndim == 2 and img2.ndim == 3:
        img1 = np.stack([img1] * img2.shape[2], axis=-1)
    elif img2.ndim == 2 and img1.ndim == 3:
        img2 = np.stack([img2] * img1.shape[2], axis=-1)

    return img1.astype(np.float32), img2.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 9.  add_images
# ═══════════════════════════════════════════════════════════════════════════

def add_images(img1: np.ndarray,
               img2: np.ndarray,
               alpha: float = 0.5) -> np.ndarray:
    """
    İki görüntünün ağırlıklı (alfa) karışımı.

    **Formül:**

        sonuç = clip( alpha · img1  +  (1 − alpha) · img2,  0, 255 )

    alpha = 0.5 olduğunda iki görüntü eşit katkıda bulunur (%50/50 karışım).
    alpha = 1.0 img1'i değiştirmeden döndürür; alpha = 0.0 img2'yi döndürür.

    img2 farklı uzamsal boyutlara sahipse karıştırma öncesinde en yakın komşu
    örnekleme ile otomatik olarak img1'e göre yeniden boyutlandırılır.

    Parametreler
    ------------
    img1  : np.ndarray   Birinci görüntü (H, W) veya (H, W, 3), dtype uint8.
    img2  : np.ndarray   İkinci görüntü — img1'in şekline göre yeniden boyutlandırılır.
    alpha : float        img1'in ağırlığı [0, 1] aralığında. Varsayılan 0.5.

    Döndürür
    --------
    np.ndarray  Karışık görüntü, img1 ile aynı şekil, dtype uint8.
    """
    alpha   = float(np.clip(alpha, 0.0, 1.0))
    a, b    = _prepare(img1, img2)
    result  = alpha * a + (1.0 - alpha) * b
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 10. multiply_images
# ═══════════════════════════════════════════════════════════════════════════

def multiply_images(img1: np.ndarray,
                    img2: np.ndarray) -> np.ndarray:
    """
    İki görüntünün piksel bazında normalleştirilmiş çarpımı.

    **Formül:**

        sonuç = clip( (img1 / 255) · (img2 / 255) · 255,  0, 255 )

    Her görüntüyü 255'e bölerek değerleri [0, 1] aralığına taşır. [0, 1]
    aralığındaki iki değerin çarpımı da [0, 1] aralığındadır; bu nedenle
    sonuç [0, 255]'e geri ölçeklenir.

    Bu işlem görüntüyü koyulaştırır (bir operand 255/saf beyaz değilse).
    Yaygın kullanım alanları:
      - Maskeleme: ikili maske ile çarparak bölgeler seçilir.
      - Modülasyon: gradyan haritası uygulanır.

    img2 farklı uzamsal boyutlara sahipse img1'e göre yeniden boyutlandırılır.

    Parametreler
    ------------
    img1 : np.ndarray   Birinci görüntü (H, W) veya (H, W, 3), dtype uint8.
    img2 : np.ndarray   İkinci görüntü — img1'in şekline göre yeniden boyutlandırılır.

    Döndürür
    --------
    np.ndarray  Çarpılmış görüntü, img1 ile aynı şekil, dtype uint8.
    """
    a, b   = _prepare(img1, img2)
    result = (a / 255.0) * (b / 255.0) * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# subtract_images  (app.py ile geriye dönük uyumluluk için korundu)
# ═══════════════════════════════════════════════════════════════════════════

def subtract_images(img1: np.ndarray,
                    img2: np.ndarray) -> np.ndarray:
    """
    İki görüntü arasındaki mutlak piksel farkı.

    **Formül:**   sonuç = clip( |img1 − img2|,  0, 255 )

    Değişim tespiti, hareket algılama ve artık hesaplama için kullanışlıdır.
    img2 farklı boyutlara sahipse img1'e göre yeniden boyutlandırılır.
    """
    a, b   = _prepare(img1, img2)
    result = np.abs(a - b)
    return np.clip(result, 0, 255).astype(np.uint8)
