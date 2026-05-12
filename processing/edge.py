"""
Yalnızca NumPy kullanılarak sıfırdan uygulanan kenar bulma algoritmaları.

Genel API
---------
sobel_edge_detection(img, threshold)  → np.ndarray  (eşiklenmiş ikili kenar haritası)
compute_gradient_direction(img)       → np.ndarray  (derece cinsinden açı, float32)

Ek yardımcılar (app.py tarafından geriye dönük uyumluluk takma adları aracılığıyla kullanılır)
----------------------------------------------------------------------------------------------
sobel(img)       → birleşik Sobel büyüklüğü (normalleştirilmiş, eşiksiz)
sobel_x(img)     → yatay gradyan büyüklüğü
sobel_y(img)     → dikey gradyan büyüklüğü
laplacian(img)   → Laplacian kenar haritası

Tüm fonksiyonlar hem (H, W) gri tonlamalı hem de (H, W, 3) renkli girişleri işler.
"""

import numpy as np
from processing.filters import _convolve2d
from processing.basic import to_grayscale


# ═══════════════════════════════════════════════════════════════════════════
# Sobel çekirdekleri (sabit, belirtildiği gibi)
# ═══════════════════════════════════════════════════════════════════════════

#  Gx dikey kenarları tespit eder (yatay yoğunluk gradyanlarına yanıt verir)
_Kx = np.array([[-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]], dtype=np.float32)

#  Gy yatay kenarları tespit eder (dikey yoğunluk gradyanlarına yanıt verir)
_Ky = np.array([[-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Dahili: gri tonlamalı float32 düzlem üzerinde ham Gx, Gy hesapla
# ═══════════════════════════════════════════════════════════════════════════

def _sobel_gradients(img: np.ndarray):
    """
    Görüntü için (Gx, Gy) float32 gradyan haritalarını döndürür.

    Adımlar
    -------
    1. Renkli ise gri tonlamaya dönüştür (BT.601 parlaklık).
    2. Gx çekirdeğini yansıma dolgulu 2 boyutlu konvolüsyon ile uygula.
    3. Gy çekirdeğini yansıma dolgulu 2 boyutlu konvolüsyon ile uygula.

    (Gx, Gy)'yi (H, W) şeklinde float32 diziler olarak döndürür.
    """
    gray = to_grayscale(img).astype(np.float32)
    gx   = _convolve2d(gray, _Kx)   # (H, W) float32
    gy   = _convolve2d(gray, _Ky)
    return gx, gy


def _to_display(result: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Orijinal ``img`` renkli ise ``result``'ı (2 boyutlu uint8) 3 kanallı
    görüntü olarak döndürür; böylece ön yüz her zaman (H, W, 3) alır.
    """
    if img.ndim == 3:
        return np.stack([result, result, result], axis=-1)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6.  sobel_edge_detection
# ═══════════════════════════════════════════════════════════════════════════

def sobel_edge_detection(img: np.ndarray,
                          threshold: int = 50) -> np.ndarray:
    """
    İkili eşikleme ile Sobel kenar tespiti.

    **Algoritma:**

    1. Girişi gri tonlamaya dönüştür (BT.601 parlaklık).

    2. Sobel Gx ve Gy çekirdeklerini 2 boyutlu konvolüsyon ile uygula:

           Gx çekirdeği: [[-1, 0, 1],   Gy çekirdeği: [[-1, -2, -1],
                          [-2, 0, 2],                   [ 0,  0,  0],
                          [-1, 0, 1]]                   [ 1,  2,  1]]

       Gx, yatay değişim hızını ölçer (dikey kenarlar); Gy, dikey değişim
       hızını ölçer.

    3. Gradyan büyüklüğünü hesapla:

           G(y, x) = sqrt(Gx(y, x)² + Gy(y, x)²)

    4. Eşik uygula (ikili çıkış):

           kenar(y, x) = 255  G(y, x) >= threshold ise
                       = 0    aksi takdirde

       ``threshold`` ham büyüklük uzayındadır (kırpmadan önce).

    5. Ham büyüklüğü uint8 [0, 255]'e kırp.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    threshold : int
        Altındaki piksellerin 0'a ayarlandığı büyüklük eşiği (varsayılan 50).

    Döndürür
    --------
    np.ndarray
        İkili kenar haritası; pikseller 0 veya ham büyüklüktedir (≤ 255),
        dtype uint8. Şekil girişle eşleşir (gri giriş → gri çıkış;
        renkli giriş → 3 kanallı çıkış).
    """
    gx, gy = _sobel_gradients(img)

    # Gradyan büyüklüğü
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Eşikleme: büyüklüğü >= threshold olan pikselleri koru, kalanları sıfırla
    thresholded = np.where(magnitude >= float(threshold), magnitude, 0.0)

    result = np.clip(thresholded, 0, 255).astype(np.uint8)
    return _to_display(result, img)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  compute_gradient_direction
# ═══════════════════════════════════════════════════════════════════════════

def compute_gradient_direction(img: np.ndarray) -> np.ndarray:
    """
    Sobel çekirdeklerini kullanarak her pikseldeki gradyan yönünü (açısını) hesaplar.

    **Formül:**

        θ(y, x) = arctan2(Gy(y, x), Gx(y, x))   [derece cinsinden, −180° ile +180° arası]

    Açı, yoğunluktaki **en dik yükseliş** yönünü temsil eder.
    Kenarlar bu yöne dik uzanır.

    Yaygın açı değerleri:
      -  0° / ±180° → dikey kenar (yatay gradyan)
      - ±90°         → yatay kenar (dikey gradyan)
      - ±45°         → çapraz kenar

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.

    Döndürür
    --------
    np.ndarray
        float32 açı haritası, şekil (H, W), değerler (−180°, +180°] aralığında.
        Giriş kanal sayısından bağımsız olarak 2 boyutlu dizi olarak döndürülür
        (yön, parlaklık kanalından türetilmiş bir skaler alandır).
    """
    gx, gy = _sobel_gradients(img)

    # arctan2 çeyreği doğru işler ve sıfıra bölmeden kaçınır
    angle_rad = np.arctan2(gy, gx)
    angle_deg = np.degrees(angle_rad).astype(np.float32)
    return angle_deg


# ═══════════════════════════════════════════════════════════════════════════
# Ek kenar dedektörleri  (app.py ile geriye dönük uyumlu)
# ═══════════════════════════════════════════════════════════════════════════

def sobel(img: np.ndarray, return_rgb: bool = True) -> np.ndarray:
    """
    Birleşik Sobel kenar haritası — büyüklük [0, 255]'e normalleştirilmiş.

    ``sobel_edge_detection``'ın aksine bu fonksiyon ikili eşik **uygulamaz**;
    bunun yerine büyüklük doğrusal olarak yeniden ölçeklenir; böylece en güçlü
    kenar 255'e eşlenir.

    Parametreler
    ------------
    img        : Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    return_rgb : True (varsayılan) ve giriş renkli ise (H, W, 3) döndür.
    """
    gx, gy  = _sobel_gradients(img)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # [0, 255]'e normalleştir
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max * 255.0

    result = np.clip(magnitude, 0, 255).astype(np.uint8)
    if return_rgb and img.ndim == 3:
        return np.stack([result, result, result], axis=-1)
    return result


def sobel_x(img: np.ndarray) -> np.ndarray:
    """
    Yatay Sobel yanıtı (dikey kenarları tespit eder).
    Yalnızca Gx'in normalleştirilmiş büyüklüğünü döndürür.
    """
    gx, _ = _sobel_gradients(img)
    gx_abs = np.abs(gx)
    mx = gx_abs.max()
    if mx > 0:
        gx_abs = gx_abs / mx * 255.0
    result = np.clip(gx_abs, 0, 255).astype(np.uint8)
    return _to_display(result, img)


def sobel_y(img: np.ndarray) -> np.ndarray:
    """
    Dikey Sobel yanıtı (yatay kenarları tespit eder).
    Yalnızca Gy'nin normalleştirilmiş büyüklüğünü döndürür.
    """
    _, gy = _sobel_gradients(img)
    gy_abs = np.abs(gy)
    my = gy_abs.max()
    if my > 0:
        gy_abs = gy_abs / my * 255.0
    result = np.clip(gy_abs, 0, 255).astype(np.uint8)
    return _to_display(result, img)


def laplacian(img: np.ndarray) -> np.ndarray:
    """
    Ayrık 4-bağlantılı çekirdek kullanan Laplacian kenar dedektörü:

        [[ 0, -1,  0],
         [-1,  4, -1],
         [ 0, -1,  0]]

    Laplacian, ikinci dereceden kısmi türevlerin toplamıdır; tüm kenar
    yönlerine aynı anda yanıt verir.
    """
    kernel = np.array([[ 0, -1,  0],
                       [-1,  4, -1],
                       [ 0, -1,  0]], dtype=np.float32)
    gray = to_grayscale(img).astype(np.float32)
    lap  = np.abs(_convolve2d(gray, kernel))
    mx   = lap.max()
    if mx > 0:
        lap = lap / mx * 255.0
    result = np.clip(lap, 0, 255).astype(np.uint8)
    return _to_display(result, img)
