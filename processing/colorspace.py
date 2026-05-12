"""
Yalnızca NumPy kullanılarak sıfırdan uygulanan renk uzayı dönüşümleri.

Genel API
---------
rgb_to_hsv(img)      → float32 dizisi  H∈[0,360), S∈[0,1], V∈[0,1]
hsv_to_rgb(hsv)      → uint8 RGB dizisi (rgb_to_hsv'nin tersi)
rgb_to_ycbcr(img)    → uint8 dizisi, Y / Cb / Cr kanalları [0,255] aralığında
rgb_to_lab(img)      → uint8 dizisi, L* / a* / b* kanalları (görüntüleme ölçekli)
pseudo_color(img)    → uint8 RGB renk haritası (jet / hot / cool)

Tüm fonksiyonlar uint8 (H, W, 3) RGB dizilerini kabul eder; rgb_to_hsv ayrıca
alt kod H, S, V üzerinde gerçek aritmetik yapabilmesi için *kanonik* HSV
aralıklarıyla float32 döndürür.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Dahili koruma kontrolü
# ═══════════════════════════════════════════════════════════════════════════

def _require_rgb(img: np.ndarray, fname: str) -> None:
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(
            f"{fname} 3 kanallı RGB görüntü (H, W, 3) gerektirir; "
            f"alınan şekil: {img.shape}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 4.  rgb_to_hsv
# ═══════════════════════════════════════════════════════════════════════════

def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """
    RGB uint8 görüntüyü kanonik kayan nokta aralıklarıyla HSV'ye dönüştürür.

    **Döndürülen kanal aralıkları:**
        H ∈ [0, 360)   derece
        S ∈ [0, 1]     kesir
        V ∈ [0, 1]     kesir

    **Algoritma (tam HSV formülü):**

    RGB'yi [0, 1] aralığına normalleştir:
        r, g, b = R/255, G/255, B/255

    Uç değerleri bul:
        C_max = max(r, g, b)
        C_min = min(r, g, b)
        Δ     = C_max − C_min

    Değer (Value):
        V = C_max

    Doygunluk (Saturation):
        S = 0          C_max = 0 ise
          = Δ / C_max  aksi takdirde

    Ton (Hue, derece cinsinden):
        H = 0°                              Δ = 0 ise (akromatik)
          = 60° · [(g−b)/Δ  mod 6]         C_max = r ise
          = 60° · [(b−r)/Δ  + 2]           C_max = g ise
          = 60° · [(r−g)/Δ  + 4]           C_max = b ise

    Parametreler
    ------------
    img : np.ndarray
        uint8 RGB görüntüsü, şekil (H, W, 3).

    Döndürür
    --------
    np.ndarray
        (H, W, 3) şeklinde float32 dizisi.
        Kanal 0 = H ∈ [0, 360), Kanal 1 = S ∈ [0, 1], Kanal 2 = V ∈ [0, 1].
    """
    _require_rgb(img, "rgb_to_hsv")

    # [0, 1] aralığına normalleştir
    rgb = img[:, :, :3].astype(np.float32) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

    c_max = np.maximum(np.maximum(r, g), b)       # V = C_max
    c_min = np.minimum(np.minimum(r, g), b)
    delta = c_max - c_min                          # kroma

    # ── Değer ────────────────────────────────────────────────────────────
    v = c_max                                      # şekil (H, W)

    # ── Doygunluk ───────────────────────────────────────────────────────
    # C_max == 0 (saf siyah) için S = 0, aksi takdirde Δ / C_max
    # np.errstate siyah pikseller için zararsız sıfıra bölmeyi bastırır —
    # np.where maskesi zaten bu sonuçları atar.
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(c_max == 0.0, 0.0, delta / c_max)

    # ── Ton ──────────────────────────────────────────────────────────────
    h = np.zeros_like(r)                           # varsayılan: 0° (akromatik)
    achromatic = (delta == 0.0)

    # Maske: C_max kırmızı kanaldır
    mask_r = (~achromatic) & (c_max == r)
    # Maske: C_max yeşil kanaldır (kırmızı değil — bağlar ilk eşleşmeye gider)
    mask_g = (~achromatic) & (c_max == g) & (~mask_r)
    # Maske: C_max mavi kanaldır
    mask_b = (~achromatic) & (c_max == b) & (~mask_r) & (~mask_g)

    # Kırmızı baskın olduğunda H:  60 · ((g−b)/Δ  mod 6)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0)

    # Yeşil baskın olduğunda H: 60 · ((b−r)/Δ + 2)
    h[mask_g] = 60.0 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0)

    # Mavi baskın olduğunda H:  60 · ((r−g)/Δ + 4)
    h[mask_b] = 60.0 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0)

    # Koruma: H'nin [0, 360) aralığında olduğundan emin ol
    h = h % 360.0

    return np.stack([h, s, v], axis=-1).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  hsv_to_rgb
# ═══════════════════════════════════════════════════════════════════════════

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """
    HSV görüntüyü uint8 RGB'ye geri dönüştürür.

    ``rgb_to_hsv`` tarafından döndürülen kanonik float32 biçimini
    (H∈[0,360), S∈[0,1], V∈[0,1]) **veya** eski görüntüleme ölçekli uint8
    biçimini (H∈[0,255], S∈[0,255], V∈[0,255]) kabul eder.
    Biçim otomatik olarak dtype ile tespit edilir.

    **Algoritma (HSV → RGB sektör ayrıştırma):**

        i  = floor(H / 60) mod 6          (sektör indeksi 0–5)
        f  = H/60 − floor(H/60)           (sektör içindeki kesirli kısım)
        p  = V · (1 − S)
        q  = V · (1 − f · S)
        t  = V · (1 − (1−f) · S)

        Sektör 0: (R,G,B) = (V, t, p)
        Sektör 1: (R,G,B) = (q, V, p)
        Sektör 2: (R,G,B) = (p, V, t)
        Sektör 3: (R,G,B) = (p, q, V)
        Sektör 4: (R,G,B) = (t, p, V)
        Sektör 5: (R,G,B) = (V, p, q)

    Parametreler
    ------------
    hsv : np.ndarray
        HSV görüntüsü, şekil (H, W, 3). Kanonik aralıklarla float32
        (H∈[0,360), S/V∈[0,1]) veya görüntüleme ölçekli uint8.

    Döndürür
    --------
    np.ndarray
        uint8 RGB görüntüsü, şekil (H, W, 3).
    """
    if hsv.ndim != 3 or hsv.shape[2] < 3:
        raise ValueError("hsv_to_rgb (H, W, 3) şeklinde bir dizi gerektirir.")

    # Biçimi otomatik tespit et
    if hsv.dtype == np.uint8:
        # Eski görüntüleme ölçekli biçim
        h = hsv[:, :, 0].astype(np.float32) / 255.0 * 360.0
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        v = hsv[:, :, 2].astype(np.float32) / 255.0
    else:
        # rgb_to_hsv'den kanonik float biçimi
        h = hsv[:, :, 0].astype(np.float32)   # [0, 360)
        s = hsv[:, :, 1].astype(np.float32)   # [0, 1]
        v = hsv[:, :, 2].astype(np.float32)   # [0, 1]

    h_div60 = h / 60.0
    i = np.floor(h_div60).astype(np.int32) % 6          # sektör 0–5
    f = h_div60 - np.floor(h_div60)                      # kesirli kısım

    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    # Vektörleştirilmiş sektör araması
    r = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [v,      q,      p,      p,      t,      v     ]
    )
    g = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [t,      v,      v,      q,      p,      p     ]
    )
    b = np.select(
        [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
        [p,      p,      t,      v,      v,      q     ]
    )

    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Yardımcı: HSV → görüntülenebilir uint8 (app.py / ön yüz için)
# ═══════════════════════════════════════════════════════════════════════════

def _hsv_to_display_uint8(hsv_float: np.ndarray) -> np.ndarray:
    """
    Kanonik float32 HSV'yi görüntüleme için uygun uint8 görüntüye ölçekler
    (H: 0–360°'yi temsil eden 0–255, S/V: 0–255).
    """
    h_disp = np.clip(hsv_float[:, :, 0] / 360.0 * 255.0, 0, 255)
    s_disp = np.clip(hsv_float[:, :, 1] * 255.0, 0, 255)
    v_disp = np.clip(hsv_float[:, :, 2] * 255.0, 0, 255)
    return np.stack([h_disp, s_disp, v_disp], axis=-1).astype(np.uint8)


def rgb_to_hsv_display(img: np.ndarray) -> np.ndarray:
    """
    Kolaylık sarmalayıcısı: RGB uint8 → HSV uint8 (görüntüleme ölçekli).
    Ön yüze göndermek için bunu kullanın; matematik için ``rgb_to_hsv`` kullanın.
    """
    return _hsv_to_display_uint8(rgb_to_hsv(img))


# ═══════════════════════════════════════════════════════════════════════════
# 6.  rgb_to_ycbcr
# ═══════════════════════════════════════════════════════════════════════════

def rgb_to_ycbcr(img: np.ndarray) -> np.ndarray:
    """
    RGB'yi YCbCr'a dönüştürür (tam aralık / JPEG kuralı, BT.601).

    **Dönüşüm matrisi (belirtildiği gibi kesin katsayılar):**

        Y  =  0.299·R + 0.587·G + 0.114·B
        Cb = −0.169·R − 0.331·G + 0.500·B + 128
        Cr =  0.500·R − 0.419·G − 0.081·B + 128

    Y  *luma* kodlar (algılanan parlaklık).
    Cb mavi fark kroma bileşenini kodlar.
    Cr kırmızı fark kroma bileşenini kodlar.
    Cb ve Cr, [0, 255] aralığında ortalanmaları için 128 ile ötelenir
    (bir ana renk için negatif değerler 128'in üzerinde pozitif değer olur).

    Parametreler
    ------------
    img : np.ndarray
        uint8 RGB görüntüsü, şekil (H, W, 3).

    Döndürür
    --------
    np.ndarray
        uint8 YCbCr görüntüsü, şekil (H, W, 3), değerler [0, 255] aralığında.
    """
    _require_rgb(img, "rgb_to_ycbcr")

    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    # BT.601 tam aralık katsayıları (belirtildiği gibi)
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 128.0
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 128.0

    return np.clip(
        np.round(np.stack([y, cb, cr], axis=-1)), 0, 255
    ).astype(np.uint8)


def ycbcr_to_rgb(img: np.ndarray) -> np.ndarray:
    """
    ``rgb_to_ycbcr``'nin tersi. YCbCr (BT.601 tam aralık) → RGB dönüşümü.

    **Ters matris:**

        R = Y                 + 1.402·(Cr − 128)
        G = Y − 0.344·(Cb − 128) − 0.714·(Cr − 128)
        B = Y + 1.772·(Cb − 128)
    """
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("ycbcr_to_rgb (H, W, 3) şeklinde bir dizi gerektirir.")

    y  = img[:, :, 0].astype(np.float32)
    cb = img[:, :, 1].astype(np.float32) - 128.0
    cr = img[:, :, 2].astype(np.float32) - 128.0

    r = y               + 1.402  * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772    * cb

    return np.clip(np.stack([r, g, b], axis=-1), 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 7.  rgb_to_lab
# ═══════════════════════════════════════════════════════════════════════════

def _srgb_gamma_expand(c: np.ndarray) -> np.ndarray:
    """
    sRGB ters sıkıştırma (gamma genişletme) uygulayarak sinyali doğrusallaştırır.

    **Formül (IEC 61966-2-1):**

        doğrusal = C / 12.92                         C ≤ 0.04045 ise
                 = ((C + 0.055) / 1.055) ^ 2.4      aksi takdirde
    """
    return np.where(
        c <= 0.04045,
        c / 12.92,
        ((c + 0.055) / 1.055) ** 2.4
    )


def _f_xyz_to_lab(t: np.ndarray) -> np.ndarray:
    """
    XYZ → L*a*b* dönüşümünde kullanılan CIE f() fonksiyonu.

    **Formül:**

        f(t) = t^(1/3)                              t > (6/29)^3 ise
             = (1/3)·(29/6)^2·t + 4/29             aksi takdirde

    Doğrusal bölüm t = 0'daki tekil durumu önler.
    """
    delta = 6.0 / 29.0          # ≈ 0.2069
    delta3 = delta ** 3         # ≈ 0.008856
    return np.where(
        t > delta3,
        np.cbrt(t),
        t / (3.0 * delta ** 2) + 4.0 / 29.0
    )


def rgb_to_lab(img: np.ndarray) -> np.ndarray:
    """
    RGB uint8 görüntüyü CIELAB'a dönüştürür (D65 aydınlatıcı, sRGB önceliği).

    **İşlem hattı:**

    Adım 1 – RGB'yi [0, 1] aralığına normalleştir:
        r, g, b = R/255, G/255, B/255

    Adım 2 – Doğrusallaştır (sRGB gamma genişletme):
        r_lin = sRGB_expand(r),   vb.

    Adım 3 – Doğrusal sRGB → CIE XYZ (D65) standart matris ile:

        [X]   [0.4124564  0.3575761  0.1804375] [r_lin]
        [Y] = [0.2126729  0.7151522  0.0721750] [g_lin]
        [Z]   [0.0193339  0.1191920  0.9503041] [b_lin]

    Adım 4 – D65 beyaz noktası (Xn, Yn, Zn) ile normalleştir:
        Xn = 0.95047,  Yn = 1.00000,  Zn = 1.08883

    Adım 5 – CIE f() uygula ve L*a*b* hesapla:
        L* = 116·f(Y/Yn) − 16            ∈ [0, 100]
        a* = 500·(f(X/Xn) − f(Y/Yn))    ∈ [−128, 127]
        b* = 200·(f(Y/Yn) − f(Z/Zn))    ∈ [−128, 127]

    Adım 6 – Görüntüleme için ölçekle (uint8):
        L_göster = L* / 100 · 255
        a_göster = a* + 128             ([0, 255] aralığına kaydır)
        b_göster = b* + 128

    Parametreler
    ------------
    img : np.ndarray
        uint8 RGB görüntüsü, şekil (H, W, 3).

    Döndürür
    --------
    np.ndarray
        uint8 L*a*b* görüntüsü, şekil (H, W, 3).
        K0 = L* [0,255],  K1 = a* 128'de merkezli,  K2 = b* 128'de merkezli.
    """
    _require_rgb(img, "rgb_to_lab")

    # Adım 1 – Normalleştir
    rgb_norm = img[:, :, :3].astype(np.float64) / 255.0

    # Adım 2 – Gamma genişletme (sRGB → doğrusal ışık)
    lin = _srgb_gamma_expand(rgb_norm)
    r_l, g_l, b_l = lin[:, :, 0], lin[:, :, 1], lin[:, :, 2]

    # Adım 3 – Doğrusal sRGB → CIE XYZ (D65) standart matris kullanarak
    x = 0.4124564 * r_l + 0.3575761 * g_l + 0.1804375 * b_l
    y = 0.2126729 * r_l + 0.7151522 * g_l + 0.0721750 * b_l
    z = 0.0193339 * r_l + 0.1191920 * g_l + 0.9503041 * b_l

    # Adım 4 – D65 beyaz noktası ile normalleştir
    fx = _f_xyz_to_lab(x / 0.95047)
    fy = _f_xyz_to_lab(y / 1.00000)
    fz = _f_xyz_to_lab(z / 1.08883)

    # Adım 5 – CIE L*a*b*
    L_star = 116.0 * fy - 16.0           # [0, 100]
    a_star = 500.0 * (fx - fy)           # [≈−128, ≈127]
    b_star = 200.0 * (fy - fz)           # [≈−128, ≈127]

    # Adım 6 – Görüntüleme ölçekleme → uint8
    L_disp = np.clip(L_star / 100.0 * 255.0, 0, 255)
    a_disp = np.clip(a_star + 128.0, 0, 255)
    b_disp = np.clip(b_star + 128.0, 0, 255)

    return np.stack([L_disp, a_disp, b_disp], axis=-1).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Sözde renk / yanlış renk haritası
# ═══════════════════════════════════════════════════════════════════════════

def pseudo_color(img: np.ndarray, colormap: str = 'jet') -> np.ndarray:
    """
    Gri tonlamalı (veya parlaklık) görüntüye yanlış renk haritası uygular.

    Desteklenen renk haritaları: ``'jet'``, ``'hot'``, ``'cool'``.

    **Jet renk haritası:** Klasik gökkuşağının doğrusal parçalı yaklaşımı
    (mavi→camgöbeği→yeşil→sarı→kırmızı).

    **Hot renk haritası:** siyah→kırmızı→sarı→beyaz (yayıcı / termal görünüm).

    **Cool renk haritası:** camgöbeği→eflatun geçişi.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü. 3 kanallı ise parlaklık (BT.601) kullanılır.
    colormap : str
        Renk haritasının adı.

    Döndürür
    --------
    np.ndarray
        uint8 RGB görüntüsü, şekil (H, W, 3).
    """
    # Tek parlaklık kanalına indir
    if img.ndim == 2:
        gray = img.astype(np.float32)
    else:
        gray = (0.299 * img[:, :, 0].astype(np.float32) +
                0.587 * img[:, :, 1].astype(np.float32) +
                0.114 * img[:, :, 2].astype(np.float32))

    t = np.clip(gray, 0.0, 255.0) / 255.0   # [0, 1] aralığına normalleştir

    if colormap == 'jet':
        # MATLAB jet renk haritasının doğrusal parçalı yaklaşımı
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    elif colormap == 'hot':
        r = np.clip(3.0 * t,       0.0, 1.0)
        g = np.clip(3.0 * t - 1.0, 0.0, 1.0)
        b = np.clip(3.0 * t - 2.0, 0.0, 1.0)
    elif colormap == 'cool':
        r = t
        g = 1.0 - t
        b = np.ones_like(t)
    else:
        r = g = b = t

    return np.clip(
        np.stack([r, g, b], axis=-1) * 255.0, 0, 255
    ).astype(np.uint8)
