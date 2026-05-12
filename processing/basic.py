"""
Temel görüntü işleme operasyonları.
Tüm algoritmalar sıfırdan yalnızca NumPy kullanılarak uygulanmıştır — PIL işleme fonksiyonu kullanılmaz.

Fonksiyonlar
------------
to_grayscale   : RGB → parlaklık gri tonlama (BT.601 katsayıları)
to_binary      : gri tonlamalı eşikleme
rotate_image   : isteğe bağlı tuval genişletmesi ile bilineer interpolasyonlu döndürme
crop_image     : dizi dilimleme ile eksene hizalı kırpma
zoom_image     : bağımsız X/Y faktörleri ile en yakın komşu ölçekleme

Tüm fonksiyonlar hem 2 boyutlu (H, W) gri tonlamalı hem de 3 boyutlu (H, W, 3)
renkli dizileri kabul eder ve girişle aynı boyutlulukta uint8 dizileri döndürür.

Geriye dönük uyumluluk takma adları olan ``rotate``, ``crop``, ``zoom`` bu modülün
alt kısmında dışa aktarılmıştır; böylece ``app.py`` değişmeden çalışmaya devam eder.
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# 1.  to_grayscale
# ═══════════════════════════════════════════════════════════════════════════

def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    RGB görüntüyü tek kanallı gri tonlamalı görüntüye dönüştürür.

    **Formül (BT.601 / ITU-R 601):**

        Y = 0.299·R + 0.587·G + 0.114·B

    Bu ağırlıklar insan renk algısını yansıtır:
      - İnsan gözü en çok yeşile duyarlı konilere sahip olduğundan
        en çok yeşil katkıda bulunur (~%59).
      - Kırmızı ~%30 ve mavi sadece ~%11 katkıda bulunur.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü. Şekil (H, W) veya (H, W, 3) olmalıdır.
        dtype'ın uint8 [0, 255] olması beklenir.

    Döndürür
    --------
    np.ndarray
        (H, W) şeklinde gri tonlamalı görüntü, dtype uint8.
        Giriş zaten 2 boyutlu ise değiştirilmeden kopyası döndürülür.
    """
    if img.ndim == 2:
        # Zaten gri tonlamalı — çağıranların güvenle değiştirebilmesi için kopya döndür
        return img.copy()

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(
            f"(H, W) veya (H, W, 3) dizisi bekleniyordu; alınan şekil: {img.shape}"
        )

    # Çarpma sırasında tamsayı taşmasını önlemek için kanalları float32 olarak çıkar
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)

    # Ağırlıklı toplam — BT.601 katsayıları
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # En yakın tamsayıya yuvarla (np.round ile bankacı yuvarlaması burada uygundur),
    # kayan nokta aşımına karşı [0, 255] aralığına kırp ve ardından uint8'e dönüştür.
    # Açık yuvarlama, Python'da veya numpy'de round(0.587 * 255) hesaplarsanız da
    # sonucun aynı olmasını sağlar.
    return np.clip(np.round(gray), 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  to_binary
# ═══════════════════════════════════════════════════════════════════════════

def to_binary(img: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Sabit bir küresel yoğunluk eşiği kullanarak görüntüyü ikili hale getirir.

    **Algoritma:**

        gray = to_grayscale(img)          # önce Y kanalına dönüştür
        binary[y, x] = 255  gray[y, x] >= threshold ise
                      = 0    aksi takdirde

    Bu en basit görüntü bölütleme biçimidir; ön plan ve arka plan açıkça
    ayrılmış yoğunluklara sahip olduğunda iyi çalışır.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü. Şekil (H, W) veya (H, W, 3), dtype uint8.
    threshold : int, isteğe bağlı
        [0, 255] aralığında yoğunluk kesme noktası. Varsayılan 128 (orta gri).

    Döndürür
    --------
    np.ndarray
        (H, W) şeklinde, yalnızca 0 veya 255 içeren, dtype uint8 ikili görüntü.
    """
    # Adım 1: tek parlaklık kanalına indirge
    gray = to_grayscale(img) if img.ndim == 3 else img.copy()

    # Adım 2: numpy karşılaştırması — mantıksal (boolean) bir dizi üretir,
    #         np.where True → 255, False → 0 olarak eşler
    binary = np.where(gray >= int(threshold), np.uint8(255), np.uint8(0))
    return binary.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  rotate_image
# ═══════════════════════════════════════════════════════════════════════════

def rotate_image(img: np.ndarray,
                 angle_degrees: float,
                 expand: bool = True) -> np.ndarray:
    """
    Bilineer interpolasyon kullanarak görüntüyü saat yönünün tersine
    ``angle_degrees`` kadar döndürür.

    **Matematiksel altyapı:**

    θ açısı kadar 2 boyutlu döndürme, şu döndürme matrisi ile tanımlanır:

        R = [[cos θ,  -sin θ],
             [sin θ,   cos θ]]

    Her *hedef* (destination) pikseli (xd, yd) *kaynak* (source) pikseline
    geri eşlemek için **ters** (transpoze) döndürme uygularız — bu, her
    çıktı pikselinin doldurulmasını garanti eden standart "geriye dönük
    eşleme" (backward mapping) stratejisidir:

        [xs]   [cos θ   sin θ] [xd - cx_dst]   [cx_src]
        [ys] = [-sin θ  cos θ] [yd - cy_dst] + [cy_src]

    Burada (cx_src, cy_src) ve (cx_dst, cy_dst) döndürme öncesi ve sonrası
    görüntü merkezleridir.

    **Bilineer interpolasyon:**

    Kesirli bir kaynak koordinatı (xs, ys) için:
      - (x0, y0) = floor(xs, ys)  ve  (x1, y1) = (x0+1, y0+1) olsun
      - dx = xs − x0,  dy = ys − y0 olsun
      - İnterpolasyonlu değer:

            P = tl·(1−dx)·(1−dy)  +  tr·dx·(1−dy)
              + bl·(1−dx)·dy      +  br·dx·dy

        burada tl/tr/bl/br çevredeki dört kaynak pikseldir.

    Kaynak koordinatları kaynak görüntünün dışına düşen pikseller
    **siyah** (sıfır) ile doldurulur.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü, şekil (H, W) veya (H, W, 3), dtype uint8.
    angle_degrees : float
        Derece cinsinden saat yönünün tersine döndürme açısı.
    expand : bool, isteğe bağlı
        True ise (varsayılan), döndürülen görüntünün tamamı görünecek
        şekilde çıktı tuvali büyütülür. False ise, tuval girişle aynı
        boyutta kalır ve köşeler kırpılır.

    Döndürür
    --------
    np.ndarray
        Girişle aynı dtype ve kanal sayısına sahip döndürülmüş görüntü.
    """
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    src_h, src_w = img.shape[:2]
    cx_src = src_w / 2.0
    cy_src = src_h / 2.0

    # ── Çıktı tuval boyutunu hesapla ────────────────────────────────────────
    if expand:
        # Tüm dört kaynak köşesini döndür ve sınırlayıcı kutuyu bul
        corners = np.array([
            [-cx_src,          -cy_src],
            [ src_w - cx_src,  -cy_src],
            [-cx_src,           src_h - cy_src],
            [ src_w - cx_src,   src_h - cy_src],
        ])
        # Köşelerin ileri doğru döndürülmesi (R · corner)
        rot_corners = corners @ np.array([[cos_a, sin_a],
                                          [-sin_a, cos_a]])
        dst_w = int(np.ceil(rot_corners[:, 0].max() - rot_corners[:, 0].min()))
        dst_h = int(np.ceil(rot_corners[:, 1].max() - rot_corners[:, 1].min()))
    else:
        dst_w, dst_h = src_w, src_h

    cx_dst = dst_w / 2.0
    cy_dst = dst_h / 2.0

    # ── Hedef piksel ızgarasını oluştur ──────────────────────────────────────
    # ys_d şekli (dst_h, dst_w), xs_d şekli (dst_h, dst_w)
    ys_d, xs_d = np.mgrid[0:dst_h, 0:dst_w].astype(np.float32)

    # Hedef pikselleri orijin merkezli olacak şekilde ötele
    xd_c = xs_d - cx_dst   # (dst_h, dst_w)
    yd_c = ys_d - cy_dst

    # ── Ters döndürme: kaynak koordinatlarını bul ─────────────────────────
    # [xs, ys] = R^T · [xd_c, yd_c]  +  [cx_src, cy_src]
    # R^T = [[cos θ, sin θ], [-sin θ, cos θ]]
    xs = cos_a * xd_c + sin_a * yd_c + cx_src   # (dst_h, dst_w)
    ys = -sin_a * xd_c + cos_a * yd_c + cy_src

    # ── Bilineer interpolasyon ────────────────────────────────────────────
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    # Kesirli mesafeler
    dx = (xs - x0).astype(np.float32)   # şekil (dst_h, dst_w)
    dy = (ys - y0).astype(np.float32)

    # Boolean maske: Dört komşunun tamamı kaynak görüntünün içindeyse True
    valid = (x0 >= 0) & (x1 < src_w) & (y0 >= 0) & (y1 < src_h)

    # Güvenli dizi erişimi için indeksleri sınırla (geçersiz pikseller daha sonra sıfırlanır)
    x0c = np.clip(x0, 0, src_w - 1)
    x1c = np.clip(x1, 0, src_w - 1)
    y0c = np.clip(y0, 0, src_h - 1)
    y1c = np.clip(y1, 0, src_h - 1)

    # Orijinal boyutluluktan bağımsız olarak (H, W, C) kaynağı üzerinde çalış
    is_gray = img.ndim == 2
    src = img[:, :, np.newaxis] if is_gray else img
    n_channels = src.shape[2]

    out = np.zeros((dst_h, dst_w, n_channels), dtype=np.float32)

    for c in range(n_channels):
        # Dört köşe değerini topla
        tl = src[y0c, x0c, c].astype(np.float32)   # sol-üst
        tr = src[y0c, x1c, c].astype(np.float32)   # sağ-üst
        bl = src[y1c, x0c, c].astype(np.float32)   # sol-alt
        br = src[y1c, x1c, c].astype(np.float32)   # sağ-alt

        # Bilineer karışım
        value = (tl * (1.0 - dx) * (1.0 - dy) +
                 tr * dx          * (1.0 - dy) +
                 bl * (1.0 - dx) * dy           +
                 br * dx          * dy)

        # Kaynak dışına eşlenen pikselleri sıfırla (siyah dolgu)
        value[~valid] = 0.0
        out[:, :, c] = value

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out[:, :, 0] if is_gray else out


# ═══════════════════════════════════════════════════════════════════════════
# 4.  crop_image
# ═══════════════════════════════════════════════════════════════════════════

def crop_image(img: np.ndarray,
               x: int,
               y: int,
               width: int,
               height: int) -> np.ndarray:
    """
    Bir görüntüden dikdörtgen biçimli bir ilgi alanı (ROI) çıkarır.

    **Algoritma:**

    NumPy dizi dilimlemesi doğrudan kullanılır:

        out = img[y : y + height,  x : x + width]

    Tüm koordinatlar geçerli görüntü sınırlarına kırpılır, böylece fonksiyon
    asla indeks hatası vermez — istenen bölgeyi yalnızca mevcut olana
    kadar kırpar.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü, şekil (H, W) veya (H, W, 3), herhangi bir dtype.
    x : int
        Kırpma dikdörtgeninin sol kenarı (sütun indeksi, 0 tabanlı).
    y : int
        Kırpma dikdörtgeninin üst kenarı (satır indeksi, 0 tabanlı).
    width : int
        Piksel cinsinden istenen çıktı genişliği.
    height : int
        Piksel cinsinden istenen çıktı yüksekliği.

    Döndürür
    --------
    np.ndarray
        Kırpılmış alt görüntü; girişle aynı dtype ve kanal sayısı.
        Çıktı bir **kopyadır** — onu değiştirmek orijinali etkilemez.
    """
    src_h, src_w = img.shape[:2]

    # Sol üst köşeyi geçerli aralığa sabitle
    x = int(np.clip(x, 0, src_w - 1))
    y = int(np.clip(y, 0, src_h - 1))

    # Sağ alt köşeyi sabitle; en az 1x1'lik bir yama garanti et
    x2 = int(np.clip(x + max(1, int(width)),  x + 1, src_w))
    y2 = int(np.clip(y + max(1, int(height)), y + 1, src_h))

    return img[y:y2, x:x2].copy()


# ═══════════════════════════════════════════════════════════════════════════
# 5.  zoom_image
# ═══════════════════════════════════════════════════════════════════════════

def zoom_image(img: np.ndarray,
               scale_x: float,
               scale_y: float) -> np.ndarray:
    """
    **En yakın komşu interpolasyonu** kullanarak bağımsız yatay ve dikey
    ölçek faktörleriyle bir görüntüyü yeniden boyutlandırır.

    **Algoritma (geriye dönük eşleme):**

    Her hedef piksel (xd, yd) için kaynak piksel indeksini şu şekilde hesaplarız:

        xs = floor( xd / scale_x )
        ys = floor( yd / scale_y )

    ve kaynak değerini doğrudan kopyalarız — hiçbir karıştırma yapılmaz.
    Bu en hızlı yeniden örnekleme yöntemidir ve karakteristik "bloklu"
    büyütmeler üretir, ancak keskin kenarları mükemmel şekilde korur.

    Eşdeğer olarak, burada uygulanan merkez hizalı eşleme varyantı kullanılarak:

        xs = floor( (xd + 0.5) / scale_x )

    Bu, yuvarlama hatasını her kaynak pikselin etrafına simetrik olarak dağıtır
    ve sistematik bir yarım piksel kaymasını önler.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü, şekil (H, W) veya (H, W, 3), dtype uint8.
    scale_x : float
        Yatay ölçek faktörü. >1 büyütür, <1 küçültür, > 0 olmalıdır.
    scale_y : float
        Dikey ölçek faktörü. >1 büyütür, <1 küçültür, > 0 olmalıdır.

    Döndürür
    --------
    np.ndarray
        Ölçeklenmiş görüntü, girişle aynı dtype ve kanal sayısı.
    """
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError(
            f"Ölçek faktörleri pozitif olmalıdır; alınan: scale_x={scale_x}, scale_y={scale_y}"
        )

    src_h, src_w = img.shape[:2]

    dst_w = max(1, int(round(src_w * scale_x)))
    dst_h = max(1, int(round(src_h * scale_y)))

    # ── Hedef piksel merkezlerini kaynak piksel indekslerine eşle ─────────────
    # Merkez hizalı eşleme kullanılarak: xs = (xd + 0.5) / scale_x − 0.5
    # sonra floor → en yakın komşu indeksi
    dst_col_idx = np.arange(dst_w)                                 # (dst_w,)
    src_col_idx = np.floor(
        (dst_col_idx + 0.5) / scale_x
    ).astype(np.int32)
    src_col_idx = np.clip(src_col_idx, 0, src_w - 1)              # kenarları koru

    dst_row_idx = np.arange(dst_h)                                 # (dst_h,)
    src_row_idx = np.floor(
        (dst_row_idx + 0.5) / scale_y
    ).astype(np.int32)
    src_row_idx = np.clip(src_row_idx, 0, src_h - 1)              # kenarları koru

    # ── Süslü indeksleme yoluyla en yakın komşu örneklemesi ─────────────────────
    # img[src_row_idx][:, src_col_idx] önce doğru satırları sonra sütunları seçer.
    # Bu hem 2 boyutlu hem de 3 boyutlu diziler için çalışır.
    out = img[src_row_idx][:, src_col_idx]

    return out.copy()


# ═══════════════════════════════════════════════════════════════════════════
# Geriye dönük uyumluluk takma adları
# ─────────────────────────────────────────────────────────────────────────
# app.py şunları çağırır: basic.rotate(...), basic.crop(...), basic.zoom(img, factor)
# Bu ince sarmalayıcılar mevcut dağıtıcının değişiklik olmadan çalışmasını sağlar.
# ═══════════════════════════════════════════════════════════════════════════

def rotate(img: np.ndarray, angle: float, expand: bool = True) -> np.ndarray:
    """``rotate_image`` için takma ad (geriye dönük uyumluluk için korundu)."""
    return rotate_image(img, angle_degrees=angle, expand=expand)


def crop(img: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """``crop_image`` için takma ad (geriye dönük uyumluluk için korundu)."""
    return crop_image(img, x=x, y=y, width=width, height=height)


def zoom(img: np.ndarray, factor: float) -> np.ndarray:
    """
    ``zoom_image`` için düzgün ölçekli takma ad (geriye dönük uyumluluk için korundu).

    Aynı ``factor`` değerini hem X hem de Y eksenlerine uygular.
    """
    return zoom_image(img, scale_x=factor, scale_y=factor)
