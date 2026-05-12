"""
Konvolüsyon tabanlı görüntü filtreleri.
Tüm algoritmalar sıfırdan yalnızca NumPy kullanılarak uygulanmıştır — scipy veya PIL filtresi yok.

Genel API
---------
convolve2d(img, kernel)                           → np.ndarray
gaussian_filter(img, kernel_size, sigma)          → np.ndarray
adaptive_equalization(img, tile_size)             → np.ndarray
brightness_adjust(img, value)                     → np.ndarray
blur_image(img, method, kernel_size)              → np.ndarray

Geriye dönük uyumluluk takma adları (app.py tarafından kullanılır)
-------------------------------------------------------------------
gaussian_blur   → gaussian_filter
box_blur        → blur_image(img, 'box', kernel_size)
sharpen         → (keskinleştirme maskesi, ayrı tutulur)
clahe           → clip_limit / tile_grid argümanları ile adaptive_equalization
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


# ═══════════════════════════════════════════════════════════════════════════
# 1.  convolve2d
# ═══════════════════════════════════════════════════════════════════════════

def convolve2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Bir görüntünün bir çekirdekle 2 boyutlu ayrık konvolüsyonu, her kanala
    bağımsız olarak uygulanır. **Sıfır dolgusu** (zero-padding) kullanır, böylece
    çıktı girdiyle aynı uzamsal boyutlara sahip olur.

    **Matematiksel tanım (tek kanal):**

        (f * g)[y, x] = Σ_m Σ_n  f[y−m, x−n] · g[m, n]

    Eşdeğer olarak — gerçek konvolüsyon için çekirdeği çevirdikten sonra —
    her çıktı pikseli, çekirdeğin (y, x) merkezli yerel görüntü komşuluğu ile
    nokta çarpımıdır.

    **Uygulama stratejisi (vektörleştirilmiş yama çıkarma):**

    İç içe Python döngüleri (O(H·W·kH·kW) saf Python yinelemeleri) yerine,
    ``yamalar[y, x]``'in (y, x) pikseli merkezli (kH, kW) komşuluğu olduğu
    (H, W, kH, kW) şeklinde bir görünüm oluşturmak için
    ``numpy.lib.stride_tricks.as_strided`` kullanırız. Konvolüsyon daha sonra
    tek bir eleman bazlı çarp-ve-topla işlemidir:

        output = (yamalar * kernel).sum(axis=(2, 3))

    Bu O(H·W·kH·kW) aritmetiğidir ancak NumPy'nin C-hızlı iç döngüsüyle çalışır.

    **Sıfır dolgusu:**

    Girdi, her iki tarafta (kH//2, kW//2) sıfırlarla doldurulur; böylece
    kenar pikselleri siyahla çevriliymiş gibi işlenir.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü, şekil (H, W) veya (H, W, C), herhangi bir sayısal dtype.
    kernel : np.ndarray
        2 boyutlu konvolüsyon çekirdeği, şekil (kH, kW). Kare veya tek olması gerekmez.

    Döndürür
    --------
    np.ndarray
        float32 olarak konvolüsyon uygulanmış görüntü, ``img`` ile aynı uzamsal şekil.
        Gerekirse bu fonksiyonun **dışında** uint8'e kırpın / dönüştürün.
    """
    if kernel.ndim != 2:
        raise ValueError(f"Çekirdek 2 boyutlu olmalıdır; alınan şekil: {kernel.shape}")

    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    k = kernel.astype(np.float32)

    def _conv_plane(plane: np.ndarray) -> np.ndarray:
        """Tek bir (H, W) float32 kanalına konvolüsyon uygular."""
        # Sıfır dolgu (yansıma değil) — şartnameye göre gereklidir
        padded = np.pad(
            plane.astype(np.float32),
            ((ph, ph), (pw, pw)),
            mode='constant',
            constant_values=0,
        )
        h, w = plane.shape

        # (H, W, kH, kW) kayan pencere görünümü oluştur — sıfır kopyalı
        shape   = (h, w, kh, kw)
        strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
        patches = as_strided(padded, shape=shape, strides=strides)

        # Çekirdek ile eleman bazlı çarp ve yama eksenleri üzerinde topla
        return (patches * k).sum(axis=(2, 3))

    if img.ndim == 2:
        return _conv_plane(img)

    # Renkli görüntü: her kanala uygula, yeniden yığınla (stack)
    channels = [_conv_plane(img[:, :, c]) for c in range(img.shape[2])]
    return np.stack(channels, axis=-1)


# ═══════════════════════════════════════════════════════════════════════════
# Dahili yardımcılar
# ═══════════════════════════════════════════════════════════════════════════

def _reflect_convolve_plane(plane: np.ndarray,
                             kernel: np.ndarray) -> np.ndarray:
    """
    convolve2d gibidir ancak **yansıma** (reflect) dolgusu kullanır (düzeltme
    filtreleri için daha iyidir — görüntü kenarlarında siyah kenar yapaylıkları olmaz).
    ``plane`` ile aynı uzamsal boyutta float32 döndürür.
    """
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(
        plane.astype(np.float32),
        ((ph, ph), (pw, pw)),
        mode='reflect',
    )
    h, w = plane.shape
    shape   = (h, w, kh, kw)
    strides = (padded.strides[0], padded.strides[1],
               padded.strides[0], padded.strides[1])
    patches = as_strided(padded, shape=shape, strides=strides)
    return (patches * kernel.astype(np.float32)).sum(axis=(2, 3))


def _apply_kernel_reflect(img: np.ndarray,
                           kernel: np.ndarray) -> np.ndarray:
    """Çekirdeği yansıma dolgusu ile her kanala uygular; uint8 döndürür."""
    if img.ndim == 2:
        out = _reflect_convolve_plane(img, kernel)
        return np.clip(out, 0, 255).astype(np.uint8)
    channels = [_reflect_convolve_plane(img[:, :, c], kernel)
                for c in range(img.shape[2])]
    return np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)


def _apply_kernel_zero(img: np.ndarray,
                        kernel: np.ndarray) -> np.ndarray:
    """Çekirdeği sıfır dolgusu ile her kanala uygular; uint8 döndürür."""
    out = convolve2d(img, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  gaussian_filter
# ═══════════════════════════════════════════════════════════════════════════

def _build_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Kare, normalleştirilmiş bir Gauss çekirdeği oluşturur.

    **Formül:**

        G(x, y) = exp(−(x² + y²) / (2σ²))   /   (2πσ²)

    Tamsayı ızgarasında [−r, +r] (burada r = size//2) örnekledikten sonra,
    tüm ağırlıkların toplamı 1 olacak şekilde normalleştiririz; bu, çekirdeği
    uygun bir olasılık dağılımı yapar (düz görüntüler için net yoğunluk değişimi olmaz).

    Parametreler
    ------------
    size : int  Çekirdek kenar uzunluğu (tek sayıya zorlanır).
    sigma : float  Piksel cinsinden standart sapma (bulanıklık yarıçapını kontrol eder).
    """
    size = max(3, size | 1)          # tek sayıya zorla, minimum 3
    r = size // 2
    ax = np.arange(-r, r + 1, dtype=np.float64)   # örn. [-2,-1,0,1,2]
    xx, yy = np.meshgrid(ax, ax)

    # Normalleştirilmemiş Gauss
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))

    # Toplam = 1 olacak şekilde normalleştir
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def gaussian_filter(img: np.ndarray,
                    kernel_size: int = 5,
                    sigma: float = 1.0) -> np.ndarray:
    """
    2 boyutlu konvolüsyon ile Gauss bulanıklığı.

    **Çekirdek formülü:**

        G(x, y) = exp(−(x² + y²) / (2σ²)) / (2πσ²)

    Çekirdek (kernel_size × kernel_size) boyutunda bir tamsayı ızgarasında
    örneklenir ve ardından girişleri toplamı 1 olacak şekilde normalleştirilir.

    Görüntü kenarında yansıma dolgusu (reflect padding) kullanılır; böylece
    kenarlar koyu bir hale oluşmadan tutarlı bir şekilde bulanıklaşır.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    kernel_size : int
        Gauss çekirdeğinin kenar uzunluğu (tek sayıya zorlanır, minimum 3).
    sigma : float
        Gauss eğrisinin piksellerdeki standart sapması. Daha büyük = daha fazla bulanıklık.

    Döndürür
    --------
    np.ndarray
        Bulanık görüntü, aynı şekil ve dtype uint8.
    """
    kernel = _build_gaussian_kernel(kernel_size, sigma)
    return _apply_kernel_reflect(img, kernel)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  adaptive_equalization  (CLAHE benzeri)
# ═══════════════════════════════════════════════════════════════════════════

def _equalize_tile(tile: np.ndarray,
                   clip_limit: float) -> np.ndarray:
    """
    İsteğe bağlı kontrast kırpma ile dikdörtgen bir döşeme (tile) için bir
    eşitleme arama tablosu (LUT) oluşturur.

    1. Histogramı ``h[v]`` say (256 bölme).
    2. Her bölmeyi kırp: fazlalık = h[v] − clip_abs  (pozitifse, aksi halde 0).
       Fazlalığı tüm bölmelere eşit olarak yeniden dağıt.
    3. CDF'yi hesapla ve LUT'u oluştur:
           LUT[v] = round((CDF[v] − CDF_min) / (N − CDF_min) * 255)

    Eşlenmiş döşemeyi değil, 256 girişli uint8 LUT'u döndürür.
    """
    n_pixels   = tile.size
    clip_abs   = max(1, int(clip_limit * n_pixels / 256))

    hist = np.bincount(tile.ravel().astype(np.uint8), minlength=256).astype(np.float64)

    # Kırp ve fazlalığı yeniden dağıt
    excess       = np.sum(np.maximum(hist - clip_abs, 0.0))
    hist         = np.minimum(hist, clip_abs)
    hist        += excess / 256.0

    # Kümülatif Dağılım Fonksiyonu (CDF)
    cdf          = hist.cumsum()
    nonzero      = cdf[cdf > 0]
    cdf_min      = float(nonzero[0]) if nonzero.size else 0.0
    denom        = float(n_pixels) - cdf_min

    if denom <= 0:
        return np.arange(256, dtype=np.uint8)      # yozlaşmış döşeme

    lut = np.round((cdf - cdf_min) / denom * 255.0)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _adaptive_equalize_channel(plane: np.ndarray,
                                tile_size: int,
                                clip_limit: float) -> np.ndarray:
    """
    Tek bir uint8 kanal için CLAHE tarzı uyarlamalı eşitleme.

    **Algoritma:**

    1. Görüntüyü (tile_size × tile_size) döşemelere böl.
       Kenar döşemeleri tüm görüntüyü kapsayacak şekilde genişletilir.

    2. Her döşeme için bir histogram eşitleme LUT'u hesapla (gürültünün
       aşırı güçlendirilmesini bastırmak için kırpma sınırı ile).

    3. Her çıktı pikseli için, onu çevreleyen dört döşeme merkezini bul
       ve LUT'ları arasında **bilineer interpolasyon** yap:

           result[y,x] = (1−α)·(1−β)·LUT_TL[v] + α·(1−β)·LUT_BL[v]
                       +  (1−α)·  β ·LUT_TR[v] +  α·  β ·LUT_BR[v]

       burada α ve β, sol üst döşeme merkezinden kesirli uzaklıklardır
       ve v = plane[y, x] değeridir.

    Bu, döşeme sınırlarında pürüzsüz, yapaysız geçişler üretir —
    sıradan (blok bazlı) uyarlamalı eşitlemeye göre temel avantajı budur.

    Parametreler
    ------------
    plane     : 2 boyutlu uint8 dizisi (H, W).
    tile_size : int  Piksel cinsinden her döşemenin yaklaşık kenar uzunluğu.
    clip_limit: float  Düzgün dağılıma göre kırpma eşiği.

    Döndürür
    --------
    np.ndarray  Eşitlenmiş kanal, aynı şekil, dtype uint8.
    """
    h, w = plane.shape

    # Her yöndeki döşeme sayısı (en az 1)
    n_rows = max(1, int(np.ceil(h / tile_size)))
    n_cols = max(1, int(np.ceil(w / tile_size)))

    # ── Her döşeme için bir LUT oluştur ───────────────────────────────────────────
    luts = np.zeros((n_rows, n_cols, 256), dtype=np.uint8)
    row_edges = np.linspace(0, h, n_rows + 1).astype(int)
    col_edges = np.linspace(0, w, n_cols + 1).astype(int)

    for tr in range(n_rows):
        for tc in range(n_cols):
            r0, r1 = row_edges[tr], row_edges[tr + 1]
            c0, c1 = col_edges[tc], col_edges[tc + 1]
            luts[tr, tc] = _equalize_tile(plane[r0:r1, c0:c1], clip_limit)

    # Döşeme merkezi koordinatları (bilineer ağırlık hesabı için kullanılır)
    row_centres = ((row_edges[:-1] + row_edges[1:]) / 2.0)   # (n_rows,)
    col_centres = ((col_edges[:-1] + col_edges[1:]) / 2.0)   # (n_cols,)

    # ── Tüm görüntü üzerinde bilineer interpolasyon (vektörleştirilmiş) ──────────
    # Her (y, x) pikseli için çevresindeki iki döşeme satırını / sütununu bul.

    # Piksel koordinat dizileri
    py = np.arange(h, dtype=np.float64)   # (H,)
    px = np.arange(w, dtype=np.float64)   # (W,)

    # Her piksel satırı için döşeme satırı parantezini bul
    # np.searchsorted ekleme dizinini verir (= sağ komşu dizini)
    row_right = np.clip(
        np.searchsorted(row_centres, py, side='right'),
        1, n_rows - 1
    ).astype(int)                          # (H,)
    row_left  = row_right - 1             # (H,)

    alpha = ((py - row_centres[row_left]) /
             (row_centres[row_right] - row_centres[row_left] + 1e-12))
    alpha = np.clip(alpha, 0.0, 1.0)     # (H,)

    # Her piksel sütunu için döşeme sütunu parantezini bul
    col_right = np.clip(
        np.searchsorted(col_centres, px, side='right'),
        1, n_cols - 1
    ).astype(int)                          # (W,)
    col_left  = col_right - 1             # (W,)

    beta = ((px - col_centres[col_left]) /
            (col_centres[col_right] - col_centres[col_left] + 1e-12))
    beta = np.clip(beta, 0.0, 1.0)       # (W,)

    # Dört döşeme köşesinin her biri için LUT araması (her birinin şekli H×W)
    v = plane.astype(np.int32)           # piksel değerleri (H, W)

    # Çevredeki dört döşeme köşesi için LUT değerlerini topla
    # luts[row_left,  col_left,  v]  →  TL (Sol Üst)
    # luts[row_left,  col_right, v]  →  TR (Sağ Üst)
    # luts[row_right, col_left,  v]  →  BL (Sol Alt)
    # luts[row_right, col_right, v]  →  BR (Sağ Alt)
    rl = row_left[:, np.newaxis]   # (H,1)   W boyunca yayınlanır
    rr = row_right[:, np.newaxis]
    cl = col_left[np.newaxis, :]   # (1,W)   H boyunca yayınlanır
    cr = col_right[np.newaxis, :]

    tl_vals = luts[rl, cl, v].astype(np.float64)   # (H, W)
    tr_vals = luts[rl, cr, v].astype(np.float64)
    bl_vals = luts[rr, cl, v].astype(np.float64)
    br_vals = luts[rr, cr, v].astype(np.float64)

    a = alpha[:, np.newaxis]   # (H, 1)
    b = beta[np.newaxis, :]    # (1, W)

    result = ((1 - a) * (1 - b) * tl_vals +
              (1 - a) *       b * tr_vals +
                    a * (1 - b) * bl_vals +
                    a *       b * br_vals)

    return np.clip(np.round(result), 0, 255).astype(np.uint8)


def adaptive_equalization(img: np.ndarray,
                           tile_size: int = 8,
                           clip_limit: float = 2.0) -> np.ndarray:
    """
    Kontrast Sınırlı Uyarlamalı Histogram Eşitleme (CLAHE benzeri).

    Tüm görüntü için tek bir eşleme kullanan küresel histogram eşitlemenin
    aksine, uyarlamalı eşitleme her (tile_size × tile_size) döşeme için yerel bir
    LUT hesaplar ve her pikselde komşu döşeme LUT'ları arasında bilineer interpolasyon yapar.

    ``clip_limit`` parametresi aşırı gürültü artışını önler:
    ``clip_limit × tile_pixels / 256`` üzerindeki histogram kutuları kırpılır ve
    fazlalıkları düzgün bir şekilde yeniden dağıtılır.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    tile_size : int
        Piksel cinsinden yaklaşık döşeme kenar uzunluğu (varsayılan 8).
    clip_limit : float
        Kontrast kırpma faktörü (varsayılan 2.0). Kırpmayı devre dışı bırakmak
        için çok yüksek ayarlayın.

    Döndürür
    --------
    np.ndarray
        Uyarlamalı eşitlenmiş görüntü, aynı şekil ve dtype uint8.
    """
    if img.ndim == 2:
        return _adaptive_equalize_channel(img, tile_size, clip_limit)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _adaptive_equalize_channel(
            img[:, :, c], tile_size, clip_limit
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 4.  brightness_adjust
# ═══════════════════════════════════════════════════════════════════════════

def brightness_adjust(img: np.ndarray, value: int) -> np.ndarray:
    """
    Her pikselin parlaklığını sabit bir ``value`` (değer) kadar kaydırır.

    **Formül:**

        new_pixel = clip(old_pixel + value, 0, 255)

    Pozitif ``value`` aydınlatır; negatif ``value`` karartır.
    Kırpma işlemi eleman bazında yapılır; böylece taşma/alt taşma oluşmaz.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    value : int
        [−255, 255] aralığında parlaklık ofseti.

    Döndürür
    --------
    np.ndarray
        Parlaklığı ayarlanmış görüntü, aynı şekil, dtype uint8.
    """
    # Kırpmadan önce işaretli toplamaya izin vermek için int16'ya yükselt
    adjusted = img.astype(np.int16) + int(value)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  blur_image
# ═══════════════════════════════════════════════════════════════════════════

def blur_image(img: np.ndarray,
               method: str = 'gaussian',
               kernel_size: int = 5,
               sigma: float = 1.0) -> np.ndarray:
    """
    Belirtilen yöntemi kullanarak bir görüntüyü bulanıklaştırır.

    Yöntemler
    ---------
    ``'box'``
        **Kutu (ortalama) bulanıklığı** — her çıktı pikseli,
        (kernel_size × kernel_size) komşuluğundaki tüm piksellerin ağırlıksız ortalamasıdır:

            kernel[i, j] = 1 / kernel_size²   tüm i, j için

        Hızlı ve basittir; tekdüze, biraz doğal olmayan bir bulanıklık üretir.

    ``'gaussian'``
        **Gauss bulanıklığı** — komşuluk pikselleri, çıktı pikseli merkezli
        2 boyutlu bir Gauss fonksiyonu ile ağırlıklandırılır (formül için
        ``gaussian_filter``'a bakın). Algısal olarak pürüzsüz, doğal görünümlü
        bir bulanıklık üretir.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    method : str
        ``'box'`` veya ``'gaussian'`` (varsayılan ``'gaussian'``).
    kernel_size : int
        Bulanıklık çekirdeğinin kenar uzunluğu (tek sayıya zorlanır, minimum 3).
    sigma : float
        Gauss sigma değeri (yalnızca yöntem ``'gaussian'`` olduğunda kullanılır).

    Döndürür
    --------
    np.ndarray
        Bulanık görüntü, aynı şekil, dtype uint8.
    """
    kernel_size = max(3, kernel_size | 1)   # tek sayı olmasını sağla

    if method == 'box':
        k = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
        return _apply_kernel_reflect(img, k)

    elif method == 'gaussian':
        return gaussian_filter(img, kernel_size=kernel_size, sigma=sigma)

    else:
        raise ValueError(
            f"Bilinmeyen bulanıklık yöntemi '{method}'. 'box' veya 'gaussian' seçin."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Unsharp-mask keskinleştirme  (app.py tarafından 'sharpen' olarak kullanılır)
# ═══════════════════════════════════════════════════════════════════════════

def sharpen(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Keskinleştirme maskesi (Unsharp-mask) yöntemi ile keskinleştirme.

    **Formül:**

        result = clip(img + strength × (img − GaussianBlur(img)), 0, 255)

    Gauss bulanıklığı uygulanmış görüntü düşük frekanslı bir referans görevi
    görür; orijinaliyle arasındaki fark yüksek frekanslı bir detay haritasıdır.
    Bu haritanın ölçeklenmiş bir versiyonunu orijinaline geri eklemek ince
    detayları geliştirir.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    strength : float
        Keskinleştirme yoğunluğu (varsayılan 1.0). > 1 değerleri aşırı
        keskinleşme üretir; (0, 1) aralığındaki değerler hafif bir geliştirme sağlar.

    Döndürür
    --------
    np.ndarray
        Keskinleştirilmiş görüntü, aynı şekil, dtype uint8.
    """
    blurred = gaussian_filter(img, kernel_size=5, sigma=1.0).astype(np.float32)
    src     = img.astype(np.float32)
    result  = src + float(strength) * (src - blurred)
    return np.clip(result, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# CLAHE  (app.py filters.clahe işlevini clip_limit / tile_grid argümanlarıyla çağırır)
# ═══════════════════════════════════════════════════════════════════════════

def clahe(img: np.ndarray,
          clip_limit: float = 2.0,
          tile_grid: tuple = (8, 8)) -> np.ndarray:
    """
    ``app.py`` için geriye dönük uyumlu CLAHE sarmalayıcısı.

    ``tile_grid = (rows, cols)`` API'sini ``adaptive_equalization`` tarafından
    kullanılan ``tile_size`` formatına dönüştürür (iki döşeme boyutundan küçük
    olanı temsilci döşeme boyutu olarak kullanılır).

    Parametreler
    ------------
    img        : Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    clip_limit : Kontrast kırpma faktörü (varsayılan 2.0).
    tile_grid  : (n_rows, n_cols) — her yöndeki döşeme sayısı.
    """
    h, w = img.shape[:2]
    gh, gw = int(tile_grid[0]), int(tile_grid[1])
    # Izgara boyutlarından yaklaşık bir tile_size türet
    tile_size = max(4, min(h // max(gh, 1), w // max(gw, 1)))
    return adaptive_equalization(img, tile_size=tile_size, clip_limit=clip_limit)


# ═══════════════════════════════════════════════════════════════════════════
# Geriye dönük uyumluluk takma adları (app.py bu adları kullanır)
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_blur(img: np.ndarray,
                  kernel_size: int = 5,
                  sigma: float = 1.0) -> np.ndarray:
    """``gaussian_filter`` için takma ad (geriye dönük uyumluluk)."""
    return gaussian_filter(img, kernel_size=kernel_size, sigma=sigma)


def box_blur(img: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """``blur_image(img, 'box', kernel_size)`` için takma ad (geriye dönük uyumluluk)."""
    return blur_image(img, method='box', kernel_size=kernel_size)


# ── Özel isim edge.py için tutulur (imports _convolve2d) ─────────────────
def _convolve2d(plane: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Dahili: tek bir float32 kanalına yansıma dolgusu ile konvolüsyon uygular.
    Bu adı doğrudan içe aktaran edge.py için korunmuştur.
    """
    return _reflect_convolve_plane(plane, kernel)
