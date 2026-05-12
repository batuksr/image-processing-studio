"""
Histogram hesaplama ve histogram tabanlı görüntü iyileştirme işlemleri.
Tüm algoritmalar sıfırdan yalnızca NumPy kullanılarak uygulanmıştır — PIL işleme fonksiyonu kullanılmaz.

Genel API
---------
compute_histogram(img)      → dict {'original': [...], 'processed': [...]} (her biri 256 değer)
stretch_histogram(img)      → np.ndarray   (doğrusal kontrast germe)
equalize_histogram(img)     → np.ndarray   (global histogram eşitleme)

Geriye dönük uyumluluk için takma adlar
----------------------------------------
histogram_stretch  → stretch_histogram
histogram_equalize → equalize_histogram
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Düşük seviyeli yardımcı fonksiyonlar
# ═══════════════════════════════════════════════════════════════════════════

def _count_histogram(plane: np.ndarray) -> np.ndarray:
    """
    Tek bir uint8 kanal için piksel frekanslarını elle sayar.

    **Algoritma:**
    256 elemanlı bir birikimli dizi oluşturur ve vektörleştirilmiş boolean maske
    kullanarak 256 olası yoğunluk değerinin tamamı üzerinde yineleme yapar —
    ``np.histogram`` veya ``np.bincount`` döngüsü kullanılmaz.

        hist[v] = v değerine eşit piksel sayısı

    Pratikte dahili olarak np.bincount kullanılır çünkü Python döngüsü olmadan
    tamsayı tekrarlarını saymak için tek standart NumPy indirgeme budur —
    şuna eşdeğerdir:

        hist = np.zeros(256, dtype=np.int64)
        for v in range(256):
            hist[v] = np.sum(plane == v)      ← tamamen manuel, O(256·H·W)

    np.bincount sürümü O(H·W) karmaşıklığına sahiptir ve özdeş sonuçlar üretir.

    Parametreler
    ------------
    plane : np.ndarray
        2 boyutlu uint8 kanal (H, W).

    Döndürür
    --------
    np.ndarray
        256 uzunluğunda 1 boyutlu int64 dizisi.
    """
    flat = plane.ravel().astype(np.uint8)
    # NumPy boolean maskeleriyle manuel sayım — açık ve okunabilir
    hist = np.zeros(256, dtype=np.int64)
    for v in range(256):
        hist[v] = int(np.sum(flat == v))
    return hist


def _fast_histogram(plane: np.ndarray) -> np.ndarray:
    """
    np.bincount ile hızlı O(N) histogram (_count_histogram ile aynı matematiksel
    sonuç, ancak 256 iterasyonlu Python döngüsü olmadan).
    Hızın önemli olduğu eşitleme işleminde dahili olarak kullanılır.
    """
    return np.bincount(plane.ravel().astype(np.uint8), minlength=256).astype(np.int64)


def _to_luminance(img: np.ndarray) -> np.ndarray:
    """Parlaklık (Y) kanalını uint8 olarak döndürür (BT.601)."""
    if img.ndim == 2:
        return img
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    return np.clip(np.round(0.299 * r + 0.587 * g + 0.114 * b), 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  compute_histogram
# ═══════════════════════════════════════════════════════════════════════════

def compute_histogram(img: np.ndarray,
                      processed: np.ndarray | None = None) -> dict:
    """
    Bir görüntü için piksel frekans histogramlarını hesaplar (isteğe bağlı olarak
    işlenmiş bir sürüm için de) ve sonuçları doğrudan JSON-serileştirilebilir
    Python listeleri olarak döndürür.

    **Algoritma:**
    v ∈ [0, 255] aralığındaki 256 olası yoğunluk değerinin her biri için:

        hist[v] = Σ  1  (piksel yoğunluğu == v olan konumlarda)

    Renkli görüntüler için parlaklık (Y) kanalı üzerinde çalışılır; böylece
    histogram kanal bazlı istatistikler yerine algılanan parlaklığı tanımlar.

    Parametreler
    ------------
    img : np.ndarray
        Orijinal görüntü, şekil (H, W) veya (H, W, 3), dtype uint8.
    processed : np.ndarray veya None, isteğe bağlı
        Verilirse histogramı 'processed' anahtarına yerleştirilir.
        Verilmezse 'processed' değeri 'original' ile aynıdır.

    Döndürür
    --------
    dict
        {
          'original':  list[int]  — `img` için 256 frekans sayısı,
          'processed': list[int]  — `processed` için 256 frekans sayısı
                                    (None ise 'original' ile aynı),
        }
    """
    lum_orig = _to_luminance(img)
    hist_orig = _count_histogram(lum_orig)

    if processed is not None:
        lum_proc = _to_luminance(processed)
        hist_proc = _count_histogram(lum_proc)
    else:
        hist_proc = hist_orig.copy()

    return {
        'original':  hist_orig.tolist(),
        'processed': hist_proc.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2.  stretch_histogram
# ═══════════════════════════════════════════════════════════════════════════

def _stretch_channel(channel: np.ndarray) -> np.ndarray:
    """
    Tek bir uint8 kanala doğrusal kontrast germe uygular.

    **Formül:**

        new_val = (val - min) / (max - min) * 255

    En koyu pikseli 0'a, en parlak pikseli 255'e eşler; histogramı tam dinamik
    aralığa yayar. min == max ise (düz kanal) kanal değiştirilmeden döndürülür.
    """
    c_min = float(channel.min())
    c_max = float(channel.max())

    if c_max == c_min:           # düz kanal — sıfıra bölmeden kaçın
        return channel.copy()

    # uint8 taşmasını önlemek için aritmetikten önce float32'ye dönüştür
    stretched = (channel.astype(np.float32) - c_min) / (c_max - c_min) * 255.0
    return np.clip(np.round(stretched), 0, 255).astype(np.uint8)


def stretch_histogram(img: np.ndarray) -> np.ndarray:
    """
    Doğrusal histogram germe (kontrast germe / normalleştirme).

    Her kanalı bağımsız olarak gerer; böylece tam [0, 255] dinamik aralığı kullanılır.

    **Kanal başına formül:**

        new_val = (val - kanal_min) / (kanal_max - kanal_min) * 255

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü, şekil (H, W) veya (H, W, 3), dtype uint8.

    Döndürür
    --------
    np.ndarray
        Kontrast gerilmiş görüntü, girişle aynı şekil ve dtype.
    """
    if img.ndim == 2:
        return _stretch_channel(img)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _stretch_channel(img[:, :, c])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 3.  equalize_histogram
# ═══════════════════════════════════════════════════════════════════════════

def _equalize_channel(channel: np.ndarray) -> np.ndarray:
    """
    Tek bir uint8 kanal için histogram eşitleme.

    **Algoritma:**

    1. Histogramı hesapla:  hist[v] = v yoğunluğuna sahip piksel sayısı

    2. Kümülatif Dağılım Fonksiyonunu (CDF) hesapla:
           CDF[v] = Σ_{i=0}^{v} hist[i]

    3. CDF_min'i bul = en küçük sıfır olmayan CDF değeri (boş kutucukları atla).

    4. Her eski v değerini yeni değere eşleyen bir Arama Tablosu (LUT) oluştur:

           new_val = round( (CDF[v] - CDF_min) / (H*W - CDF_min) * 255 )

       Bu standart eşitleme eşlemesidir; yoğunlukları yeniden dağıtır, böylece
       çıkış CDF'si (yaklaşık olarak) doğrusal olur, yani tüm yoğunluk seviyeleri
       eşit sıklıkta kullanılır.

    5. LUT'u uygula: output[y, x] = LUT[ input[y, x] ]

    Parametreler
    ------------
    channel : np.ndarray
        2 boyutlu uint8 dizisi (H, W).

    Döndürür
    --------
    np.ndarray
        Eşitlenmiş kanal, aynı şekil, dtype uint8.
    """
    # Adım 1 – histogram (hızlı yol; _count_histogram ile aynı sonuç)
    hist = _fast_histogram(channel)                     # şekil (256,)

    # Adım 2 – CDF
    cdf = hist.cumsum().astype(np.float64)              # şekil (256,)

    # Adım 3 – CDF_min: ilk sıfır olmayan giriş
    nonzero_cdf = cdf[cdf > 0]
    if nonzero_cdf.size == 0:
        return channel.copy()                           # boş görüntü
    cdf_min = float(nonzero_cdf[0])

    # Toplam piksel sayısı
    n_pixels = float(channel.size)                      # H * W

    # Adım 4 – LUT
    denom = n_pixels - cdf_min
    if denom == 0:
        return channel.copy()                           # tüm pikseller aynı

    lut = np.round((cdf - cdf_min) / denom * 255.0)
    lut = np.clip(lut, 0, 255).astype(np.uint8)        # şekil (256,)

    # Adım 5 – LUT'u süslü indeksleme ile uygula (O(H*W), Python döngüsü yok)
    return lut[channel]


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    Global histogram eşitleme.

    Piksel yoğunluklarını, çıkış histogramının (yaklaşık olarak) düzgün olacağı
    şekilde yeniden dağıtır — her gri seviyesi eşit sıklıkta kullanılır.

    **Eşitleme eşlemesi (kanal başına):**

        new_val[v] = round( (CDF(v) - CDF_min) / (H·W - CDF_min) * 255 )

    RGB görüntülerde işlem her kanala bağımsız olarak uygulanır.
    Bu, kontrası maksimize eder ancak ton kaymasına yol açabilir. Renkli
    görüntülerde algısal açıdan daha iyi bir sonuç için CLAHE (``filters.py``)
    kullanın.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü, şekil (H, W) veya (H, W, 3), dtype uint8.

    Döndürür
    --------
    np.ndarray
        Eşitlenmiş görüntü, girişle aynı şekil ve dtype.
    """
    if img.ndim == 2:
        return _equalize_channel(img)

    out = np.empty_like(img)
    for c in range(img.shape[2]):
        out[:, :, c] = _equalize_channel(img[:, :, c])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Geriye dönük uyumluluk için takma adlar (app.py bu isimleri çağırır)
# ═══════════════════════════════════════════════════════════════════════════

def histogram_stretch(img: np.ndarray) -> np.ndarray:
    """``stretch_histogram`` için takma ad (geriye dönük uyumluluk)."""
    return stretch_histogram(img)


def histogram_equalize(img: np.ndarray) -> np.ndarray:
    """``equalize_histogram`` için takma ad (geriye dönük uyumluluk)."""
    return equalize_histogram(img)
