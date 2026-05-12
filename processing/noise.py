"""
Gürültü ekleme ve gürültü giderme filtreleri.
Tüm algoritmalar sıfırdan yalnızca NumPy kullanılarak uygulanmıştır — scipy veya PIL filtresi yok.

Genel API
---------
add_salt_pepper_noise(img, density)          → np.ndarray
mean_filter(img, kernel_size)                → np.ndarray
median_filter(img, kernel_size)              → np.ndarray

Geriye dönük uyumluluk için korundu (app.py)
---------------------------------------------
salt_and_pepper  → add_salt_pepper_noise (salt_vs_pepper + seed argümanları ile)
gaussian_noise   → (bağımsız, değiştirilmedi)
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided


# ═══════════════════════════════════════════════════════════════════════════
# 1.  add_salt_pepper_noise
# ═══════════════════════════════════════════════════════════════════════════

def add_salt_pepper_noise(img: np.ndarray,
                           density: float = 0.05,
                           salt_ratio: float = 0.5,
                           seed: int = 42) -> np.ndarray:
    """
    Piksellerin rastgele bir bölümünü darbe (tuz ve biber) gürültüsü ile bozar.

    **Algoritma:**

    1. Bozulacak piksel sayısını hesapla:
           n_bozuk = round(density × H × W)

    2. Düzleştirilmiş görüntü dizin uzayından [0, H×W) rastgele (tekrarsız)
       ``n_bozuk`` piksel konumu çiz.

    3. Bozuk konumları tuz ve biber alt kümelerine böl:
           n_tuz    = round(salt_ratio × n_bozuk)
           n_biber  = n_bozuk − n_tuz

    4. Tuz piksellerini 255'e (beyaz), biber piksellerini 0'a (siyah) ayarla;
       renkli görüntülerde **tüm kanallara** aynı anda uygulanır; böylece
       bozuk pikseller renkli nokta değil saf beyaz veya siyah olur.

    **Bu gürültü modeli neden?**
    Tuz-biber gürültüsü sensör hatalarını, iletim kesintilerini ve ADC
    doygunluğunu taklit eder. Medyan filtresi (aşağıda) özellikle bu tür
    gürültüyü kenarları bulanıklaştırmadan temizlemek için tasarlanmıştır.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    density : float
        Bozulacak toplam piksel oranı, [0, 1] aralığında. Varsayılan 0.05 (%5).
    salt_ratio : float
        Bozuk piksellerin kaçının tuz (beyaz) olacağı oranı.
        Varsayılan 0.5 (eşit tuz ve biber).
    seed : int
        Tekrarlanabilirlik için rastgele tohum. Varsayılan 42.

    Döndürür
    --------
    np.ndarray
        Gürültülü görüntü, girişle aynı şekil ve dtype.
    """
    if not (0.0 <= density <= 1.0):
        raise ValueError(f"density [0, 1] aralığında olmalıdır; alınan: {density}")

    rng        = np.random.default_rng(int(seed))
    out        = img.copy()
    h, w       = out.shape[:2]
    n_pixels   = h * w
    n_corrupt  = int(round(density * n_pixels))

    if n_corrupt == 0:
        return out

    # Benzersiz düz indeksler çiz
    flat_idx   = rng.choice(n_pixels, size=n_corrupt, replace=False)
    rows       = flat_idx // w
    cols       = flat_idx %  w

    n_salt     = int(round(salt_ratio * n_corrupt))
    salt_r, salt_c     = rows[:n_salt],  cols[:n_salt]
    pepper_r, pepper_c = rows[n_salt:],  cols[n_salt:]

    if out.ndim == 2:
        out[salt_r,   salt_c]   = 255
        out[pepper_r, pepper_c] = 0
    else:
        # Tüm kanallara uygula; bozuk pikseller beyaz / siyah olsun,
        # rastgele renkli değil
        out[salt_r,   salt_c,   :] = 255
        out[pepper_r, pepper_c, :] = 0

    return out


# ═══════════════════════════════════════════════════════════════════════════
# 2.  mean_filter
# ═══════════════════════════════════════════════════════════════════════════

def mean_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Ortalama (kutu) filtresi — her pikseli (kernel_size × kernel_size)
    komşuluğundaki tüm piksellerin aritmetik ortalaması ile değiştirir.

    **Algoritma (manuel kayan pencere konvolüsyonu):**

    k × k boyutunda tekdüze bir kutu çekirdeğin her girdisi 1/k²'ye eşittir.
    1/k² ile çarpıp toplamak yerine, yansıma dolgulu görüntünün kaydırılmış
    kopyalarını biriktirerek pencereyi kaydırırız:

        birikimli = Σ_{dr=0}^{k-1}  Σ_{dc=0}^{k-1}
                        dolgulu[y+dr, x+dc]

        çıkış[y, x] = birikimli[y, x] / k²

    Bu O(H·W·k²) karmaşıklığındadır ancak yalnızca NumPy dilimleme
    toplamları kullanır — Python piksel döngüsü yok, harici kütüphane yok.

    **Yansıma dolgusu** kenarlarda kullanılır; kenar pikseller simetrik
    komşuluk ile bulanıklaştırılır (koyu kenar yapay görüntüsü olmaz).

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    kernel_size : int
        Ortalama alma penceresinin kenar uzunluğu (tek sayıya zorlanır, min 3).

    Döndürür
    --------
    np.ndarray
        Ortalama filtreli görüntü, aynı şekil, dtype uint8.
    """
    kernel_size = max(3, kernel_size | 1)    # tek sayıya zorla
    k           = kernel_size
    ph = pw     = k // 2

    def _mean_plane(plane: np.ndarray) -> np.ndarray:
        """Tek bir (H, W) uint8 kanalını ortala."""
        padded = np.pad(
            plane.astype(np.float32),
            ((ph, ph), (pw, pw)),
            mode='reflect',
        )
        h, w   = plane.shape
        accum  = np.zeros((h, w), dtype=np.float32)

        # k² kaydırılmış dilimi biriktir — bu, konvolüsyon fonksiyonu
        # kullanmadan kutu çekirdeği ile manuel konvolüsyondur
        for dr in range(k):
            for dc in range(k):
                accum += padded[dr:dr + h, dc:dc + w]

        return np.clip(np.round(accum / (k * k)), 0, 255).astype(np.uint8)

    if img.ndim == 2:
        return _mean_plane(img)
    return np.stack(
        [_mean_plane(img[:, :, c]) for c in range(img.shape[2])],
        axis=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  median_filter
# ═══════════════════════════════════════════════════════════════════════════

def median_filter(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Medyan filtresi — her pikseli (kernel_size × kernel_size) komşuluğunun
    **medyan** değeri ile değiştirir.

    **Algoritma:**

    1. Görüntüyü her kenardan (k//2) piksel yansıma dolgusu ile genişlet.

    2. ``numpy.lib.stride_tricks.as_strided`` kullanarak (H, W, k, k)
       şeklinde sıfır kopyalı bir görünüm oluştur; ``yamalar[y, x]``,
       (y, x) piksel merkezli k×k komşuluğudur:

           şekil    = (H, W, k, k)
           adımlar  = (satır_adım, sütun_adım, satır_adım, sütun_adım)

    3. Piksel başına düz pencere elde etmek için (H, W, k²) şekline dönüştür.

    4. Son eksen boyunca ``numpy.sort`` ile her pencereyi sırala ve orta
       elemanı (k²//2 indeksi) seç:

           medyan[y, x] = sıralı_pencere[y, x, k²//2]

       Bu O(H·W·k²·log(k²)) karmaşıklığındadır ama tamamen vektörleştirilmiş —
       **scipy.ndimage.median_filter yok**, Python piksel döngüsü yok.

    **Tuz-biber için medyan neden?**
    Aşırı darbe değerleri (0 veya 255) sıralı pencerenin uçlarına itilir;
    medyan ortada oturur ve onları tamamen görmezden gelir.

    Parametreler
    ------------
    img : np.ndarray
        Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    kernel_size : int
        Medyan penceresinin kenar uzunluğu (tek sayıya zorlanır, min 3).

    Döndürür
    --------
    np.ndarray
        Medyan filtreli görüntü, aynı şekil, dtype uint8.
    """
    kernel_size = max(3, kernel_size | 1)
    k           = kernel_size
    ph = pw     = k // 2

    def _median_plane(plane: np.ndarray) -> np.ndarray:
        """Tek bir (H, W) uint8 kanalına medyan filtresi uygula."""
        padded = np.pad(
            plane,
            ((ph, ph), (pw, pw)),
            mode='reflect',
        )
        h, w = plane.shape

        # (H, W, k, k) kayan pencere görünümü oluştur (sıfır kopyalı)
        shape   = (h, w, k, k)
        strides = (padded.strides[0], padded.strides[1],
                   padded.strides[0], padded.strides[1])
        patches = as_strided(padded, shape=shape, strides=strides)

        # Her yamayı (H, W, k²) şekline düzleştir, sırala, ortayı seç
        flat    = patches.reshape(h, w, k * k).astype(np.int32)
        sorted_ = np.sort(flat, axis=2)          # numpy.sort, scipy değil
        median_ = sorted_[:, :, (k * k) // 2]   # orta eleman

        return median_.astype(np.uint8)

    if img.ndim == 2:
        return _median_plane(img)
    return np.stack(
        [_median_plane(img[:, :, c]) for c in range(img.shape[2])],
        axis=-1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Gauss gürültüsü  (değiştirilmedi — app.py geriye dönük uyumluluk için korundu)
# ═══════════════════════════════════════════════════════════════════════════

def gaussian_noise(img: np.ndarray,
                   mean: float = 0.0,
                   std:  float = 25.0,
                   seed: int   = 42) -> np.ndarray:
    """
    Her piksele sıfır ortalımlı Gauss (beyaz) gürültüsü ekler.

    **Formül:**

        çıkış = clip(img + N(ortalama, std²), 0, 255)

    Gauss gürültüsü görüntü sensörlerindeki ısıl/okuma gürültüsünü modeller.
    Tuz-biberden farklı olarak seyrek bir alt kümeyi şiddetle bozmak yerine
    her pikseli hafifçe etkiler.

    Parametreler
    ------------
    img  : np.ndarray   Giriş görüntüsü (H, W) veya (H, W, 3), dtype uint8.
    mean : float        Gürültü dağılımının ortalaması (varsayılan 0.0).
    std  : float        Yoğunluk birimlerinde standart sapma (varsayılan 25.0).
    seed : int          Rastgele tohum (varsayılan 42).

    Döndürür
    --------
    np.ndarray  Gürültülü görüntü, aynı şekil, dtype uint8.
    """
    rng   = np.random.default_rng(int(seed))
    noise = rng.normal(float(mean), float(std), img.shape).astype(np.float32)
    out   = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Geriye dönük uyumluluk için takma ad (app.py noise.salt_and_pepper çağırır)
# ═══════════════════════════════════════════════════════════════════════════

def salt_and_pepper(img: np.ndarray,
                    amount: float = 0.05,
                    salt_vs_pepper: float = 0.5,
                    seed: int = 42) -> np.ndarray:
    """
    Orijinal parametre adlarını kullanan ``add_salt_pepper_noise`` takma adı.
    ``amount`` → ``density``, ``salt_vs_pepper`` → ``salt_ratio`` ile eşleşir.
    """
    return add_salt_pepper_noise(
        img,
        density=float(amount),
        salt_ratio=float(salt_vs_pepper),
        seed=int(seed),
    )
