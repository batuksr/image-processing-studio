"""
Görüntü İşleme Web Uygulaması için Flask uygulaması.

Uç Noktalar (Endpoints):
  POST /upload  – bir görüntü dosyası yükler (PNG/JPG/JPEG, max 10 MB)
  POST /apply   – base64 kodlu bir görüntüye işlem operasyonu uygular
  GET  /        – tek sayfalık ön yüzü sunar
"""
import base64
import io
import json
import os
import time
import traceback

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image

# ------------------------------------------------------------------
# Tüm işlem modüllerini içe aktar (saf NumPy uygulamaları)
# ------------------------------------------------------------------
from processing import basic, histogram, arithmetic, filters, edge, noise, morphology, colorspace

# ------------------------------------------------------------------
# Uygulama yapılandırması
# ------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}


# ------------------------------------------------------------------
# Yardımcı fonksiyonlar
# ------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _pil_to_ndarray(pil_img: Image.Image) -> np.ndarray:
    """Bir PIL Görüntüsünü NumPy uint8 dizisine dönüştürür (RGB veya gri tonlamalı)."""
    pil_img = pil_img.convert("RGB")
    return np.array(pil_img, dtype=np.uint8)


def _ndarray_to_b64(arr: np.ndarray) -> str:
    """Numpy dizisini (uint8) base64 PNG dizesine dönüştürür."""
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    else:
        raise ValueError(f"Desteklenmeyen dizi şekli: {arr.shape}")
    pil_img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def _b64_to_ndarray(b64: str) -> np.ndarray:
    """Base64 kodlu bir görüntü dizesini NumPy uint8 RGB dizisine çözer."""
    # Varsa data-URL önekini kaldır
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil, dtype=np.uint8)


def _ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Görüntüleme için 2 boyutlu gri tonlamalı veya 1 kanallı diziyi RGB'ye dönüştürür."""
    if arr.ndim == 2:
        return np.stack([arr, arr, arr], axis=-1)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.concatenate([arr, arr, arr], axis=-1)
    return arr


def _angle_to_uint8(angle_deg: np.ndarray) -> np.ndarray:
    """
    Float32 açı haritasını (−180° … +180° aralığındaki değerler) uint8 [0, 255] aralığına ölçekler.
    Aralığı [0, 360] yapmak için önce 180 ile ötele, ardından 360'a böl.
    """
    scaled = (angle_deg + 180.0) / 360.0 * 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Rotalar (Routes)
# ------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Bir görüntü yükler ve base64 PNG olarak döndürür."""
    if "image" not in request.files:
        return jsonify({"error": "İstekte image (görüntü) alanı yok."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Dosya adı boş."}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Yalnızca PNG/JPG/JPEG dosyaları kabul edilir."}), 400

    try:
        pil = Image.open(io.BytesIO(file.read())).convert("RGB")
        # >4000px ise otomatik olarak alt örnekleme yap
        if pil.width > 4000 or pil.height > 4000:
            pil.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
        arr = np.array(pil, dtype=np.uint8)
        b64 = _ndarray_to_b64(arr)
        return jsonify({
            "image_data": b64,
            "width": arr.shape[1],
            "height": arr.shape[0],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ------------------------------------------------------------------
# Operasyon dağıtıcısı (Operation dispatcher)
# ------------------------------------------------------------------

OPERATIONS = {
    # Temel (Basic)
    "grayscale":          lambda img, p: _ensure_rgb(basic.to_grayscale(img)),
    "binary":             lambda img, p: _ensure_rgb(basic.to_binary(img, int(p.get("threshold", 128)))),
    "rotate":             lambda img, p: _ensure_rgb(basic.rotate_image(img, float(p.get("angle", 90)), bool(p.get("expand", True)))),
    "crop":               lambda img, p: _ensure_rgb(basic.crop_image(img,
                                                     int(p.get("x", 0)), int(p.get("y", 0)),
                                                     int(p.get("width", img.shape[1])),
                                                     int(p.get("height", img.shape[0])))),
    "zoom":               lambda img, p: _ensure_rgb(basic.zoom_image(img, float(p.get("scale_x", 1.5)), float(p.get("scale_y", 1.5)))),

    # Histogram  ((işlenmiş_görüntü, hist_verisi) demetleri döndürür; böylece
    #             /apply rotası histogram verilerini yanıta ekleyebilir)
    "hist_stretch":  lambda img, p: (
        histogram.stretch_histogram(img),
        histogram.compute_histogram(img, histogram.stretch_histogram(img)),
    ),
    "hist_equalize": lambda img, p: (
        histogram.equalize_histogram(img),
        histogram.compute_histogram(img, histogram.equalize_histogram(img)),
    ),

    # Aritmetik (Arithmetic) (ikinci görüntü gerektirir)
    "add_images":         lambda img, p: arithmetic.add_images(img, _b64_to_ndarray(p["image2"]),
                                                                float(p.get("alpha", 0.5))),
    "multiply_images":    lambda img, p: arithmetic.multiply_images(img, _b64_to_ndarray(p["image2"])),
    "subtract_images":    lambda img, p: arithmetic.subtract_images(img, _b64_to_ndarray(p["image2"])),

    # Filtreler (Filters)
    "blur_unified":       lambda img, p: filters.blur_image(img, p.get("method", "gaussian"),
                                                             int(p.get("kernel_size", 5)),
                                                             float(p.get("sigma", 1.0))),
    "gaussian_filter":    lambda img, p: filters.gaussian_filter(img,
                                                                  int(p.get("kernel_size", 5)),
                                                                  float(p.get("sigma", 1.0))),
    "box_blur":           lambda img, p: filters.blur_image(img, 'box',
                                                             int(p.get("kernel_size", 5))),
    "sharpen":            lambda img, p: filters.sharpen(img, float(p.get("strength", 1.0))),
    "brightness":         lambda img, p: filters.brightness_adjust(img, int(p.get("value", 30))),
    "clahe":              lambda img, p: filters.clahe(img,
                                                        float(p.get("clip_limit", 2.0)),
                                                        (int(p.get("tile_rows", 8)),
                                                         int(p.get("tile_cols", 8)))),
    "adaptive_eq":        lambda img, p: filters.adaptive_equalization(img,
                                                                         int(p.get("tile_size", 8)),
                                                                         float(p.get("clip_limit", 2.0))),

    # Kenar tespiti (Edge detection)
    "sobel":                    lambda img, p: edge.sobel(img),
    "sobel_x":                  lambda img, p: edge.sobel_x(img),
    "sobel_y":                  lambda img, p: edge.sobel_y(img),
    "laplacian":                lambda img, p: edge.laplacian(img),
    "sobel_edge":               lambda img, p: edge.sobel_edge_detection(
                                                   img,
                                                   int(p.get("threshold", 50))),
    "gradient_direction":       lambda img, p: _ensure_rgb(
                                                   _angle_to_uint8(
                                                       edge.compute_gradient_direction(img))),

    # Gürültü (Noise)
    "salt_pepper_noise":  lambda img, p: noise.salt_and_pepper(img,
                                                                float(p.get("amount", 0.05)),
                                                                float(p.get("salt_vs_pepper", 0.5))),
    "add_salt_pepper":    lambda img, p: noise.add_salt_pepper_noise(img,
                                                                      float(p.get("density", 0.05)),
                                                                      float(p.get("salt_ratio", 0.5))),
    "gaussian_noise":     lambda img, p: noise.gaussian_noise(img,
                                                               float(p.get("mean", 0.0)),
                                                               float(p.get("std", 25.0))),
    "denoise_unified":    lambda img, p: noise.mean_filter(img, int(p.get("kernel_size", 3))) if p.get("method") == "mean" else noise.median_filter(img, int(p.get("kernel_size", 3))),

    # Morfoloji — op parametresi değerleri morphology.py'deki gerçek fonksiyon adlarıyla eşleşir:
    # dilate, erode, opening, closing, gradient
    "morphology_unified": lambda img, p: (
        morphology.dilate(img,  kernel_size=int(p.get("kernel_size", 3)), shape=p.get("shape", "rect")) if p.get("op","dilate") == "dilate"   else
        morphology.erode(img,   kernel_size=int(p.get("kernel_size", 3)), shape=p.get("shape", "rect")) if p.get("op") == "erode"   else
        morphology.opening(img, int(p.get("kernel_size", 3)),             p.get("shape", "rect"))       if p.get("op") == "opening" else
        morphology.closing(img, int(p.get("kernel_size", 3)),             p.get("shape", "rect"))       if p.get("op") == "closing" else
        morphology.gradient(img,int(p.get("kernel_size", 3)),             p.get("shape", "rect"))
    ),

    # Renk uzayı (Color space)
    "color_space_unified":lambda img, p: getattr(colorspace, p.get("conversion", "rgb_to_hsv_display"))(img) if p.get("conversion") in ["rgb_to_hsv_display", "rgb_to_ycbcr", "rgb_to_lab"] else colorspace.pseudo_color(img, p.get("conversion").replace("pseudo_color_", "")) if p.get("conversion").startswith("pseudo_color_") else colorspace.rgb_to_hsv_display(img),
}


@app.route("/apply", methods=["POST"])
def apply_operation():
    """Base64 bir görüntüye bir işlem operasyonu uygular."""
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Geçersiz JSON gövdesi."}), 400

    operation = body.get("operation", "")
    params = body.get("params", {})
    image_b64 = body.get("image_data", "")

    if not operation:
        return jsonify({"error": "İşlem belirtilmedi."}), 400
    if not image_b64:
        return jsonify({"error": "Görüntü verisi (image_data) sağlanmadı."}), 400
    if operation not in OPERATIONS:
        return jsonify({"error": f"Bilinmeyen işlem '{operation}'"}), 400

    try:
        t0 = time.time()
        img = _b64_to_ndarray(image_b64)
        
        # Renk operasyonları için gri tonlamayı otomatik olarak RGB'ye dönüştür
        if operation == "color_space_unified" and img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)

        warning_msg = None
        # İkili (binary) değilse morfolojik işlem için otomatik eşikleme yap
        if operation == "morphology_unified":
            unique_vals = np.unique(img)
            if len(unique_vals) > 2:
                warning_msg = "Morfolojik işlem için ikili olmayan görüntüye otomatik eşikleme uygulandı."
                img = basic.to_binary(img, 128)

        op_func = OPERATIONS[operation]
        result = op_func(img, params)

        hist_data = None
        if isinstance(result, tuple) and len(result) == 2:
            processed_img, hist_data = result
        else:
            processed_img = result

        out_b64 = _ndarray_to_b64(processed_img)
        
        t1 = time.time()
        time_ms = int((t1 - t0) * 1000)

        response_data = {
            "image_data": out_b64,
            "width": processed_img.shape[1],
            "height": processed_img.shape[0],
            "time_ms": time_ms
        }
        if hist_data is not None:
            response_data["histogram"] = hist_data
        if warning_msg:
            response_data["warning"] = warning_msg

        return jsonify(response_data)
        
    except KeyError as exc:
        return jsonify({"error": f"Gerekli parametre eksik: {exc}"}), 400
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/operations", methods=["GET"])
def list_operations():
    """Kullanılabilir tüm işlemleri döndürür."""
    return jsonify({"operations": list(OPERATIONS.keys())})


# ------------------------------------------------------------------
# Giriş noktası (Entry point)
# ------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
