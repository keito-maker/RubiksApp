# app.py
from PIL import Image, ImageFilter
import numpy as np
from skimage import color
import streamlit as st

# =========================
# 設定
# =========================
GRID = 60
CUBE = 3
OUTPUT_SIZE = 900
THIN = 1
THICK = 4

palette = {
    "red": (200, 0, 0),
    "orange": (255, 120, 0),
    "yellow": (255, 235, 60),
    "green": (0, 150, 0),
    "blue": (0, 70, 200),
    "white": (255, 255, 255),
}

palette_rgb = np.array(list(palette.values()), dtype=np.float32) / 255.0
palette_lab = color.rgb2lab(palette_rgb.reshape(1, -1, 3)).reshape(-1, 3)

EDGE_THRESHOLD = 120

# =========================
# 色分類
# =========================
def classify_pixel(rgb):
    rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb_arr.reshape(1, 1, 3)).reshape(3)
    L, a, b_lab = lab
    r, g, b = rgb

    # 黒髪 → 青
    if L < 28 and r < 70 and g < 70 and b < 70:
        return palette["blue"]

    # 肌色
    if (r > g > b) and (r > 95) and (L > 55):
        if L > 74:
            return palette["yellow"]
        if 60 < L <= 74:
            return palette["orange"]
        if 58 < L <= 59:
            return palette["red"]

    # Lab最近色
    distances = np.linalg.norm(palette_lab - lab, axis=1)
    white_index = list(palette.keys()).index("white")
    distances[white_index] *= 1.20
    idx = np.argmin(distances)
    return tuple((palette_rgb[idx] * 255).astype(np.uint8))

# =========================
# Streamlit UI
# =========================
st.title("モザイク化ルービックアプリ")
st.write("画像をアップロードするとモザイク化されます。")

uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    # 正方形トリミング
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))

    # エッジ
    edge_img = img.filter(ImageFilter.FIND_EDGES).convert("L")
    edge_small = edge_img.resize((GRID, GRID))

    small = img.resize((GRID, GRID), Image.BILINEAR)
    quantized = Image.new("RGB", (GRID, GRID))

    for y in range(GRID):
        for x in range(GRID):
            edge_val = edge_small.getpixel((x, y))

            if edge_val > EDGE_THRESHOLD and edge_val < 240:
                if edge_val > 230:
                    quantized.putpixel((x, y), palette["red"])
                else:
                    quantized.putpixel((x, y), palette["orange"])
            else:
                rgb = small.getpixel((x, y))
                quantized.putpixel((x, y), classify_pixel(rgb))

    # 拡大
    final = quantized.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.NEAREST)
    pixels = final.load()
    cell = OUTPUT_SIZE // GRID
    cube = cell * CUBE

    # 細線
    for i in range(0, OUTPUT_SIZE, cell):
        for t in range(THIN):
            if i + t < OUTPUT_SIZE:
                for j in range(OUTPUT_SIZE):
                    pixels[i+t, j] = (0, 0, 0)
                    pixels[j, i+t] = (0, 0, 0)

    # 太線
    for i in range(0, OUTPUT_SIZE, cube):
        for t in range(THICK):
            if i + t < OUTPUT_SIZE:
                for j in range(OUTPUT_SIZE):
                    pixels[i+t, j] = (0, 0, 0)
                    pixels[j, i+t] = (0, 0, 0)

    st.image(final, caption="モザイク化された画像", use_column_width=True)