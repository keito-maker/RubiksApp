import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from skimage import color

st.title("ルービックキューブ モザイク作成")

# =========================
# ルービックキューブ色
# =========================
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

# =========================
# 色分類
# =========================
def classify_pixel(rgb):
    rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb_arr.reshape(1, 1, 3)).reshape(3)
    L, a, b_lab = lab
    r, g, b = rgb

    # 黒髪補正
    if L < 28 and r < 70 and g < 70 and b < 70:
        return palette["blue"]

    # 肌色補正
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
# 入力
# =========================
cols = st.number_input("横のキューブ数", min_value=1, value=20, key="cols")
rows = st.number_input("縦のキューブ数", min_value=1, value=20, key="rows")

st.divider()

uploaded_file = st.file_uploader(
    "画像をアップロード", type=["png", "jpg", "jpeg"], key="uploader"
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = img.size

    st.write("### 外枠の位置とサイズを入力")

    x = st.number_input("左上X", min_value=0, max_value=img_w, value=0, key="x")
    y = st.number_input("左上Y", min_value=0, max_value=img_h, value=0, key="y")
    w = st.number_input("幅", min_value=10, max_value=img_w, value=img_w, key="w")
    h = st.number_input("高さ", min_value=10, max_value=img_h, value=img_h, key="h")

    w = min(w, img_w - x)
    h = min(h, img_h - y)

    preview = img.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    st.image(preview, caption="外枠プレビュー", width="stretch")

    if st.button("この範囲でモザイク生成", key="generate"):
        crop = img.crop((x, y, x + w, y + h))

        # =========================
        # 小さいモザイク作成
        # =========================
        small_cols = cols * 3
        small_rows = rows * 3

        small = crop.resize((small_cols, small_rows), Image.BILINEAR)

        quantized = Image.new("RGB", (small_cols, small_rows))
        for yy in range(small_rows):
            for xx in range(small_cols):
                rgb = small.getpixel((xx, yy))
                quantized.putpixel((xx, yy), classify_pixel(rgb))

        # =========================
        # 本来サイズへ拡大
        # =========================
        mosaic = quantized.resize((w, h), Image.NEAREST)

        # 表示用にさらに拡大
        DISPLAY_SCALE = 6
        display_w = w * DISPLAY_SCALE
        display_h = h * DISPLAY_SCALE

        mosaic_display = mosaic.resize(
            (display_w, display_h),
            Image.NEAREST
        )

        # =========================
        # 線描画（拡大後！）
        # =========================
        pixels = mosaic_display.load()
        width, height = mosaic_display.size

        cell = max(1, width // small_cols)
        cube = cell * 3  # ← ここが太線間隔の核心

        THIN = 1
        THICK = 3
        LINE_COLOR = (0, 0, 0)

        # -------- 細線 --------
        for i in range(0, width, cell):
            for t in range(THIN):
                xi = i + t
                if xi >= width:
                    continue
                for j in range(height):
                    pixels[xi, j] = LINE_COLOR

        for j in range(0, height, cell):
            for t in range(THIN):
                yj = j + t
                if yj >= height:
                    continue
                for i in range(width):
                    pixels[i, yj] = LINE_COLOR

        # -------- 太線（3マスごと）--------
        for i in range(0, width, cube):
            for t in range(THICK):
                xi = i + t
                if xi >= width:
                    continue
                for j in range(height):
                    pixels[xi, j] = LINE_COLOR

        for j in range(0, height, cube):
            for t in range(THICK):
                yj = j + t
                if yj >= height:
                    continue
                for i in range(width):
                    pixels[i, yj] = LINE_COLOR

        st.image(mosaic_display, caption="ルービックモザイク結果", width="stretch")