# app.py
from PIL import Image, ImageFilter
import numpy as np
from skimage import color
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# =========================
# 設定
# =========================
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
# 色分類関数
# =========================
def classify_pixel(rgb):
    rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb_arr.reshape(1, 1, 3)).reshape(3)
    L, a, b_lab = lab
    r, g, b = rgb

    if L < 28 and r < 70 and g < 70 and b < 70:
        return palette["blue"]
    if (r > g > b) and (r > 95) and (L > 55):
        if L > 74:
            return palette["yellow"]
        if 60 < L <= 74:
            return palette["orange"]
        if 58 < L <= 59:
            return palette["red"]
    distances = np.linalg.norm(palette_lab - lab, axis=1)
    white_index = list(palette.keys()).index("white")
    distances[white_index] *= 1.20
    idx = np.argmin(distances)
    return tuple((palette_rgb[idx] * 255).astype(np.uint8))

# =========================
# Streamlit UI
# =========================
st.title("モザイク化ルービックキューブアートアプリ")
st.write("画像をアップロードして、モザイク化する範囲とキューブ数を指定できます。")

uploaded_file = st.file_uploader("画像を選択してください", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = img.size

    # キューブ数入力
    col1, col2 = st.columns(2)
    with col1:
        GRID_X = st.number_input("横のキューブ数", min_value=1, max_value=100, value=20)
    with col2:
        GRID_Y = st.number_input("縦のキューブ数", min_value=1, max_value=100, value=20)

    st.write("画像上で範囲を選択してください。")

    # Canvasで枠選択
    canvas_result = st_canvas(
        fill_color="rgba(0,0,0,0)",
        stroke_color="red",
        stroke_width=3,
        background_image=img,
        height=img_h,
        width=img_w,
        drawing_mode="rect",
        key="canvas",
    )

    if canvas_result.json_data is not None and st.button("範囲を決定してモザイク化"):
        objects = canvas_result.json_data["objects"]
        if len(objects) > 0:
            rect = objects[-1]  # 最後に描いた矩形を使用
            x0 = int(rect["left"])
            y0 = int(rect["top"])
            x1 = int(x0 + rect["width"])
            y1 = int(y0 + rect["height"])
            x1 = min(x1, img_w)
            y1 = min(y1, img_h)

            cropped = img.crop((x0, y0, x1, y1))

            # エッジ画像
            edge_img = cropped.filter(ImageFilter.FIND_EDGES).convert("L")
            edge_small = edge_img.resize((GRID_X, GRID_Y))

            small = cropped.resize((GRID_X, GRID_Y), Image.BILINEAR)
            quantized = Image.new("RGB", (GRID_X, GRID_Y))

            for y in range(GRID_Y):
                for x in range(GRID_X):
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
            cell_x = OUTPUT_SIZE // GRID_X
            cell_y = OUTPUT_SIZE // GRID_Y
            cube_x = cell_x * CUBE
            cube_y = cell_y * CUBE

            # 細線
            for i in range(0, OUTPUT_SIZE, cell_x):
                for t in range(THIN):
                    if i+t < OUTPUT_SIZE:
                        for j in range(OUTPUT_SIZE):
                            pixels[i+t, j] = (0, 0, 0)
                            pixels[j, i+t] = (0, 0, 0)
            # 太線
            for i in range(0, OUTPUT_SIZE, cube_x):
                for t in range(THICK):
                    if i+t < OUTPUT_SIZE:
                        for j in range(OUTPUT_SIZE):
                            pixels[i+t, j] = (0, 0, 0)
                            pixels[j, i+t] = (0, 0, 0)

            st.image(final, caption="モザイク化された画像", use_column_width=True)
        else:
            st.warning("まず画像上で範囲を描いてください。")