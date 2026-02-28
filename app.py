import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
from skimage import color

# =========================================================
# アプリタイトル
# =========================================================
st.title("ルービックキューブ モザイク作成")

# =========================================================
# ルービックキューブの基本カラーパレット
# =========================================================
palette = {
    "red": (200, 0, 0),
    "orange": (255, 120, 0),
    "yellow": (255, 235, 60),
    "green": (0, 150, 0),
    "blue": (0, 70, 200),
    "white": (255, 255, 255),
}

# Lab色空間で距離計算するための前処理
palette_rgb = np.array(list(palette.values()), dtype=np.float32) / 255.0
palette_lab = color.rgb2lab(palette_rgb.reshape(1, -1, 3)).reshape(-1, 3)


# =========================================================
# 色分類関数（RGB → 最も近いキューブ色）
# =========================================================
def classify_pixel(rgb):
    """
    入力RGBをルービックキューブの近似色に変換する。
    判定ロジックは元コードを完全維持。
    """
    rgb_arr = np.array(rgb, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb_arr.reshape(1, 1, 3)).reshape(3)
    L, a, b_lab = lab
    r, g, b = rgb

    # --- 暗すぎる青の補正 ---
    if L < 28 and r < 70 and g < 70 and b < 70:
        return palette["blue"]

    # --- 赤〜黄系の特別判定 ---
    if (r > g > b) and (r > 95) and (L > 55):
        if L > 74:
            return palette["yellow"]
        if 60 < L <= 74:
            return palette["orange"]
        if 58 < L <= 59:
            return palette["red"]

    # --- Lab距離による最近傍色選択 ---
    distances = np.linalg.norm(palette_lab - lab, axis=1)

    # 白は選ばれすぎるため少し不利にする
    white_index = list(palette.keys()).index("white")
    distances[white_index] *= 1.20

    idx = np.argmin(distances)
    return tuple((palette_rgb[idx] * 255).astype(np.uint8))


# =========================================================
# モザイクのキューブ数入力
# =========================================================
cols = st.number_input("横のキューブ数", min_value=1, value=20, key="cols")
rows = st.number_input("縦のキューブ数", min_value=1, value=20, key="rows")

st.divider()

# =========================================================
# 画像アップロード
# =========================================================
uploaded_file = st.file_uploader(
    "画像をアップロード",
    type=["png", "jpg", "jpeg"],
    key="uploader",
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_w, img_h = img.size

    st.write("### 外枠の位置とサイズを入力")

    # =====================================================
    # セッション状態の初期化
    # =====================================================
    if "x" not in st.session_state:
        st.session_state.x = 0
    if "y" not in st.session_state:
        st.session_state.y = 0
    if "width" not in st.session_state:
        st.session_state.width = img_w
    if "height" not in st.session_state:
        st.session_state.height = int(st.session_state.width * rows / cols)

    # =====================================================
    # 初回のみ縦横比を同期
    # =====================================================
    if "ratio_synced_for_image" not in st.session_state:
        st.session_state.ratio_synced_for_image = True

        max_w0 = max(1, img_w - st.session_state.x)
        max_h0 = max(1, img_h - st.session_state.y)

        target_h = int(st.session_state.width * rows / cols)

        if target_h > max_h0:
            st.session_state.height = max_h0
            st.session_state.width = int(max_h0 * cols / rows)
        else:
            st.session_state.height = target_h

    # =====================================================
    # サイズ・位置の制約処理（クランプ）
    # =====================================================
    def clamp_all():
        """位置変更時：画像内に収まるよう全体を調整"""
        max_w = max(1, img_w - st.session_state.x)
        max_h = max(1, img_h - st.session_state.y)

        calc_h = int(st.session_state.width * rows / cols)

        if calc_h > max_h:
            st.session_state.height = max_h
            st.session_state.width = int(max_h * cols / rows)
        else:
            st.session_state.height = calc_h

        st.session_state.width = min(st.session_state.width, max_w)
        st.session_state.height = min(st.session_state.height, max_h)

    def clamp_size_from_width():
        """幅変更時の比率維持調整"""
        clamp_all()

    def clamp_size_from_height():
        """高さ変更時の比率維持調整"""
        max_w = max(1, img_w - st.session_state.x)
        max_h = max(1, img_h - st.session_state.y)

        calc_w = int(st.session_state.height * cols / rows)

        if calc_w > max_w:
            st.session_state.width = max_w
            st.session_state.height = int(max_w * rows / cols)
            st.warning("⚠️ 指定サイズが大きすぎたため、最大サイズに調整しました。")
        else:
            st.session_state.width = calc_w

        st.session_state.height = min(st.session_state.height, max_h)

    # コールバック
    def width_changed():
        clamp_size_from_width()

    def height_changed():
        clamp_size_from_height()

    def xy_changed():
        clamp_all()

    max_w = max(1, img_w - st.session_state.x)
    max_h = max(1, img_h - st.session_state.y)

    # =====================================================
    # 入力UI（切り取り範囲）
    # =====================================================
    x = st.number_input("左上X", 0, img_w - 1, st.session_state.x, key="x", on_change=xy_changed)
    y = st.number_input("左上Y", 0, img_h - 1, st.session_state.y, key="y", on_change=xy_changed)

    width = st.number_input(
        "幅",
        min_value=10,
        max_value=max_w,
        value=min(st.session_state.width, max_w),
        key="width",
        on_change=width_changed,
    )

    height = st.number_input(
        "高さ",
        min_value=10 if max_h >= 10 else 1,
        max_value=max_h,
        value=min(st.session_state.height, max_h),
        key="height",
        on_change=height_changed,
    )

    # =====================================================
    # 外枠プレビュー表示
    # =====================================================
    width = min(width, img_w - x)
    height = min(height, img_h - y)

    preview = img.copy()
    draw = ImageDraw.Draw(preview)
    draw.rectangle([x, y, x + width, y + height], outline="red", width=2)

    st.image(preview, caption="外枠プレビュー", width=200)

    # =====================================================
    # モザイク生成（高速化版）
    # =====================================================
    if st.button("この範囲でモザイク生成", key="generate"):
        crop = img.crop((x, y, x + width, y + height))

        # --- 3×3単位に縮小 ---
        small_cols = cols * 3
        small_rows = rows * 3
        small = crop.resize((small_cols, small_rows), Image.BILINEAR)

        # --- NumPyで高速色量子化 ---
        small_np = np.array(small)
        out_np = np.zeros_like(small_np)

        for yy in range(small_rows):
            for xx in range(small_cols):
                out_np[yy, xx] = classify_pixel(tuple(small_np[yy, xx]))

        quantized = Image.fromarray(out_np.astype(np.uint8))

        # =================================================
        # 表示用に拡大
        # =================================================
        DISPLAY_SCALE = 6
        display_w = small_cols * DISPLAY_SCALE
        display_h = small_rows * DISPLAY_SCALE
        mosaic_display = quantized.resize((display_w, display_h), Image.NEAREST)

        # =================================================
        # グリッド線描画
        # =================================================
        pixels = mosaic_display.load()
        w_disp, h_disp = mosaic_display.size

        cell = DISPLAY_SCALE
        cube = cell * 3

        THIN = 1
        THICK = 2
        LINE_COLOR = (0, 0, 0)

        # 細線（ステッカー境界）
        for i in range(0, w_disp, cell):
            for t in range(THIN):
                xi = i + t
                if xi < w_disp:
                    for j in range(h_disp):
                        pixels[xi, j] = LINE_COLOR

        for j in range(0, h_disp, cell):
            for t in range(THIN):
                yj = j + t
                if yj < h_disp:
                    for i in range(w_disp):
                        pixels[i, yj] = LINE_COLOR

        # 太線（キューブ境界）
        for i in range(0, w_disp, cube):
            for t in range(THICK):
                xi = i + t
                if xi < w_disp:
                    for j in range(h_disp):
                        pixels[xi, j] = LINE_COLOR

        for j in range(0, h_disp, cube):
            for t in range(THICK):
                yj = j + t
                if yj < h_disp:
                    for i in range(w_disp):
                        pixels[i, yj] = LINE_COLOR

        # =================================================
        # 結果表示
        # =================================================
        st.image(
            mosaic_display,
            caption=f"ルービックモザイク結果 ({cols}×{rows})",
            width=700,
        )