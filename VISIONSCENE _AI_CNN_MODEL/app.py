import io
import os
import html
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.models import load_model


# =========================================================
# APP SETTINGS
# =========================================================
st.set_page_config(
    page_title="VisionScene AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "intel_image_cnn.keras"
IMAGE_SIZE = (256, 256)

# Mixed-scene heuristic for the existing single-label Softmax CNN.
# A second class is shown as part of the final scene only when it has
# meaningful probability and is reasonably close to the primary class.
MIXED_MIN_SECONDARY = 0.20
MIXED_MAX_GAP = 0.30
SECONDARY_NOTE_MIN = 0.10

CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]

CLASS_ICONS = {
    "buildings": "🏙️",
    "forest": "🌲",
    "glacier": "❄️",
    "mountain": "⛰️",
    "sea": "🌊",
    "street": "🛣️",
}


# =========================================================
# THEME STATE
# =========================================================
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"



# =========================================================
# 3D NEON / AI DASHBOARD UI
# =========================================================
st.markdown(
    """
<style>
html, body, [class*="st-"] {
    font-family: "Segoe UI", Inter, Arial, sans-serif;
}

/* ---------- Hide Streamlit top toolbar ---------- */
[data-testid="stHeader"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    display: none !important;
}

#MainMenu {
    visibility: hidden !important;
}

footer {
    visibility: hidden !important;
}

[data-testid="stToolbar"] {
    display: none !important;
}

html, body {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

/* ---------------------------------------------------------
   3D-style neon futuristic background
   Inspired by holographic dashboards + glowing node network
   --------------------------------------------------------- */
.stApp {
    color: #F8FBFF;
    background:
        radial-gradient(circle at 15% 18%, rgba(28, 214, 255, 0.20), transparent 24%),
        radial-gradient(circle at 82% 16%, rgba(131, 92, 255, 0.20), transparent 24%),
        radial-gradient(circle at 72% 72%, rgba(255, 61, 187, 0.14), transparent 26%),
        radial-gradient(circle at 30% 76%, rgba(0, 255, 190, 0.08), transparent 22%),
        linear-gradient(145deg, #02030a 0%, #050913 28%, #08111e 58%, #0b1220 100%);
    overflow-x: hidden;
}

/* floating neon panels + node glows */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: 0;
    pointer-events: none;
    opacity: 0.95;

    background:
        radial-gradient(circle at 18% 20%, rgba(35, 219, 255, 0.20), transparent 18%),
        radial-gradient(circle at 80% 18%, rgba(141, 99, 255, 0.18), transparent 18%),
        radial-gradient(circle at 66% 72%, rgba(255, 60, 190, 0.12), transparent 20%),
        radial-gradient(circle at 28% 74%, rgba(0, 255, 200, 0.07), transparent 18%),

        radial-gradient(circle at 10% 22%, rgba(255, 203, 90, .92) 0 2px, transparent 4px),
        radial-gradient(circle at 16% 30%, rgba(72, 238, 180, .90) 0 2px, transparent 4px),
        radial-gradient(circle at 24% 16%, rgba(75, 230, 255, .88) 0 2px, transparent 4px),
        radial-gradient(circle at 33% 24%, rgba(255, 170, 80, .86) 0 2px, transparent 4px),
        radial-gradient(circle at 46% 18%, rgba(75, 230, 255, .85) 0 2px, transparent 4px),
        radial-gradient(circle at 57% 29%, rgba(160, 110, 255, .82) 0 2px, transparent 4px),
        radial-gradient(circle at 69% 17%, rgba(255, 198, 88, .82) 0 2px, transparent 4px),
        radial-gradient(circle at 82% 26%, rgba(75, 230, 255, .85) 0 2px, transparent 4px),

        linear-gradient(24deg,
            transparent 49.65%,
            rgba(50, 217, 255, .06) 49.82%,
            rgba(50, 217, 255, .06) 50.15%,
            transparent 50.32%
        ),
        linear-gradient(-24deg,
            transparent 49.65%,
            rgba(150, 104, 255, .055) 49.82%,
            rgba(150, 104, 255, .055) 50.15%,
            transparent 50.32%
        );
    background-size:
        auto, auto, auto, auto,
        auto, auto, auto, auto, auto, auto, auto, auto,
        240px 175px,
        280px 195px;
    filter:
        drop-shadow(0 0 8px rgba(50, 217, 255, .12))
        drop-shadow(0 0 12px rgba(150, 104, 255, .08));
    animation: auroraFloat 16s ease-in-out infinite alternate;
}

/* perspective floor + network connection lines */
.stApp::after {
    content: "";
    position: fixed;
    left: -12%;
    right: -12%;
    bottom: -34%;
    height: 74%;
    z-index: 0;
    pointer-events: none;

    background-image:
        linear-gradient(rgba(42, 214, 255, .08) 1px, transparent 1px),
        linear-gradient(90deg, rgba(126, 91, 255, .07) 1px, transparent 1px);

    background-size: 56px 56px;
    transform:
        perspective(820px)
        rotateX(66deg)
        scale(1.18);
    transform-origin: center bottom;
    opacity: .42;
    filter:
        drop-shadow(0 0 10px rgba(42, 214, 255, .06));
    animation: gridPulse 7s ease-in-out infinite;
}

.block-container {
    position: relative;
    z-index: 2;
    max-width: 1180px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ---------- Hero ---------- */
.hero {
    border-radius: 30px;
    padding: 2.25rem 2.4rem;
    background:
        linear-gradient(145deg, rgba(6, 14, 24, 0.97), rgba(10, 16, 26, 0.95));
    border: 1px solid rgba(135, 227, 255, 0.20);
    box-shadow:
        0 28px 75px rgba(0,0,0,.38),
        0 0 38px rgba(39,228,255,.06),
        inset 0 1px 0 rgba(255,255,255,.06);
    margin-bottom: 1rem;
    backdrop-filter: blur(14px);
}

.greeting {
    color: #90EDFF;
    font-size: .97rem;
    font-weight: 900;
    letter-spacing: .10em;
    margin-bottom: .55rem;
}

.hero-title {
    font-size: 3.25rem;
    line-height: 1.03;
    font-weight: 950;
    letter-spacing: -1.45px;
    margin: 0;
    background: linear-gradient(90deg, #FFFFFF 0%, #90EDFF 35%, #D1C3FF 68%, #FF90E9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-text {
    color: #C9D9EA;
    font-size: 1.08rem;
    line-height: 1.68;
    max-width: 850px;
    margin-top: .8rem;
}

/* ---------- Chips ---------- */
.chip {
    display: inline-block;
    padding: .50rem .84rem;
    margin: .22rem .22rem .22rem 0;
    border-radius: 999px;
    color: #F5FBFF;
    background: rgba(255,255,255,.09);
    border: 1px solid rgba(140, 228, 255, .16);
    font-weight: 780;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
}

/* ---------- Glass cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(145deg, rgba(6, 15, 26, 0.97), rgba(10, 16, 25, 0.95));
    border: 1px solid rgba(118, 214, 255, .18) !important;
    border-radius: 24px !important;
    box-shadow:
        0 24px 60px rgba(0,0,0,.32),
        inset 0 1px 0 rgba(255,255,255,.045);
    backdrop-filter: blur(15px);
}

.step-title {
    font-size: 1.36rem;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: .9rem;
}

.step-number {
    display: inline-flex;
    width: 40px;
    height: 40px;
    align-items: center;
    justify-content: center;
    margin-right: 10px;
    border-radius: 50%;
    background: linear-gradient(145deg, #29DEFF, #7A60FF);
    color: white;
    font-weight: 950;
    box-shadow: 0 10px 24px rgba(41, 222, 255, .25);
}

/* ---------- Image source buttons ---------- */
.source-helper {
    color: #AFC6DB;
    font-size: .94rem;
    margin: .15rem 0 .85rem 0;
}

/* Primary = selected source */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #17C9F4 0%, #586CFF 56%, #9950FF 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(137, 229, 255, .65) !important;
    box-shadow:
        0 12px 28px rgba(42, 184, 255, .18),
        0 0 18px rgba(104, 92, 255, .12) !important;
}

div.stButton > button[kind="primary"] p,
div.stButton > button[kind="primary"] span {
    color: #FFFFFF !important;
    font-weight: 900 !important;
}

/* Secondary = unselected source */
div.stButton > button[kind="secondary"] {
    background: linear-gradient(145deg, rgba(12, 24, 40, .98), rgba(15, 22, 38, .98)) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(133, 158, 190, .30) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.04),
        0 8px 20px rgba(0,0,0,.16) !important;
}

div.stButton > button[kind="secondary"] p,
div.stButton > button[kind="secondary"] span {
    color: #FFFFFF !important;
    font-weight: 850 !important;
}

div.stButton > button[kind="secondary"]:hover {
    border-color: rgba(80, 221, 255, .55) !important;
    background: linear-gradient(145deg, rgba(14, 32, 52, .98), rgba(21, 28, 48, .98)) !important;
    transform: translateY(-1px);
}

/* ---------- Widget labels ---------- */
[data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    font-weight: 850 !important;
}

/* ---------- Dark attractive upload box ---------- */
[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(145deg, rgba(7, 18, 32, .98), rgba(10, 25, 43, .98)) !important;
    border: 2px dashed rgba(67, 219, 255, .55) !important;
    border-radius: 20px !important;
    padding: 1.15rem !important;
    min-height: 145px !important;
    box-shadow:
        inset 0 0 0 1px rgba(255,255,255,.025),
        0 0 22px rgba(38, 215, 255, .055);
}

[data-testid="stFileUploaderDropzone"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: .9rem !important;
}

[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #EAF6FF !important;
}

[data-testid="stFileUploaderDropzone"] button {
    width: auto !important;
    min-width: 130px !important;
    min-height: 42px !important;
    padding: .45rem .95rem !important;
    margin: 0 !important;
    border-radius: 11px !important;
    background: linear-gradient(90deg, #14C7F2, #6370FF) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,.15) !important;
    box-shadow: 0 8px 18px rgba(54, 174, 255, .14) !important;
}

[data-testid="stFileUploaderDropzone"] button * {
    color: #FFFFFF !important;
}

/* Hide Streamlit's default selected-file card because it turns white */
[data-testid="stFileUploaderFile"] {
    display: none !important;
}

/* Our own attractive selected-file card */
.selected-file-card {
    display: flex;
    align-items: center;
    gap: .85rem;
    margin: .85rem 0 .75rem 0;
    padding: .90rem 1rem;
    border-radius: 16px;
    background:
        linear-gradient(145deg, rgba(11, 30, 50, .96), rgba(15, 22, 39, .96));
    border: 1px solid rgba(78, 217, 255, .25);
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.04),
        0 10px 26px rgba(0,0,0,.20);
}

.selected-file-icon {
    width: 46px;
    height: 46px;
    min-width: 46px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 13px;
    background: linear-gradient(145deg, rgba(32, 211, 255, .18), rgba(114, 94, 255, .18));
    border: 1px solid rgba(91, 221, 255, .24);
    font-size: 1.35rem;
}

.selected-file-name {
    color: #FFFFFF;
    font-size: 1rem;
    font-weight: 850;
    word-break: break-word;
}

.selected-file-meta {
    color: #AFC6DA;
    font-size: .90rem;
    margin-top: .12rem;
}

/* ---------- Text inputs ---------- */
[data-testid="stTextInput"] input {
    background: #071525 !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(114, 210, 255, .30) !important;
    border-radius: 12px !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #96ABC1 !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(115,211,255,.12);
    border-radius: 16px;
    padding: .95rem 1rem;
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03);
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {
    color: #A9C9DF !important;
    font-size: .97rem !important;
    font-weight: 780 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
    font-size: 2.25rem !important;
    font-weight: 950 !important;
}

[data-testid="stCaptionContainer"] p {
    color: #BED1E4 !important;
    font-size: .98rem !important;
    font-weight: 650 !important;
}

/* ---------- Buttons ---------- */
div.stButton > button {
    width: 100%;
    min-height: 3.2rem;
    border-radius: 14px;
    font-size: 1.06rem;
    font-weight: 900;
    background: linear-gradient(90deg, #10C8F4, #5A6DFF 56%, #A14EFF);
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,.15);
    box-shadow:
        0 14px 30px rgba(50,135,255,.24),
        0 0 24px rgba(39,228,255,.06);
}

div.stButton > button:hover {
    transform: translateY(-1px);
    border-color: #AAEDFF;
    box-shadow:
        0 18px 36px rgba(60,150,255,.30),
        0 0 28px rgba(255,53,215,.06);
}

div.stButton > button:disabled {
    background: rgba(255,255,255,.08) !important;
    color: #8D9CAF !important;
    box-shadow: none !important;
}

/* ---------- Download ---------- */
div.stDownloadButton > button {
    width: 100%;
    min-height: 3rem;
    border-radius: 14px;
    font-weight: 900;
    background: linear-gradient(90deg, #0B9CCA, #5B59E1, #9049D5) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,.16) !important;
}

div.stDownloadButton > button * {
    color: #FFFFFF !important;
}

/* ---------- Alert boxes ---------- */
[data-testid="stAlert"] {
    border-radius: 14px !important;
}

[data-testid="stAlert"] p {
    color: #F8FBFF !important;
    font-size: 1rem !important;
}

/* ---------- Result ---------- */
.result-card {
    border-radius: 22px;
    padding: 1.5rem 1.55rem;
    background:
        linear-gradient(145deg, rgba(10, 45, 72, 0.76), rgba(53, 28, 92, 0.76));
    border: 1px solid rgba(108, 221, 255, .24);
    box-shadow:
        0 22px 50px rgba(0,0,0,.26),
        0 0 28px rgba(39,228,255,.05);
}

.result-label {
    color: #9DE1FF;
    font-size: .92rem;
    font-weight: 900;
    letter-spacing: .14em;
    margin-bottom: .45rem;
}

.result-class {
    color: #FFFFFF;
    font-size: 3.65rem;
    line-height: 1.02;
    font-weight: 950;
    margin: 0 0 .65rem 0;
    text-shadow: 0 0 24px rgba(38,217,255,.13);
}

.result-confidence {
    color: #F3FBFF;
    font-size: 1.70rem;
    font-weight: 900;
    margin-bottom: .35rem;
}

.result-secondary {
    color: #BFD8EA;
    font-size: 1.02rem;
    font-weight: 750;
    margin-top: .40rem;
    line-height: 1.45;
}

.result-mixed-note {
    color: #93B9D4;
    font-size: .86rem;
    font-weight: 650;
    margin-top: .55rem;
    line-height: 1.45;
}

.status-high {
    color: #79F7B1;
    font-size: 1.10rem;
    font-weight: 900;
}

.status-mid {
    color: #FFD166;
    font-size: 1.10rem;
    font-weight: 900;
}

.status-low {
    color: #FF9D9D;
    font-size: 1.10rem;
    font-weight: 900;
}

[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #22D9FF, #7064FF, #FF2BD6);
}

.small-note {
    color: #B3C8DD;
    font-size: .95rem;
    line-height: 1.55;
}

.top3-wrapper {
    margin-top: .40rem;
    border-radius: 20px;
    padding: 1rem 1rem .45rem 1rem;
    background: linear-gradient(145deg, rgba(9, 20, 35, 0.78), rgba(12, 18, 32, 0.82));
    border: 1px solid rgba(111, 211, 255, .12);
}

.top3-title {
    color: #EAF6FF;
    font-size: 1.08rem;
    font-weight: 900;
    margin-bottom: .8rem;
}

.pred-row {
    margin-bottom: 1rem;
}

.pred-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: .45rem;
}

.pred-name {
    color: #F5FBFF;
    font-size: 1rem;
    font-weight: 800;
}

.pred-score {
    color: #EAF5FF;
    font-size: 1rem;
    font-weight: 900;
}

.progress-track {
    width: 100%;
    height: 12px;
    border-radius: 999px;
    overflow: hidden;
    background: linear-gradient(90deg, rgba(255,255,255,.08), rgba(255,255,255,.06));
    border: 1px solid rgba(111, 211, 255, .10);
}

.progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #23D9FF, #6670FF, #FF3ACB);
}


/* ---------- Animated 3D dark background ---------- */
@keyframes auroraFloat {
    0% {
        transform: translate3d(0px, 0px, 0px) scale(1);
        filter:
            drop-shadow(0 0 8px rgba(50, 217, 255, .12))
            drop-shadow(0 0 12px rgba(150, 104, 255, .08));
    }
    50% {
        transform: translate3d(-8px, 6px, 0px) scale(1.03);
        filter:
            drop-shadow(0 0 12px rgba(50, 217, 255, .16))
            drop-shadow(0 0 16px rgba(255, 85, 200, .08));
    }
    100% {
        transform: translate3d(10px, -8px, 0px) scale(1.05);
        filter:
            drop-shadow(0 0 10px rgba(50, 217, 255, .14))
            drop-shadow(0 0 16px rgba(150, 104, 255, .10));
    }
}

@keyframes gridPulse {
    0%, 100% {
        opacity: .34;
        transform: perspective(820px) rotateX(66deg) scale(1.16);
    }
    50% {
        opacity: .46;
        transform: perspective(820px) rotateX(66deg) scale(1.20);
    }
}

.hero::after {
    content: "";
    position: absolute;
    top: -28px;
    right: 34px;
    width: 210px;
    height: 210px;
    border-radius: 50%;
    background:
        radial-gradient(circle at 35% 35%, rgba(255,255,255,.28), transparent 10%),
        radial-gradient(circle, rgba(36, 214, 255, .20), rgba(130, 92, 255, .12) 50%, transparent 72%);
    filter: blur(2px);
    box-shadow: 0 0 60px rgba(36, 214, 255, .08);
    pointer-events: none;
}

@media (max-width: 800px) {
    .hero-title { font-size: 2.4rem; }
    .result-class { font-size: 2.65rem; }
    .result-confidence { font-size: 1.35rem; }
}
</style>

""",
    unsafe_allow_html=True
)

# =========================================================
# LIGHT THEME OVERRIDES
# =========================================================
if st.session_state.theme_mode == "light":
    st.markdown(
        """
<style>
html, body {
    background: #EEF4FA !important;
    color-scheme: light !important;
}

.stApp {
    color: #142033 !important;
    background:
        radial-gradient(circle at 14% 12%, rgba(44, 188, 255, .18), transparent 25%),
        radial-gradient(circle at 84% 16%, rgba(132, 104, 255, .16), transparent 24%),
        radial-gradient(circle at 68% 76%, rgba(255, 105, 190, .10), transparent 28%),
        linear-gradient(145deg, #F8FBFF 0%, #EEF5FC 38%, #EAF1FA 70%, #F6F2FF 100%) !important;
}

.stApp::before {
    opacity: .34 !important;
    filter: none !important;
}

.stApp::after {
    opacity: .17 !important;
}

/* Main text */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span {
    color: #1B2B40;
}

/* Hero */
.hero {
    background:
        linear-gradient(145deg, rgba(255,255,255,.94), rgba(242,247,253,.92)) !important;
    border: 1px solid rgba(72, 153, 210, .20) !important;
    box-shadow:
        0 24px 55px rgba(44, 79, 120, .12),
        inset 0 1px 0 rgba(255,255,255,.95) !important;
}

.greeting {
    color: #087FA8 !important;
}

.hero-title {
    background: linear-gradient(90deg, #13243A 0%, #167FA6 36%, #6357C9 70%, #A34085 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.hero-text {
    color: #445B73 !important;
}

/* Chips */
.chip {
    color: #25384D !important;
    background: rgba(255,255,255,.78) !important;
    border: 1px solid rgba(78, 151, 204, .18) !important;
    box-shadow: 0 8px 18px rgba(50, 78, 110, .08) !important;
}

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,.94), rgba(245,249,253,.92)) !important;
    border: 1px solid rgba(76, 154, 208, .18) !important;
    box-shadow:
        0 20px 45px rgba(40, 72, 110, .10),
        inset 0 1px 0 rgba(255,255,255,.92) !important;
}

.step-title {
    color: #17283D !important;
}

.source-helper,
.small-note {
    color: #5B7188 !important;
}

/* Source buttons */
div.stButton > button[kind="secondary"] {
    background: linear-gradient(145deg, #FFFFFF, #EFF4F9) !important;
    border: 1px solid rgba(82, 128, 170, .26) !important;
    color: #1E3148 !important;
    box-shadow: 0 7px 18px rgba(45, 74, 105, .08) !important;
}

div.stButton > button[kind="secondary"] p,
div.stButton > button[kind="secondary"] span {
    color: #1E3148 !important;
}

/* Widget labels */
[data-testid="stWidgetLabel"] p {
    color: #1C2F46 !important;
}

/* Upload area */
[data-testid="stFileUploaderDropzone"] {
    background:
        linear-gradient(145deg, #FFFFFF, #F1F6FB) !important;
    border: 2px dashed rgba(49, 164, 211, .48) !important;
    box-shadow: 0 10px 24px rgba(48, 83, 120, .07) !important;
}

[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small {
    color: #294059 !important;
}

[data-testid="stFileUploaderFile"] {
    background: #FFFFFF !important;
    border-color: rgba(62, 150, 205, .20) !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #243B53 !important;
}

/* Text inputs */
[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    color: #1B2C40 !important;
    border: 1px solid rgba(66, 142, 195, .28) !important;
}

[data-testid="stTextInput"] input::placeholder {
    color: #8091A4 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,.82) !important;
    border-color: rgba(74, 148, 202, .16) !important;
}

[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] div {
    color: #597086 !important;
}

[data-testid="stMetricValue"] {
    color: #17283D !important;
}

[data-testid="stCaptionContainer"] p {
    color: #61768B !important;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(239, 248, 255, .92) !important;
    border: 1px solid rgba(62, 147, 201, .18) !important;
}

[data-testid="stAlert"] p {
    color: #23425D !important;
}

/* Result */
.result-card {
    background:
        linear-gradient(145deg, rgba(242,250,255,.96), rgba(244,238,255,.94)) !important;
    border-color: rgba(64, 157, 211, .20) !important;
    box-shadow: 0 18px 42px rgba(52, 81, 115, .10) !important;
}

.result-label {
    color: #267FA4 !important;
}

.result-class {
    color: #17283D !important;
    text-shadow: none !important;
}

.result-confidence {
    color: #2C4258 !important;
}

.result-secondary {
    color: #536B82 !important;
}

.result-mixed-note {
    color: #6A7F93 !important;
}

.top3-wrapper {
    background: linear-gradient(145deg, #FFFFFF, #F4F8FC) !important;
    border-color: rgba(72, 148, 201, .16) !important;
}

.top3-title,
.pred-name,
.pred-score {
    color: #1B2F45 !important;
}

.progress-track {
    background: #DDE7F0 !important;
    border-color: rgba(65, 134, 184, .12) !important;
}
</style>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HELPERS
# =========================================================
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def load_from_path(text: str):
    """
    Load an image from a Windows/local path.

    Supports:
    C:\\Users\\Hp\\Pictures\\forest.jpg
    C:/Users/Hp/Pictures/forest.jpg
    "C:\\Users\\Hp\\Pictures\\forest.jpg"
    file:///C:/Users/Hp/Pictures/forest.jpg
    """
    clean = text.strip().strip('"').strip("'")

    if clean.lower().startswith("file://"):
        parsed = urlparse(clean)
        clean = unquote(parsed.path)

        # file:///C:/... on Windows becomes /C:/...
        if os.name == "nt" and len(clean) >= 3 and clean[0] == "/" and clean[2] == ":":
            clean = clean[1:]

    clean = os.path.expandvars(os.path.expanduser(clean))
    path = Path(clean)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found at this path: {clean}"
        )

    if not path.is_file():
        raise ValueError(
            f"This path is not a file: {clean}"
        )

    try:
        image = Image.open(path).convert("RGB")
    except Exception as error:
        raise ValueError(
            "The file exists, but it is not a readable JPG/PNG/WEBP image."
        ) from error

    return image, path.name


def normalize_image_url(url: str):
    """
    Accept normal web URLs and local development URLs.

    Examples:
    https://site.com/image.jpg
    http://localhost:8000/image.jpg
    localhost:8000/image.jpg
    127.0.0.1:8000/image.jpg
    """
    url = url.strip()

    if not url:
        raise ValueError("Please enter an image URL.")

    lower = url.lower()

    if lower.startswith(("localhost:", "127.0.0.1:", "0.0.0.0:")):
        url = "http://" + url

    elif not lower.startswith(("http://", "https://")):
        url = "https://" + url

    return url


def load_from_url(url: str):
    url = normalize_image_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(5, 20),
            allow_redirects=True,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError as error:
        raise ConnectionError(
            "Could not connect to this URL. "
            "For localhost, make sure the local server is running and use "
            "http://localhost:PORT/image.jpg."
        ) from error

    except requests.exceptions.Timeout as error:
        raise TimeoutError(
            "The image URL took too long to respond."
        ) from error

    except requests.exceptions.RequestException as error:
        raise ValueError(
            f"URL request failed: {error}"
        ) from error

    content_type = response.headers.get("Content-Type", "").lower()

    # Some servers omit Content-Type, so PIL still gets the final say.
    try:
        image = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

    except Exception as error:
        if "text/html" in content_type:
            raise ValueError(
                "This URL returned a web page, not an image. "
                "Use the direct image URL ending in .jpg, .jpeg, .png, .webp, etc."
            ) from error

        raise ValueError(
            "The URL responded, but the returned content is not a readable image."
        ) from error

    parsed = urlparse(response.url)
    name = Path(parsed.path).name or "url_image"

    return image, name


def analyze_scene_probabilities(prediction):
    """
    Interpret the Softmax output conservatively.

    The CNN is still a single-label model. This helper does NOT turn it
    into a true multi-label model. It only shows a secondary scene when
    the second-highest probability is strong enough and close enough to
    the primary prediction.
    """
    order = np.argsort(prediction)[::-1]

    primary_index = int(order[0])
    secondary_index = int(order[1])

    primary_probability = float(prediction[primary_index])
    secondary_probability = float(prediction[secondary_index])

    gap = primary_probability - secondary_probability

    is_mixed = (
        secondary_probability >= MIXED_MIN_SECONDARY
        and gap <= MIXED_MAX_GAP
    )

    return {
        "primary_index": primary_index,
        "secondary_index": secondary_index,
        "primary_class": CLASS_NAMES[primary_index],
        "secondary_class": CLASS_NAMES[secondary_index],
        "primary_probability": primary_probability,
        "secondary_probability": secondary_probability,
        "is_mixed": is_mixed,
    }


def get_font(size: int, bold=False):
    candidates = (
        ["DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"]
        if bold
        else ["DejaVuSans.ttf", "arial.ttf", "Arial.ttf", "DejaVuSans-Bold.ttf"]
    )

    for font_name in candidates:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            pass

    return ImageFont.load_default()


def add_class_label(image: Image.Image, class_name: str):
    """
    Add only a clean class badge on the image.
    Confidence is shown in the result panel below the image.
    """
    result = image.convert("RGB").copy()

    if result.width > 1400:
        ratio = 1400 / result.width
        result = result.resize((1400, int(result.height * ratio)))

    rgba = result.convert("RGBA")
    overlay = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    class_text = class_name.upper()

    pad_x = max(20, int(rgba.width * 0.02))
    pad_y = max(14, int(rgba.height * 0.018))

    left = max(18, int(rgba.width * 0.025))
    top = max(18, int(rgba.height * 0.025))

    # Automatically shrink a mixed label such as "MOUNTAIN + SEA"
    # so that it always stays inside the image.
    font_size = max(30, int(rgba.width * 0.050))
    max_text_width = max(80, rgba.width - left - (pad_x * 2) - 20)

    while True:
        class_font = get_font(font_size, bold=True)
        class_box = draw.textbbox((0, 0), class_text, font=class_font)

        class_w = class_box[2] - class_box[0]
        class_h = class_box[3] - class_box[1]

        if class_w <= max_text_width or font_size <= 18:
            break

        font_size -= 2

    right = min(
        rgba.width - 10,
        left + class_w + (pad_x * 2)
    )

    bottom = top + class_h + (pad_y * 2)

    draw.rounded_rectangle(
        (left + 7, top + 7, right + 7, bottom + 7),
        radius=22,
        fill=(0, 0, 0, 90)
    )

    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=22,
        fill=(5, 17, 32, 225),
        outline=(62, 228, 255, 185),
        width=3
    )

    draw.text(
        (left + pad_x, top + pad_y - 2),
        class_text,
        font=class_font,
        fill=(255, 255, 255, 255)
    )

    return Image.alpha_composite(rgba, overlay).convert("RGB")


def image_bytes(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# =========================================================
# SESSION STATE
# =========================================================
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

if "source_name" not in st.session_state:
    st.session_state.source_name = None

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "source_mode" not in st.session_state:
    st.session_state.source_mode = "upload"


def set_image(image, name):
    st.session_state.selected_image = image
    st.session_state.source_name = name
    st.session_state.prediction_data = None


# =========================================================
# THEME CONTROLS
# =========================================================
theme_spacer, dark_col, light_col = st.columns([0.72, 0.14, 0.14])

with dark_col:
    if st.button(
        "🌙 Dark",
        key="theme_dark",
        type="primary" if st.session_state.theme_mode == "dark" else "secondary",
        use_container_width=True
    ):
        st.session_state.theme_mode = "dark"
        st.rerun()

with light_col:
    if st.button(
        "☀️ Light",
        key="theme_light",
        type="primary" if st.session_state.theme_mode == "light" else "secondary",
        use_container_width=True
    ):
        st.session_state.theme_mode = "light"
        st.rerun()

st.write("")


# =========================================================
# HERO
# =========================================================
st.markdown(
    (
        '<div class="hero">'
        '<div class="greeting">👋 WELCOME TO VISIONSCENE AI</div>'
        '<div class="hero-title">Hello! Ready to classify your image?</div>'
        '<div class="hero-text">'
        'Choose an image using upload, local path, or image URL. '
        'Load it, preview it, and let your trained CNN identify the scene in a futuristic AI dashboard interface.'
        '</div>'
        '</div>'
    ),
    unsafe_allow_html=True
)

chips_html = "".join(
    f'<span class="chip">{CLASS_ICONS[name]} {name.title()}</span>'
    for name in CLASS_NAMES
)

st.markdown(chips_html, unsafe_allow_html=True)
st.write("")


# =========================================================
# STEP 1 — LOAD IMAGE
# =========================================================
with st.container(border=True):
    st.markdown(
        '<div class="step-title"><span class="step-number">1</span>Choose and load an image</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="source-helper">Select one method below, then load your image.</div>',
        unsafe_allow_html=True
    )

    source_col1, source_col2, source_col3 = st.columns(3)

    with source_col1:
        if st.button(
            "📤 Upload Image",
            key="source_upload",
            type="primary" if st.session_state.source_mode == "upload" else "secondary",
            use_container_width=True
        ):
            st.session_state.source_mode = "upload"
            st.rerun()

    with source_col2:
        if st.button(
            "📁 Local Path",
            key="source_path",
            type="primary" if st.session_state.source_mode == "path" else "secondary",
            use_container_width=True
        ):
            st.session_state.source_mode = "path"
            st.rerun()

    with source_col3:
        if st.button(
            "🔗 Image URL",
            key="source_url",
            type="primary" if st.session_state.source_mode == "url" else "secondary",
            use_container_width=True
        ):
            st.session_state.source_mode = "url"
            st.rerun()

    st.write("")

    # -----------------------------------------------------
    # UPLOAD IMAGE
    # -----------------------------------------------------
    if st.session_state.source_mode == "upload":
        st.markdown("### Upload an image from your device")

        uploaded_file = st.file_uploader(
            "Choose JPG, JPEG, PNG or WEBP image",
            type=["jpg", "jpeg", "png", "webp"],
            key="uploaded_file"
        )

        if uploaded_file is not None:
            safe_name = html.escape(uploaded_file.name)
            size_mb = uploaded_file.size / (1024 * 1024)

            st.markdown(
                (
                    '<div class="selected-file-card">'
                    '<div class="selected-file-icon">🖼️</div>'
                    '<div>'
                    f'<div class="selected-file-name">{safe_name}</div>'
                    f'<div class="selected-file-meta">{size_mb:.2f} MB • Ready to load</div>'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True
            )

        if st.button(
            "Load Uploaded Image",
            key="load_uploaded",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True
        ):
            try:
                set_image(
                    Image.open(uploaded_file).convert("RGB"),
                    uploaded_file.name
                )
            except Exception:
                st.error("The selected file could not be opened as an image.")

    # -----------------------------------------------------
    # LOCAL PATH
    # -----------------------------------------------------
    elif st.session_state.source_mode == "path":
        st.markdown("### Load an image using a local path")

        local_path = st.text_input(
            "Paste copied image path",
            placeholder=r'C:\Users\Hp\Pictures\forest.jpg',
            key="local_path"
        )

        st.markdown(
            '<div class="small-note">Works when Streamlit is running on this same computer. Right-click the image → Copy as path → paste it here. You can also use a file:/// path.</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Load Path Image",
            key="load_path",
            type="primary",
            disabled=not local_path.strip(),
            use_container_width=True
        ):
            try:
                image, name = load_from_path(local_path)
                set_image(image, name)
            except Exception as error:
                st.error(str(error))

    # -----------------------------------------------------
    # IMAGE URL
    # -----------------------------------------------------
    else:
        st.markdown("### Load an image from a URL")

        image_url = st.text_input(
            "Paste direct image URL",
            placeholder="https://example.com/image.jpg  or  localhost:8000/image.jpg",
            key="image_url"
        )

        st.markdown(
            '<div class="small-note">Supports https:// links and local development URLs such as http://localhost:8000/image.jpg. The URL must return the image itself, not a normal webpage.</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "Load URL Image",
            key="load_url",
            type="primary",
            disabled=not image_url.strip(),
            use_container_width=True
        ):
            try:
                with st.spinner("Loading image from URL..."):
                    image, name = load_from_url(image_url)

                set_image(image, name)

            except Exception:
                st.error(
                    "Could not load this image URL. Please use a direct image link."
                )

    if st.session_state.selected_image is not None:
        st.write("")

        left_status, right_status = st.columns([0.78, 0.22])

        with left_status:
            st.success(
                f"✓ Image loaded: {st.session_state.source_name}"
            )

        with right_status:
            if st.button(
                "Clear Image",
                key="clear_image",
                type="secondary",
                use_container_width=True
            ):
                st.session_state.selected_image = None
                st.session_state.source_name = None
                st.session_state.prediction_data = None
                st.rerun()


# =========================================================
# STEP 2 — PREVIEW & PREDICT
# =========================================================
st.write("")
with st.container(border=True):
    st.markdown(
        '<div class="step-title"><span class="step-number">2</span>Preview and analyze</div>',
        unsafe_allow_html=True
    )

    if st.session_state.selected_image is None:
        st.info("Load an image in Step 1 to continue.")
    else:
        preview_col, action_col = st.columns([1.18, 0.82], gap="large")

        with preview_col:
            st.image(
                st.session_state.selected_image,
                caption=st.session_state.source_name,
                use_container_width=True
            )

        with action_col:
            image = st.session_state.selected_image

            st.metric(
                "Image Resolution",
                f"{image.width} × {image.height}"
            )

            st.caption("CNN input size: 256 × 256")

            if st.button("✨ Predict Scene", key="predict_scene", type="primary", use_container_width=True):
                if not Path(MODEL_PATH).exists():
                    st.error(f"'{MODEL_PATH}' was not found. Keep it in the same folder as this app.")
                else:
                    try:
                        with st.spinner("CNN is analyzing the image..."):
                            model = get_model()
                            prediction = model.predict(preprocess_image(image), verbose=0)[0]

                        scene_info = analyze_scene_probabilities(prediction)

                        st.session_state.prediction_data = {
                            "prediction": prediction,
                            "predicted_class": scene_info["primary_class"],
                            "confidence": scene_info["primary_probability"] * 100,
                            "secondary_class": scene_info["secondary_class"],
                            "secondary_confidence": scene_info["secondary_probability"] * 100,
                            "is_mixed": scene_info["is_mixed"],
                        }
                    except Exception as error:
                        st.error("The model could not make a prediction.")
                        st.exception(error)


# =========================================================
# STEP 3 — RESULT
# =========================================================
st.write("")
with st.container(border=True):
    st.markdown(
        '<div class="step-title"><span class="step-number">3</span>Prediction result</div>',
        unsafe_allow_html=True
    )

    if st.session_state.prediction_data is None:
        st.info("Click Predict Scene in Step 2 to see the result.")
    else:
        data = st.session_state.prediction_data
        prediction = data["prediction"]
        predicted_class = data["predicted_class"]
        confidence = data["confidence"]
        secondary_class = data.get("secondary_class")
        secondary_confidence = data.get("secondary_confidence", 0.0)
        is_mixed = data.get("is_mixed", False)

        if is_mixed:
            display_class = f"{predicted_class} + {secondary_class}"
        else:
            display_class = predicted_class

        result_image = add_class_label(
            st.session_state.selected_image,
            display_class
        )

        if confidence >= 80:
            status_text = "✓ High Confidence"
            status_class = "status-high"
        elif confidence >= 60:
            status_text = "• Moderate Confidence"
            status_class = "status-mid"
        else:
            status_text = "! Low Confidence"
            status_class = "status-low"

        image_col, details_col = st.columns([1.18, 0.82], gap="large")

        with image_col:
            st.image(
                result_image,
                caption="Final prediction image",
                use_container_width=True
            )

            st.download_button(
                "⬇️ Download Result Image",
                data=image_bytes(result_image),
                file_name=f"{display_class.replace(' + ', '_')}_prediction.png",
                mime="image/png",
                use_container_width=True
            )

        with details_col:
            primary_icon = CLASS_ICONS[predicted_class]

            if is_mixed:
                secondary_icon = CLASS_ICONS[secondary_class]

                result_html = (
                    '<div class="result-card">'
                    '<div class="result-label">MIXED SCENE ESTIMATE</div>'
                    f'<div class="result-class">{primary_icon} {predicted_class.title()} + '
                    f'{secondary_icon} {secondary_class.title()}</div>'
                    f'<div class="result-confidence">Primary: {confidence:.2f}%</div>'
                    f'<div class="result-secondary">Secondary: '
                    f'{secondary_class.title()} ({secondary_confidence:.2f}%)</div>'
                    f'<div class="{status_class}">{status_text}</div>'
                    '<div class="result-mixed-note">'
                    'The CNN is still a single-label Softmax model; this combined result '
                    'is shown only when the top two probabilities are both strong and close.'
                    '</div>'
                    '</div>'
                )

            else:
                secondary_html = ""

                if secondary_confidence >= (SECONDARY_NOTE_MIN * 100):
                    secondary_html = (
                        f'<div class="result-secondary">Possible secondary scene: '
                        f'{CLASS_ICONS[secondary_class]} {secondary_class.title()} '
                        f'({secondary_confidence:.2f}%)</div>'
                    )

                result_html = (
                    '<div class="result-card">'
                    '<div class="result-label">PREDICTED CLASS</div>'
                    f'<div class="result-class">{primary_icon} {predicted_class.title()}</div>'
                    f'<div class="result-confidence">Confidence: {confidence:.2f}%</div>'
                    f'{secondary_html}'
                    f'<div class="{status_class}">{status_text}</div>'
                    '</div>'
                )

            st.markdown(result_html, unsafe_allow_html=True)

            st.write("")

            st.markdown('<div class="top3-wrapper"><div class="top3-title">Top 3 Predictions</div>', unsafe_allow_html=True)

            top_indices = np.argsort(prediction)[::-1][:3]

            for index in top_indices:
                name = CLASS_NAMES[int(index)]
                probability = float(prediction[index])
                percentage = probability * 100

                bar_html = f"""
<div class="pred-row">
  <div class="pred-head">
    <div class="pred-name">{CLASS_ICONS[name]} {name.title()}</div>
    <div class="pred-score">{percentage:.2f}%</div>
  </div>
  <div class="progress-track">
    <div class="progress-fill" style="width:{percentage:.2f}%;"></div>
  </div>
</div>
"""
                st.markdown(bar_html, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
