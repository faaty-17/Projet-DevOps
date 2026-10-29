import base64
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from PIL import Image


# ============================================================
# PERSONNALISATION
# ============================================================

APP_NAME = "NeuroScan"
APP_NAME_ACCENT = "AI"

TAGLINE = "Brain MRI Classification Platform"

AUTHOR = "Fatma Nouira"
SCHOOL = "ENSTAB × ALYOS Digital"


# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(

    page_title=f"{APP_NAME} {APP_NAME_ACCENT}",

    page_icon="🧠",

    layout="wide",

    initial_sidebar_state="expanded",
)


# ============================================================
# API
# ============================================================

# LOCAL :
# http://127.0.0.1:8000/predict
#
# ONLINE :
# La variable API_URL sera configurée sur le serveur
# Streamlit sans modifier ce code.

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

HEALTH_URL = (
    API_URL.rsplit("/", 1)[0]
    + "/health"
)


# ============================================================
# ASSETS
# ============================================================

ASSETS = (
    Path(__file__).parent
    / "assets"
)


# ============================================================
# CLASSES
# ============================================================

CLASS_META = {

    "glioma":
        ("Glioma", "#F43F5E"),

    "meningioma":
        ("Meningioma", "#3B82F6"),

    "pituitary":
        ("Pituitary", "#A855F7"),

    "notumor":
        ("No Tumor", "#22C55E"),
}


CLASS_ORDER = [

    "glioma",

    "meningioma",

    "pituitary",

    "notumor",

]


# ============================================================
# MODELS
# ============================================================

MODELS = {

    "DenseNet121":
        "densenet",

    "ResNet50":
        "resnet",

}


# ============================================================
# MIME
# ============================================================

MIME = {

    ".jpg":
        "image/jpeg",

    ".jpeg":
        "image/jpeg",

    ".png":
        "image/png",

    ".webp":
        "image/webp",

}


# ============================================================
# ICONS
# ============================================================

_ICONS = {

    "upload":
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/>'
        '<line x1="12" y1="3" x2="12" y2="15"/>',

    "image":
        '<rect x="3" y="3" width="18" height="18" rx="2"/>'
        '<circle cx="8.5" cy="8.5" r="1.5"/>'
        '<polyline points="21 15 16 10 5 21"/>',

    "activity":
        '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',

    "scan":
        '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<line x1="7" y1="12" x2="17" y2="12"/>',

    "file":
        '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
        '<polyline points="14 2 14 8 20 8"/>'
        '<line x1="16" y1="13" x2="8" y2="13"/>'
        '<line x1="16" y1="17" x2="8" y2="17"/>',

    "clock":
        '<circle cx="12" cy="12" r="10"/>'
        '<polyline points="12 6 12 12 16 14"/>',

    "info":
        '<circle cx="12" cy="12" r="10"/>'
        '<line x1="12" y1="16" x2="12" y2="12"/>'
        '<line x1="12" y1="8" x2="12.01" y2="8"/>',

    "target":
        '<circle cx="12" cy="12" r="10"/>'
        '<circle cx="12" cy="12" r="6"/>'
        '<circle cx="12" cy="12" r="2"/>',

    "zap":
        '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',

    "shield":
        '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',

    "alert":
        '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>'
        '<line x1="12" y1="9" x2="12" y2="13"/>'
        '<line x1="12" y1="17" x2="12.01" y2="17"/>',

    "check":
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>',

    "tick":
        '<polyline points="20 6 9 17 4 12"/>',

}


def icon(
    name: str,
    size: int = 20
) -> str:

    return (

        f'<svg class="ic" '
        f'width="{size}" '
        f'height="{size}" '
        f'viewBox="0 0 24 24" '
        f'fill="none" '
        f'stroke="currentColor" '
        f'stroke-width="1.8" '
        f'stroke-linecap="round" '
        f'stroke-linejoin="round">'

        f'{_ICONS[name]}'

        '</svg>'

    )


# ============================================================
# LOGO
# ============================================================

LOGO_SVG = (

    '<svg width="34" height="34" '
    'viewBox="0 0 48 48" '
    'fill="none" '
    'stroke="#fff" '
    'stroke-width="2" '
    'stroke-linecap="round">'

    '<line x1="24" y1="10" x2="10" y2="22"/>'
    '<line x1="24" y1="10" x2="38" y2="22"/>'

    '<line x1="10" y1="22" x2="16" y2="38"/>'
    '<line x1="38" y1="22" x2="32" y2="38"/>'

    '<line x1="16" y1="38" x2="32" y2="38"/>'
    '<line x1="10" y1="22" x2="38" y2="22"/>'

    '<line x1="24" y1="10" x2="16" y2="38"/>'
    '<line x1="24" y1="10" x2="32" y2="38"/>'

    '<g fill="#fff" stroke="none">'

    '<circle cx="24" cy="10" r="4"/>'
    '<circle cx="10" cy="22" r="4"/>'
    '<circle cx="38" cy="22" r="4"/>'
    '<circle cx="16" cy="38" r="4"/>'
    '<circle cx="32" cy="38" r="4"/>'

    '</g>'

    '</svg>'

)


# ============================================================
# HELPERS
# ============================================================

def html(s: str) -> None:

    markup = " ".join(
        line.strip()
        for line in s.splitlines()
        if line.strip()
    )

    st.markdown(
        markup,
        unsafe_allow_html=True,
    )


def norm(name: str) -> str:

    return (
        str(name)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def to_percent(v) -> float:

    try:

        v = float(v)

    except (
        TypeError,
        ValueError
    ):

        return 0.0

    return (
        v * 100
        if v <= 1.0
        else v
    )


@st.cache_data(
    ttl=5,
    show_spinner=False
)
def api_online() -> bool:

    try:

        response = requests.get(
            HEALTH_URL,
            timeout=2
        )

        return response.status_code == 200

    except requests.exceptions.RequestException:

        return False


def human_size(n: int) -> str:

    if n < 1024 * 1024:

        return f"{n / 1024:.1f} KB"

    return (
        f"{n / (1024 * 1024):.1f} MB"
    )


@st.cache_data(
    show_spinner=False
)
def find_asset(stem: str):

    for ext in (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    ):

        p = ASSETS / f"{stem}{ext}"

        if p.exists():

            data = base64.b64encode(
                p.read_bytes()
            ).decode()

            return (
                f"data:{MIME[ext]};"
                f"base64,{data}"
            )

    return None


@st.cache_data(
    show_spinner=False
)
def neural_background() -> str:

    rnd = random.Random(11)

    w, h = 1600, 900

    pts = [
        (
            rnd.uniform(0, w),
            rnd.uniform(0, h)
        )
        for _ in range(85)
    ]

    parts = [

        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}">'

    ]

    for i, (x1, y1) in enumerate(pts):

        for x2, y2 in pts[i + 1:]:

            if math.hypot(
                x1 - x2,
                y1 - y2
            ) < 190:

                parts.append(

                    f'<line '
                    f'x1="{x1:.0f}" '
                    f'y1="{y1:.0f}" '
                    f'x2="{x2:.0f}" '
                    f'y2="{y2:.0f}" '
                    f'stroke="#38A8FF" '
                    f'stroke-opacity="0.16" '
                    f'stroke-width="1"/>'

                )

    for x, y in pts:

        r = rnd.choice(
            [2, 2.5, 3, 4]
        )

        parts.append(

            f'<circle '
            f'cx="{x:.0f}" '
            f'cy="{y:.0f}" '
            f'r="{r}" '
            f'fill="#5CC0FF" '
            f'fill-opacity="0.45"/>'

        )

    parts.append("</svg>")

    return (
        "data:image/svg+xml;base64,"
        + base64.b64encode(
            "".join(parts).encode()
        ).decode()
    )


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault(
    "history",
    []
)

st.session_state.setdefault(
    "result",
    None
)


# ============================================================
# BACKGROUND
# ============================================================

bg_uri = find_asset(
    "background"
)

if bg_uri:

    bg_css = (

        "linear-gradient("
        "180deg,"
        "rgba(4,9,24,.78) 0%,"
        "rgba(4,9,24,.90) 100%"
        "),"

        f"url('{bg_uri}') "
        "center / cover "
        "no-repeat fixed"

    )

else:

    bg_css = (

        "radial-gradient("
        "1100px 500px at 75% -5%,"
        "rgba(30,100,255,.28) 0%,"
        "transparent 60%"
        "),"

        "radial-gradient("
        "900px 500px at 0% 100%,"
        "rgba(124,92,255,.16) 0%,"
        "transparent 60%"
        "),"

        f"url('{neural_background()}') "
        "center / cover "
        "no-repeat fixed,"

        "#050B1C"

    )


# ============================================================
# CSS
# ============================================================

CSS = """

<style>

@import url(
'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap'
);

:root {

    --bg: #050B1C;

    --glass: rgba(9, 19, 46, .66);

    --glass-2: rgba(16, 32, 72, .55);

    --line: rgba(90, 140, 255, .20);

    --text: #EAF0FF;

    --muted: #93A7D3;

    --accent: #38A8FF;

    --accent-2: #7C5CFF;

}

html,
body,
[class*="css"],
.stApp {

    font-family:
        'Plus Jakarta Sans',
        sans-serif;

}

.stApp {

    background:
        __BG__;

    color:
        var(--text);

}

#MainMenu,
footer,
header[data-testid="stHeader"] {

    visibility: hidden;
    height: 0;

}

.block-container {

    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1500px;

}

.ic {

    vertical-align: middle;
    flex-shrink: 0;

}

.topbar {

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 4px 4px 16px 4px;
    margin-bottom: 20px;

    border-bottom:
        1px solid var(--line);

}

.brand {

    display: flex;
    align-items: center;
    gap: 16px;

}

.brand-logo {

    width: 56px;
    height: 56px;

    border-radius: 16px;

    display: flex;
    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #1244B8,
            #2F8BFF 60%,
            #7C5CFF
        );

    box-shadow:
        0 0 28px
        rgba(56,168,255,.45),

        inset 0 0 0 1px
        rgba(255,255,255,.18);

}

.brand-name {

    font-size: 28px;
    font-weight: 800;

}

.brand-name span {

    background:
        linear-gradient(
            90deg,
            #38A8FF,
            #8B5CF6
        );

    -webkit-background-clip: text;
    background-clip: text;

    color: transparent;

}

.brand-sub {

    color: var(--muted);
    font-size: 14px;

}

.top-right {

    text-align: right;

}

.status {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    border:
        1px solid
        rgba(34,197,94,.35);

    background:
        rgba(34,197,94,.10);

    color:
        #D1FADF;

    padding:
        6px 14px;

    border-radius:
        999px;

    font-size:
        13px;

}

.status.off {

    border-color:
        rgba(239,68,68,.4);

    background:
        rgba(239,68,68,.10);

    color:
        #FECACA;

}

.dot {

    width: 8px;
    height: 8px;

    border-radius: 50%;

    background:
        #22C55E;

    box-shadow:
        0 0 8px
        #22C55E;

}

.status.off .dot {

    background:
        #EF4444;

    box-shadow:
        0 0 8px
        #EF4444;

}

.slogan {

    color: var(--muted);

    font-size: 12.5px;

    margin-top: 9px;

}

.hero {

    position: relative;

    overflow: hidden;

    display: flex;

    align-items: center;

    justify-content:
        space-between;

    gap: 28px;

    flex-wrap: wrap;

    background:
        linear-gradient(
            105deg,
            rgba(14,40,110,.78) 0%,
            rgba(9,19,46,.70) 60%
        );

    backdrop-filter:
        blur(10px);

    border:
        1px solid var(--line);

    border-radius:
        20px;

    padding:
        26px 34px;

    margin-bottom:
        22px;

}

.hero::before {

    content: "";

    position: absolute;

    left: 0;
    top: 0;
    bottom: 0;

    width: 4px;

    background:
        linear-gradient(
            180deg,
            #38A8FF,
            #8B5CF6
        );

}

.hero-left {

    display: flex;

    align-items: center;

    gap: 28px;

}

.hero-img {

    width: 150px;
    height: 130px;

    object-fit: contain;

    filter:
        drop-shadow(
            0 0 26px
            rgba(56,168,255,.60)
        );

}

.hero-mark {

    width: 110px;
    height: 110px;

    border-radius: 28px;

    display: flex;

    align-items: center;
    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #1244B8,
            #2F8BFF 60%,
            #7C5CFF
        );

}

.hero h1 {

    font-size: 34px;

    font-weight: 800;

    margin: 0;

}

.hero h1 span {

    background:
        linear-gradient(
            90deg,
            #38A8FF,
            #8B5CF6
        );

    -webkit-background-clip: text;

    background-clip: text;

    color: transparent;

}

.hero p {

    color:
        var(--muted);

    margin:
        8px 0 0 0;

}

.hero-feats {

    display: flex;

    gap: 30px;

}

.feat {

    display: flex;

    gap: 12px;

    align-items: center;

}

.feat-ic {

    width: 44px;
    height: 44px;

    border-radius: 12px;

    color:
        var(--accent);

    display: flex;

    align-items: center;
    justify-content: center;

    background:
        rgba(56,168,255,.10);

    border:
        1px solid var(--line);

}

.feat b {

    display: block;

}

.feat small {

    color:
        var(--muted);

}

div[data-testid="stVerticalBlockBorderWrapper"] {

    background:
        var(--glass);

    backdrop-filter:
        blur(14px);

    border:
        1px solid var(--line) !important;

    border-radius:
        18px !important;

    padding:
        8px 12px;

    box-shadow:
        0 10px 40px
        rgba(0,0,0,.35);

}

.card-title {

    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 18px;

    font-weight: 700;

    color:
        var(--text);

    margin-bottom:
        12px;

}

.card-title .ic {

    color:
        var(--accent);

}

.card-title.sm {

    font-size: 15px;

    color:
        var(--accent);

}

section[data-testid="stFileUploaderDropzone"] {

    background:
        rgba(16,32,72,.45);

    border:
        1.5px dashed
        rgba(56,168,255,.55);

    border-radius:
        14px;

    padding:
        26px;

}

section[data-testid="stFileUploaderDropzone"] button {

    background:
        transparent;

    color:
        var(--accent);

    border:
        1px solid
        rgba(56,168,255,.55);

    border-radius:
        8px;

}

div[data-testid="stFileUploader"] label {

    display: none;

}

div[data-testid="stFileUploaderFile"] {

    display: none;

}

.chip {

    display: flex;

    align-items: center;

    gap: 12px;

    background:
        var(--glass-2);

    border:
        1px solid var(--line);

    border-radius:
        12px;

    padding:
        10px 14px;

    margin-top:
        10px;

}

.chip-ok {

    width: 24px;
    height: 24px;

    border-radius: 50%;

    background:
        #22C55E;

    color:
        #052e16;

    display: flex;

    align-items: center;
    justify-content: center;

}

.chip b {

    font-size: 14px;

    display: block;

    word-break:
        break-all;

}

.chip small {

    color:
        var(--muted);

}

.stButton > button {

    width: 100%;

    background:
        linear-gradient(
            90deg,
            #2F6BFF 0%,
            #8B5CF6 100%
        );

    color: #fff;

    border: none;

    border-radius:
        12px;

    padding:
        14px;

    font-weight:
        700;

    font-size:
        16px;

}

.stButton > button:hover {

    filter:
        brightness(1.12);

    color:
        #fff;

}

.res-grid {

    display: grid;

    grid-template-columns:
        1.05fr 1fr;

    gap:
        16px;

}

.res-main {

    border-radius:
        16px;

    padding:
        22px;

    display: flex;

    gap:
        18px;

    align-items:
        flex-start;

}

.res-icon {

    width: 64px;
    height: 64px;

    border-radius:
        50%;

    flex-shrink:
        0;

    display: flex;

    align-items:
        center;

    justify-content:
        center;

}

.badge {

    display:
        inline-block;

    padding:
        4px 14px;

    border-radius:
        999px;

    font-size:
        13px;

    font-weight:
        600;

    margin-bottom:
        8px;

}

.res-name {

    font-size:
        30px;

    font-weight:
        800;

    margin:
        0 0 14px 0;

}

.conf-row {

    display:
        flex;

    align-items:
        center;

    gap:
        12px;

}

.conf-row .lbl {

    color:
        var(--muted);

}

.bar-track {

    background:
        rgba(255,255,255,.08);

    border-radius:
        99px;

    height:
        8px;

    overflow:
        hidden;

    flex:
        1;

}

.bar-fill {

    height:
        100%;

    border-radius:
        99px;

}

.probs {

    border:
        1px solid var(--line);

    border-radius:
        16px;

    padding:
        16px 18px;

    background:
        var(--glass-2);

}

.prob-row {

    margin-bottom:
        12px;

}

.prob-head {

    display:
        flex;

    justify-content:
        space-between;

    font-size:
        14px;

    margin-bottom:
        5px;

}

.img-title {

    color:
        var(--accent);

    font-weight:
        600;

    font-size:
        15px;

    margin:
        18px 0 8px 0;

}

.heat-ph {

    height:
        190px;

    border:
        1px dashed var(--line);

    border-radius:
        12px;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    text-align:
        center;

    color:
        var(--muted);

    font-size:
        13px;

}

.pill {

    display:
        inline-flex;

    align-items:
        center;

    gap:
        8px;

    float:
        right;

    background:
        var(--glass-2);

    border:
        1px solid var(--line);

    border-radius:
        999px;

    padding:
        6px 14px;

    font-size:
        13px;

    color:
        var(--muted);

}

.empty {

    text-align:
        center;

    padding:
        64px 20px;

    color:
        var(--muted);

}

.empty .mark {

    width:
        84px;

    height:
        84px;

    border-radius:
        24px;

    margin:
        0 auto 14px auto;

    display:
        flex;

    align-items:
        center;

    justify-content:
        center;

    background:
        linear-gradient(
            135deg,
            #1244B8,
            #2F8BFF 60%,
            #7C5CFF
        );

}

.det-grid {

    display:
        grid;

    grid-template-columns:
        repeat(4, 1fr) 1.7fr;

    gap:
        18px;

    align-items:
        center;

}

.det-item small {

    color:
        var(--accent);

    display:
        block;

}

.det-item b {

    font-size:
        16px;

}

.notice {

    display:
        flex;

    gap:
        10px;

    align-items:
        flex-start;

    border:
        1px solid var(--line);

    background:
        rgba(56,168,255,.07);

    border-radius:
        12px;

    padding:
        14px 16px;

    color:
        var(--muted);

    font-size:
        13px;

}

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            rgba(8,17,43,.92) 0%,
            rgba(5,11,28,.96) 100%
        );

    border-right:
        1px solid var(--line);

}

.side-box {

    border:
        1px solid var(--line);

    border-radius:
        14px;

    background:
        var(--glass-2);

    padding:
        14px 16px;

    margin-top:
        14px;

}

.side-box h4 {

    margin:
        0 0 10px 0;

    font-size:
        14px;

    color:
        var(--accent);

}

.kv {

    display:
        flex;

    justify-content:
        space-between;

    font-size:
        13px;

    padding:
        5px 0;

}

.kv span:first-child {

    color:
        var(--muted);

}

.side-project {

    display:
        flex;

    gap:
        10px;

    align-items:
        center;

    border:
        1px solid var(--line);

    border-radius:
        14px;

    padding:
        14px 16px;

    margin-top:
        26px;

    background:
        rgba(56,168,255,.06);

}

.side-project .ic {

    color:
        var(--accent);

}

.side-project b {

    color:
        var(--accent);

    display:
        block;

}

.side-project small {

    color:
        var(--muted);

}

.htable {

    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        14px;

}

.htable th {

    text-align:
        left;

    color:
        var(--muted);

    font-weight:
        500;

    padding:
        10px 8px;

    border-bottom:
        1px solid var(--line);

}

.htable td {

    padding:
        12px 8px;

    border-bottom:
        1px solid
        rgba(90,140,255,.10);

}

.footer {

    text-align:
        center;

    color:
        var(--muted);

    font-size:
        12px;

    margin-top:
        26px;

}

@media (max-width: 900px) {

    .res-grid,
    .det-grid {

        grid-template-columns:
            1fr;

    }

    .hero-feats {

        flex-wrap:
            wrap;

    }

}

</style>

"""


st.markdown(
    CSS.replace(
        "__BG__",
        bg_css
    ),
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

PAGES = [
    "Accueil",
    "Détection",
    "Historique",
    "À propos",
]


with st.sidebar:

    page = st.radio(
        "Navigation",
        PAGES,
        index=1,
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        "**Modèle**"
    )

    model_choice = st.selectbox(
        "Modèle",
        list(MODELS.keys()),
        label_visibility="collapsed",
    )

    model_param = MODELS[
        model_choice
    ]

    html(

        """
        <div class="side-box">

            <h4>
                Informations techniques
            </h4>

            <div class="kv">
                <span>Frontend</span>
                <span>Streamlit</span>
            </div>

            <div class="kv">
                <span>Backend</span>
                <span>FastAPI</span>
            </div>

            <div class="kv">
                <span>Framework</span>
                <span>PyTorch</span>
            </div>

            <div class="kv">
                <span>Input</span>
                <span>224 × 224</span>
            </div>

        </div>
        """

    )

    html(

        f"""

        <div class="side-project">

            {icon("shield", 22)}

            <div>

                <b>
                    Projet de stage
                </b>

                <small>
                    Non destiné à un usage médical
                </small>

            </div>

        </div>

        """

    )


# ============================================================
# TOP BAR
# ============================================================

online = api_online()

html(

    f"""

    <div class="topbar">

        <div class="brand">

            <div class="brand-logo">

                {LOGO_SVG}

            </div>

            <div>

                <div class="brand-name">

                    {APP_NAME}
                    <span>
                        {APP_NAME_ACCENT}
                    </span>

                </div>

                <div class="brand-sub">

                    {TAGLINE}

                </div>

            </div>

        </div>

        <div class="top-right">

            <div class="status
                {'' if online else 'off'}">

                <span class="dot"></span>

                {
                    'System Online'
                    if online
                    else
                    'API Offline'
                }

            </div>

            <div class="slogan">

                Smarter Analysis
                •
                Better Decisions
                •
                Healthier Tomorrow

            </div>

        </div>

    </div>

    """

)


# ============================================================
# HERO
# ============================================================

def render_hero():

    brain_uri = find_asset(
        "brain"
    )

    if brain_uri:

        visual = (

            f'<img '
            f'class="hero-img" '
            f'src="{brain_uri}" '
            f'alt="brain">'

        )

    else:

        visual = (

            f'<div class="hero-mark">'

            f'{LOGO_SVG}'

            f'</div>'

        )

    html(

        f"""

        <div class="hero">

            <div class="hero-left">

                {visual}

                <div>

                    <h1>

                        Détection des
                        <span>
                            Tumeurs Cérébrales
                        </span>

                    </h1>

                    <p>

                        Analyse automatique
                        des images IRM du cerveau
                        par Deep Learning

                    </p>

                </div>

            </div>

            <div class="hero-feats">

                <div class="feat">

                    <div class="feat-ic">
                        {icon("target", 22)}
                    </div>

                    <div>

                        <b>
                            Précis
                        </b>

                        <small>
                            IA avancée
                        </small>

                    </div>

                </div>

                <div class="feat">

                    <div class="feat-ic">
                        {icon("zap", 22)}
                    </div>

                    <div>

                        <b>
                            Rapide
                        </b>

                        <small>
                            Résultats en secondes
                        </small>

                    </div>

                </div>

                <div class="feat">

                    <div class="feat-ic">
                        {icon("shield", 22)}
                    </div>

                    <div>

                        <b>
                            Fiable
                        </b>

                        <small>
                            Architecture éprouvée
                        </small>

                    </div>

                </div>

            </div>

        </div>

        """

    )


# ============================================================
# API CALL
# ============================================================

def run_analysis(
    uploaded_file
):

    files = {

        "file": (

            uploaded_file.name,

            uploaded_file.getvalue(),

            uploaded_file.type

        )

    }

    params = {

        "model_choice":
            model_param

    }

    try:

        start = time.time()

        response = requests.post(

            API_URL,

            files=files,

            params=params,

            timeout=120,

        )

        elapsed = (
            time.time()
            - start
        )

    except requests.exceptions.ConnectionError:

        st.error(
            "Le serveur d'analyse "
            "est injoignable."
        )

        st.info(
            f"API : {API_URL}"
        )

        return

    except requests.exceptions.Timeout:

        st.error(
            "Le serveur a mis "
            "trop de temps à répondre."
        )

        return

    except requests.exceptions.RequestException as e:

        st.error(
            f"Erreur réseau : {e}"
        )

        return

    # --------------------------------------------------------
    # Vérification réponse
    # --------------------------------------------------------

    if response.status_code != 200:

        try:

            error_data = (
                response.json()
            )

            error_message = (
                error_data.get(
                    "detail",
                    "Erreur inconnue"
                )
            )

        except Exception:

            error_message = (
                response.text[:500]
            )

        st.error(

            f"Erreur API "
            f"({response.status_code}) : "
            f"{error_message}"

        )

        return

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:

        data = response.json()

    except Exception:

        st.error(
            "La réponse de l'API "
            "n'est pas un JSON valide."
        )

        return

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = data.get(
        "prediction",
        "Unknown"
    )

    key = norm(
        prediction
    )

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    raw_probs = data.get(
        "probabilities",
        {}
    )

    probs = {

        norm(k):
            to_percent(v)

        for k, v
        in raw_probs.items()

    }

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = data.get(
        "confidence"
    )

    if confidence is not None:

        confidence = to_percent(
            confidence
        )

    else:

        confidence = probs.get(
            key
        )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    st.session_state.result = {

        "file":
            uploaded_file.name,

        "prediction":
            prediction,

        "key":
            key,

        "probs":
            probs,

        "confidence":
            confidence,

        "heatmap":
            data.get("heatmap"),

        "elapsed":
            elapsed,

        "model":
            model_choice,

    }

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    st.session_state.history.insert(

        0,

        {

            "date":
                datetime.now().strftime(
                    "%d/%m/%Y %H:%M"
                ),

            "file":
                uploaded_file.name,

            "model":
                model_choice,

            "prediction":
                prediction,

            "confidence":
                confidence,

            "elapsed":
                elapsed,

        }

    )


# ============================================================
# RESULT
# ============================================================

def render_result(
    res: dict,
    image: Image.Image
):

    label, color = CLASS_META.get(

        res["key"],

        (
            res["prediction"],
            "#38A8FF"
        )

    )

    is_tumor = (
        res["key"] != "notumor"
    )

    badge = (

        "Tumeur détectée"
        if is_tumor
        else
        "Aucune tumeur détectée"

    )

    ic = (

        icon("alert", 30)
        if is_tumor
        else
        icon("check", 30)

    )

    conf = res[
        "confidence"
    ]

    html(

        f"""

        <span class="pill">

            {icon("clock", 15)}

            Temps d'inférence :
            {res["elapsed"]:.2f} s

        </span>

        """

    )

    html(

        f"""

        <div class="card-title">

            {icon("activity")}

            Résultat de l'analyse

        </div>

        """

    )

    if conf is not None:

        conf_html = f"""

        <div class="conf-row">

            <span class="lbl">
                Confiance
            </span>

            <div class="bar-track">

                <div
                    class="bar-fill"
                    style="
                        width:{min(conf,100):.1f}%;
                        background:
                        linear-gradient(
                            90deg,
                            #06B6D4,
                            #22D3EE
                        );
                    ">
                </div>

            </div>

            <b>
                {conf:.1f}%
            </b>

        </div>

        """

    else:

        conf_html = """

        <div class="conf-row">

            <span class="lbl">

                Confiance non fournie
                par l'API

            </span>

        </div>

        """

    main_card = f"""

    <div class="res-main"
        style="
            border:1px solid {color}66;
            background:{color}18;
        ">

        <div
            class="res-icon"
            style="
                background:{color}30;
                color:{color};
            ">

            {ic}

        </div>

        <div style="flex:1;">

            <span
                class="badge"
                style="
                    background:{color}30;
                    color:{color};
                    border:
                        1px solid {color}66;
                "
            >

                {badge}

            </span>

            <div class="res-name">

                {label}

            </div>

            {conf_html}

        </div>

    </div>

    """

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    if res["probs"]:

        rows = ""

        for k in CLASS_ORDER:

            name, c = CLASS_META[k]

            p = res[
                "probs"
            ].get(
                k,
                0.0
            )

            p = max(
                0,
                min(
                    100,
                    p
                )
            )

            rows += f"""

            <div class="prob-row">

                <div class="prob-head">

                    <span>
                        {name}
                    </span>

                    <span>
                        {p:.1f}%
                    </span>

                </div>

                <div class="bar-track">

                    <div
                        class="bar-fill"
                        style="
                            width:{p:.1f}%;
                            background:{c};
                        "
                    >
                    </div>

                </div>

            </div>

            """

        probs_card = (

            f'<div class="probs">'
            f'{rows}'
            f'</div>'

        )

    else:

        probs_card = (

            '<div class="probs" '
            'style="'
            'color:#93A7D3;'
            'font-size:13px;'
            '">'

            'Les probabilités par classe '
            'ne sont pas renvoyées par l\'API.'

            '</div>'

        )

    html(

        f"""

        <div class="res-grid">

            {main_card}

            {probs_card}

        </div>

        """

    )

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    c1, c2 = st.columns(
        2,
        gap="medium"
    )

    with c1:

        html(
            '<div class="img-title">'
            'Image d\'origine'
            '</div>'
        )

        st.image(
            image,
            use_container_width=True
        )

    with c2:

        html(
            '<div class="img-title">'
            'Heatmap (Grad-CAM)'
            '</div>'
        )

        if res.get("heatmap"):

            try:

                st.image(

                    base64.b64decode(
                        res["heatmap"]
                    ),

                    use_container_width=True

                )

            except Exception:

                html(
                    '<div class="heat-ph">'
                    'Heatmap illisible.'
                    '</div>'
                )

        else:

            html(

                '<div class="heat-ph">'
                'Heatmap non disponible.'
                '<br>'
                'Grad-CAM non configuré côté API.'
                '</div>'

            )


# ============================================================
# DETAILS
# ============================================================

def render_details(
    res: dict
):

    label, _ = CLASS_META.get(

        res["key"],

        (
            res["prediction"],
            ""
        )

    )

    conf = (

        f'{res["confidence"]:.1f}%'
        if res["confidence"] is not None
        else
        "—"

    )

    html(

        f"""

        <div class="card-title"
            style="margin-top:6px;">

            {icon("file")}

            Détails de l'analyse

        </div>

        <div class="det-grid">

            <div class="det-item">

                <small>
                    Modèle utilisé
                </small>

                <b>
                    {res["model"]}
                </b>

            </div>

            <div class="det-item">

                <small>
                    Classe prédite
                </small>

                <b>
                    {label}
                </b>

            </div>

            <div class="det-item">

                <small>
                    Résolution
                </small>

                <b>
                    224 × 224
                </b>

            </div>

            <div class="det-item">

                <small>
                    Confiance
                </small>

                <b>
                    {conf}
                </b>

            </div>

            <div class="notice">

                {icon("info", 18)}

                <span>

                    Cette prédiction est fournie
                    à titre informatif uniquement
                    et ne remplace pas l'avis
                    d'un professionnel de santé.

                </span>

            </div>

        </div>

        """

    )


# ============================================================
# PAGE DETECTION
# ============================================================

def page_detection():

    render_hero()

    left, right = st.columns(
        [1, 1.25],
        gap="large"
    )

    uploaded = None
    image = None
    clicked = False

    with left:

        with st.container(
            border=True
        ):

            html(

                f"""

                <div class="card-title">

                    {icon("upload")}

                    1. Charger une image IRM

                </div>

                """

            )

            uploaded = st.file_uploader(

                "Image IRM",

                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],

                help=
                    "Formats acceptés : JPG, PNG",

            )

            html(

                '<div style="'
                'color:#93A7D3;'
                'font-size:12px;'
                'text-align:center;'
                'margin-top:-6px;'
                '">'
                'Formats acceptés : '
                'JPG, JPEG, PNG'
                '</div>'

            )

            if uploaded is not None:

                try:

                    image = Image.open(
                        uploaded
                    ).convert(
                        "RGB"
                    )

                except Exception:

                    st.error(
                        "Image invalide."
                    )

                    image = None

                if image is not None:

                    html(

                        f"""

                        <div
                            class="card-title sm"
                            style="
                                margin-top:16px;
                            "
                        >

                            {icon("image", 18)}

                            Aperçu de l'image

                        </div>

                        """

                    )

                    st.image(
                        image,
                        use_container_width=True
                    )

                    html(

                        f"""

                        <div class="chip">

                            <div class="chip-ok">

                                {icon("tick", 14)}

                            </div>

                            <div>

                                <b>
                                    {uploaded.name}
                                </b>

                                <small>

                                    {human_size(uploaded.size)}
                                    •
                                    {image.width}×{image.height}px

                                </small>

                            </div>

                        </div>

                        """

                    )

            html(

                f"""

                <div
                    class="card-title sm"
                    style="margin-top:18px;"
                >

                    {icon("scan", 18)}

                    2. Lancer l'analyse

                </div>

                """

            )

            clicked = st.button(

                "Analyser l'image",

                disabled=(
                    uploaded is None
                    or image is None
                ),

                use_container_width=True,

            )

        if (
            clicked
            and uploaded is not None
            and image is not None
        ):

            with st.spinner(
                "Analyse de l'IRM en cours..."
            ):

                run_analysis(
                    uploaded
                )

    # ========================================================
    # RESULTAT
    # ========================================================

    with right:

        res = (
            st.session_state.result
        )

        if (
            res
            and uploaded is not None
            and res["file"] != uploaded.name
        ):

            res = None

        with st.container(
            border=True
        ):

            if (
                res
                and image is not None
            ):

                render_result(
                    res,
                    image
                )

            else:

                html(

                    f"""

                    <div class="empty">

                        <div class="mark">

                            {LOGO_SVG}

                        </div>

                        <b>

                            En attente d'une image IRM

                        </b>

                        Charge une image puis clique
                        sur « Analyser l'image ».

                    </div>

                    """

                )

        if (
            res
            and image is not None
        ):

            with st.container(
                border=True
            ):

                render_details(
                    res
                )


# ============================================================
# ACCUEIL
# ============================================================

def page_accueil():

    render_hero()

    c1, c2, c3 = st.columns(
        3,
        gap="medium"
    )

    cards = [

        (
            "upload",
            "Charge une IRM",
            "Importe une image cérébrale "
            "au format JPG ou PNG."
        ),

        (
            "activity",
            "Choisis un modèle",
            "DenseNet121 ou ResNet50, "
            "depuis le menu de gauche."
        ),

        (
            "target",
            "Lis le résultat",
            "Classe prédite, confiance "
            "et probabilités."
        ),

    ]

    for col, (ic_name, title, description) in zip(

        (c1, c2, c3),

        cards

    ):

        with col:

            with st.container(
                border=True
            ):

                html(

                    f"""

                    <div class="card-title">

                        {icon(ic_name)}

                        {title}

                    </div>

                    <div
                        style="
                            color:#93A7D3;
                            font-size:14px;
                        "
                    >

                        {description}

                    </div>

                    """

                )

    html(

        f"""

        <div
            class="notice"
            style="margin-top:18px;"
        >

            {icon("info", 18)}

            <span>

                Ouvre l'onglet
                « Détection »
                dans le menu de gauche
                pour commencer.

            </span>

        </div>

        """

    )


# ============================================================
# HISTORIQUE
# ============================================================

def page_historique():

    with st.container(
        border=True
    ):

        html(

            f"""

            <div class="card-title">

                {icon("clock")}

                Historique des analyses

            </div>

            """

        )

        hist = (
            st.session_state.history
        )

        if not hist:

            html(

                """

                <div class="empty">

                    <b>
                        Aucune analyse pour le moment
                    </b>

                    Les résultats apparaîtront ici.

                </div>

                """

            )

            return

        rows = ""

        for h in hist:

            label, color = CLASS_META.get(

                norm(
                    h["prediction"]
                ),

                (
                    h["prediction"],
                    "#38A8FF"
                )

            )

            conf = (

                f'{h["confidence"]:.1f}%'
                if h["confidence"] is not None
                else
                "—"

            )

            rows += (

                f"<tr>"

                f"<td>{h['date']}</td>"

                f"<td>{h['file']}</td>"

                f"<td>{h['model']}</td>"

                f"<td style='"
                f"color:{color};"
                f"font-weight:700;"
                f"'>"
                f"{label}"
                f"</td>"

                f"<td>{conf}</td>"

                f"<td>{h['elapsed']:.2f} s</td>"

                f"</tr>"

            )

        html(

            "<table class='htable'>"

            "<thead>"

            "<tr>"

            "<th>Date</th>"

            "<th>Fichier</th>"

            "<th>Modèle</th>"

            "<th>Classe</th>"

            "<th>Confiance</th>"

            "<th>Temps</th>"

            "</tr>"

            "</thead>"

            "<tbody>"

            f"{rows}"

            "</tbody>"

            "</table>"

        )

        if st.button(
            "Vider l'historique"
        ):

            st.session_state.history = []

            st.rerun()


# ============================================================
# A PROPOS
# ============================================================

def page_apropos():

    with st.container(
        border=True
    ):

        html(

            f"""

            <div class="card-title">

                {icon("info")}

                À propos

            </div>

            <div
                style="
                    color:#93A7D3;
                    font-size:14px;
                    line-height:1.75;
                "
            >

                {APP_NAME} {APP_NAME_ACCENT}
                classe les IRM cérébrales
                en 4 catégories :

                Glioma,
                Meningioma,
                Pituitary
                et No Tumor.

                <br><br>

                Les modèles
                DenseNet121 et ResNet50
                sont entraînés avec PyTorch
                et exposés via une API FastAPI.

                L'interface est développée
                avec Streamlit.

                <br><br>

                <b style="color:#EAF0FF;">
                    Avertissement :
                </b>

                projet à but éducatif.
                Il ne constitue pas
                un diagnostic médical.

            </div>

            """

        )


# ============================================================
# ROUTER
# ============================================================

if page == "Accueil":

    page_accueil()

elif page == "Détection":

    page_detection()

elif page == "Historique":

    page_historique()

else:

    page_apropos()


# ============================================================
# FOOTER
# ============================================================

html(

    f"""

    <div class="footer">

        {APP_NAME} {APP_NAME_ACCENT}
        •
        Réalisé par {AUTHOR}
        •
        {SCHOOL}

    </div>

    """

)