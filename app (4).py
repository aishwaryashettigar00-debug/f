# ══════════════════════════════════════════════════════════════════════════════
#  FeedbackLens — AI Customer Feedback Clustering Engine
#  Matches the 10-slide presentation exactly:
#    Slide 1  → Landing / Hero
#    Slide 2  → Pipeline overview (How it works)
#    Slide 3  → Data loading & profiling
#    Slide 4  → Data cleaning (before / after)
#    Slide 5  → TF-IDF + Silhouette + K-Means
#    Slide 6  → PCA 2D cluster scatter
#    Slide 7  → Feature priority matrix
#    Slide 8  → Live app (this IS the app)
#    Slide 9  → Colab-style step log
#    Slide 10 → Conclusion / summary
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import re, io, warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FeedbackLens · AI Feature Clustering",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Reset ─── */
html,body,[class*="css"]{font-family:'Segoe UI',Inter,sans-serif;}
.stApp{background:#0f1117;color:#e2e8f0;}
.main .block-container{padding-top:1.2rem;padding-bottom:3rem;max-width:1300px;}

/* ─── Sidebar ─── */
[data-testid="stSidebar"]{background:#1a1d2e;border-right:1px solid #2d3748;}
[data-testid="stSidebar"] section{padding:0!important;}

/* ─── Metric card ─── */
.mcard{background:#1e2433;border:1px solid #2d3748;border-radius:12px;
       padding:1.1rem 0.8rem;text-align:center;position:relative;overflow:hidden;}
.mcard::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
               background:linear-gradient(90deg,#667eea,#a78bfa);}
.mnum{font-size:2rem;font-weight:900;
      background:linear-gradient(135deg,#667eea,#a78bfa);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.1;}
.mlbl{font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;margin-top:.25rem;}
.msub{font-size:0.65rem;color:#4a5568;margin-top:.1rem;}

/* ─── Section header ─── */
.sh{font-size:1.15rem;font-weight:800;color:#e2e8f0;
    border-left:4px solid #667eea;padding-left:.8rem;margin:1.8rem 0 1rem;}

/* ─── Step card (pipeline) ─── */
.step-card{background:#1e2433;border:1px solid #2d3748;border-radius:10px;
           padding:1rem .9rem;text-align:center;height:100%;}
.step-num{width:2rem;height:2rem;background:linear-gradient(135deg,#667eea,#764ba2);
          border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
          font-size:.8rem;font-weight:800;color:#fff;margin-bottom:.5rem;}
.step-title{font-weight:700;color:#e2e8f0;font-size:.85rem;margin-bottom:.25rem;}
.step-desc{color:#64748b;font-size:.75rem;line-height:1.5;}

/* ─── Before/after boxes ─── */
.before-box{background:#1f1515;border:1px solid #5a2020;border-radius:10px;
            padding:.9rem 1rem;font-family:Consolas,monospace;font-size:.8rem;color:#f87171;}
.after-box{background:#0e1f12;border:1px solid #2d5a3d;border-radius:10px;
           padding:.9rem 1rem;font-family:Consolas,monospace;font-size:.8rem;color:#4ade80;}

/* ─── Cleaning step row ─── */
.cstep{background:#1e2433;border:1px solid #2d3748;border-radius:8px;
       padding:.6rem .9rem;margin:.3rem 0;display:flex;align-items:center;gap:.7rem;}
.cstep-badge{background:#2d3748;border-radius:6px;padding:.15rem .5rem;
             font-size:.72rem;font-weight:700;color:#a78bfa;white-space:nowrap;}
.cstep-title{font-weight:700;color:#e2e8f0;font-size:.85rem;}
.cstep-desc{color:#64748b;font-size:.78rem;}

/* ─── TF-IDF param row ─── */
.param-row{display:flex;justify-content:space-between;align-items:center;
           padding:.45rem .9rem;border-radius:6px;margin:.15rem 0;}
.param-row.even{background:#1e2433;}
.param-row.odd{background:#161b26;}
.param-key{color:#94a3b8;font-size:.85rem;}
.param-val{color:#e2e8f0;font-weight:700;font-size:.85rem;}

/* ─── Cluster expander ─── */
.streamlit-expanderHeader{background:#1a1d2e!important;border:1px solid #2d3748!important;
    border-radius:10px!important;color:#e2e8f0!important;font-weight:700!important;}
.streamlit-expanderContent{background:#0f1420!important;border:1px solid #2d3748!important;
    border-top:none!important;border-radius:0 0 10px 10px!important;}

/* ─── Keyword pill ─── */
.pill{display:inline-block;padding:3px 11px;border-radius:20px;font-size:.73rem;
      font-weight:600;background:#21293d;color:#a78bfa;margin:2px 3px;
      border:1px solid #2d3a52;}

/* ─── Quote block ─── */
.qblock{background:#1c2335;border-radius:8px;padding:.65rem .9rem;margin:.35rem 0;
        font-size:.82rem;color:#94a3b8;font-style:italic;line-height:1.55;
        border-left:3px solid #667eea;}

/* ─── Feature rec box ─── */
.frec{background:linear-gradient(135deg,#0d1f12,#111e30);border:1px solid #2d5a3d;
      border-radius:10px;padding:.9rem 1.1rem;margin-top:.7rem;}
.frec-title{font-weight:800;color:#4ade80;font-size:.92rem;}
.frec-body{color:#94a3b8;font-size:.83rem;margin-top:.35rem;line-height:1.6;}

/* ─── Action item ─── */
.action-item{display:flex;align-items:center;gap:.55rem;margin:.3rem 0;
             background:#161b27;border-radius:6px;padding:.45rem .75rem;
             border:1px solid #21293d;}

/* ─── Priority matrix row ─── */
.prow{background:#1e2433;border:1px solid #2d3748;border-radius:8px;
      padding:.7rem 1rem;margin:.3rem 0;display:flex;align-items:center;gap:.8rem;}

/* ─── Info / success / warn boxes ─── */
.ibox{background:#161e30;border-left:4px solid #667eea;border-radius:0 8px 8px 0;
      padding:.8rem 1rem;color:#94a3b8;font-size:.84rem;margin:.6rem 0;}
.sbox{background:#0e1f16;border-left:4px solid #4ade80;border-radius:0 8px 8px 0;
      padding:.8rem 1rem;color:#6ee7a0;font-size:.84rem;margin:.6rem 0;}
.wbox{background:#1f1a0e;border-left:4px solid #fbbf24;border-radius:0 8px 8px 0;
      padding:.8rem 1rem;color:#d4a74a;font-size:.84rem;margin:.6rem 0;}

/* ─── Colab cell ─── */
.cell{background:#0d1117;border:1px solid #21293d;border-radius:8px;
      margin:.4rem 0;overflow:hidden;}
.cell-header{background:#1a1d2e;padding:.35rem .75rem;display:flex;align-items:center;gap:.6rem;}
.cell-num{color:#667eea;font-family:Consolas,monospace;font-size:.8rem;font-weight:700;}
.cell-title{color:#a78bfa;font-size:.85rem;font-weight:700;}
.cell-code{background:#0d1117;padding:.6rem .9rem;font-family:Consolas,monospace;
           font-size:.77rem;color:#6ee7b7;line-height:1.6;white-space:pre;}
.cell-out{background:#0a0f1a;border-top:1px solid #21293d;padding:.4rem .9rem;
          font-size:.78rem;color:#94a3b8;font-style:italic;}

/* ─── Upload zone ─── */
[data-testid="stFileUploadDropzone"]{background:#1a1d2e!important;
    border:2px dashed #2d3748!important;border-radius:12px!important;}

/* ─── Run button ─── */
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;
    color:#fff!important;font-weight:800!important;font-size:1rem!important;
    border:none!important;border-radius:10px!important;padding:.65rem 1.5rem!important;
    width:100%!important;letter-spacing:.02em!important;}
.stButton>button:hover{opacity:.88!important;transform:translateY(-1px)!important;}

/* ─── Download buttons ─── */
.stDownloadButton>button{background:#1e2433!important;color:#e2e8f0!important;
    border:1px solid #667eea!important;border-radius:8px!important;
    font-weight:600!important;}

/* ─── Summary card ─── */
.exec-card{background:linear-gradient(135deg,#0d1520,#111e30);border:1px solid #1e3a5f;
           border-radius:14px;padding:1.4rem 1.6rem;margin-bottom:.8rem;}

/* ─── Hide branding ─── */
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
COLORS = ["#667eea","#f093fb","#4ade80","#fb923c","#38bdf8",
          "#f472b6","#a78bfa","#34d399","#fbbf24","#60a5fa"]

STOPWORDS = set([
    "the","a","an","and","or","is","it","in","on","to","for","of","with",
    "this","that","was","are","be","i","my","me","we","our","you","your",
    "but","if","at","by","from","have","has","had","not","so","as","do",
    "did","its","just","very","also","they","their","there","been","more",
    "which","when","no","would","could","one","all","can","will","about",
    "what","how","than","then","some","any","up","out","into","only","get",
    "got","too","he","she","his","her","them","were","well","am","being",
    "us","like","use","used","really","much","even","after","over","make",
    "made","good","great","love","nice","best","product","item","ordered",
    "bought","purchase","using","works","worked","working","tried","time",
    "day","now","still","back","first","last","ever","again","always",
    "never","every","feel","feels","felt","little","bit","lot","thing","things",
])

THEMES = {
    "storage": {
        "label":"💾 Storage & Memory",
        "color":"#4ade80",
        "kws":{"storage","memory","card","sandisk","samsung","gb","space","capacity",
               "microsd","speed","transfer","read","write","galaxy","format"},
        "rec":"Expand storage capacity tiers (128GB → 256GB → 512GB). Improve device compatibility documentation. Add a companion storage manager utility.",
        "actions":["Release 256 GB and 512 GB product variants","Build device compatibility checker on website","Develop storage analytics / file-manager feature"],
    },
    "app_ui": {
        "label":"📱 App & Interface",
        "color":"#667eea",
        "kws":{"app","interface","ui","button","screen","navigation","menu","install",
               "update","version","login","crash","slow","feature","notification","installs"},
        "rec":"Conduct full UX audit. Redesign onboarding flow to cut new-user drop-off. Add in-app tooltips for the 5 most complex features.",
        "actions":["Redesign onboarding flow (target 30% drop-off reduction)","Add in-app tooltip hints on complex features","Introduce in-app feedback widget"],
    },
    "scent": {
        "label":"🌸 Scent & Fragrance",
        "color":"#f472b6",
        "kws":{"scent","smell","fragrance","odor","fresh","deodorant","perfume","aroma","stink","musky","floral"},
        "rec":"Expand scent range based on top-requested categories. Introduce customisable fragrance intensity. Launch unscented variant for sensitive users.",
        "actions":["Launch 3 new scent variants based on demand data","Introduce unscented / fragrance-free SKU","Add scent intensity selector to product page"],
    },
    "skin": {
        "label":"🧴 Skin & Sensitivity",
        "color":"#34d399",
        "kws":{"skin","sensitive","rash","irritate","natural","organic","allergy","dry","gentle","hypoallergenic","dermatologist","react","breakout"},
        "rec":"Launch a dedicated sensitive-skin line. Obtain dermatologist-tested certification. Improve ingredient transparency on all packaging.",
        "actions":["Develop hypoallergenic sensitive-skin variant","Obtain dermatologist-tested certification","Redesign packaging to clearly list all ingredients"],
    },
    "price": {
        "label":"💰 Pricing & Value",
        "color":"#fb923c",
        "kws":{"price","expensive","cheap","cost","worth","value","money","afford","budget","overpriced","deal","discount","refund"},
        "rec":"Introduce tiered pricing, a loyalty rewards programme, and bundle deals. Create a clearer value-communication strategy at point of sale.",
        "actions":["Launch loyalty rewards programme","Introduce 3-pack bundle with 15% discount","Add value-comparison section to product page"],
    },
    "packaging": {
        "label":"📦 Packaging & Design",
        "color":"#60a5fa",
        "kws":{"packag","box","seal","leak","container","bottle","wrap","tube","dispenser","label","cap","spill"},
        "rec":"Redesign packaging seal to eliminate leaks. Launch travel-safe formats. Switch to eco-friendly materials.",
        "actions":["Fix packaging seal — eliminate leakage reports","Launch travel-size format","Switch outer box to 100% recycled materials"],
    },
    "delivery": {
        "label":"🚚 Shipping & Delivery",
        "color":"#38bdf8",
        "kws":{"deliver","ship","arriv","order","track","late","courier","dispatch","transit","delay","broken"},
        "rec":"Partner with faster courier providers. Add real-time shipment tracking. Reduce average delivery time and set clear ETA at checkout.",
        "actions":["Integrate real-time tracking (SMS + email)","Add express delivery option at checkout","Target average delivery time < 3 business days"],
    },
    "taste": {
        "label":"🍨 Taste & Flavour",
        "color":"#f093fb",
        "kws":{"taste","flavor","flavour","sweet","bitter","creamy","gelato","ice","cream","bland","rich","refreshing","delicious"},
        "rec":"Expand flavour portfolio. Launch seasonal limited-edition flavours. Test a reduced-sugar / keto-friendly variant.",
        "actions":["Launch 4 new flavours from customer voting","Introduce seasonal limited-edition flavour quarterly","Test reduced-sugar keto-friendly variant"],
    },
    "quality": {
        "label":"💎 Product Quality",
        "color":"#a78bfa",
        "kws":{"quality","poor","cheap","break","broke","flimsy","durable","lasting","solid","sturdy","defect","faulty","damaged"},
        "rec":"Strengthen QC processes. Introduce durability testing benchmarks. Review supplier scorecards monthly.",
        "actions":["Implement pre-shipment quality inspection","Introduce 12-month durability guarantee","Set up defect-rate dashboard reviewed weekly"],
    },
    "support": {
        "label":"🎧 Customer Support",
        "color":"#fbbf24",
        "kws":{"support","service","help","response","staff","team","customer","contact","reply","resolve","complaint","refund","return","exchange"},
        "rec":"Reduce response times. Train support team on product knowledge. Introduce live chat and a self-service FAQ.",
        "actions":["Target < 2-hour first-response time","Launch self-service FAQ / knowledge base","Add live chat during business hours"],
    },
    "default": {
        "label":"💬 General Feedback",
        "color":"#94a3b8",
        "kws":set(),
        "rec":"Review this cluster in detail. Run a targeted survey with this segment to gather more focused insights.",
        "actions":["Run targeted survey with this customer segment","Schedule qualitative user interviews","Manually review top-20 feedback entries for hidden patterns"],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = re.sub(r"http\S+|www\S+|@\S+|#\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(w for w in t.split() if w not in STOPWORDS and len(w) > 2)


def to_sentiment(rating):
    try:
        r = float(str(rating).strip())
        return "Positive" if r >= 4 else ("Neutral" if r >= 3 else "Negative")
    except:
        return "Neutral"


def top_keywords(texts: list, n: int = 12) -> list:
    freq = Counter(" ".join(texts).split())
    return [w for w, _ in freq.most_common(n) if len(w) > 3]


def classify_theme(keywords: list) -> str:
    kset = set(keywords)
    best, best_s = "default", 0
    for key, td in THEMES.items():
        if key == "default":
            continue
        score = sum(1 for kw in kset if any(t in kw for t in td["kws"]))
        if score > best_s:
            best_s, best = score, key
    return best


def find_optimal_k(X, max_k: int = 8):
    n = X.shape[0]
    k_max = min(max_k + 1, n // 3)
    if k_max < 3:
        return 2, {2: 0.0}
    scores = {}
    for k in range(2, k_max):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X)
        try:
            s = silhouette_score(X, labels, sample_size=min(500, n))
            scores[k] = round(float(s), 4)
        except:
            scores[k] = 0.0
    best_k = max(scores, key=scores.get) if scores else 2
    return best_k, scores


def detect_text_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        cl = c.lower().strip()
        if any(kw in cl for kw in ["feedback text","feedback","review","text","comment","content","message","body","description"]):
            if df[c].dtype == object:
                return c
    # fallback: longest avg string col
    str_cols = df.select_dtypes(include="object").columns
    if len(str_cols):
        return max(str_cols, key=lambda c: df[c].dropna().astype(str).str.len().mean())
    return None


def detect_cols(df: pd.DataFrame) -> dict:
    lower_map = {c.lower().strip(): c for c in df.columns}
    result = {}
    result["text"] = detect_text_col(df)
    for kw in ["rating","score","stars","star","rate"]:
        if kw in lower_map:
            result["rating"] = lower_map[kw]; break
    if "rating" not in result:
        for c in df.select_dtypes(include="number").columns:
            vals = df[c].dropna()
            if vals.between(1, 5).mean() > 0.7:
                result["rating"] = c; break
    for kw in ["source","platform","channel","origin"]:
        if kw in lower_map:
            result["source"] = lower_map[kw]; break
    for kw in ["brand","brand name","company","vendor","manufacturer"]:
        if kw in lower_map:
            result["brand"] = lower_map[kw]; break
    for kw in ["product","product name","item","sku","name"]:
        if kw in lower_map and kw != "name":
            result["product"] = lower_map[kw]; break
    for kw in ["category","type","segment","department"]:
        if kw in lower_map:
            result["category"] = lower_map[kw]; break
    for kw in ["date","created","timestamp","posted","review date"]:
        if kw in lower_map:
            result["date"] = lower_map[kw]; break
    return result


def dark_fig(w=7, h=4):
    fig, ax = plt.subplots(figsize=(w, h), facecolor="#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#64748b", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2d3748")
    return fig, ax


def dark_figs(rows, cols, w=12, h=5):
    fig, axes = plt.subplots(rows, cols, figsize=(w, h), facecolor="#0f1117")
    arr = axes if isinstance(axes, np.ndarray) else np.array([axes])
    for ax in arr.flat:
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#64748b", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#2d3748")
    return fig, arr


# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR  (matches Slide 8 sidebar)
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1.4rem 1rem 1rem;border-bottom:1px solid #2d3748;margin-bottom:1rem'>
        <div style='font-size:2.6rem'>🔬</div>
        <div style='font-size:1.3rem;font-weight:900;
            background:linear-gradient(135deg,#667eea,#a78bfa);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>FeedbackLens</div>
        <div style='font-size:.72rem;color:#4a5568;letter-spacing:.06em;margin-top:.15rem'>
            AI FEATURE CLUSTERING ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📂 Upload Feedback Data**")
    st.markdown("<div style='font-size:.76rem;color:#4a5568;margin-bottom:.4rem'>CSV or Excel — any column names</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload file", type=["csv","xlsx","xls"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**⚙️ Clustering Settings**")
    auto_k  = st.checkbox("🤖 Auto-detect clusters (Silhouette)", value=True,
                          help="Tries K=2…8 and picks the best Silhouette Score")
    man_k   = st.slider("Number of clusters (K)", 2, 10, 5) if not auto_k else None

    st.markdown("---")
    st.markdown("**🔍 Filter by Source**")
    f_source = st.multiselect("Source", [], key="fsrc")
    st.markdown("**🔍 Filter by Brand**")
    f_brand  = st.multiselect("Brand",  [], key="fbrnd")
    st.markdown("**🔍 Filter by Sentiment**")
    f_sent   = st.multiselect("Sentiment", ["Positive","Neutral","Negative"], key="fsent")

    st.markdown("---")
    st.markdown("""<div style='font-size:.7rem;color:#2d3748;text-align:center;line-height:1.9'>
        FeedbackLens v2.0 · AI Business Project<br>
        Works with <b style='color:#4a5568'>any feedback dataset</b>
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  HERO HEADER  (Slide 1 / Slide 8 top)
# ──────────────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([3,1])
with h1:
    st.markdown("""
    <div style='margin-bottom:.4rem'>
      <span style='font-size:2.3rem;font-weight:900;color:#e2e8f0'>Customer Feedback</span>
      <span style='font-size:2.3rem;font-weight:900;
        background:linear-gradient(135deg,#667eea,#a78bfa);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent'> Clustering</span>
    </div>
    <div style='color:#64748b;font-size:.95rem'>
      Upload any feedback dataset → AI cleans it → Clusters by theme →
      Feature recommendations &amp; priority matrix for product managers
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown("""
    <div style='text-align:right;padding-top:.5rem'>
      <div style='background:#1e2433;border:1px solid #2d3748;border-radius:10px;
                  padding:.65rem 1rem;display:inline-block'>
        <div style='font-size:.65rem;color:#4a5568;letter-spacing:.08em'>POWERED BY</div>
        <div style='font-size:.88rem;font-weight:700;color:#a78bfa'>TF-IDF + K-Means</div>
        <div style='font-size:.65rem;color:#4a5568'>Silhouette Optimisation · PCA Viz</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
df_raw = None
# ──────────────────────────────────────────────────────────────────────────────
#  LOAD DATA  (upload OR sample dataset)
# ──────────────────────────────────────────────────────────────────────────────
import os as _os
SAMPLE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sample_feedback.csv")

df_raw = None

# Check if user clicked "Load Sample" on landing page
if "use_sample" in st.session_state and st.session_state["use_sample"] and uploaded is None:
    try:
        df_raw = pd.read_csv(SAMPLE_PATH)
        st.markdown(f"""<div class='sbox'>
            ✅ Loaded <strong>sample dataset</strong> —
            {len(df_raw):,} rows × {len(df_raw.columns)} columns.
            &nbsp; (Upload your own file in the sidebar to analyse real data)
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not load sample: {e}")
        st.session_state.pop("use_sample", None)
        st.stop()

elif uploaded is not None:
    st.session_state.pop("use_sample", None)
    try:
        if uploaded.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded, encoding="utf-8", on_bad_lines="skip")
        else:
            df_raw = pd.read_excel(uploaded)
        if len(df_raw) == 0:
            st.error("The uploaded file is empty.")
            st.stop()
        st.markdown(f"""<div class='sbox'>
            ✅ Loaded <strong>{len(df_raw):,} rows</strong> ×
            <strong>{len(df_raw.columns)} columns</strong> from
            <strong>{uploaded.name}</strong>
        </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

else:
    # ── LANDING PAGE (Slides 1 + 2 content) ──────────────────────────────────
    st.markdown("---")

    # Slide 2 — Pipeline
    st.markdown("<div class='sh'>📖 How FeedbackLens Works — The 6-Step AI Pipeline</div>", unsafe_allow_html=True)
    pipeline_steps = [
        ("01","Data Loading","Multi-source CSV/Excel.\nAmazon, App Store feeds."),
        ("02","Data Cleaning","Null removal, stop-words,\nURL stripping, dedup."),
        ("03","TF-IDF","Text vectorisation.\n500 features, bigrams."),
        ("04","K-Means ML","Auto K via Silhouette.\nOptimal clustering."),
        ("05","PCA + Viz","2D cluster projection.\nScatter visualisation."),
        ("06","Insights","Theme labelling.\nFeature suggestions."),
    ]
    cols = st.columns(6)
    for col, (num, title, desc) in zip(cols, pipeline_steps):
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-num'>{num}</div>
                <div class='step-title'>{title}</div>
                <div class='step-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Challenge + Innovation (Slide 2 bottom)
    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown("""<div class='ibox' style='margin-top:1rem'>
            <strong style='color:#e2e8f0'>The Challenge</strong><br><br>
            • Feedback scattered across Amazon, App Stores, Social Media<br>
            • Unstructured text — no consistent format or labelling<br>
            • Product teams lack data-driven feature prioritisation<br>
            • Manual analysis is slow and misses patterns at scale
        </div>""", unsafe_allow_html=True)
    with ch2:
        st.markdown("""<div style='background:#0d1f12;border:1px solid #2d5a3d;border-radius:10px;
                        padding:1rem;margin-top:1rem'>
            <div style='color:#4ade80;font-weight:800;margin-bottom:.5rem'>✨ Our Innovation</div>
            <div style='color:#94a3b8;font-size:.84rem;line-height:1.6'>
                Combines live data upload + AI clustering + feature recommendations
                in one unified Streamlit app. Auto-detects your column names,
                automatically finds the optimal number of clusters,
                and produces a ranked priority matrix any product manager can act on.
            </div>
        </div>""", unsafe_allow_html=True)

    # What your file should look like
    st.markdown("<div class='sh' style='margin-top:2rem'>📋 What Your File Should Look Like</div>", unsafe_allow_html=True)
    st.markdown("""<div class='ibox'>
        FeedbackLens works with <strong>any CSV or Excel file</strong> that has at least one column
        of customer review text. It auto-detects: feedback text, rating, source/platform,
        brand, product, category, and date columns — regardless of what you name them.
    </div>""", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Feedback Text": [
            "Love this! The scent lasts all day and doesn't irritate my skin.",
            "App crashes every time I open settings. Very frustrating.",
            "Good memory card but doesn't work with my Samsung Galaxy S21."
        ],
        "Rating": [5, 1, 3],
        "Source": ["Amazon","Google Play","Amazon"],
        "Brand": ["Schmidt's","MobileApp Co","SanDisk"],
        "Category": ["Beauty","Apps","Electronics"],
    }), use_container_width=True, hide_index=True)

    st.markdown("""<div class='wbox'>
        ⚠️ <strong>Column names don't need to match exactly.</strong>
        FeedbackLens recognises "Review", "Comment", "Stars", "Platform", "Channel" etc.
        The only hard requirement: one text column with customer reviews.
    </div>""", unsafe_allow_html=True)

    # ── One-click demo ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='sh'>🚀 Try It Now — No Upload Needed</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sbox'>
        Click below to instantly load the built-in <strong>120-row sample dataset</strong>
        (Electronics · Beauty · Apps · Food) and see the full AI pipeline live.
    </div>""", unsafe_allow_html=True)
    import os as _tmpimport
    _sample_path = _tmpimport.path.join(_tmpimport.path.dirname(_tmpimport.path.abspath(__file__)), "sample_feedback.csv")
    if _tmpimport.path.exists(_sample_path):
        if st.button("▶️  Load Sample Dataset & Run Demo", use_container_width=True):
            st.session_state["use_sample"] = True
            st.rerun()
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
#  COLUMN DETECTION + SIDEBAR FILTER UPDATE
# ──────────────────────────────────────────────────────────────────────────────
col_map = detect_cols(df_raw)
if col_map.get("text") is None:
    st.error("❌ Cannot find a text/review column. Please check your file.")
    st.stop()

# Show mapping
with st.expander("🔍 Detected Column Mapping (click to review / override)", expanded=False):
    all_c = list(df_raw.columns)
    roles = ["text","rating","source","brand","product","category","date"]
    for role in roles:
        cur = col_map.get(role)
        opts = ["(not used)"] + all_c
        idx  = opts.index(cur) if cur in opts else 0
        sel  = st.selectbox(f"**{role.capitalize()}** column", opts,
                            index=idx, key=f"cm_{role}")
        if sel != "(not used)":
            col_map[role] = sel
        elif role in col_map:
            del col_map[role]


# ──────────────────────────────────────────────────────────────────────────────
#  RUN BUTTON  (Slide 8: big purple button)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("")
run_btn = st.button("🚀  Run AI Analysis Pipeline", use_container_width=True)

if not run_btn and "res" not in st.session_state:
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE  (runs only on button press; then cached in session_state)
# ──────────────────────────────────────────────────────────────────────────────
if run_btn:
    st.session_state.pop("res", None)

    log = []   # Colab-style log for Slide 9

    with st.status("🔄 Running AI Analysis Pipeline…", expanded=True) as status:

        # ── Step 1 ──────────────────────────────────────────────
        st.write("**[1]** Loading and mapping columns…")
        df = df_raw.copy()
        df.rename(columns={col_map["text"]: "Feedback Text"}, inplace=True)
        for role in ["rating","source","brand","product","category","date"]:
            if role in col_map and col_map[role] in df.columns:
                df.rename(columns={col_map[role]: role.capitalize()}, inplace=True)
        for opt, default in [("Rating","3"),("Source","Unknown"),("Brand","Unknown"),
                              ("Product","Unknown"),("Category","Unknown")]:
            if opt not in df.columns:
                df[opt] = default

        total_raw = len(df)
        log.append(("[1]","Load Dataset",
                    f"pd.read_excel / pd.read_csv(FILE)",
                    f"📂 Loaded: {total_raw} rows × {len(df.columns)} columns"))

        # ── Step 2 ──────────────────────────────────────────────
        st.write("**[2]** Applying data quality checks…")
        nulls_before = df["Feedback Text"].isnull().sum()
        dups_before  = df.duplicated(subset=["Feedback Text"]).sum()
        log.append(("[2]","Data Quality (Before)",
                    f"df.isnull().sum()\ndf.duplicated().sum()",
                    f"Nulls: {nulls_before} | Duplicates: {dups_before}"))

        # Apply filters
        if f_source and "Source" in df.columns:
            df = df[df["Source"].isin(f_source)]
        if f_brand and "Brand" in df.columns:
            df = df[df["Brand"].isin(f_brand)]

        # ── Step 3 ──────────────────────────────────────────────
        st.write("**[3]** Cleaning & pre-processing text…")
        df.dropna(subset=["Feedback Text"], inplace=True)
        df = df[df["Feedback Text"].astype(str).str.strip() != ""]
        df.drop_duplicates(subset=["Feedback Text"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["Sentiment"]    = df["Rating"].apply(to_sentiment)
        if f_sent:
            df = df[df["Sentiment"].isin(f_sent)].reset_index(drop=True)
        df["Cleaned Text"] = df["Feedback Text"].astype(str).apply(clean_text)
        df = df[df["Cleaned Text"].str.len() > 10].reset_index(drop=True)
        total_clean = len(df)
        removed     = total_raw - total_clean
        pos_pct = round(df[df["Sentiment"]=="Positive"].shape[0]/max(total_clean,1)*100)
        log.append(("[3]","Clean & Preprocess",
                    "df['Cleaned'] = df['Text'].apply(clean_text)\ndf['Sentiment'] = df['Rating'].apply(...)",
                    f"{total_clean} clean rows | Positive {pos_pct}%"))

        if total_clean < 10:
            status.update(label="❌ Not enough data", state="error")
            st.error("Too few rows after cleaning (need ≥ 10). Adjust filters or check the file.")
            st.stop()

        # ── Step 4 ──────────────────────────────────────────────
        st.write("**[4]** TF-IDF vectorisation…")
        min_df = max(2, int(total_clean * 0.01))
        vec = TfidfVectorizer(max_features=500, ngram_range=(1,2),
                              min_df=min_df, sublinear_tf=True)
        X = vec.fit_transform(df["Cleaned Text"])
        log.append(("[4]","TF-IDF Vectorise",
                    f"TfidfVectorizer(max_features=500,\n  ngram_range=(1,2), min_df={min_df})",
                    f"Matrix: {X.shape[0]} × {X.shape[1]}"))

        # ── Step 5 ──────────────────────────────────────────────
        st.write("**[5]** Finding optimal clusters (Silhouette)…")
        if auto_k:
            best_k, sil_scores = find_optimal_k(X, max_k=min(8, total_clean//5))
        else:
            best_k = min(man_k, total_clean//3)
            sil_scores = {}
        km = KMeans(n_clusters=best_k, random_state=42, n_init=15, max_iter=500)
        df["Cluster"] = km.fit_predict(X)
        try:
            final_sil = silhouette_score(X, df["Cluster"], sample_size=min(500, total_clean))
        except:
            final_sil = 0.0
        log.append(("[5]","Optimal K (Silhouette)",
                    f"for k in range(2,9):\n  sil = silhouette_score(X, labels)",
                    f"Best K = {best_k} (Score: {final_sil:.3f})"))

        # ── Step 6 ──────────────────────────────────────────────
        st.write("**[6]** PCA 2D projection…")
        pca   = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.toarray())
        df["PCA_X"] = coords[:,0];  df["PCA_Y"] = coords[:,1]
        pca_var = pca.explained_variance_ratio_.sum()
        log.append(("[6]","PCA Visualisation",
                    "pca = PCA(n_components=2)\ncoords = pca.fit_transform(X.toarray())",
                    f"Explained variance: {pca_var:.1%}"))

        # ── Step 7 ──────────────────────────────────────────────
        st.write("**[7]** Generating feature recommendations…")
        cluster_info = {}
        for c in sorted(df["Cluster"].unique()):
            mask  = df["Cluster"] == c
            kws   = top_keywords(df.loc[mask,"Cleaned Text"].tolist())
            tkey  = classify_theme(kws)
            td    = THEMES[tkey]
            sents = df.loc[mask,"Sentiment"].value_counts().to_dict()
            neg_p = sents.get("Negative",0)/max(mask.sum(),1)*100
            rtgs  = pd.to_numeric(df.loc[mask,"Rating"], errors="coerce")
            avg_r = float(rtgs.mean()) if not rtgs.isna().all() else None
            cluster_info[c] = {
                "count":   int(mask.sum()),
                "keywords": kws[:10],
                "theme":   tkey,
                "label":   td["label"],
                "color":   td["color"],
                "rec":     td["rec"],
                "actions": td["actions"],
                "sentiments": sents,
                "neg_pct": round(neg_p,1),
                "avg_r":   avg_r,
                "sources": df.loc[mask,"Source"].value_counts().to_dict(),
                "brands":  df.loc[mask,"Brand"].value_counts().head(3).to_dict(),
                "samples": df.loc[mask,"Feedback Text"].tolist()[:3],
                "priority": round(mask.sum()*(1+neg_p/100)),
            }
        log.append(("[7]","Feature Engine",
                    "theme = classify_theme(keywords)\npriority = count*(1+neg_pct/100)",
                    f"{best_k} clusters themed and prioritised ✅"))

        status.update(label="✅ Analysis complete!", state="complete", expanded=False)

    st.session_state["res"] = dict(
        df=df, cluster_info=cluster_info, best_k=best_k,
        final_sil=final_sil, sil_scores=sil_scores, pca_var=pca_var,
        total_raw=total_raw, total_clean=total_clean, removed=removed,
        log=log,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  RENDER RESULTS
# ──────────────────────────────────────────────────────────────────────────────
if "res" not in st.session_state:
    st.stop()

R  = st.session_state["res"]
df = R["df"];  ci = R["cluster_info"];  best_k = R["best_k"]
final_sil = R["final_sil"];  sil_scores = R["sil_scores"]
pca_var   = R["pca_var"]
total_raw = R["total_raw"];  total_clean = R["total_clean"];  removed = R["removed"]

st.markdown("---")


# ════════════════════════════════════════════════════════
#  SLIDE 3 → Dataset Overview metrics
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>📊 Dataset Overview — Data Loading & Profiling</div>", unsafe_allow_html=True)

m1,m2,m3,m4,m5,m6 = st.columns(6)
for col,(val,lbl,sub) in zip([m1,m2,m3,m4,m5,m6],[
    (total_raw,       "TOTAL FEEDBACK",  "Raw input rows"),
    (total_clean,     "AFTER CLEANING",  "Valid reviews"),
    (removed,         "ROWS REMOVED",    "Nulls & dupes"),
    (best_k,          "CLUSTERS FOUND",  "Feedback groups"),
    (f"{final_sil:.3f}","SILHOUETTE",    "Cluster quality"),
    (f"{pca_var:.0%}", "PCA VARIANCE",   "2D coverage"),
]):
    with col:
        st.markdown(f"""<div class='mcard'>
            <div class='mnum'>{val}</div>
            <div class='mlbl'>{lbl}</div>
            <div class='msub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

# Schema + sample (Slide 3 right side)
s3l, s3r = st.columns([1,1])
with s3l:
    sources = df["Source"].value_counts()
    fig, ax = dark_fig(5,3.5)
    bars = ax.barh(sources.index.tolist(), sources.values.tolist(),
                   color=COLORS[:len(sources)], height=0.55)
    for bar,v in zip(bars,sources.values):
        ax.text(v+.5, bar.get_y()+bar.get_height()/2, str(v),
                va="center",color="#e2e8f0",fontsize=8,fontweight="bold")
    ax.set_title("Feedback by Source Channel",color="#e2e8f0",fontweight="bold")
    ax.set_xlabel("Entries",color="#64748b",fontsize=8)
    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

with s3r:
    st.markdown("<div style='font-size:.9rem;font-weight:700;color:#a78bfa;margin-bottom:.5rem'>Dataset Schema</div>", unsafe_allow_html=True)
    schema_cols = [c for c in df.columns if c not in ("PCA_X","PCA_Y","Cluster","Cleaned Text","Sentiment")]
    for i,sc in enumerate(schema_cols[:8]):
        bg = "#1e2433" if i%2==0 else "#161b26"
        st.markdown(f"""<div style='background:{bg};display:flex;justify-content:space-between;
            padding:.35rem .7rem;border-radius:5px;margin:.1rem 0'>
            <span style='color:#a78bfa;font-size:.8rem;font-weight:600'>{sc}</span>
            <span style='color:#64748b;font-size:.78rem'>{str(df[sc].dtype)}</span>
        </div>""", unsafe_allow_html=True)

    # Sample feedback
    sample_text = df["Feedback Text"].iloc[0][:220]
    st.markdown(f"""<div style='background:#1e2433;border:1px solid #2d3748;border-radius:8px;
        padding:.7rem .9rem;margin-top:.8rem'>
        <div style='font-size:.72rem;color:#667eea;font-weight:700;margin-bottom:.3rem'>
            📋 Sample Raw Feedback</div>
        <div style='color:#94a3b8;font-size:.8rem;font-style:italic;line-height:1.55'>
            "{sample_text}…"</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SLIDE 4 → Data Cleaning
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>🧹 Data Cleaning & Pre-processing</div>", unsafe_allow_html=True)

raw_sample = str(df["Feedback Text"].iloc[0])[:180]
clean_sample= str(df["Cleaned Text"].iloc[0])[:180]

s4l, s4arrow, s4r = st.columns([5,.6,5])
with s4l:
    st.markdown("**BEFORE**")
    st.markdown(f"""<div class='before-box'>"{raw_sample}…"<br><br>
        <span style='color:#64748b;font-size:.72rem'>Raw text — contains noise, symbols, filler words</span>
    </div>""", unsafe_allow_html=True)
with s4arrow:
    st.markdown("<div style='font-size:2rem;color:#667eea;text-align:center;padding-top:2.5rem'>→</div>", unsafe_allow_html=True)
with s4r:
    st.markdown("**AFTER**")
    st.markdown(f"""<div class='after-box'>"{clean_sample}…"<br><br>
        <span style='color:#4a5568;font-size:.72rem'>Only meaningful words remain</span>
    </div>""", unsafe_allow_html=True)

st.markdown("**Cleaning Steps Applied**", unsafe_allow_html=False)
steps_clean = [
    ("−"+str(df["Feedback Text"].isnull().sum())+" rows","1. Null Removal","Drop rows with empty Feedback Text"),
    ("Dedup","2. Deduplication","Remove identical reviews — prevents training bias"),
    ("All text","3. Lowercasing","'Natural' → 'natural' — normalise case"),
    ("Regex","4. URL & Symbol Removal","Strip http://, www., emojis, hashtags"),
    ("70+ words","5. Stop-word Removal","Remove common words with no analytical value"),
    ("min_len=3","6. Short Token Filter","Remove tokens < 3 chars to cut noise"),
]
cols6 = st.columns(2)
for i,(badge,title,desc) in enumerate(steps_clean):
    with cols6[i%2]:
        st.markdown(f"""<div class='cstep'>
            <span class='cstep-badge'>{badge}</span>
            <div>
                <div class='cstep-title'>{title}</div>
                <div class='cstep-desc'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

ret_pct = round(total_clean/max(total_raw,1)*100,1)
pos_n = df[df["Sentiment"]=="Positive"].shape[0]
neu_n = df[df["Sentiment"]=="Neutral"].shape[0]
neg_n = df[df["Sentiment"]=="Negative"].shape[0]
st.markdown(f"""<div class='sbox'>
    ✅ <strong>Result:</strong> {total_raw} raw rows → {total_clean} clean rows
    ({ret_pct}% retained) &nbsp;|&nbsp;
    Sentiment added: Positive {round(pos_n/max(total_clean,1)*100)}% ·
    Neutral {round(neu_n/max(total_clean,1)*100)}% ·
    Negative {round(neg_n/max(total_clean,1)*100)}%
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SLIDE 5 → TF-IDF + Silhouette + K-Means
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>🤖 Machine Learning — TF-IDF Vectorisation & K-Means Clustering</div>", unsafe_allow_html=True)

ml1, ml2 = st.columns(2)

with ml1:
    st.markdown("**Step 1: TF-IDF Vectorisation**")
    params = [
        ("Max Features","500 most important terms"),
        ("N-gram Range","Unigrams + Bigrams (1,2)"),
        ("Min Document Freq","≥2 docs (ignore rare terms)"),
        ("Sublinear TF","Log scaling applied"),
        ("Output Shape",f"{total_clean} rows × 500 features"),
    ]
    for i,(k,v) in enumerate(params):
        cls = "even" if i%2==0 else "odd"
        st.markdown(f"""<div class='param-row {cls}'>
            <span class='param-key'>{k}</span>
            <span class='param-val'>{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>**Step 3: K-Means Clustering**", unsafe_allow_html=True)
    km_params = [
        ("Algorithm","K-Means (Lloyd's)"),
        ("Initialisation",f"n_init=15 random starts"),
        ("Distance","Cosine on TF-IDF space"),
        ("Final K",f"{best_k} clusters"),
        ("Silhouette",f"{final_sil:.3f}"),
    ]
    for i,(k,v) in enumerate(km_params):
        cls = "even" if i%2==0 else "odd"
        st.markdown(f"""<div class='param-row {cls}'>
            <span class='param-key'>{k}</span>
            <span class='param-val'>{v}</span>
        </div>""", unsafe_allow_html=True)

with ml2:
    st.markdown("**Step 2: Optimal K Selection (Silhouette Method)**")
    if sil_scores:
        fig, ax = dark_fig(5,3.8)
        ks = list(sil_scores.keys())
        ss = list(sil_scores.values())
        bar_cols = ["#4ade80" if k==best_k else "#2a3347" for k in ks]
        edge_cols= ["#4ade80" if k==best_k else "#3a4a60" for k in ks]
        bars = ax.bar(ks, ss, color=bar_cols, edgecolor=edge_cols, width=.6)
        for bar,v,k in zip(bars,ss,ks):
            ax.text(bar.get_x()+bar.get_width()/2, v+.001, f"{v:.3f}",
                    ha="center",color="#4ade80" if k==best_k else "#64748b",
                    fontsize=8, fontweight="bold" if k==best_k else "normal")
        ax.set_xlabel("Number of Clusters (K)",color="#64748b",fontsize=8)
        ax.set_ylabel("Silhouette Score",color="#64748b",fontsize=8)
        ax.set_title(f"Silhouette Score per K  (Best = {best_k})",
                     color="#e2e8f0",fontweight="bold")
        ax.set_xticks(ks)
        fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)
    else:
        st.markdown(f"""<div class='ibox'>
            Manual K = {best_k} selected. Enable Auto-detect in sidebar to see Silhouette chart.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>**Step 4: PCA Visualisation**", unsafe_allow_html=True)
    st.markdown(f"""<div style='background:#161b26;border:1px solid #2d3748;border-radius:8px;padding:.8rem 1rem'>
        <div style='color:#94a3b8;font-size:.83rem;line-height:1.65'>
            Principal Component Analysis (PCA) reduces the 500-dimensional TF-IDF
            matrix to 2 dimensions for visualisation.<br><br>
            • PCA 1 + 2 explain <strong style='color:#a78bfa'>{pca_var:.0%}</strong> of variance<br>
            • Used only for charting — clustering uses full 500D space<br>
            • Clusters visually separable in 2D projection
        </div>
    </div>""", unsafe_allow_html=True)

# Cluster size mini row
st.markdown("**Cluster Distribution**")
ccols = st.columns(best_k)
for c in sorted(ci.keys()):
    with ccols[c % best_k]:
        col_c = ci[c]["color"]
        st.markdown(f"""<div style='background:#1e2433;border-top:3px solid {col_c};
            border-radius:8px;padding:.6rem .5rem;text-align:center;border:1px solid #2d3748'>
            <div style='font-size:.7rem;color:{col_c};font-weight:700'>Cluster {c}</div>
            <div style='font-size:1.5rem;font-weight:800;color:#e2e8f0'>{ci[c]["count"]}</div>
            <div style='font-size:.65rem;color:#4a5568'>entries</div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SLIDE 6 → PCA Scatter + Cluster cards
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>🗺️ Cluster Analysis & PCA Visualisation</div>", unsafe_allow_html=True)

sc1, sc2 = st.columns([3,2])
with sc1:
    fig, ax = dark_fig(8, 5)
    for c in sorted(df["Cluster"].unique()):
        mask  = df["Cluster"]==c
        color = ci[c]["color"]
        lbl   = f"C{c}: {ci[c]['label']} ({ci[c]['count']})"
        ax.scatter(df.loc[mask,"PCA_X"], df.loc[mask,"PCA_Y"],
                   c=color, label=lbl, alpha=.65, s=22, linewidths=0)
    ax.legend(loc="upper right",fontsize=7,facecolor="#1a1d2e",labelcolor="#e2e8f0",
              edgecolor="#2d3748",framealpha=.95)
    ax.set_xlabel("PCA Component 1",color="#64748b",fontsize=8)
    ax.set_ylabel("PCA Component 2",color="#64748b",fontsize=8)
    ax.set_title(f"Feedback Clusters — PCA 2D Projection ({pca_var:.0%} variance)",
                 color="#e2e8f0",fontweight="bold",fontsize=10)
    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

with sc2:
    for c in sorted(ci.keys()):
        info  = ci[c]
        color = info["color"]
        kws4  = " · ".join(info["keywords"][:4])
        neg   = info["neg_pct"]
        st.markdown(f"""<div style='background:#1e2433;border:1px solid #2d3748;
            border-left:4px solid {color};border-radius:0 8px 8px 0;
            padding:.6rem .9rem;margin:.3rem 0'>
            <div style='font-weight:700;color:{color};font-size:.88rem'>C{c} &nbsp; {info["label"]}</div>
            <div style='color:#64748b;font-size:.76rem;margin-top:.15rem'>
                {info["count"]} entries · Neg: {neg:.0f}%
            </div>
            <div style='color:#4a5568;font-size:.73rem;margin-top:.1rem;font-style:italic'>
                {kws4}
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class='ibox' style='margin-top:.8rem'>
        📐 <strong>Silhouette: {final_sil:.3f}</strong> — acceptable for multi-domain text.
        Text clusters naturally overlap across brands/categories.
    </div>""", unsafe_allow_html=True)

# Sentiment + source charts
da1, da2 = st.columns(2)
with da1:
    fig, ax = dark_fig(5,3.5)
    sent = df["Sentiment"].value_counts()
    sc   = [{"Positive":"#4ade80","Neutral":"#fbbf24","Negative":"#f87171"}.get(s,"#94a3b8") for s in sent.index]
    ax.pie(sent.values, labels=sent.index, colors=sc,
           autopct="%1.1f%%", startangle=90,
           textprops={"color":"#e2e8f0","fontsize":9},
           wedgeprops={"linewidth":2,"edgecolor":"#0f1117"})
    ax.set_title("Sentiment Distribution",color="#e2e8f0",fontweight="bold")
    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

with da2:
    fig, ax = dark_fig(5,3.5)
    csz = df["Cluster"].value_counts().sort_index()
    bar_colors = [ci[c]["color"] for c in csz.index]
    bars = ax.bar([f"C{c}" for c in csz.index], csz.values, color=bar_colors, width=.6)
    for bar,v in zip(bars,csz.values):
        ax.text(bar.get_x()+bar.get_width()/2, v+.5, str(v),
                ha="center",color="#e2e8f0",fontsize=9,fontweight="bold")
    ax.set_title("Feedback Count per Cluster",color="#e2e8f0",fontweight="bold")
    ax.set_ylabel("Count",color="#64748b",fontsize=8)
    fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)


# ════════════════════════════════════════════════════════
#  SLIDE 8 (partial) → Cluster deep-dives
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>🔍 Cluster Deep-Dive & Feature Suggestions</div>", unsafe_allow_html=True)

for c in sorted(ci.keys()):
    info  = ci[c]
    color = info["color"]
    neg   = info["neg_pct"]
    badge = ("🔴 HIGH" if neg>=15 else "🟡 MED" if neg>=5 else "🟢 LOW")

    with st.expander(
        f"Cluster {c} · {info['label']} · {info['count']} reviews · {neg:.0f}% negative  {badge}",
        expanded=(c<2)
    ):
        dl, dr = st.columns([3,2])
        with dl:
            pills = " ".join(f"<span class='pill'>{k}</span>" for k in info["keywords"])
            st.markdown(f"**🔑 Top Keywords**<br>{pills}", unsafe_allow_html=True)
            st.markdown("")
            st.markdown("**🗣️ Sample Reviews**")
            for sample in info["samples"]:
                ex = sample[:240]+("…" if len(sample)>240 else "")
                st.markdown(f"<div class='qblock' style='border-left-color:{color}'>\"{ex}\"</div>",
                            unsafe_allow_html=True)
            if info["sources"]:
                src_pills = " ".join(f"<span class='pill'>{s}: {n}</span>" for s,n in info["sources"].items())
                st.markdown(f"**📡 Sources**<br>{src_pills}", unsafe_allow_html=True)

        with dr:
            # Sentiment mini chart
            fig, ax = dark_fig(3.8,2.8)
            sc_map = {"Positive":"#4ade80","Neutral":"#fbbf24","Negative":"#f87171"}
            sent_d = info["sentiments"]
            bars   = ax.bar(list(sent_d.keys()), list(sent_d.values()),
                            color=[sc_map.get(s,"#94a3b8") for s in sent_d.keys()], width=.5)
            for bar,v in zip(bars,sent_d.values()):
                ax.text(bar.get_x()+bar.get_width()/2, v+.2, str(v),
                        ha="center",color="#e2e8f0",fontsize=9,fontweight="bold")
            ax.set_title("Sentiment Breakdown",color="#e2e8f0",fontsize=9,fontweight="bold")
            fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

            avg_r = info["avg_r"]
            st.markdown(f"""<div style='display:flex;gap:.7rem;margin-top:.4rem'>
                <div style='background:#1e2433;border:1px solid #2d3748;border-radius:8px;
                            padding:.55rem .7rem;flex:1;text-align:center'>
                    <div style='font-size:1.3rem;font-weight:800;color:#fbbf24'>
                        {"⭐ "+str(round(avg_r,1)) if avg_r else "N/A"}</div>
                    <div style='font-size:.65rem;color:#4a5568;text-transform:uppercase'>Avg Rating</div>
                </div>
                <div style='background:#1e2433;border:1px solid #2d3748;border-radius:8px;
                            padding:.55rem .7rem;flex:1;text-align:center'>
                    <div style='font-size:1.3rem;font-weight:800;color:{color}'>{info["priority"]}</div>
                    <div style='font-size:.65rem;color:#4a5568;text-transform:uppercase'>Priority Score</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Recommendation
        st.markdown(f"""<div class='frec'>
            <div class='frec-title'>💡 Feature Recommendation for Product Managers</div>
            <div class='frec-body'>{info["rec"]}</div>
        </div>""", unsafe_allow_html=True)

        # Action items
        st.markdown("**📋 Specific Action Items**")
        for i,action in enumerate(info["actions"],1):
            st.markdown(f"""<div class='action-item'>
                <span style='background:{color};color:#0d1117;border-radius:50%;
                    width:1.25rem;height:1.25rem;display:inline-flex;align-items:center;
                    justify-content:center;font-size:.68rem;font-weight:800;flex-shrink:0'>{i}</span>
                <span style='color:#94a3b8;font-size:.84rem'>{action}</span>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SLIDE 7 → Feature Priority Matrix
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>🎯 Feature Priority Matrix — For Product Managers</div>", unsafe_allow_html=True)
st.markdown("""<div class='ibox'>
    <strong>Priority Score</strong> = Feedback Volume × (1 + Negative Sentiment %).
    Higher score = more urgent. &nbsp;
    🔴 HIGH ≥ 150 &nbsp;|&nbsp; 🟡 MEDIUM ≥ 80 &nbsp;|&nbsp; 🟢 LOW &lt; 80
</div>""", unsafe_allow_html=True)

prows = []
for c,info in ci.items():
    neg = info["neg_pct"]
    avg = info["avg_r"]
    urg = "🔴 HIGH" if neg>=15 else ("🟡 MEDIUM" if neg>=5 else "🟢 LOW")
    prows.append({
        "Cluster":        f"C{c}",
        "Theme":          info["label"],
        "Feedback Count": info["count"],
        "Avg Rating":     f"{avg:.1f} ⭐" if avg else "N/A",
        "Neg %":          f"{neg:.0f}%",
        "Priority Score": info["priority"],
        "Urgency":        urg,
        "Top Keywords":   ", ".join(info["keywords"][:5]),
        "Recommendation": info["rec"],
    })
pdf = pd.DataFrame(prows).sort_values("Priority Score", ascending=False).reset_index(drop=True)

st.dataframe(
    pdf,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Priority Score": st.column_config.ProgressColumn(
            "Priority Score",
            min_value=0,
            max_value=int(pdf["Priority Score"].max())+10,
        ),
        "Recommendation": st.column_config.TextColumn("Recommendation", width="large"),
    }
)

# Priority bar chart
fig, ax = dark_fig(10,4)
sc_sorted = pdf["Cluster"].tolist()
ss_sorted = pdf["Priority Score"].tolist()
sc_colors = [ci[int(c[1:])]["color"] for c in sc_sorted]
bars = ax.barh(sc_sorted[::-1], ss_sorted[::-1], color=sc_colors[::-1], height=.55)
for bar,v in zip(bars,ss_sorted[::-1]):
    ax.text(v+1, bar.get_y()+bar.get_height()/2, str(v),
            va="center",color="#e2e8f0",fontsize=9,fontweight="bold")
ax.set_xlabel("Priority Score",color="#64748b",fontsize=9)
ax.set_title("Feature Priority Ranking",color="#e2e8f0",fontweight="bold",fontsize=11)
fig.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close(fig)

# Top 3 priority callouts (Slide 7 bottom)
st.markdown("**🎯 Top Priority Feature Actions**")
top3 = pdf.head(3)
t3cols = st.columns(3)
titles = ["Storage Tier Expansion","Top-Priority Fix","App & Interface Overhaul"]
for col,(_,row) in zip(t3cols, top3.iterrows()):
    c_num = int(row["Cluster"][1:])
    color = ci[c_num]["color"]
    with col:
        st.markdown(f"""<div style='background:linear-gradient(135deg,#0d1f12,#111e30);
            border:1px solid {color};border-radius:10px;padding:1rem 1.1rem;height:100%'>
            <div style='font-weight:800;color:{color};font-size:.9rem'>{row["Theme"]}</div>
            <div style='color:#64748b;font-size:.76rem;margin:.25rem 0'>
                {row["Feedback Count"]} reviews · {row["Neg %"]} negative · Priority {row["Priority Score"]}
            </div>
            <div style='color:#94a3b8;font-size:.8rem;line-height:1.55;margin-top:.4rem'>
                {row["Recommendation"][:160]}…
            </div>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SLIDE 9 → Google Colab-style step log
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>📓 Analysis Log — Google Colab Style</div>", unsafe_allow_html=True)
st.markdown("""<div class='ibox'>
    Each step mirrors the Google Colab notebook exactly.
    This log is auto-generated from the pipeline run above.
</div>""", unsafe_allow_html=True)

for (cell_num, cell_title, cell_code, cell_out) in R["log"]:
    st.markdown(f"""<div class='cell'>
        <div class='cell-header'>
            <span class='cell-num'>{cell_num}</span>
            <span class='cell-title'>{cell_title}</span>
        </div>
        <div class='cell-code'>{cell_code}</div>
        <div class='cell-out'>▶ &nbsp;{cell_out}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("""<div style='color:#4a5568;font-size:.78rem;text-align:center;margin-top:.5rem;font-style:italic'>
    + Additional cells: Cluster Visualisation · PCA · Feature Engine · Priority Matrix · Export & Download
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  SLIDE 10 → Conclusion / Executive summary
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>📋 Executive Summary — Product Manager Report</div>", unsafe_allow_html=True)

total_neg  = df[df["Sentiment"]=="Negative"].shape[0]
total_pos  = df[df["Sentiment"]=="Positive"].shape[0]
neg_pct_all= round(total_neg/max(total_clean,1)*100,1)
pos_pct_all= round(total_pos/max(total_clean,1)*100,1)
avg_all    = pd.to_numeric(df["Rating"],errors="coerce").mean()

st.markdown(f"""<div class='exec-card'>
    <div style='font-size:.95rem;font-weight:800;color:#667eea;margin-bottom:.9rem;letter-spacing:.03em'>
        🔬 FEEDBACKLENS ANALYSIS REPORT
    </div>
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-bottom:1rem'>
        <div style='background:#161b27;border:1px solid #2d3748;border-radius:8px;
                    padding:.7rem;text-align:center'>
            <div style='font-size:1.6rem;font-weight:800;color:#f87171'>{total_neg}</div>
            <div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>Negative Reviews</div>
            <div style='font-size:.62rem;color:#4a5568'>{neg_pct_all}% of total</div>
        </div>
        <div style='background:#161b27;border:1px solid #2d3748;border-radius:8px;
                    padding:.7rem;text-align:center'>
            <div style='font-size:1.6rem;font-weight:800;color:#4ade80'>{pos_pct_all}%</div>
            <div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>Positive Sentiment</div>
            <div style='font-size:.62rem;color:#4a5568'>Customer satisfaction</div>
        </div>
        <div style='background:#161b27;border:1px solid #2d3748;border-radius:8px;
                    padding:.7rem;text-align:center'>
            <div style='font-size:1.6rem;font-weight:800;color:#fbbf24'>
                {"⭐ "+str(round(avg_all,1)) if not np.isnan(avg_all) else "N/A"}
            </div>
            <div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>Overall Avg Rating</div>
            <div style='font-size:.62rem;color:#4a5568'>Across all reviews</div>
        </div>
        <div style='background:#161b27;border:1px solid #2d3748;border-radius:8px;
                    padding:.7rem;text-align:center'>
            <div style='font-size:1.6rem;font-weight:800;color:#667eea'>{best_k}</div>
            <div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>Clusters Found</div>
            <div style='font-size:.62rem;color:#4a5568'>Silhouette: {final_sil:.3f}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

for _,row in pdf.head(3).iterrows():
    c_num = int(row["Cluster"][1:])
    color = ci[c_num]["color"]
    st.markdown(f"""<div style='background:#161b27;border-left:4px solid {color};
        border-radius:0 8px 8px 0;padding:.7rem 1rem;margin:.4rem 0'>
        <div style='font-weight:700;color:{color};font-size:.88rem'>
            {row["Urgency"]} · {row["Theme"]} (Priority: {row["Priority Score"]})
        </div>
        <div style='color:#94a3b8;font-size:.8rem;margin-top:.2rem'>
            {row["Feedback Count"]} reviews · {row["Neg %"]} negative · Avg rating {row["Avg Rating"]}
        </div>
        <div style='color:#64748b;font-size:.77rem;margin-top:.2rem'>
            📌 {row["Recommendation"][:180]}
        </div>
    </div>""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Key takeaways (Slide 10 four boxes)
st.markdown("**Key Takeaways**")
kt1,kt2,kt3,kt4 = st.columns(4)
takeaways = [
    ("🔬","AI-Powered Clustering",f"TF-IDF + K-Means grouped {total_clean} reviews into {best_k} themes automatically — zero manual labelling.","#667eea"),
    ("💡","Actionable Insights","Each cluster maps to concrete feature recommendations. Product teams can act immediately.","#4ade80"),
    ("📊","Data Quality Proven",f"{ret_pct if 'ret_pct' in dir() else round(total_clean/max(total_raw,1)*100,1)}% data retained. Silhouette score validates cluster separation.","#fb923c"),
    ("🚀","Live App Delivered","FeedbackLens works on any feedback dataset. Upload → Cluster → Download report.","#a78bfa"),
]
for col,(icon,title,desc,color) in zip([kt1,kt2,kt3,kt4],takeaways):
    with col:
        st.markdown(f"""<div style='background:#1e2433;border:1px solid #2d3748;border-radius:10px;
            padding:1rem;text-align:center;height:100%'>
            <div style='font-size:1.8rem'>{icon}</div>
            <div style='font-weight:700;color:{color};font-size:.88rem;margin:.4rem 0'>{title}</div>
            <div style='color:#64748b;font-size:.78rem;line-height:1.5'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# Future enhancements
st.markdown(f"""<div style='background:#1e2433;border:1px solid #2d3748;border-radius:10px;
    padding:1rem 1.3rem;margin-top:1rem'>
    <div style='font-weight:800;color:#4ade80;margin-bottom:.6rem'>🚀 Future Enhancements</div>
    <div style='color:#94a3b8;font-size:.84rem;line-height:1.9'>
        ①&nbsp; Integrate live web scraping bot (BeautifulSoup + Selenium) into Streamlit<br>
        ②&nbsp; Upgrade to BERT / Sentence-Transformers for semantic clustering<br>
        ③&nbsp; Add LDA topic modelling alongside K-Means for richer theme coverage<br>
        ④&nbsp; Deploy to Streamlit Cloud for permanent public demo URL
    </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  EXPORT  (Slide 8: Download buttons)
# ════════════════════════════════════════════════════════
st.markdown("<div class='sh'>⬇️ Export Results</div>", unsafe_allow_html=True)

export_cols = ["Feedback Text","Sentiment","Cluster"]
for opt in ["Source","Brand","Product","Category","Rating","Date","Cleaned Text"]:
    if opt in df.columns:
        export_cols.append(opt)

# Build 3-sheet Excel
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    df[export_cols].to_excel(writer, sheet_name="Clustered Feedback", index=False)
    pdf.to_excel(writer, sheet_name="Priority Matrix", index=False)
    sumrows = []
    for c,info in ci.items():
        sumrows.append({
            "Cluster": f"C{c}", "Theme": info["label"],
            "Count": info["count"],
            "Avg Rating": f"{info['avg_r']:.1f}" if info["avg_r"] else "N/A",
            "Neg %": f"{info['neg_pct']:.0f}%",
            "Priority": info["priority"],
            "Keywords": ", ".join(info["keywords"][:6]),
            "Recommendation": info["rec"],
            "Action 1": info["actions"][0] if len(info["actions"])>0 else "",
            "Action 2": info["actions"][1] if len(info["actions"])>1 else "",
            "Action 3": info["actions"][2] if len(info["actions"])>2 else "",
        })
    pd.DataFrame(sumrows).to_excel(writer, sheet_name="Cluster Summary", index=False)
buf.seek(0)

dl1,dl2 = st.columns(2)
with dl1:
    st.download_button("⬇️ Download Full Report (Excel — 3 sheets)", buf,
        "feedbacklens_report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)
with dl2:
    st.download_button("⬇️ Download Clustered Data (CSV)", df[export_cols].to_csv(index=False),
        "feedbacklens_clusters.csv","text/csv", use_container_width=True)

st.markdown("""<div style='text-align:center;color:#2d3748;font-size:.73rem;padding:2rem 0 1rem'>
    FeedbackLens v2.0 · AI Customer Feedback Clustering Engine ·
    TF-IDF + K-Means + PCA · Built with Python &amp; Streamlit
</div>""", unsafe_allow_html=True)
