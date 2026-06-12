import time
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.input_form import render_input_form
from app.components.result_display import render_result
from src.predict import load_model, predict_single

st.set_page_config(
    page_title="JobVerify AI — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
#  MASTER CSS  (no blank lines inside the style block — CommonMark rule)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
:root {
  --cyan:#06B6D4; --purple:#8B5CF6; --green:#10B981; --red:#EF4444; --amber:#F59E0B;
  --text0:#FFFFFF; --text1:rgba(255,255,255,0.82); --text2:rgba(255,255,255,0.50); --text3:rgba(255,255,255,0.28);
  --mono:'JetBrains Mono',monospace; --sans:'Space Grotesk',sans-serif; --ui:'Inter',sans-serif;
  --border:rgba(255,255,255,0.07); --border-hov:rgba(255,255,255,0.14);
  --r-sm:8px; --r-md:12px; --r-lg:16px; --r-xl:24px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
html,body{
  background:
    radial-gradient(ellipse 70% 45% at 8% 0%,rgba(6,182,212,0.09) 0%,transparent 65%),
    radial-gradient(ellipse 60% 40% at 92% 0%,rgba(124,58,237,0.07) 0%,transparent 65%),
    radial-gradient(ellipse 50% 35% at 50% 100%,rgba(37,99,235,0.06) 0%,transparent 65%),
    #080810 !important;
  color:var(--text1) !important;
  font-family:var(--ui) !important;
}
[data-testid="stAppViewContainer"],[data-testid="stMain"],.main{background:transparent !important;color:var(--text1) !important;font-family:var(--ui) !important;}
.block-container{padding:0 !important;max-width:100% !important;background:transparent !important;}
#MainMenu,footer,header,[data-testid="stToolbar"],[data-testid="stSidebar"],[data-testid="stDecoration"]{display:none !important;}
::-webkit-scrollbar{width:4px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:2px;}
/* ── COLUMN PANELS ── */
div[data-testid="column"]{background:rgba(8,8,20,0.60) !important;backdrop-filter:blur(28px) saturate(160%) !important;-webkit-backdrop-filter:blur(28px) saturate(160%) !important;border:1px solid var(--border) !important;border-radius:var(--r-xl) !important;padding:2rem !important;transition:border-color 0.35s ease,box-shadow 0.35s ease !important;}
div[data-testid="column"]:hover{border-color:rgba(255,255,255,0.12) !important;box-shadow:0 0 60px rgba(6,182,212,0.05),inset 0 1px 0 rgba(255,255,255,0.05) !important;}
div[data-testid="column"] div[data-testid="column"]{background:transparent !important;backdrop-filter:none !important;-webkit-backdrop-filter:none !important;border:none !important;border-radius:0 !important;padding:0 0.25rem !important;box-shadow:none !important;}
/* ── FORM FIELD LABELS — big, visible, coloured ── */
.stTextInput>label,.stTextArea>label,.stSelectbox>label{font-family:var(--sans) !important;font-size:0.88rem !important;font-weight:600 !important;letter-spacing:0.01em !important;color:rgba(255,255,255,0.80) !important;margin-bottom:7px !important;text-transform:none !important;}
.stTextInput>div>div>input,.stTextArea>div>textarea{background:rgba(0,0,0,0.3) !important;border:1px solid rgba(255,255,255,0.11) !important;border-radius:var(--r-md) !important;color:var(--text1) !important;font-family:var(--ui) !important;font-size:0.88rem !important;transition:border-color 0.22s,box-shadow 0.22s !important;caret-color:var(--cyan) !important;}
.stTextInput>div>div>input:focus,.stTextArea>div>textarea:focus{border-color:rgba(6,182,212,0.60) !important;box-shadow:0 0 0 1px rgba(6,182,212,0.35),0 0 18px rgba(6,182,212,0.12) !important;background:rgba(0,0,0,0.40) !important;outline:none !important;}
.stTextInput>div>div>input::placeholder,.stTextArea>div>textarea::placeholder{color:rgba(255,255,255,0.12) !important;}
/* Select */
div[data-baseweb="select"]>div{background:rgba(0,0,0,0.3) !important;border:1px solid rgba(255,255,255,0.11) !important;border-radius:var(--r-md) !important;color:var(--text1) !important;font-family:var(--ui) !important;font-size:0.88rem !important;transition:border-color 0.22s !important;}
div[data-baseweb="select"]>div:focus-within{border-color:rgba(6,182,212,0.60) !important;box-shadow:0 0 0 1px rgba(6,182,212,0.35),0 0 18px rgba(6,182,212,0.12) !important;}
div[data-baseweb="popover"]{background:#0C0C14 !important;}
div[data-baseweb="popover"] ul{background:#0C0C14 !important;border:1px solid rgba(255,255,255,0.08) !important;border-radius:var(--r-md) !important;}
div[data-baseweb="popover"] li{font-family:var(--ui) !important;font-size:0.84rem !important;color:var(--text2) !important;background:transparent !important;}
div[data-baseweb="popover"] li:hover,div[data-baseweb="popover"] li[aria-selected="true"]{background:rgba(6,182,212,0.08) !important;color:var(--cyan) !important;}
/* Checkboxes */
div[data-testid="stCheckbox"]{background:rgba(0,0,0,0.22) !important;border:1px solid rgba(255,255,255,0.10) !important;border-radius:var(--r-md) !important;padding:10px 14px !important;transition:border-color 0.2s,background-color 0.2s,box-shadow 0.2s !important;}
div[data-testid="stCheckbox"]:hover{border-color:rgba(6,182,212,0.38) !important;background:rgba(6,182,212,0.04) !important;box-shadow:0 0 12px rgba(6,182,212,0.05) !important;}
div[data-testid="stCheckbox"]>label{font-family:var(--ui) !important;font-size:0.85rem !important;color:var(--text1) !important;font-weight:500 !important;}
/* Button */
.stButton>button{position:relative !important;background:linear-gradient(135deg,rgba(6,182,212,0.13) 0%,rgba(139,92,246,0.13) 100%) !important;color:var(--text0) !important;border:1px solid rgba(6,182,212,0.30) !important;border-radius:var(--r-lg) !important;font-family:var(--sans) !important;font-size:0.92rem !important;font-weight:600 !important;letter-spacing:0.01em !important;padding:0.9rem 1.5rem !important;width:100% !important;transition:all 0.3s cubic-bezier(0.4,0,0.2,1) !important;}
.stButton>button:hover{border-color:rgba(6,182,212,0.65) !important;box-shadow:0 0 32px rgba(6,182,212,0.2),0 0 80px rgba(139,92,246,0.09),inset 0 1px 0 rgba(255,255,255,0.09) !important;transform:translateY(-2px) !important;background:linear-gradient(135deg,rgba(6,182,212,0.20) 0%,rgba(139,92,246,0.20) 100%) !important;}
.stButton>button:active{transform:translateY(0) scale(0.99) !important;}
/* Plotly */
.stPlotlyChart,.stPyplot{background:transparent !important;}
/* Pill */
.jv-pill{display:inline-flex;align-items:center;gap:6px;padding:5px 14px 5px 9px;border-radius:100px;background:rgba(16,185,129,0.07);border:1px solid rgba(16,185,129,0.22);font-family:var(--mono);font-size:0.6rem;color:#6EE7B7;letter-spacing:0.06em;}
.jv-pill-dot{width:6px;height:6px;border-radius:50%;background:#10B981;animation:pillBlink 2.4s ease-in-out infinite;}
@keyframes pillBlink{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(16,185,129,0.6);}50%{opacity:0.5;box-shadow:0 0 0 5px rgba(16,185,129,0);}}
/* Section labels in form */
.jv-sec{display:flex;align-items:center;gap:10px;margin:1.6rem 0 1rem;padding-left:10px;border-left:2px solid;font-family:var(--sans);font-size:0.9rem;font-weight:600;letter-spacing:0.01em;}
.jv-sec-line{flex:1;height:1px;opacity:0.15;}
/* Intel panel */
.jv-intel-row{display:flex;justify-content:space-between;align-items:flex-start;padding:11px 0;border-bottom:1px solid rgba(255,255,255,0.05);}
.jv-intel-row:last-child{border-bottom:none;}
.jv-intel-key{font-family:var(--mono);font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text3);padding-top:3px;}
.jv-intel-val{font-family:var(--sans);font-size:0.92rem;font-weight:600;color:var(--text1);text-align:right;}
.jv-accent-c{color:var(--cyan);}
.jv-accent-g{color:var(--green);}
/* Step animation */
.jv-step{display:flex;align-items:center;gap:12px;padding:8px 12px;border-radius:var(--r-md);font-family:var(--mono);font-size:0.72rem;letter-spacing:0.06em;color:var(--text2);background:rgba(6,182,212,0.04);border:1px solid rgba(6,182,212,0.1);margin-bottom:6px;animation:stepFadeIn 0.25s ease;}
@keyframes stepFadeIn{from{opacity:0;transform:translateX(-6px);}to{opacity:1;transform:none;}}
/* Signal chips */
.jv-sig{padding:13px 15px;background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.08);border-radius:var(--r-md);transition:border-color 0.2s,transform 0.2s;cursor:default;}
.jv-sig:hover{border-color:rgba(6,182,212,0.20);transform:translateY(-1px);}
.jv-sig-name{font-family:var(--mono);font-size:0.58rem;text-transform:uppercase;letter-spacing:0.1em;color:var(--text3);margin-bottom:7px;}
.jv-sig-val{font-family:var(--sans);font-size:1.55rem;font-weight:700;letter-spacing:-0.03em;}
.jv-divider{display:flex;align-items:center;gap:10px;font-family:var(--mono);font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;color:var(--text3);margin:16px 0 12px;}
.jv-divider::after{content:'';flex:1;height:1px;background:rgba(255,255,255,0.07);}
/* Responsive stacking */
@media (max-width: 768px) {
  div[data-testid="column"]{margin-bottom:1rem !important;}
}
</style>""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Aurora Background */
.stApp {
    background:
        radial-gradient(circle at 15% 20%,
            rgba(6,182,212,0.12),
            transparent 35%),

        radial-gradient(circle at 85% 15%,
            rgba(139,92,246,0.12),
            transparent 35%),

        radial-gradient(circle at 50% 90%,
            rgba(37,99,235,0.10),
            transparent 40%),

        #080810 !important;
}

/* Floating glow layer */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;

    background:
        radial-gradient(circle at 30% 40%,
            rgba(6,182,212,0.06),
            transparent 20%),

        radial-gradient(circle at 70% 60%,
            rgba(139,92,246,0.06),
            transparent 20%);

    filter: blur(80px);
    z-index: 0;
}

/* subtle grid */
.stApp::after {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;

    background-image:
        linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px);

    background-size: 50px 50px;

    z-index: 0;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising model…")
def get_model():
    return load_model()


def _analysis_sequence():
    steps = ["Extracting features", "Computing TF-IDF embeddings",
             "Running logistic regression", "Generating SHAP explanations"]
    ph = st.empty()
    done = []
    for s in steps:
        done.append(s)
        rows = "".join(
            '<div class="jv-step"><span style="color:{c};">{ic}</span>'
            '<span style="color:{tc};">{x}</span></div>'.format(
                c="#10B981" if i < len(done)-1 else "#06B6D4",
                ic="✓" if i < len(done)-1 else "⟳",
                tc="rgba(16,185,129,0.9)" if i < len(done)-1 else "rgba(255,255,255,0.65)",
                x=x
            )
            for i, x in enumerate(done)
        )
        ph.markdown(rows, unsafe_allow_html=True)
        time.sleep(0.28)
    time.sleep(0.15)
    ph.empty()


def main():
    pipeline, threshold = get_model()
    thresh_pct  = int(threshold * 100)
    recall_pct  = 96

    st.markdown('<div style="position:relative;z-index:2;padding:2.4rem 2.8rem 3rem;">', unsafe_allow_html=True)

    # ── HERO ── (no blank lines inside HTML — CommonMark type-7 block rule)
    st.markdown(
        '<div style="margin-bottom:16px;">'
        '<span class="jv-pill"><span class="jv-pill-dot"></span>'
        f'Model online &nbsp;·&nbsp; threshold {threshold:.2f}'
        '</span></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="display:flex;align-items:flex-start;justify-content:space-between;'
        'flex-wrap:wrap;gap:2rem;margin-bottom:3rem;">'
        # Left: headline + tagline
        '<div style="flex:1;min-width:300px;">'
        '<div style="font-family:var(--sans);font-size:clamp(2.4rem,4.2vw,3.6rem);'
        'font-weight:800;letter-spacing:-0.05em;line-height:1.06;color:var(--text0);margin-bottom:16px;">'
        'Detect fraudulent<br>'
        '<span style="color:var(--cyan);">job listings</span>'
        '<span style="color:var(--purple);"> with AI</span>'
        '</div>'
        '<div style="font-family:var(--ui);font-size:1.02rem;color:var(--text2);'
        'font-weight:300;letter-spacing:0.01em;line-height:1.65;'
        'max-width:520px;margin-bottom:20px;">'
        'Paste any job posting below. Our model cross-checks 11 signals using '
        'Logistic Regression + TF-IDF and explains every prediction with SHAP.'
        '</div>'
        '<div style="font-family:var(--mono);font-size:0.58rem;color:rgba(6,182,212,0.48);'
        'letter-spacing:0.16em;text-transform:uppercase;">'
        'EMSCAD Dataset &nbsp;·&nbsp; Neural Fraud Detection System &nbsp;·&nbsp; v1.0'
        '</div>'
        '</div>'
        # Right: stats glass card
        '<div style="display:flex;align-items:flex-start;padding-top:6px;">'
        '<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.09);'
        'border-radius:20px;padding:20px 28px;display:flex;gap:32px;flex-wrap:wrap;'
        'backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);">'
        '<div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">'
        '<div style="font-family:var(--sans);font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:var(--text0);line-height:1;">17,880</div>'
        '<div style="font-family:var(--mono);font-size:0.52rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text3);">postings trained</div>'
        '</div>'
        '<div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">'
        '<div style="font-family:var(--sans);font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:var(--cyan);line-height:1;">96.5%</div>'
        '<div style="font-family:var(--mono);font-size:0.52rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text3);">recall score</div>'
        '</div>'
        '<div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px;">'
        '<div style="font-family:var(--sans);font-size:2rem;font-weight:700;letter-spacing:-0.04em;color:var(--purple);line-height:1;">4.8%</div>'
        '<div style="font-family:var(--mono);font-size:0.52rem;letter-spacing:0.1em;text-transform:uppercase;color:var(--text3);">base fraud rate</div>'
        '</div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── TWO COLUMNS ──
    left, right = st.columns([1.38, 0.62], gap="large")

    with left:
        st.markdown(
            '<div style="font-family:var(--mono);font-size:0.6rem;text-transform:uppercase;'
            'letter-spacing:0.18em;color:rgba(6,182,212,0.70);margin-bottom:20px;padding-top:4px;">'
            '01 &mdash; Job Listing Input</div>',
            unsafe_allow_html=True
        )
        listing = render_input_form()
        st.markdown('<div style="height:16px;"></div>', unsafe_allow_html=True)
        submitted = st.button("⟶  Run Authenticity Analysis")

    with right:
        if submitted and listing:
            _analysis_sequence()
            result = predict_single(listing, pipeline=pipeline, threshold=threshold)
            render_result(result, listing)
        else:
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.6rem;text-transform:uppercase;'
                'letter-spacing:0.18em;color:rgba(139,92,246,0.70);margin-bottom:20px;padding-top:4px;">'
                '02 &mdash; Intelligence Panel</div>',
                unsafe_allow_html=True
            )
            # ── INTELLIGENCE PANEL ── (no blank lines inside HTML)
            st.markdown(
                '<div style="margin-bottom:20px;border:1px solid rgba(255,255,255,0.06);'
                'border-radius:16px;padding:4px 0;overflow:hidden;">'
                '<div class="jv-intel-row" style="padding:12px 18px;">'
                '<span class="jv-intel-key">Architecture</span>'
                '<div class="jv-intel-val" style="font-size:0.84rem;font-weight:500;">'
                'Logistic Regression<br>'
                '<span style="font-size:0.72rem;color:var(--text3);font-weight:400;">+ TF-IDF · SMOTE resampling</span>'
                '</div></div>'
                '<div class="jv-intel-row" style="padding:12px 18px;">'
                '<span class="jv-intel-key">Training corpus</span>'
                '<span class="jv-intel-val"><span class="jv-accent-c">17,880</span> postings</span>'
                '</div>'
                f'<div class="jv-intel-row" style="padding:12px 18px;">'
                f'<span class="jv-intel-key">Threshold</span>'
                f'<div class="jv-intel-val"><span class="jv-accent-c">{threshold:.2f}</span>'
                f'<div style="margin-top:5px;height:3px;width:90px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">'
                f'<div style="height:100%;width:{thresh_pct}%;background:var(--cyan);box-shadow:0 0 8px rgba(6,182,212,0.6);border-radius:2px;"></div>'
                f'</div></div></div>'
                f'<div class="jv-intel-row" style="padding:12px 18px;">'
                f'<span class="jv-intel-key">Recall</span>'
                f'<div class="jv-intel-val"><span class="jv-accent-g">{recall_pct}%</span>'
                f'<div style="margin-top:5px;height:3px;width:90px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">'
                f'<div style="height:100%;width:{recall_pct}%;background:var(--green);box-shadow:0 0 8px rgba(16,185,129,0.6);border-radius:2px;"></div>'
                f'</div></div></div>'
                '</div>',
                unsafe_allow_html=True
            )
            # Capabilities — 2×2 grid
            st.markdown(
                '<div style="font-family:var(--mono);font-size:0.58rem;text-transform:uppercase;'
                'letter-spacing:0.12em;color:var(--text3);margin-bottom:10px;">Capabilities</div>'
                '<div style="display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-bottom:18px;">'
                + "".join(
                    '<div style="display:flex;align-items:center;gap:8px;padding:8px 11px;'
                    'background:rgba(16,185,129,0.04);border:1px solid rgba(16,185,129,0.12);'
                    'border-radius:10px;font-family:var(--mono);font-size:0.6rem;color:var(--text2);">'
                    '<span style="color:var(--green);font-size:0.72rem;">✓</span>'
                    f'<span>{cap}</span></div>'
                    for cap in ["Probability scoring", "SHAP attribution", "4 Eng. signals", "Recall-tuned"]
                )
                + '</div>',
                unsafe_allow_html=True
            )
            # System ready chip
            st.markdown(
                '<div style="padding:13px 16px;background:rgba(6,182,212,0.04);'
                'border:1px solid rgba(6,182,212,0.14);border-radius:var(--r-md);'
                'display:flex;align-items:center;gap:12px;">'
                '<div style="width:7px;height:7px;border-radius:50%;background:var(--cyan);'
                'box-shadow:0 0 10px rgba(6,182,212,0.9);'
                'animation:pillBlink 2s ease-in-out infinite;flex-shrink:0;"></div>'
                '<div><div style="font-family:var(--mono);font-size:0.62rem;text-transform:uppercase;'
                'letter-spacing:0.12em;color:rgba(6,182,212,0.75);">System Ready</div>'
                '<div style="font-family:var(--ui);font-size:0.78rem;color:var(--text3);margin-top:2px;">'
                'Fill in the form and click Analyse</div></div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ── FOOTER ──
    st.markdown(
        f'<div style="margin-top:3rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.05);'
        f'display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'
        f'font-family:var(--mono);font-size:0.54rem;color:var(--text3);letter-spacing:0.08em;">'
        f'<span>JOBVERIFY AI &nbsp;·&nbsp; EMSCAD DATASET &nbsp;·&nbsp; FAKE JOB DETECTION</span>'
        f'<span>THRESHOLD {threshold:.2f} &nbsp;·&nbsp; MAX RECALL &nbsp;·&nbsp; SHAP EXPLAINABILITY</span>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()