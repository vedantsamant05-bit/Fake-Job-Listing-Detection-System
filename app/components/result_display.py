import streamlit as st
import pandas as pd
import numpy as np
from src.feature_engineering import (
    salary_anomaly_score, vagueness_score,
    missing_field_ratio, req_comp_mismatch,
)

_VERDICT = {
    "High":   ("⚠  LIKELY FRAUDULENT",  "#EF4444", "rgba(239,68,68,0.07)",  "rgba(239,68,68,0.22)"),
    "Medium": ("◈  REVIEW RECOMMENDED", "#F59E0B", "rgba(245,158,11,0.07)", "rgba(245,158,11,0.20)"),
    "Low":    ("✓  APPEARS LEGITIMATE", "#10B981", "rgba(16,185,129,0.07)", "rgba(16,185,129,0.20)"),
}

def _sig_colour(val: float, invert: bool = False) -> str:
    if invert:
        return "#10B981" if val >= 0.7 else ("#F59E0B" if val >= 0.4 else "#EF4444")
    return "#EF4444" if val >= 0.6 else ("#F59E0B" if val >= 0.3 else "#10B981")


def render_result(result: dict, listing: dict):
    prob   = result["probability_fake"]
    risk   = result["risk_level"]
    pct    = prob * 100
    thresh = result["threshold_used"]

    label, accent, bg, border = _VERDICT[risk]

    # ── VERDICT BANNER ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:18px 22px;background:{bg};border:1px solid {border};
                border-radius:16px;margin-bottom:18px;text-align:center;
                position:relative;overflow:hidden;">
      <div style="position:absolute;top:-30px;right:-30px;width:100px;height:100px;
                  border-radius:50%;background:{accent};opacity:0.12;filter:blur(20px);"></div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
                  text-transform:uppercase;letter-spacing:0.18em;
                  color:{accent};opacity:0.75;margin-bottom:6px;">
        Fraud Risk Assessment
      </div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.15rem;
                  font-weight:700;color:{accent};letter-spacing:-0.02em;">
        {label}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ANIMATED SVG RING + FILL BAR ────────────────────────────────────────
    r         = 54
    stroke_w  = 9
    circ      = 2 * 3.14159265 * r
    dash      = circ * pct / 100
    gap       = circ - dash
    offset    = circ - dash        # stroke-dashoffset target for animation

    # Unique ID so multiple renders don't collide
    uid = f"ring_{int(pct*10)}"

    st.markdown(f"""
    <style>
    @keyframes ringDraw_{uid} {{
      from {{ stroke-dashoffset: {circ:.1f}; opacity: 0.3; }}
      to   {{ stroke-dashoffset: {offset:.1f}; opacity: 1; }}
    }}
    @keyframes numCount_{uid} {{
      from {{ opacity: 0; transform: translateY(6px) scale(0.9); }}
      to   {{ opacity: 1; transform: none; }}
    }}
    </style>

    <div style="display:flex;align-items:center;gap:24px;
                padding:20px 22px;background:rgba(0,0,0,0.2);
                border:1px solid rgba(255,255,255,0.07);border-radius:16px;
                margin-bottom:16px;">

      <!-- Ring -->
      <div style="position:relative;width:128px;height:128px;flex-shrink:0;">
        <svg width="128" height="128" viewBox="0 0 128 128"
             style="transform:rotate(-90deg);overflow:visible;">
          <!-- track -->
          <circle cx="64" cy="64" r="{r}"
                  fill="none" stroke="rgba(255,255,255,0.05)"
                  stroke-width="{stroke_w}"/>
          <!-- animated fill -->
          <circle cx="64" cy="64" r="{r}"
                  fill="none" stroke="{accent}"
                  stroke-width="{stroke_w}" stroke-linecap="round"
                  stroke-dasharray="{circ:.1f}"
                  stroke-dashoffset="{circ:.1f}"
                  style="filter:drop-shadow(0 0 8px {accent});
                         animation:ringDraw_{uid} 1.2s ease-out 0.1s forwards;"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;
                    align-items:center;justify-content:center;
                    animation:numCount_{uid} 0.5s ease 0.9s both;">
          <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;
                      font-weight:800;letter-spacing:-0.05em;color:{accent};
                      line-height:1;">{pct:.1f}</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:0.5rem;
                      letter-spacing:0.12em;color:rgba(255,255,255,0.22);
                      text-transform:uppercase;margin-top:2px;">%</div>
        </div>
      </div>

      <!-- Right side: label + fill bar -->
      <div style="flex:1;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.57rem;
                    text-transform:uppercase;letter-spacing:0.12em;
                    color:rgba(255,255,255,0.22);margin-bottom:10px;">
          Probability fake
        </div>
        <div style="height:5px;background:rgba(255,255,255,0.05);
                    border-radius:3px;overflow:hidden;margin-bottom:14px;">
          <div style="height:100%;width:{pct:.1f}%;
                      background:linear-gradient(90deg,{accent}66,{accent});
                      border-radius:3px;box-shadow:0 0 12px {accent}66;
                      animation:ringDraw_{uid} 1.2s ease-out 0.1s both;"></div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.58rem;
                    color:rgba(255,255,255,0.2);letter-spacing:0.07em;">
          Threshold <span style="color:rgba(255,255,255,0.4);">{thresh:.2f}</span>
          &nbsp;·&nbsp; Risk <span style="color:{accent};">{risk.upper()}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ENGINEERED SIGNALS ───────────────────────────────────────────────────
    st.markdown("""
    <div class="jv-divider" style="color:rgba(6,182,212,0.55);">
      Engineered Signals
    </div>
    """, unsafe_allow_html=True)

    row = pd.Series(listing)
    combined = " ".join(str(listing.get(k,"")) for k in
                        ["title","description","requirements","benefits"])

    sal  = salary_anomaly_score(str(listing.get("salary_range","")))
    vag  = vagueness_score(combined)
    miss = missing_field_ratio(row)
    mism = req_comp_mismatch(str(listing.get("requirements","")),
                             str(listing.get("salary_range","")))

    signals = [
        ("Salary Anomaly", f"{sal:.2f}", _sig_colour(sal, invert=True),
         "Higher = salary present & realistic"),
        ("Vagueness",      f"{vag:.2f}", _sig_colour(vag),
         "Higher = more vague/suspicious phrases"),
        ("Missing Fields", f"{miss:.2f}", _sig_colour(miss),
         "Fraction of important fields empty"),
        ("Req–Comp Gap",   f"{mism:.0f}", "#EF4444" if mism else "#10B981",
         "High skill req + no/low salary"),
    ]

    cols = st.columns(2)
    for i, (name, val, col, tip) in enumerate(signals):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="jv-sig" title="{tip}">
              <div class="jv-sig-name">{name}</div>
              <div class="jv-sig-val" style="color:{col};
                   text-shadow:0 0 10px {col}aa,0 0 24px {col}33;">{val}</div>
            </div>
            """, unsafe_allow_html=True)
