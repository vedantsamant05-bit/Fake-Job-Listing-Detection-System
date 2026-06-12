import streamlit as st
import numpy as np
import pandas as pd
import shap
import plotly.graph_objects as go

from src.data_loader import basic_clean
try:
    from src.config import ENGINEERED_FEATURES
except Exception:
    ENGINEERED_FEATURES = []

try:
    from src.preprocessing import BINARY_COLUMNS
except Exception:
    BINARY_COLUMNS = []


def render_shap_explanation(pipeline, listing: dict, top_n: int = 14):

    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;
                font-family:'JetBrains Mono',monospace;font-size:0.57rem;
                text-transform:uppercase;letter-spacing:0.16em;
                color:rgba(139,92,246,0.65);margin:16px 0 12px;">
      SHAP · Feature Attribution
      <span style="flex:1;height:1px;background:linear-gradient(90deg,rgba(139,92,246,0.18),transparent);"></span>
    </div>
    """, unsafe_allow_html=True)

    try:
        df = pd.DataFrame([listing])
        df = basic_clean(df)
        df = df.drop(columns=["fraudulent", "job_id"], errors="ignore")

        pre = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        X_t = pre.transform(df)

        try:
            exp = shap.TreeExplainer(clf)
            sv  = exp.shap_values(X_t)
            if isinstance(sv, list): sv = sv[1]
        except Exception:
            exp = shap.LinearExplainer(clf, X_t)
            sv  = exp.shap_values(X_t)

        row     = np.asarray(sv[0]).flatten()
        abs_vals= np.abs(row)
        top_idx = np.argsort(abs_vals)[::-1][:top_n]

        try:
            tfidf_names = pre.named_transformers_["text"] \
                .named_steps["tfidf"].get_feature_names_out().tolist()
        except Exception:
            tfidf_names = []
        try:
            cat_names = pre.named_transformers_["cat"] \
                .named_steps["ohe"].get_feature_names_out().tolist()
        except Exception:
            cat_names = []

        feat_names = tfidf_names + cat_names + list(BINARY_COLUMNS) + list(ENGINEERED_FEATURES)
        if len(feat_names) < X_t.shape[1]:
            feat_names += [f"feat_{i}" for i in range(len(feat_names), X_t.shape[1])]

        names = [feat_names[i] if i < len(feat_names) else f"feat_{i}" for i in top_idx]
        vals  = np.asarray(row[top_idx]).flatten()

        FAKE_COL  = "#EF4444"
        REAL_COL  = "#06B6D4"
        BG        = "rgba(0,0,0,0)"

        def hex_to_rgba(hex_val, alpha):
            hex_val = hex_val.lstrip('#')
            r = int(hex_val[0:2], 16)
            g = int(hex_val[2:4], 16)
            b = int(hex_val[4:6], 16)
            return f"rgba({r}, {g}, {b}, {alpha})"

        # Build two bar traces: glow (wide, low opacity) + solid (narrow, full opacity)
        colours = [FAKE_COL if v > 0 else REAL_COL for v in vals]
        hover_texts = [
            f"<b>{n}</b><br>SHAP: {v:+.4f}<br>{'↑ pushes FAKE' if v>0 else '↓ pushes REAL'}"
            for n, v in zip(names[::-1], vals[::-1])
        ]

        fig = go.Figure()

        # Glow bars (wide, translucent)
        fig.add_trace(go.Bar(
            x=vals[::-1],
            y=[n[:40] for n in names[::-1]],
            orientation="h",
            marker=dict(
                color=[hex_to_rgba(c, 0.15) for c in colours[::-1]],
                line=dict(width=0),
            ),
            width=0.72,
            showlegend=False,
            hoverinfo="skip",
        ))

        # Solid bars
        fig.add_trace(go.Bar(
            x=vals[::-1],
            y=[n[:40] for n in names[::-1]],
            orientation="h",
            marker=dict(
                color=colours[::-1],
                line=dict(width=0),
                opacity=0.9,
            ),
            width=0.44,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        # Zero line
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.12)", line_width=0.8)

        fig.update_layout(
            paper_bgcolor=BG,
            plot_bgcolor="rgba(10,10,16,0.0)",
            font=dict(family="'JetBrains Mono', monospace", color="rgba(255,255,255,0.35)", size=9),
            height=max(220, top_n * 30 + 60),
            margin=dict(t=8, b=30, l=0, r=8),
            barmode="overlay",
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                tickfont=dict(family="'JetBrains Mono', monospace", size=8, color="rgba(255,255,255,0.22)"),
                title=dict(
                    text="← pushes REAL      |      pushes FAKE →",
                    font=dict(family="'JetBrains Mono', monospace", size=7.5, color="rgba(255,255,255,0.2)"),
                ),
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(family="'JetBrains Mono', monospace", size=8.5, color="rgba(255,255,255,0.38)"),
                ticklen=0,
            ),
            hoverlabel=dict(
                bgcolor="#0C0C14",
                bordercolor="rgba(255,255,255,0.1)",
                font=dict(family="'JetBrains Mono', monospace", size=10, color="rgba(255,255,255,0.8)"),
            ),
        )

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Legend
        st.markdown("""
        <div style="display:flex;gap:16px;justify-content:flex-end;
                    font-family:'JetBrains Mono',monospace;font-size:0.54rem;
                    color:rgba(255,255,255,0.3);letter-spacing:0.06em;margin-top:-6px;">
          <span><span style="color:#EF4444;">■</span> pushes FAKE</span>
          <span><span style="color:#06B6D4;">■</span> pushes REAL</span>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;'
            f'color:rgba(255,255,255,0.15);padding:16px;'
            f'border:1px dashed rgba(255,255,255,0.06);border-radius:10px;">'
            f'SHAP unavailable: {e}</div>',
            unsafe_allow_html=True
        )