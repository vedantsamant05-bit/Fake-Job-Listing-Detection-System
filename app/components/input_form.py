import streamlit as st

EMPLOYMENT_TYPES  = ["", "Full-time", "Part-time", "Contract", "Temporary", "Other"]
EXPERIENCE_LEVELS = ["", "Entry level", "Mid-Senior level", "Director",
                     "Executive", "Associate", "Not Applicable", "Internship"]
EDUCATION_LEVELS  = ["", "Bachelor's Degree", "Master's Degree",
                     "High School or equivalent", "Some College(s)",
                     "Certification", "Doctorate", "Associate Degree",
                     "Professional", "Unspecified"]

# Colour palette for section icons & borders
_SEC_COLOURS = {
    "cyan":   {"border": "#06B6D4", "text": "#06B6D4", "bg": "rgba(6,182,212,0.06)"},
    "purple": {"border": "#8B5CF6", "text": "#8B5CF6", "bg": "rgba(139,92,246,0.06)"},
    "green":  {"border": "#10B981", "text": "#10B981", "bg": "rgba(16,185,129,0.06)"},
}


def _section(num: str, icon: str, label: str, scheme: str = "cyan"):
    c = _SEC_COLOURS[scheme]
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;'
        f'margin:1.8rem 0 1.1rem;padding:10px 14px;'
        f'background:{c["bg"]};'
        f'border-left:3px solid {c["border"]};'
        f'border-radius:0 10px 10px 0;">'
        f'<span style="font-size:1rem;color:{c["text"]};">{icon}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:0.58rem;'
        f'color:rgba(255,255,255,0.28);letter-spacing:0.1em;">{num}</span>'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:1.05rem;'
        f'font-weight:700;color:rgba(255,255,255,0.88);letter-spacing:-0.01em;">{label}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


def _hint(text: str):
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.56rem;'
        f'color:rgba(255,255,255,0.20);letter-spacing:0.04em;'
        f'margin-top:-4px;margin-bottom:12px;">{text}</div>',
        unsafe_allow_html=True
    )


def _counter(n: int):
    col = "#06B6D4" if n > 60 else "rgba(255,255,255,0.20)"
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.55rem;'
        f'color:{col};text-align:right;margin-top:-4px;margin-bottom:10px;'
        f'letter-spacing:0.04em;">{n} chars</div>',
        unsafe_allow_html=True
    )


def render_input_form() -> dict | None:

    # Override Streamlit's default label style inside the form
    st.markdown("""<style>
.stTextArea textarea{resize:vertical !important;min-height:80px !important;}
.stSelectbox svg{color:rgba(255,255,255,0.25) !important;}
div[data-testid="stTextInputRootElement"] label,
div[data-testid="stTextAreaRootElement"] label,
div[data-testid="stSelectboxContainer"] label{
  font-family:'Space Grotesk',sans-serif !important;
  font-size:0.88rem !important;
  font-weight:600 !important;
  color:rgba(255,255,255,0.80) !important;
  text-transform:none !important;
  letter-spacing:0.01em !important;
}
</style>""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    #  SECTION 01 — Job Information
    # ═══════════════════════════════════════════════════════
    _section("01", "⬡", "Job Information", "cyan")

    title = st.text_input("Job Title", placeholder="e.g.  Senior Data Engineer")

    col_a, col_b = st.columns(2)
    with col_a:
        employment_type = st.selectbox("Employment Type", EMPLOYMENT_TYPES)
    with col_b:
        salary_range = st.text_input("Salary Range", placeholder="e.g.  80,000 – 120,000")

    col_c, col_d = st.columns(2)
    with col_c:
        required_experience = st.selectbox("Experience Level", EXPERIENCE_LEVELS)
    with col_d:
        required_education = st.selectbox("Education Required", EDUCATION_LEVELS)

    col_e, col_f = st.columns(2)
    with col_e:
        industry = st.text_input("Industry", placeholder="e.g.  Information Technology")
    with col_f:
        function = st.text_input("Job Function", placeholder="e.g.  Engineering")

    # ═══════════════════════════════════════════════════════
    #  SECTION 02 — Company & Content
    # ═══════════════════════════════════════════════════════
    _section("02", "◈", "Company & Content", "purple")
    _hint("Richer content → higher model confidence")

    company_profile = st.text_area(
        "Company Profile", height=90,
        placeholder="Company mission, culture, size, values…"
    )

    description = st.text_area(
        "Job Description", height=135,
        placeholder="Role responsibilities, team, day-to-day work…"
    )
    _counter(len(description))

    col_g, col_h = st.columns(2)
    with col_g:
        requirements = st.text_area(
            "Requirements", height=115,
            placeholder="Skills, qualifications, years of experience…"
        )
        _counter(len(requirements))
    with col_h:
        benefits = st.text_area(
            "Benefits", height=115,
            placeholder="Health, equity, PTO, remote policy…"
        )
        _counter(len(benefits))

    # ═══════════════════════════════════════════════════════
    #  SECTION 03 — Listing Metadata
    # ═══════════════════════════════════════════════════════
    _section("03", "⬟", "Listing Metadata", "green")
    _hint("Binary signals — strong fraud indicators when absent")

    col_i, col_j, col_k = st.columns(3)
    with col_i:
        telecommuting    = st.checkbox("Remote / Telecommuting")
    with col_j:
        has_company_logo = st.checkbox("Has Company Logo")
    with col_k:
        has_questions    = st.checkbox("Has Screening Questions")

    # ═══════════════════════════════════════════════════════
    #  COMPLETENESS METER
    # ═══════════════════════════════════════════════════════
    filled = sum([
        bool(title.strip()), bool(salary_range.strip()),
        bool(company_profile.strip()), bool(description.strip()),
        bool(requirements.strip()), bool(benefits.strip()),
        bool(industry.strip()), bool(function.strip()),
        bool(employment_type), bool(required_experience),
        bool(required_education),
    ])
    total = 11
    pct   = int(filled / total * 100)

    if pct >= 70:
        bar_col, bar_glow, status = "#06B6D4", "rgba(6,182,212,0.45)", "High confidence"
    elif pct >= 40:
        bar_col, bar_glow, status = "#F59E0B", "rgba(245,158,11,0.45)", "Moderate confidence"
    else:
        bar_col, bar_glow, status = "#EF4444", "rgba(239,68,68,0.45)", "Low confidence"

    advice = "Add more fields for higher accuracy" if pct < 70 else "Good coverage — model will return reliable score"

    st.markdown(
        f'<div style="margin-top:1.6rem;padding:16px 18px;'
        f'background:rgba(0,0,0,0.22);border:1px solid rgba(255,255,255,0.08);border-radius:14px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'font-family:\'JetBrains Mono\',monospace;font-size:0.57rem;letter-spacing:0.08em;margin-bottom:9px;">'
        f'<span style="color:rgba(255,255,255,0.28);">COMPLETENESS</span>'
        f'<span style="color:{bar_col};">{status} &nbsp;·&nbsp; {pct}%</span></div>'
        f'<div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">'
        f'<div style="height:100%;width:{pct}%;'
        f'background:linear-gradient(90deg,{bar_col}66,{bar_col});border-radius:2px;'
        f'box-shadow:0 0 10px {bar_glow};'
        f'transition:width 0.6s cubic-bezier(0.4,0,0.2,1);"></div></div>'
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.52rem;'
        f'color:rgba(255,255,255,0.18);margin-top:7px;letter-spacing:0.04em;">'
        f'{filled}/{total} fields &nbsp;·&nbsp; {advice}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ═══════════════════════════════════════════════════════
    #  VALIDATION
    # ═══════════════════════════════════════════════════════
    if not title.strip() and not description.strip():
        st.markdown(
            '<div style="margin-top:14px;padding:12px 16px;'
            'background:rgba(245,158,11,0.05);border:1px solid rgba(245,158,11,0.20);'
            'border-radius:11px;display:flex;align-items:center;gap:10px;'
            'font-family:\'JetBrains Mono\',monospace;font-size:0.62rem;'
            'color:rgba(245,158,11,0.85);letter-spacing:0.06em;">'
            '<span>⚠</span>'
            '<span>Provide at least a job title or description to run analysis</span>'
            '</div>',
            unsafe_allow_html=True
        )
        return None

    return {
        "title":               title,
        "salary_range":        salary_range,
        "company_profile":     company_profile,
        "description":         description,
        "requirements":        requirements,
        "benefits":            benefits,
        "telecommuting":       int(telecommuting),
        "has_company_logo":    int(has_company_logo),
        "has_questions":       int(has_questions),
        "employment_type":     employment_type     or "unknown",
        "required_experience": required_experience or "unknown",
        "required_education":  required_education  or "unknown",
        "industry":            industry            or "unknown",
        "function":            function            or "unknown",
        "location":            "",
        "department":          "",
    }