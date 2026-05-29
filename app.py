"""MCA Pulse: Indian Company Registrations Dashboard — May 2019."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from inference_network import (
    DEFAULT_MODEL_PATH,
    infer_lead_signals,
    load_model_artifact,
    predict_single,
)
from train_model import train_and_save_model
from utils import (
    CITY_COORDINATES,
    DATA_AS_OF,
    DEFAULT_DATA_PATH,
    VALID_STATES,
    calculate_lead_scores,
    export_excel,
    format_inr_crore,
    load_company_data,
    make_prediction_record,
)

st.set_page_config(
    page_title="MCA Pulse — Indian Company Dashboard",
    page_icon="📊",
    layout="wide",
)

_FONTS_URL = os.environ.get(
    "MCA_FONTS_URL",
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
)

st.markdown(
    f"""
<style>
@import url('{_FONTS_URL}');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.metric-card {{
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.5rem;
}}
.metric-label {{ color: #6b7280; font-size: 0.78rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }}
.metric-value {{ color: #111827; font-size: 1.9rem; font-weight: 700; margin: 0.3rem 0 0.1rem; }}
.metric-note {{ color: #9ca3af; font-size: 0.78rem; }}
.lead-card {{
    border-left: 4px solid #14b8a6;
    background: #f0fdfa;
    border-radius: 0 10px 10px 0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
}}
.lead-card.hot {{ border-left-color: #ef4444; background: #fef2f2; }}
.lead-card.warm {{ border-left-color: #f59e0b; background: #fffbeb; }}
.lead-card.cold {{ border-left-color: #6b7280; background: #f9fafb; }}
.lead-name {{ font-weight: 600; color: #111827; font-size: 0.92rem; }}
.lead-meta {{ color: #6b7280; font-size: 0.78rem; margin-top: 0.2rem; }}
.section-header {{
    border-top: 2px solid #e5e7eb;
    padding-top: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 0.5rem;
}}
.tip-box {{
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #1e40af;
    font-size: 0.83rem;
    margin-bottom: 1rem;
}}
</style>
""",
    unsafe_allow_html=True,
)


# ── Data & model loading ──────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def source_data() -> pd.DataFrame:
    df = calculate_lead_scores(load_company_data(DEFAULT_DATA_PATH))
    df["lead_category"] = df["lead_category"].replace(
        {"Hot": "High", "Warm": "Medium", "Cold": "Low"}
    )
    return df.drop(columns=["sub_industry_code"], errors="ignore")


@st.cache_data(show_spinner=False)
def sub_industry_map() -> pd.Series:
    raw = load_company_data(DEFAULT_DATA_PATH)
    return raw["sub_industry"].reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def model_artifact() -> dict:
    if not Path(DEFAULT_MODEL_PATH).exists():
        try:
            train_and_save_model()
        except Exception as e:
            st.error(f"Model training failed: {e}")
            st.stop()
    try:
        return load_model_artifact()
    except Exception as e:
        st.error(f"Model load failed: {e}")
        st.stop()


# ── Load data ─────────────────────────────────────────────────────────────────

with st.spinner("Loading company data..."):
    try:
        base = source_data()
    except Exception as e:
        st.error(f"Could not load data: {e}")
        st.stop()

artifact = model_artifact()
sub_codes = sub_industry_map()

try:
    scored = infer_lead_signals(base, artifact)
    scored["lead_category"] = scored["lead_category"].replace(
        {"Hot": "High", "Warm": "Medium", "Cold": "Low"}
    )
    scored["sub_industry"] = sub_codes.values
except Exception as e:
    st.error(f"Scoring failed: {e}")
    st.stop()

all_industries = sorted(scored["industry"].unique())
all_cities = sorted(scored["roc_city"].unique())
all_states = sorted(scored["state"].unique())


# ── Sidebar filters ───────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/48/combo-chart.png", width=40)
    st.title("MCA Pulse")
    st.caption("Indian Company Registrations · May 2019")
    st.divider()

    st.subheader("🔍 Filter Companies")

    selected_state = st.multiselect(
        "State",
        VALID_STATES,
        default=[],
        help="Pick one or more states to filter. Leave empty to show all states.",
    )
    selected_industry = st.multiselect(
        "Industry / Sector",
        all_industries,
        default=[],
        help="Pick one or more industries. Leave empty to show all.",
    )
    # sub-industries depend on selected industries
    industry_pool = (
        scored[scored["industry"].isin(selected_industry)]
        if selected_industry
        else scored
    )
    all_sub = sorted(industry_pool["sub_industry"].unique())
    selected_sub = st.multiselect(
        "Company Sub-category",
        all_sub,
        default=[],
        help="Filter by company sub-category within the selected industry. Leave empty to show all.",
    )
    selected_segments = st.multiselect(
        "Lead Quality",
        ["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
        help="High = best prospects, Medium = decent, Low = lower priority.",
    )

    st.divider()
    st.caption(
        "💡 **What is a lead?** A company that looks like a good business opportunity based on its size, location and registration details. Leave all filters empty to see everything."
    )


# ── Apply filters ─────────────────────────────────────────────────────────────

seg_map = {"High": "High", "Medium": "Medium", "Low": "Low"}
selected_seg_values = (
    selected_segments if selected_segments else ["High", "Medium", "Low"]
)

filtered = scored.copy()
if selected_industry:
    filtered = filtered[filtered["industry"].isin(selected_industry)]
if selected_sub:
    filtered = filtered[filtered["sub_industry"].isin(selected_sub)]
if selected_state:
    filtered = filtered[filtered["state"].isin(selected_state)]
filtered = filtered[filtered["lead_category"].isin(selected_seg_values)]

if filtered.empty:
    st.warning(
        "⚠️ No companies match your filters. Try selecting more options in the sidebar."
    )
    st.stop()


# ── Page header ───────────────────────────────────────────────────────────────

st.title("📊 Indian Company Registrations — May 2019")
st.markdown(
    f"Showing **{len(filtered):,} companies** registered in India during May 2019. "
    "Use the sidebar to filter by industry, state or lead quality."
)
st.markdown(
    '<div class="tip-box">💡 <b>How to use this dashboard:</b> Start by picking an industry or state in the sidebar. '
    "The charts and tables below will update automatically to show only matching companies.</div>",
    unsafe_allow_html=True,
)


# ── Summary numbers ───────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>📌 Quick Summary</h3></div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4, c5 = st.columns(5)


def metric_card(col, label, value, note):
    col.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


metric_card(c1, "Total Companies", f"{len(filtered):,}", "Registered in May 2019")
metric_card(c2, "States Covered", f"{filtered['state'].nunique():,}", "Unique states")
metric_card(
    c3,
    "Cities Covered",
    f"{filtered['roc_city'].nunique():,}",
    "Unique registration offices",
)
metric_card(
    c4, "Industries Covered", f"{filtered['industry'].nunique():,}", "Unique sectors"
)
metric_card(
    c5,
    "Total Capital",
    format_inr_crore(filtered["authorized_capital_rs"].sum()),
    "Authorized capital declared",
)


# ── Industries & Capital ──────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>🏭 Industries & Capital</h3></div>',
    unsafe_allow_html=True,
)
st.caption("Click an industry slice to drill into its sub-sectors and companies.")

industry_summary = (
    filtered.groupby("industry", as_index=False)
    .agg(
        companies=("cin", "count"),
        authorized_capital_rs=("authorized_capital_rs", "sum"),
    )
    .sort_values("companies", ascending=False)
    .reset_index(drop=True)
)
industry_summary["capital_cr"] = (
    industry_summary["authorized_capital_rs"] / 10_000_000
).round(1)
industry_summary["share"] = (
    industry_summary["companies"] / industry_summary["companies"].sum() * 100
).round(1)

PASTEL_COLORS = [
    "#a8d8ea",
    "#b8e0d2",
    "#d6eadf",
    "#ffd3b6",
    "#ffaaa5",
    "#c7ceea",
    "#ffeaa7",
    "#dfe6e9",
    "#b2bec3",
    "#81ecec",
    "#fab1a0",
    "#a29bfe",
    "#fd79a8",
    "#55efc4",
    "#74b9ff",
    "#e17055",
    "#6c5ce7",
    "#00cec9",
    "#fdcb6e",
    "#e84393",
]

# selected industry state
if "selected_ind" not in st.session_state:
    st.session_state.selected_ind = None

col_donut, col_panel = st.columns([0.45, 0.55])

with col_donut:
    sel = st.session_state.selected_ind
    center_label = sel if sel else "All Industries"
    center_count = (
        industry_summary[industry_summary["industry"] == sel]["companies"].values[0]
        if sel
        else filtered["industry"].count()
    )

    donut_fig = go.Figure(
        go.Pie(
            labels=industry_summary["industry"],
            values=industry_summary["companies"],
            hole=0.62,
            marker=dict(
                colors=PASTEL_COLORS[: len(industry_summary)],
                line=dict(color="#ffffff", width=2),
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>Companies: %{value:,}<br>Share: %{percent}<extra></extra>",
            pull=[
                0.04 if row["industry"] == sel else 0
                for _, row in industry_summary.iterrows()
            ],
        )
    )
    donut_fig.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=420,
        paper_bgcolor="#ffffff",
        showlegend=False,
        annotations=[
            dict(
                text=f"<b style='font-size:13px'>{center_label[:22]}</b><br>"
                f"<span style='font-size:22px;font-weight:700'>{center_count:,}</span><br>"
                f"<span style='font-size:11px;color:#9ca3af'>companies</span>",
                x=0.5,
                y=0.5,
                showarrow=False,
                align="center",
                font=dict(family="Inter", color="#111827"),
            )
        ],
    )
    st.plotly_chart(donut_fig, use_container_width=True)

    # click selector below chart
    ind_options = ["All Industries"] + industry_summary["industry"].tolist()
    chosen = st.selectbox(
        "Click to drill into an industry",
        ind_options,
        index=0 if not sel else ind_options.index(sel),
        label_visibility="collapsed",
    )
    st.session_state.selected_ind = None if chosen == "All Industries" else chosen

with col_panel:
    sel = st.session_state.selected_ind
    if sel:
        panel_df = (
            filtered[filtered["industry"] == sel]
            .groupby("sub_industry", as_index=False)
            .agg(
                companies=("cin", "count"), capital_rs=("authorized_capital_rs", "sum")
            )
            .sort_values("companies", ascending=False)
            .reset_index(drop=True)
        )
        header_title = f"SUB-SECTORS / {sel.upper()}"
        source_df = panel_df
        name_col = "sub_industry"
    else:
        source_df = industry_summary.copy()
        source_df = source_df.rename(columns={"authorized_capital_rs": "capital_rs"})
        header_title = "ALL INDUSTRIES / RANKED BY COMPANIES"
        name_col = "industry"

    max_co = source_df["companies"].max() if len(source_df) else 1
    total_co = filtered["cin"].count()

    rows_html = ""
    for idx, row in source_df.iterrows():
        bar_w = int(row["companies"] / max_co * 100)
        share_pct = round(row["companies"] / total_co * 100, 1)
        cap_cr = round(row["capital_rs"] / 10_000_000, 1)
        rank_num = str(idx + 1).zfill(2)
        # top 3 get accent color, rest get lighter blue
        bar_color = (
            "#059669"
            if idx == 0
            else ("#10b981" if idx == 1 else ("#34d399" if idx == 2 else "#a7f3d0"))
        )
        rows_html += f"""
<div class="si-row" style="display:flex;align-items:center;gap:12px;padding:10px 14px;
     border-radius:10px;margin-bottom:4px;transition:background .2s;
     cursor:default;" onmouseover="this.style.background='#f0fdf4'" onmouseout="this.style.background='transparent'">
  <div style="font-size:.68rem;font-weight:700;color:#cbd5e1;width:20px;flex-shrink:0;font-family:monospace">{rank_num}</div>
  <div style="flex:1;min-width:0">
    <div style="font-size:.82rem;font-weight:600;color:#1e293b;margin-bottom:5px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{row[name_col]}</div>
    <div style="background:#f1f5f9;border-radius:99px;height:5px;overflow:hidden">
      <div style="background:{bar_color};width:{bar_w}%;height:5px;border-radius:99px;
                  animation:growBar .6s ease forwards"></div>
    </div>
  </div>
  <div style="text-align:right;flex-shrink:0;min-width:64px">
    <div style="font-size:.9rem;font-weight:700;color:#0f172a;letter-spacing:-.02em">{int(row['companies']):,}</div>
    <div style="font-size:.65rem;color:#94a3b8;margin-top:1px">{share_pct}%</div>
  </div>
</div>"""

    st.markdown(
        f"""
<style>
@keyframes growBar {{
  from {{ width: 0%; }}
  to {{ width: var(--target-w); }}
}}
</style>
<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;
            box-shadow:0 1px 8px rgba(0,0,0,.06);padding:1.2rem 0.4rem 0.8rem;">
  <div style="padding:0 14px 10px;display:flex;align-items:center;justify-content:space-between;">
    <div>
      <div style="font-size:.62rem;font-weight:700;letter-spacing:.14em;color:#94a3b8;margin-bottom:2px">
        {header_title}
      </div>
      <div style="font-size:.78rem;color:#64748b">
        {len(source_df)} {'sub-sectors' if sel else 'industries'} &nbsp;·&nbsp;
        {total_co:,} companies total
      </div>
    </div>
    <div style="background:#ecfdf5;border-radius:8px;padding:4px 10px;
                font-size:.7rem;font-weight:600;color:#059669">
      {'↓ Drill-down' if sel else '← Select industry'}
    </div>
  </div>
  <div style="height:1px;background:#f1f5f9;margin:0 14px 8px"></div>
  <div style="max-height:380px;overflow-y:auto;padding:0 4px">
    {rows_html}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ── Cities ────────────────────────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>📍 Cities & Locations</h3></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Geographic distribution and market concentration of company registrations across India"
)

city_summary = (
    filtered.groupby("roc_city", as_index=False)
    .agg(
        companies=("cin", "count"),
        authorized_capital_rs=("authorized_capital_rs", "sum"),
        avg_lead=("lead_probability", "mean"),
        industries=("industry", "nunique"),
    )
    .sort_values("companies", ascending=False)
)
top_city = city_summary.iloc[0]
top_cap_city = city_summary.nlargest(1, "authorized_capital_rs").iloc[0]
best_lead_city = city_summary.nlargest(1, "avg_lead").iloc[0]
highest_opp = city_summary.nlargest(1, "industries").iloc[0]
total_co = filtered["cin"].count()

# ── TOP KPI STRIP ──
strip_cols = st.columns(4)
for col, (label, value, note) in zip(
    strip_cols,
    [
        (
            "CITIES COVERED",
            f"{city_summary['roc_city'].nunique():,}",
            "Unique ROC offices",
        ),
        (
            "TOP REGISTRATION HUB",
            top_city["roc_city"],
            f"{int(top_city['companies']):,} companies",
        ),
        (
            "CAPITAL LEADER",
            top_cap_city["roc_city"],
            format_inr_crore(top_cap_city["authorized_capital_rs"]),
        ),
        ("TOTAL COMPANIES", f"{total_co:,}", "In current filter"),
    ],
):
    col.markdown(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
        f"padding:.65rem 1rem;min-height:74px;box-shadow:0 1px 3px rgba(0,0,0,.04);"
        f'transition:box-shadow .18s,transform .18s;cursor:default"'
        f" onmouseover=\"this.style.boxShadow='0 4px 14px rgba(0,0,0,.09)';this.style.transform='translateY(-1px)'\""
        f" onmouseout=\"this.style.boxShadow='0 1px 3px rgba(0,0,0,.04)';this.style.transform='none'\">"
        f'<div style="font-size:.57rem;font-weight:700;color:#94a3b8;letter-spacing:.13em;margin-bottom:.28rem">{label}</div>'
        f'<div style="font-size:1.18rem;font-weight:700;color:#0f172a;line-height:1.15;margin-bottom:.18rem">{value}</div>'
        f'<div style="font-size:.65rem;color:#94a3b8">{note}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

# ── MAIN ROW: ranked list + map ──
col_left, col_right = st.columns([0.60, 0.40])

with col_left:
    st.markdown(
        '<div style="font-size:.68rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:.1em;margin-bottom:.75rem">Top Registration Markets</div>',
        unsafe_allow_html=True,
    )
    top8 = city_summary.head(8).reset_index(drop=True)
    max_co_city = top8["companies"].max()
    rows = ""
    for i, row in top8.iterrows():
        w = int(row["companies"] / max_co_city * 100)
        lightness = int(42 + (100 - w) * 0.10)
        bar_color = f"hsl(168,58%,{lightness}%)"
        rows += f"""
<div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid #f1f5f9">
  <div style="font-size:.64rem;font-weight:700;color:#94a3b8;width:20px;flex-shrink:0;font-family:monospace">{str(i+1).zfill(2)}</div>
  <div style="font-size:.82rem;font-weight:700;color:#0f172a;min-width:0;flex:0 0 98px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{row['roc_city']}</div>
  <div style="flex:1;background:#f8fafc;border-radius:99px;height:5px;overflow:hidden">
    <div style="background:{bar_color};width:{w}%;height:5px;border-radius:99px"></div>
  </div>
  <div style="font-size:.78rem;font-weight:700;color:#0f172a;flex-shrink:0;width:54px;text-align:right">{int(row['companies']):,}</div>
</div>"""
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:22px;padding:18px 18px;box-shadow:0 10px 24px rgba(15,23,42,.05);">{rows}</div>',
        unsafe_allow_html=True,
    )

with col_right:
    st.markdown(
        '<div style="font-size:.68rem;font-weight:700;color:#475569;text-transform:uppercase;'
        'letter-spacing:.1em;margin-bottom:.75rem">Live Registration Density</div>',
        unsafe_allow_html=True,
    )
    mapped = city_summary.head(12).copy()
    mapped["lat"] = mapped["roc_city"].map(
        lambda x: CITY_COORDINATES.get(x, (None, None))[0]
    )
    mapped["lon"] = mapped["roc_city"].map(
        lambda x: CITY_COORDINATES.get(x, (None, None))[1]
    )
    mapped = mapped.dropna(subset=["lat", "lon"])
    geo = px.scatter_geo(
        mapped,
        lat="lat",
        lon="lon",
        size="companies",
        color="avg_lead",
        hover_name="roc_city",
        color_continuous_scale=["#d9f5f3", "#22c1b5", "#0b7669"],
        size_max=28,
        opacity=0.72,
        hover_data={
            "companies": ":,",
            "avg_lead": ":.1f",
            "authorized_capital_rs": ":,.0f",
            "industries": True,
            "lat": False,
            "lon": False,
        },
        labels={
            "companies": "Companies",
            "avg_lead": "Lead Score",
            "authorized_capital_rs": "Capital (Rs)",
            "industries": "Industries",
        },
    )
    geo.update_traces(marker=dict(line=dict(width=1.5, color="#ffffff")))
    geo.update_geos(
        visible=False,
        scope="asia",
        lataxis_range=[6, 37],
        lonaxis_range=[67, 98],
        showland=True,
        landcolor="#f8fafc",
        oceancolor="#f8fafc",
        showcountries=True,
        countrycolor="#e2e8f0",
        showcoastlines=False,
        showframe=False,
    )
    geo.update_layout(
        coloraxis_showscale=False,
        margin=dict(t=0, b=0, l=0, r=0),
        height=360,
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(geo, use_container_width=True)

insight_cards = [
    (
        "📈",
        "Fastest Growth Market",
        top_city["roc_city"],
        f"{int(top_city['companies']):,} registrations",
    ),
    (
        "🎯",
        "Best Lead Density",
        best_lead_city["roc_city"],
        f"Score {best_lead_city['avg_lead']:.1f}%",
    ),
    (
        "🧭",
        "Highest Opportunity Region",
        highest_opp["roc_city"],
        f"{int(highest_opp['industries'])} industries",
    ),
    (
        "🚀",
        "Expansion Recommendation",
        top_cap_city["roc_city"],
        format_inr_crore(top_cap_city["authorized_capital_rs"]),
    ),
]

cards_html = ""
for icon, label, city, detail in insight_cards:
    cards_html += (
        f'<div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:22px;'
        f'padding:16px 18px;box-shadow:0 8px 18px rgba(15,23,42,.04);min-height:120px;display:flex;flex-direction:column;justify-content:space-between;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'<div style="width:34px;height:34px;border-radius:14px;background:#ecfdf5;display:flex;align-items:center;justify-content:center;font-size:1rem;">{icon}</div>'
        f'<div style="font-size:.68rem;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.14em;">{label}</div>'
        f"</div>"
        f'<div style="font-size:1rem;font-weight:800;color:#0f172a;line-height:1.2;margin-bottom:6px;">{city}</div>'
        f'<div style="font-size:.82rem;color:#64748b;">{detail}</div>'
        f"</div>"
    )

st.markdown(
    f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:18px;">{cards_html}</div>',
    unsafe_allow_html=True,
)


# ── Lead Quality ────────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)

lq_head, lq_desc = st.columns([0.55, 0.45])
with lq_head:
    st.markdown(
        '<div style="font-size:1.65rem;font-weight:700;color:#0f172a;line-height:1.2;margin-bottom:.5rem">'
        "<h3>🌍 What high-value companies are actually building.</h3></div>",
        unsafe_allow_html=True,
    )
with lq_desc:
    st.markdown(
        '<div style="font-size:.78rem;color:#64748b;line-height:1.7;padding-top:.3rem">'
        "Lead themes extracted from company registrations, industry classification, "
        "lead scores and capital strength. Interactive themes help identify opportunity clusters.</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:.9rem'></div>", unsafe_allow_html=True)

# ── lead cluster pills ──
cluster_df = (
    filtered.groupby("industry", as_index=False)
    .agg(companies=("cin", "count"), avg_score=("lead_probability", "mean"))
    .sort_values("companies", ascending=False)
)
max_cluster = cluster_df["companies"].max()

pills_html = ""
for _, row in cluster_df.iterrows():
    ratio = row["companies"] / max_cluster
    if ratio >= 0.5:
        bg, border, text, size = "#0f766e", "#0d9488", "#ffffff", ".82rem"
    elif ratio >= 0.2:
        bg, border, text, size = "#f0fdf4", "#6ee7b7", "#065f46", ".78rem"
    else:
        bg, border, text, size = "#f8fafc", "#e2e8f0", "#475569", ".74rem"
    pills_html += (
        f'<div style="display:inline-flex;align-items:center;gap:.4rem;'
        f"background:{bg};border:1px solid {border};border-radius:99px;"
        f'padding:.28rem .75rem;margin:.22rem;cursor:default">'
        f'<span style="font-size:{size};font-weight:600;color:{text}">{row["industry"]}</span>'
        f'<span style="font-size:.65rem;font-weight:700;color:{text};opacity:.7;margin-left:.15rem">{int(row["companies"]):,}</span>'
        f"</div>"
    )

st.markdown(
    f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:14px;'
    f'padding:1.1rem 1.2rem;box-shadow:0 1px 4px rgba(0,0,0,.04);line-height:2">'
    f'<div style="font-size:.57rem;font-weight:700;letter-spacing:.13em;color:#94a3b8;margin-bottom:.7rem">LEAD CLUSTERS — SIZE REFLECTS VOLUME</div>'
    f"{pills_html}</div>",
    unsafe_allow_html=True,
)

st.markdown("<div style='margin-top:.8rem'></div>", unsafe_allow_html=True)

# ── top lead companies table ──


# ── Registration Trends (Premium Timeline) ─────────────────────────────────────


# ── Main Timeline Card Container ──
st.markdown(
    """
<style>
.trend-hero {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 24px;
    width: 100%;
    margin-top: 10px;
}

.trend-hero-left {
    flex: 1 1 0;
    min-width: 0;
}

.trend-hero-left {
    flex: 1 1 0;
    min-width: 0;
}

.trend-hero-title {
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0;
    color: #0F172A;
    line-height: 1.15;
}

.trend-hero-copy {
    font-size: 0.92rem;
    color: #64748B;
    margin: 8px 0 0;
    line-height: 1.6;
    max-width: 560px;
}

.trend-hero-note {
    margin-left: auto;
    width: min(450px, 100%);
    min-width: 380px;
    max-width: 450px;

    font-size: 0.95rem;
    color: #64748B;
    line-height: 1.55;
    text-align: right;
    margin: 0;
    padding-top: 4px;
}

.trend-divider {
    border: 0;
    height: 1px;
    background: #E2E8F0;
    margin: 0 0 12px;
}
.trend-chart-title {
    font-size: 1rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0;
}
.trend-chart-copy {
    font-size: 0.82rem;
    color: #64748B;
    margin: 6px 0 0;
}
.trend-toggle {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    flex-wrap: nowrap;
}
.trend-toggle button {
    appearance: none;
    border: 1px solid #E2E8F0;
    background: #ffffff;
    color: #64748B;
    border-radius: 999px;
    padding: 8px 14px;
    font-size: 0.82rem;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
    min-width: 120px;
}
.trend-toggle button:hover {
    background: #effdfa;
    border-color: #c6f3e9;
}
.trend-toggle button.active {
    background: #22C1B5;
    border-color: #22C1B5;
    color: #ffffff;
}
.trend-kpis {
    display: flex;
    gap: 24px;
    justify-content: flex-end;
    align-items: center;
    flex-wrap: nowrap;
    white-space: nowrap;
}
.trend-kpi {
    display: flex;
    flex-direction: column;
    gap: 3px;
    padding: 3px 0;
}
.trend-kpi-label {
    font-size: 0.625rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #94A3B8;
    margin: 0;
}
.trend-kpi-value {
    font-size: 1.15rem;
    font-weight: 800;
    color: #0F172A;
    margin: 0;
}
.trend-chart-panel {
    border: 1px solid #E2E8F0;
    border-radius: 20px;
    overflow: hidden;
    background: #ffffff;
    padding: 10px;
}
.trend-insights {
    display: grid;
    grid-template-columns: repeat(4, minmax(160px, 1fr));
    gap: 1px;
    background: #E2E8F0;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 18px;
}
.trend-insight {
    background: #ffffff;
    padding: 16px 18px;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.trend-insight-label {
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 8px;
}
.trend-insight-value {
    font-size: 1.45rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 4px;
    line-height: 1.1;
}
.trend-insight-note {
    font-size: 0.82rem;
    color: #64748B;
    line-height: 1.5;
}
</style>

""",
    unsafe_allow_html=True,
)

# ── Compute peak data ──
daily_regs = filtered.groupby("registration_date", as_index=False).agg(
    companies=("cin", "count"),
    high_leads=("lead_category", lambda x: (x == "High").sum()),
)
peak_date = daily_regs.loc[daily_regs["companies"].idxmax(), "registration_date"]
peak_count = int(daily_regs["companies"].max())
total_companies = len(filtered)
total_high_leads = int((filtered["lead_category"] == "High").sum())
unique_industries = int(filtered["industry"].nunique())
active_rate = int(
    (filtered[filtered["lead_category"].isin(["High", "Medium"])].shape[0])
    / max(len(filtered), 1)
    * 100
)

# ── Registration Trends ──
trend_options = ["Total Registrations", "High Leads", "Medium Leads"]
if "trend_metric" not in st.session_state:
    st.session_state.trend_metric = "Total Registrations"

st.markdown(
    """
<div class="trend-hero">
  <div class="trend-hero-left">
    <h3 class="trend-hero-title">📈 Registration Trends</h3>
    
  </div>
  <div class="trend-hero-note">Peak registration activity reflects emerging business momentum and lead generation opportunities.</div>
</div>
<hr class="trend-divider" />
""",
    unsafe_allow_html=True,
)

row_left, row_center, row_right = st.columns([1.5, 1, 1])
with row_left:
    st.markdown(
        """
<div>
  <div class="trend-chart-title">Companies registered by timeline</div>
  <div class="trend-chart-copy">Daily registrations vs lead activity</div>
</div>
""",
        unsafe_allow_html=True,
    )
with row_center:
    trend_metric = st.radio(
        "",
        options=trend_options,
        index=trend_options.index(st.session_state.trend_metric),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.trend_metric = trend_metric
with row_right:
    st.markdown(
        f"""
<div class="trend-kpis">
  <div class="trend-kpi">
    <div class="trend-kpi-label">TOTAL</div>
    <div class="trend-kpi-value">{total_companies:,}</div>
  </div>
  <div class="trend-kpi">
    <div class="trend-kpi-label">HIGH LEADS</div>
    <div class="trend-kpi-value">{total_high_leads:,}</div>
  </div>
  <div class="trend-kpi">
    <div class="trend-kpi-label">ACTIVE</div>
    <div class="trend-kpi-value">{active_rate}%</div>
  </div>
  <div class="trend-kpi">
    <div class="trend-kpi-label">PEAK</div>
    <div class="trend-kpi-value">{peak_date.strftime('%b %d')}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

# ── Map trend_metric to data
if trend_metric == "High Leads":
    chart_data = daily_regs.copy()
    chart_data["value"] = chart_data["high_leads"]
    y_label = "High Lead Companies"
    chart_title = "High Lead Companies"
elif trend_metric == "Medium Leads":
    chart_data = filtered.groupby("registration_date", as_index=False).agg(
        value=("lead_category", lambda x: (x == "Medium").sum()),
    )
    y_label = "Medium Lead Companies"
    chart_title = "Medium Lead Companies"
else:
    chart_data = daily_regs.copy()
    chart_data["value"] = chart_data["companies"]
    y_label = "Companies Registered"
    chart_title = "Total Companies Registered"

# ── Main smooth area chart ──
chart_data = chart_data.sort_values("registration_date")
area_chart = px.area(
    chart_data,
    x="registration_date",
    y="value",
    labels={"registration_date": "Date", "value": y_label},
    line_shape="spline",
    color_discrete_sequence=["#22c1b5"],
)

area_chart.update_traces(
    fill="tozeroy",
    fillcolor="rgba(34, 193, 181, 0.16)",
    line=dict(color="#22c1b5", width=3),
    hovertemplate="<b>%{x|%b %d}</b><br>%{y:,.0f} "
    + y_label.split()[-1]
    + "s<extra></extra>",
)

area_chart.update_layout(
    height=360,
    margin=dict(t=10, b=24, l=34, r=20),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(family="Inter", color="#64748b", size=12),
    hovermode="x unified",
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(226, 232, 240, 0.38)",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#94a3b8"),
        tickformat="%b %d",
        ticks="outside",
        ticklen=4,
        tickcolor="#E2E8F0",
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor="rgba(226, 232, 240, 0.38)",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11, color="#94a3b8"),
    ),
    legend=dict(visible=False),
)

st.markdown('<div class="trend-chart-panel">', unsafe_allow_html=True)
st.plotly_chart(area_chart, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── Capital scatter ───────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>💰 Capital Analysis</h3></div>',
    unsafe_allow_html=True,
)
st.caption(
    "How much capital did companies declare, and how does it relate to lead quality?"
)

col_j, col_k = st.columns(2)

with col_j:
    st.markdown(
        "**Authorized vs Paid-up Capital** — each dot is a company, colored by lead quality"
    )
    sample = filtered.nlargest(min(2500, len(filtered)), "authorized_capital_rs").copy()
    sample["auth_plot"] = sample["authorized_capital_lakh"].clip(lower=0.01)
    sample["paid_plot"] = sample["paid_up_capital_lakh"].clip(lower=0.01)
    scatter = px.scatter(
        sample,
        x="auth_plot",
        y="paid_plot",
        color="lead_category",
        size="business_opportunity_score",
        hover_name="company_name",
        log_x=True,
        log_y=True,
        color_discrete_map={"High": "#16a34a", "Medium": "#d97706", "Low": "#9ca3af"},
        labels={
            "auth_plot": "Authorized Capital (Rs lakh)",
            "paid_plot": "Paid-up Capital (Rs lakh)",
        },
    )
    scatter.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=340)
    st.plotly_chart(scatter, use_container_width=True)

with col_k:
    st.markdown(
        "**How much of the authorized capital was actually paid up?** (by lead group)"
    )
    ratios = filtered.copy()
    ratios["conversion_pct"] = (ratios["capital_conversion"] * 100).clip(0, 120)
    box = px.box(
        ratios,
        x="lead_category",
        y="conversion_pct",
        color="lead_category",
        category_orders={"lead_category": ["High", "Medium", "Low"]},
        color_discrete_map={"High": "#16a34a", "Medium": "#d97706", "Low": "#9ca3af"},
        labels={"lead_category": "Lead Group", "conversion_pct": "Capital Paid Up (%)"},
    )
    box.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=340)
    st.plotly_chart(box, use_container_width=True)


# ── Lead Score Calculator ─────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>🤖 Check Any Company\'s Lead Score</h3></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Enter details about a company and our AI model will estimate how good a lead it is. "
    "This uses the same model trained on all 11,281 May 2019 registrations."
)

with st.form("lead_score_form"):
    st.markdown("**Fill in the company details:**")
    fa, fb = st.columns(2)
    with fa:
        auth_lakh = st.number_input(
            "Authorized Capital (Rs Lakh)",
            min_value=0.0,
            value=10.0,
            step=1.0,
            help="Total capital the company is allowed to raise.",
        )
        lead_class = st.selectbox(
            "Company Type",
            sorted(scored["company_class"].unique()),
            help="e.g. Private Limited, Public Limited, One Person Company.",
        )
        lead_city = st.selectbox("City (ROC Office)", all_cities)
        lead_industry = st.selectbox("Industry / Sector", all_industries)
    with fb:
        paid_lakh = st.number_input(
            "Paid-up Capital (Rs Lakh)",
            min_value=0.0,
            value=5.0,
            step=1.0,
            help="Capital actually paid in by shareholders so far.",
        )
        lead_listed = st.selectbox("Listed on Stock Exchange?", ["Unlisted", "Listed"])
        lead_state = st.selectbox("State", VALID_STATES)
        lead_date = st.date_input(
            "Registration Date",
            value=DATA_AS_OF.date(),
            min_value=pd.Timestamp("2019-05-01").date(),
            max_value=DATA_AS_OF.date(),
        )
    submitted = st.form_submit_button(
        "🔍 Calculate Lead Score", use_container_width=True
    )

if submitted:
    try:
        record = make_prediction_record(
            auth_lakh,
            paid_lakh,
            lead_class,
            lead_listed,
            lead_date,
            lead_city,
            lead_state,
            lead_industry,
        )
        result = predict_single(record, artifact)
        st.success("✅ Lead score calculated!")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(
            "Lead Probability",
            f"{result['lead_probability']:.1f}%",
            help="How likely this is a good lead.",
        )
        r2.metric(
            "Lead Strength",
            result["lead_strength"],
            help="Emerging / Promising / Priority.",
        )
        r3.metric(
            "Opportunity Score",
            f"{result['business_opportunity_score']:.1f} / 100",
            help="Overall business opportunity rating.",
        )
        r4.metric(
            "Growth Potential",
            result["growth_potential"],
            help="Developing / Moderate / High.",
        )
    except Exception as e:
        st.error(f"Could not calculate score: {e}")


# ── Company Search & Table ────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>🔎 Search & Browse Companies</h3></div>',
    unsafe_allow_html=True,
)
st.caption(
    "Search by company name, CIN number, city or industry. Results are sorted by opportunity score."
)

query = st.text_input(
    "Search companies", placeholder="e.g. Mumbai, Technology, Private Limited..."
)
explorer = filtered.copy()
if query:
    token = query.strip()
    explorer = explorer[
        explorer["company_name"].str.contains(token, case=False, na=False)
        | explorer["cin"].str.contains(token, case=False, na=False)
        | explorer["roc_city"].str.contains(token, case=False, na=False)
        | explorer["industry"].str.contains(token, case=False, na=False)
    ]

st.caption(f"Showing {len(explorer):,} companies")
view_cols = [
    "company_name",
    "roc_city",
    "state",
    "industry",
    "authorized_capital_lakh",
    "paid_up_capital_lakh",
    "lead_category",
    "lead_probability",
    "business_opportunity_score",
    "priority_rank",
]
st.dataframe(
    explorer[view_cols].sort_values("business_opportunity_score", ascending=False),
    use_container_width=True,
    hide_index=True,
    height=380,
    column_config={
        "company_name": "Company Name",
        "roc_city": "City",
        "state": "State",
        "industry": "Industry",
        "authorized_capital_lakh": st.column_config.NumberColumn(
            "Auth. Capital (Lakh)", format="%.1f"
        ),
        "paid_up_capital_lakh": st.column_config.NumberColumn(
            "Paid-up Capital (Lakh)", format="%.1f"
        ),
        "lead_category": "Lead Quality",
        "lead_probability": st.column_config.ProgressColumn(
            "Lead Score %", min_value=0, max_value=100, format="%.1f%%"
        ),
        "business_opportunity_score": st.column_config.NumberColumn(
            "Opportunity (0–100)", format="%.1f"
        ),
        "priority_rank": "Rank",
    },
)


# ── Downloads ─────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="section-header"><h3>⬇️ Download Your Data</h3></div>',
    unsafe_allow_html=True,
)
st.caption("Download the filtered results in your preferred format.")

summary_export = pd.DataFrame(
    [
        {"Metric": "Total Companies", "Value": len(filtered)},
        {"Metric": "Cities", "Value": filtered["roc_city"].nunique()},
        {"Metric": "Industries", "Value": filtered["industry"].nunique()},
        {
            "Metric": "Authorized Capital (Rs)",
            "Value": filtered["authorized_capital_rs"].sum(),
        },
        {
            "Metric": "Paid-up Capital (Rs)",
            "Value": filtered["paid_up_capital_rs"].sum(),
        },
        {
            "Metric": "High Leads",
            "Value": int((filtered["lead_category"] == "High").sum()),
        },
        {
            "Metric": "Avg Lead Score (%)",
            "Value": round(filtered["lead_probability"].mean(), 1),
        },
    ]
)

try:
    csv_data = explorer[view_cols].to_csv(index=False).encode("utf-8")
except Exception as e:
    st.error(f"CSV export failed: {e}")
    csv_data = b""

try:
    xlsx_data = export_excel(filtered, summary_export)
except Exception as e:
    st.error(f"Excel export failed: {e}")
    xlsx_data = b""

try:
    lead_data = (
        filtered.nlargest(250, "business_opportunity_score")[
            [
                "company_name",
                "cin",
                "roc_city",
                "state",
                "industry",
                "lead_category",
                "lead_probability",
                "business_opportunity_score",
                "priority_rank",
            ]
        ]
        .to_csv(index=False)
        .encode("utf-8")
    )
except Exception as e:
    st.error(f"Lead report export failed: {e}")
    lead_data = b""

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button(
        "📄 Download as CSV",
        csv_data,
        "companies.csv",
        "text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "📊 Download as Excel",
        xlsx_data,
        "companies.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with dl3:
    st.download_button(
        "🎯 Download Top 250 Leads",
        lead_data,
        "top_leads.csv",
        "text/csv",
        use_container_width=True,
    )

st.divider()
st.caption(
    "Source: Ministry of Corporate Affairs (MCA) · Indian company registrations · May 2019 · "
    "Lead scores are AI estimates based on capital, company type and location — not guaranteed outcomes."
)
