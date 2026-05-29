"""MCA Pulse: Premium Executive Intelligence Dashboard for Indian Market Analysis."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from urllib.parse import unquote_plus

from utils import (
    DATA_AS_OF,
    DEFAULT_DATA_PATH,
    VALID_STATES,
    calculate_lead_scores,
    export_excel,
    format_inr_crore,
    load_company_data,
)

st.set_page_config(
    page_title="MCA Pulse — Market Intelligence Platform",
    page_icon="📊",
    layout="wide",
)

_PREMIUM_STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=Crimson+Text:wght@600;700&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, [class*="css"] {
    font-family: 'Sora', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

body {
    background: #F8FAFC;
    color: #0F172A;
}

.stApp {
    background: #F8FAFC;
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #F8FAFC 0%, #F0F9FC 100%);
    padding: 60px 0;
    margin-bottom: 40px;
    border-bottom: 1px solid #E2E8F0;
}

.hero-title {
    font-family: 'Crimson Text', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.1;
    margin-bottom: 12px;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #64748B;
    font-weight: 500;
    line-height: 1.6;
    max-width: 600px;
}

/* Section Headers */
.section-wrapper {
    margin: 60px 0;
    padding: 0;
}

.section-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #22C1B5;
    font-weight: 700;
    margin-bottom: 8px;
    display: block;
}

.section-title {
    font-family: 'Crimson Text', serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 6px;
    line-height: 1.2;
}

.section-description {
    font-size: 1rem;
    color: #64748B;
    font-weight: 500;
    margin-bottom: 28px;
    line-height: 1.6;
}

/* Premium Cards */
.premium-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.premium-card:hover {
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    border-color: #CAF0F8;
}

/* KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    border-left: 3px solid #22C1B5;
}

.kpi-card:hover {
    box-shadow: 0 4px 16px rgba(34, 193, 181, 0.12);
    transform: translateY(-2px);
}

.kpi-label {
    font-size: 0.8rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
    font-weight: 600;
    margin-bottom: 8px;
    display: block;
}

.kpi-value {
    font-family: 'Sora', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 4px;
}

.kpi-note {
    font-size: 0.85rem;
    color: #64748B;
    font-weight: 500;
}

/* Divider */
.premium-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
    margin: 48px 0;
}

/* Chart Container */
.chart-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 28px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

.chart-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 4px;
}

.chart-subtitle {
    font-size: 0.9rem;
    color: #94A3B8;
    margin-bottom: 20px;
}

/* Insight Badge */
.insight-badge {
    background: #F0FDFA;
    border: 1px solid #CCFBF1;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.9rem;
    color: #0D9488;
    font-weight: 500;
    margin-top: 16px;
}

/* Grid System */
.grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
    margin-bottom: 24px;
}

.grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-bottom: 24px;
}

.grid-6 {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 16px;
    margin-bottom: 32px;
}

@media (max-width: 1200px) {
    .grid-2 { grid-template-columns: 1fr; }
    .grid-3 { grid-template-columns: repeat(2, 1fr); }
    .grid-6 { grid-template-columns: repeat(3, 1fr); }
}

/* Company Table */
.company-table {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    overflow: hidden;
}

/* Lead Insight Card */
.lead-insight {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 3px solid #22C1B5;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.lead-company {
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 4px;
}

.lead-meta {
    font-size: 0.85rem;
    color: #94A3B8;
}

/* Sidebar Enhancement */
.sidebar-section {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.sidebar-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #22C1B5;
    font-weight: 700;
    margin-bottom: 12px;
}

/* Metric Row */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 12px;
}

.metric-small {
    flex: 1;
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}

.metric-small-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: #94A3B8;
    font-weight: 600;
    margin-bottom: 4px;
}

.metric-small-value {
    font-size: 1.4rem;
    font-weight: 700;
    color: #22C1B5;
}

/* Color Scheme */
.text-teal { color: #22C1B5; }
.text-navy { color: #0F172A; }
.text-gray { color: #94A3B8; }
.bg-light-teal { background: #F0FDFA; }

</style>
"""

st.markdown(_PREMIUM_STYLE, unsafe_allow_html=True)

THEME_OPTIONS = [
    "Default",
    "Fleet Management",
    "IT Services & Software",
    "Manufacturing",
    "Healthcare",
    "Financial Services",
]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = load_company_data(DEFAULT_DATA_PATH)
    scored = calculate_lead_scores(data)
    scored["lead_category"] = scored["lead_category"].fillna("Low")
    for col in [
        "company_name",
        "industry",
        "roc_city",
        "state",
        "sub_industry",
        "lead_category",
    ]:
        if col in scored.columns:
            scored[col] = scored[col].fillna("Not specified")
    scored["authorized_capital_lakh"] = scored["authorized_capital_lakh"].fillna(0)
    scored["paid_up_capital_lakh"] = scored["paid_up_capital_lakh"].fillna(0)
    return scored


def filter_leads(
    frame: pd.DataFrame,
    selected_state: list[str],
    selected_industry: list[str],
    selected_sub: list[str],
    selected_quality: list[str],
    capital_range: tuple[int, int],
    score_range: tuple[int, int],
    search_text: str,
) -> pd.DataFrame:
    filtered = frame.copy()
    if selected_state:
        filtered = filtered[filtered["state"].isin(selected_state)]
    if selected_industry:
        filtered = filtered[filtered["industry"].isin(selected_industry)]
    if selected_sub:
        filtered = filtered[filtered["sub_industry"].isin(selected_sub)]
    if selected_quality:
        filtered = filtered[filtered["lead_category"].isin(selected_quality)]
    filtered = filtered[
        filtered["authorized_capital_lakh"].between(capital_range[0], capital_range[1])
    ]
    filtered = filtered[filtered["lead_score"].between(score_range[0], score_range[1])]
    if search_text:
        term = search_text.strip()
        mask = (
            filtered["company_name"].str.contains(term, case=False, na=False)
            | filtered["roc_city"].str.contains(term, case=False, na=False)
            | filtered["industry"].str.contains(term, case=False, na=False)
            | filtered["state"].str.contains(term, case=False, na=False)
        )
        filtered = filtered[mask]
    return filtered


def create_registration_chart(frame: pd.DataFrame) -> go.Figure:
    timeseries = (
        frame.groupby("registration_date", as_index=False)
        .agg(
            companies=("cin", "count"),
            high_leads=("lead_category", lambda x: (x == "High").sum()),
        )
        .sort_values("registration_date")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeseries["registration_date"],
            y=timeseries["companies"],
            fill="tozeroy",
            name="Registrations",
            line=dict(color="#22C1B5", width=3),
            fillcolor="rgba(34, 193, 181, 0.1)",
            hovertemplate="<b>%{x|%b %d}</b><br>Registrations: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeseries["registration_date"],
            y=timeseries["high_leads"],
            name="High Leads",
            line=dict(color="#0F172A", width=2, dash="dash"),
            hovertemplate="<b>%{x|%b %d}</b><br>High Leads: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(t=10, b=30, l=40, r=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#F0F0F0", zeroline=False),
        height=380,
    )
    return fig


def create_industry_rankings(frame: pd.DataFrame) -> go.Figure:
    industry_data = (
        frame.groupby("industry", as_index=False)
        .agg(companies=("cin", "count"), capital=("authorized_capital_rs", "sum"))
        .sort_values("companies", ascending=False)
        .head(8)
    )
    fig = px.bar(
        industry_data,
        y="industry",
        x="companies",
        orientation="h",
        text="companies",
        color_discrete_sequence=["#22C1B5"],
        height=340,
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="inside", marker_line_width=0
    )
    fig.update_layout(
        margin=dict(t=10, b=30, l=150, r=10),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def create_lead_quality_donut(frame: pd.DataFrame) -> go.Figure:
    quality = (
        frame["lead_category"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )
    fig = px.pie(
        values=quality.values,
        names=quality.index,
        hole=0.55,
        color_discrete_map={"High": "#22C1B5", "Medium": "#A7F3D0", "Low": "#E0F2FE"},
        height=340,
    )
    fig.update_traces(
        textinfo="label+percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="#FFFFFF",
        showlegend=True,
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0),
    )
    return fig


def create_city_map(frame: pd.DataFrame) -> go.Figure:
    city_data = (
        frame.groupby("roc_city")
        .agg(companies=("cin", "count"), capital=("authorized_capital_rs", "sum"))
        .reset_index()
    )
    top_cities = city_data.nlargest(10, "companies")
    fig = px.bar(
        top_cities,
        x="companies",
        y="roc_city",
        orientation="h",
        color="capital",
        color_continuous_scale=["#E0F2FE", "#22C1B5"],
        text="companies",
        height=380,
    )
    fig.update_traces(
        texttemplate="%{text:,}", textposition="inside", marker_line_width=0
    )
    fig.update_layout(
        margin=dict(t=10, b=30, l=120, r=60),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        coloraxis_colorbar=dict(title="Capital (₹)", tickformat="$,.0f"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    return fig


def parse_query_params() -> tuple[list[str], str]:
    try:
        params = st.query_params
    except AttributeError:
        params = {}
    industries = []
    if "industry" in params:
        industry_val = params["industry"]
        if isinstance(industry_val, list):
            industries = [unquote_plus(v) for v in industry_val if v]
        elif industry_val:
            industries = [unquote_plus(industry_val)]
    theme = ""
    if "theme" in params:
        theme_val = params["theme"]
        theme = (
            unquote_plus(theme_val)
            if isinstance(theme_val, str)
            else (unquote_plus(theme_val[0]) if theme_val else "")
        )
    return industries, theme


def main() -> None:
    data = load_data()

    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-label">Platform Settings</div>', unsafe_allow_html=True
        )
        query_industries, query_theme = parse_query_params()
        selected_theme = st.selectbox(
            "Dashboard Theme",
            THEME_OPTIONS,
            index=(
                THEME_OPTIONS.index(query_theme) if query_theme in THEME_OPTIONS else 0
            ),
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-label">Data Filters</div>', unsafe_allow_html=True
        )
        selected_state = st.multiselect("State", VALID_STATES, default=[])
        industry_options = sorted(data["industry"].unique())
        selected_industry = st.multiselect(
            "Industry",
            industry_options,
            default=[i for i in query_industries if i in industry_options],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        industry_pool = (
            data[data["industry"].isin(selected_industry)]
            if selected_industry
            else data
        )
        all_sub = (
            sorted(industry_pool["sub_industry"].dropna().unique())
            if "sub_industry" in data.columns
            else []
        )
        selected_sub = st.multiselect("Sub-industry", all_sub, default=[])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        selected_quality = st.multiselect(
            "Lead Category",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        min_cap = int(data["authorized_capital_lakh"].min())
        max_cap = int(data["authorized_capital_lakh"].quantile(0.98))
        selected_capital = st.slider(
            "Capital Range (₹ Lakh)",
            min_value=min_cap,
            max_value=max_cap,
            value=(min_cap, max_cap),
        )
        selected_score = st.slider("Lead Score", 0, 100, (0, 100))
        st.markdown("</div>", unsafe_allow_html=True)

    filtered = filter_leads(
        data,
        selected_state,
        selected_industry,
        selected_sub,
        selected_quality,
        selected_capital,
        selected_score,
        "",
    )

    # Hero Section
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">MCA Pulse</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Executive intelligence platform for Indian company registrations, market signals, and investment opportunities.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Overview KPIs
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    total = len(filtered)
    high_count = int((filtered["lead_category"] == "High").sum())
    industries_count = filtered["industry"].nunique()
    cities_count = filtered["roc_city"].nunique()
    total_cap = filtered["authorized_capital_rs"].sum()
    avg_score = filtered["lead_score"].mean()

    col1, col2, col3, col4, col5, col6 = st.columns(6, gap="small")
    with col1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Total Companies</div><div class="kpi-value">{total:,}</div><div class="kpi-note">In dataset</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">High Quality Leads</div><div class="kpi-value">{high_count:,}</div><div class="kpi-note">Priority targets</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Industry Coverage</div><div class="kpi-value">{industries_count}</div><div class="kpi-note">Sectors</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Active Cities</div><div class="kpi-value">{cities_count}</div><div class="kpi-note">Locations</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Capital Pool</div><div class="kpi-value">{format_inr_crore(total_cap)}</div><div class="kpi-note">Authorized</div></div>',
            unsafe_allow_html=True,
        )
    with col6:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-label">Avg Lead Score</div><div class="kpi-value">{avg_score:.0f}%</div><div class="kpi-note">Quality index</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # Registration Trends
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown(
        '<span class="section-label">Market Activity</span>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-title">Registration Momentum</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">Track the trajectory of company registrations and high-potential leads over time.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.plotly_chart(
        create_registration_chart(filtered),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # Industry & Cities
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
        st.markdown(
            '<span class="section-label">Sector Analysis</span>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-title">Industry Leaders</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-description">Discover which sectors dominate the registration landscape.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(
            create_industry_rankings(filtered),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
        st.markdown(
            '<span class="section-label">Geographic Insights</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">City Concentration</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-description">Identify the regional hubs of business activity and capital deployment.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(
            create_city_map(filtered),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # Lead Quality
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    with col1:
        st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
        st.markdown(
            '<span class="section-label">Quality Assessment</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">Lead Distribution</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-description">Understand the composition and strength of qualified opportunities.</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(
            create_lead_quality_donut(filtered),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
        st.markdown(
            '<span class="section-label">Priority Pipeline</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-title">Top Opportunities</div>', unsafe_allow_html=True
        )
        st.markdown(
            '<div class="section-description">The highest-scoring companies ranked by investment potential.</div>',
            unsafe_allow_html=True,
        )
        top_leads = filtered.nlargest(6, "lead_score")[
            ["company_name", "roc_city", "industry", "lead_score", "lead_category"]
        ]
        for _, row in top_leads.iterrows():
            st.markdown(
                f'<div class="lead-insight"><div class="lead-company">{row["company_name"]}</div><div class="lead-meta">{row["industry"]} • {row["roc_city"]} • Score: {row["lead_score"]:.0f}% ({row["lead_category"]})</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # Company Explorer
    st.markdown('<div class="section-wrapper">', unsafe_allow_html=True)
    st.markdown(
        '<span class="section-label">Data Export</span>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-title">Company Intelligence</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-description">Access complete company records and export for deeper analysis.</div>',
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "🔍 Search company, city, or industry...", key="explorer_search"
    )
    explorer = filter_leads(
        filtered,
        selected_state,
        selected_industry,
        selected_sub,
        selected_quality,
        selected_capital,
        selected_score,
        search,
    )

    st.markdown(
        f'<div class="insight-badge">📊 {len(explorer):,} companies · {explorer["lead_category"].value_counts().get("High", 0):,} high-quality leads</div>',
        unsafe_allow_html=True,
    )

    cols_to_display = [
        "company_name",
        "roc_city",
        "state",
        "industry",
        "authorized_capital_lakh",
        "lead_score",
        "lead_category",
    ]
    explorer_view = explorer[cols_to_display].rename(
        columns={
            "company_name": "Company",
            "roc_city": "City",
            "state": "State",
            "industry": "Industry",
            "authorized_capital_lakh": "Capital (₹ Lakh)",
            "lead_score": "Score",
            "lead_category": "Category",
        }
    )
    st.dataframe(explorer_view, use_container_width=True, height=400, hide_index=True)

    # Export
    st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
    summary_df = pd.DataFrame(
        {
            "Metric": [
                "Total Companies",
                "High Leads",
                "Industries",
                "Cities",
                "Total Capital",
            ],
            "Value": [
                total,
                high_count,
                industries_count,
                cities_count,
                format_inr_crore(total_cap),
            ],
        }
    )
    report = export_excel(explorer, summary_df)
    st.download_button(
        "📥 Download Intelligence Report",
        data=report,
        file_name="mca_pulse_intelligence.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
