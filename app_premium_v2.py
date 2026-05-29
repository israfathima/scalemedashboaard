"""MCA Lead Intelligence Platform: premium investor-grade business dashboard."""

from __future__ import annotations

import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils import (
    CITY_COORDINATES,
    DATA_AS_OF,
    DEFAULT_DATA_PATH,
    VALID_STATES,
    calculate_lead_scores,
    export_excel,
    format_inr_crore,
    load_company_data,
)

st.set_page_config(
    page_title="MCA Lead Intelligence Platform",
    page_icon="📈",
    layout="wide",
)

_STYLE = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

body {
    background: #F8FAFC;
}

.stApp {
    background: #F8FAFC;
}

.hero-block {
    padding: 56px 0 16px;
}

.hero-title {
    font-family: 'Crimson Text', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.02;
    margin-bottom: 14px;
}

.hero-text {
    font-size: 1.05rem;
    color: #64748B;
    max-width: 720px;
    line-height: 1.8;
}

.section-header {
    margin-top: 80px;
    margin-bottom: 20px;
}

.section-label {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #22C1B5;
    margin-bottom: 10px;
}

.section-title {
    font-family: 'Crimson Text', serif;
    font-size: 2.3rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 10px;
    line-height: 1.08;
}

.section-copy {
    color: #64748B;
    font-size: 1rem;
    line-height: 1.75;
    max-width: 760px;
}

.premium-card,
.chart-card,
.table-card,
.filter-card,
.lead-card,
.final-card,
.kpi-card,
.mini-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 24px;
    box-shadow: 0 14px 30px rgba(15, 23, 42, 0.06);
}

.premium-card,
.chart-card,
.table-card,
.filter-card,
.lead-card,
.final-card {
    padding: 26px;
}

.kpi-card {
    padding: 24px;
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}

.kpi-card:hover,
.premium-card:hover,
.chart-card:hover,
.table-card:hover,
.lead-card:hover,
.final-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 20px 45px rgba(15, 23, 42, 0.1);
}

.kpi-label {
    color: #94A3B8;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 14px;
    font-weight: 700;
}

.kpi-value {
    font-size: 2.3rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 6px;
}

.kpi-note {
    color: #64748B;
    font-size: 0.95rem;
    line-height: 1.6;
}

.chart-title,
.table-title,
.final-card-title,
.filter-heading {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 10px;
}

.chart-copy,
.table-copy,
.filter-copy,
.final-card-copy {
    color: #64748B;
    font-size: 0.95rem;
    line-height: 1.7;
    margin-bottom: 18px;
}

.mini-card {
    padding: 22px;
    min-height: 130px;
}

.mini-card-label {
    color: #94A3B8;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-weight: 700;
}

.mini-card-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #0F172A;
}

.mini-card-note {
    color: #64748B;
    font-size: 0.92rem;
    margin-top: 10px;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 16px;
}

.chip {
    padding: 10px 18px;
    border-radius: 999px;
    border: 1px solid #E2E8F0;
    color: #0F172A;
    font-size: 0.92rem;
    cursor: pointer;
}

.chip-active {
    background: #E0FCF9;
    border-color: #22C1B5;
    color: #0F172A;
}

.lead-card {
    padding: 20px;
    margin-bottom: 14px;
}

.lead-name {
    font-weight: 700;
    color: #0F172A;
    font-size: 1rem;
    margin-bottom: 6px;
}

.lead-meta {
    color: #64748B;
    font-size: 0.92rem;
    margin-bottom: 12px;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: #F0FDFA;
    color: #0F172A;
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 700;
    border: 1px solid #CCFBF1;
}

.filter-card {
    margin-bottom: 24px;
}

.table-card {
    padding: 24px;
}

.stButton>button {
    border-radius: 999px;
}

@media (max-width: 1150px) {
    .grid-6,
    .grid-4,
    .grid-3,
    .grid-2 {
        display: block;
    }
}

</style>
'''

st.markdown(_STYLE, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    raw = load_company_data(DEFAULT_DATA_PATH)
    scored = calculate_lead_scores(raw)
    scored["lead_category"] = scored["lead_category"].fillna("Low")
    scored["company_age_years"] = ((pd.Timestamp(DATA_AS_OF) - scored["registration_date"]).dt.days / 365.25).fillna(0).clip(lower=0).round(1)
    scored["lead_probability"] = scored.get("lead_probability", scored["lead_score"])
    scored["authorized_capital_lakh"] = scored["authorized_capital_lakh"].fillna(0)
    scored["paid_up_capital_lakh"] = scored["paid_up_capital_lakh"].fillna(0)
    for col in ["company_name", "industry", "roc_city", "state", "sub_industry", "lead_category"]:
        if col in scored.columns:
            scored[col] = scored[col].fillna("Not specified")
    return scored


def safe_list(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique())


def get_city_coordinates(city: str) -> tuple[float, float] | tuple[None, None]:
    return CITY_COORDINATES.get(city, (None, None))


def filter_leads(
    frame: pd.DataFrame,
    selected_state: list[str],
    selected_industry: list[str],
    selected_sub: list[str],
    selected_quality: list[str],
    age_range: tuple[float, float],
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
    filtered = filtered[filtered["company_age_years"].between(age_range[0], age_range[1])]
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


def build_kpi_metrics(frame: pd.DataFrame) -> list[dict]:
    total = len(frame)
    high_leads = int((frame["lead_category"] == "High").sum())
    industries = frame["industry"].nunique()
    total_cap = format_inr_crore(frame["authorized_capital_rs"].sum())
    cities = frame["roc_city"].nunique()
    avg_score = frame["lead_score"].mean().round(1)
    return [
        {"title": "Total Companies", "value": f"{total:,}", "note": "Current filtered universe"},
        {"title": "High Leads", "value": f"{high_leads:,}", "note": "Priority lead opportunities"},
        {"title": "Industries", "value": f"{industries:,}", "note": "Sector count"},
        {"title": "Capital Strength", "value": total_cap, "note": "Authorized capital"},
        {"title": "Active Cities", "value": f"{cities:,}", "note": "Regional hubs"},
        {"title": "Avg Lead Score", "value": f"{avg_score:.1f}%", "note": "Lead quality average"},
    ]


def registration_trend_chart(frame: pd.DataFrame) -> go.Figure:
    timeseries = (
        frame.assign(month=frame["registration_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)
        .agg(companies=("cin", "count"), high_leads=("lead_category", lambda x: (x == "High").sum()))
        .sort_values("month")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=timeseries["month"],
            y=timeseries["companies"],
            mode="lines",
            line=dict(color="#22C1B5", width=4),
            fill="tozeroy",
            fillcolor="rgba(34, 193, 181, 0.14)",
            name="Registrations",
            hovertemplate="%{x|%b %Y}<br>Registrations: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=timeseries["month"],
            y=timeseries["high_leads"],
            mode="lines+markers",
            line=dict(color="#0F172A", width=2, dash="dash"),
            marker=dict(size=7, color="#0F172A"),
            name="High Leads",
            hovertemplate="%{x|%b %Y}<br>High Leads: %{y:,}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(t=24, b=34, l=40, r=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor="#E2E8F0", tickformat="%b %Y"),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False),
        height=420,
    )
    return fig


def top_cities_bar(frame: pd.DataFrame) -> go.Figure:
    city_summary = (
        frame.groupby("roc_city", as_index=False)
        .agg(companies=("cin", "count"), high_leads=("lead_category", lambda x: (x == "High").sum()))
        .sort_values("companies", ascending=True)
        .tail(8)
    )
    fig = go.Figure(go.Bar(
        x=city_summary["companies"],
        y=city_summary["roc_city"],
        orientation="h",
        marker_color="#22C1B5",
        text=city_summary["companies"].map('{:,}'.format),
        textposition="inside",
    ))
    fig.update_layout(
        margin=dict(t=16, b=30, l=120, r=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False),
        height=380,
        showlegend=False,
    )
    return fig


def city_map(frame: pd.DataFrame) -> go.Figure:
    city_summary = (
        frame.groupby("roc_city", as_index=False)
        .agg(companies=("cin", "count"), high_leads=("lead_category", lambda x: (x == "High").sum()))
    )
    coords = city_summary["roc_city"].map(get_city_coordinates)
    city_summary["lat"] = coords.map(lambda x: x[0])
    city_summary["lon"] = coords.map(lambda x: x[1])
    city_summary = city_summary.dropna(subset=["lat", "lon"]).nlargest(20, "companies")
    if city_summary.empty:
        return go.Figure()
    city_summary["lead_density"] = city_summary["high_leads"] / city_summary["companies"]
    fig = px.scatter_geo(
        city_summary,
        lat="lat",
        lon="lon",
        size="companies",
        color="lead_density",
        hover_name="roc_city",
        hover_data={"companies": True, "high_leads": True, "lead_density": True, "lat": False, "lon": False},
        color_continuous_scale=["#A7F3D0", "#22C1B5"],
        projection="natural earth",
        scope="asia",
        height=420,
    )
    fig.update_geos(
        showcountries=True,
        showland=False,
        showocean=False,
        showcoastlines=False,
        lataxis_range=[6, 38],
        lonaxis_range=[66, 98],
    )
    fig.update_layout(margin=dict(t=12, b=12, l=12, r=12), coloraxis_colorbar=dict(title="Lead density"))
    return fig


def industry_donut(frame: pd.DataFrame) -> go.Figure:
    industry_summary = (
        frame.groupby("industry", as_index=False)
        .agg(companies=("cin", "count"))
        .sort_values("companies", ascending=False)
        .head(8)
    )
    fig = px.pie(
        industry_summary,
        values="companies",
        names="industry",
        hole=0.58,
        color_discrete_sequence=["#22C1B5", "#0F172A", "#94A3B8", "#A7F3D0", "#5EEAD4", "#38BDF8", "#22C55E", "#0E7490"],
        height=420,
    )
    fig.update_traces(textinfo="percent+label", textposition="inside", hovertemplate="%{label}: %{value:,} companies<extra></extra>")
    fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    return fig


def engagement_bars(frame: pd.DataFrame) -> go.Figure:
    engagement = (
        frame.groupby("industry", as_index=False)
        .agg(avg_score=("lead_score", "mean"), companies=("cin", "count"))
        .sort_values("avg_score", ascending=True)
        .tail(10)
    )
    fig = go.Figure(go.Bar(
        x=engagement["avg_score"],
        y=engagement["industry"],
        orientation="h",
        marker_color="#22C1B5",
        text=engagement["avg_score"].round(1).astype(str) + "%",
        textposition="inside",
    ))
    fig.update_layout(
        margin=dict(t=16, b=30, l=160, r=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(showgrid=False, zeroline=False, title="Avg Lead Score"),
        yaxis=dict(showgrid=False),
        height=420,
        showlegend=False,
    )
    return fig


def capital_segment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    boundaries = frame["authorized_capital_lakh"].quantile([0.33, 0.66]).values
    def segment(value: float) -> str:
        if value <= boundaries[0]:
            return "Small"
        if value <= boundaries[1]:
            return "Medium"
        return "Large"
    frame = frame.copy()
    frame["segment"] = frame["authorized_capital_lakh"].map(segment)
    summary = (
        frame.groupby("segment", as_index=False)
        .agg(
            companies=("cin", "count"),
            authorized_capital_lakh=("authorized_capital_lakh", "sum"),
            paid_up_capital_lakh=("paid_up_capital_lakh", "sum"),
        )
    )
    summary["avg_capital_lakh"] = (summary["authorized_capital_lakh"] / summary["companies"]).round(1)
    summary["segment_order"] = summary["segment"].map({"Small": 0, "Medium": 1, "Large": 2})
    return summary.sort_values("segment_order")


def capital_segment_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["authorized_capital_lakh"],
        y=summary["segment"],
        orientation="h",
        name="Authorized",
        marker_color="#22C1B5",
        opacity=0.82,
    ))
    fig.add_trace(go.Bar(
        x=summary["paid_up_capital_lakh"],
        y=summary["segment"],
        orientation="h",
        name="Paid-up",
        marker_color="#0F172A",
        opacity=0.6,
    ))
    fig.update_layout(
        margin=dict(t=16, b=30, l=120, r=20),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        barmode="overlay",
        xaxis=dict(showgrid=False, zeroline=False, title="Capital (₹ Lakh)"),
        yaxis=dict(showgrid=False),
        height=380,
        legend=dict(orientation="h", y=-0.18, x=0.05),
    )
    return fig


def paginate_frame(frame: pd.DataFrame, page: int, page_size: int) -> pd.DataFrame:
    start = (page - 1) * page_size
    return frame.iloc[start : start + page_size]


def render_kpi_cards(cards: list[dict]) -> None:
    cols = st.columns(6, gap="large")
    for card, col in zip(cards, cols):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-label">{card["title"]}</div><div class="kpi-value">{card["value"]}</div><div class="kpi-note">{card["note"]}</div></div>',
            unsafe_allow_html=True,
        )


def render_mini_cards(cards: list[dict]) -> None:
    cols = st.columns(4, gap="large")
    for card, col in zip(cards, cols):
        col.markdown(
            f'<div class="mini-card"><div class="mini-card-label">{card["title"]}</div><div class="mini-card-value">{card["value"]}</div><div class="mini-card-note">{card["note"]}</div></div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    data = load_data()
    sidebar = st.sidebar

    with sidebar:
        st.markdown('<div class="filter-card">', unsafe_allow_html=True)
        st.markdown('<div class="filter-heading">Dashboard Filters</div>', unsafe_allow_html=True)
        st.markdown('<div class="filter-copy">Refine results with location, industry, age, lead score and company search.</div>', unsafe_allow_html=True)

        selected_state = st.multiselect("Location", VALID_STATES, default=[])
        industry_options = safe_list(data["industry"])
        selected_industry = st.multiselect("Industry", industry_options, default=[])
        sub_options = safe_list(data["sub_industry"])
        selected_sub = st.multiselect("Sub Industry", sub_options, default=[])
        selected_quality = st.multiselect(
            "Lead Type",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"],
        )

        min_age = int(math.floor(data["company_age_years"].min()))
        max_age = int(math.ceil(data["company_age_years"].quantile(0.98)))
        selected_age = st.slider(
            "Company Age (Years)",
            min_value=min_age,
            max_value=max_age,
            value=(min_age, max_age),
            step=1,
        )

        selected_score = st.slider("Lead Score", 0, 100, (0, 100), step=1)
        search_company = st.text_input("Search Company", value="")
        st.markdown('</div>', unsafe_allow_html=True)

    filtered = filter_leads(
        data,
        selected_state,
        selected_industry,
        selected_sub,
        selected_quality,
        selected_age,
        selected_score,
        search_company,
    )

    st.markdown('<div class="hero-block">', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">MCA Lead Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-text">Analyze Indian company registrations, identify high-value leads and explore market intelligence insights.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    render_kpi_cards(build_kpi_metrics(filtered))

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Registration Trends</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Registration Trends</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Smooth company registration insights with high-lead momentum over time.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(registration_trend_chart(filtered), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    registration_mini = [
        {"title": "Peak Month", "value": filtered["registration_date"].dt.to_period("M").value_counts().idxmax().strftime("%b %Y") if not filtered.empty else "N/A", "note": "Strongest registration period"},
        {"title": "Total Registrations", "value": f"{len(filtered):,}", "note": "Filtered registrations"},
        {"title": "High Leads", "value": f"{int((filtered["lead_category"] == "High").sum()):,}", "note": "Qualified leads"},
        {"title": "Active Industries", "value": f"{filtered["industry"].nunique():,}", "note": "Sector coverage"},
    ]
    render_mini_cards(registration_mini)

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Cities & Location Intelligence</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">City Clusters & Market Density</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Rank top cities and explore location intelligence with lead-quality bubbles.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.55, 0.45], gap="large")
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Top Registration Cities</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-copy">Horizontal ranking view of the most active cities.</div>', unsafe_allow_html=True)
        st.plotly_chart(top_cities_bar(filtered), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">India Location Intelligence</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-copy">Bubble markers represent company count and lead quality density.</div>', unsafe_allow_html=True)
        st.plotly_chart(city_map(filtered), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    city_insights = [
        {"title": "Fastest Growth City", "note": filtered.groupby("roc_city")["cin"].count().pct_change().fillna(0).idxmax() if not filtered.empty else "N/A"},
        {"title": "Highest Capital City", "note": filtered.groupby("roc_city")["authorized_capital_rs"].sum().idxmax() if not filtered.empty else "N/A"},
        {"title": "Best Lead Density", "note": filtered.groupby("roc_city")["lead_score"].mean().idxmax() if not filtered.empty else "N/A"},
        {"title": "Recommended Expansion Market", "note": filtered["roc_city"].mode().iloc[0] if not filtered.empty else "N/A"},
    ]
    render_mini_cards(city_insights)

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Industry Intelligence</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sector Opportunity Snapshot</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Minimal, elegant industry insights for decision-ready analysis.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    industry_themes = ["Finance", "Technology", "Manufacturing", "Healthcare", "Business Services", "Trading", "Construction"]
    selected_theme = st.radio("Theme Chips", industry_themes, index=0)
    st.markdown(f'<div class="chart-copy">Selected theme: <strong>{selected_theme}</strong></div>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.42, 0.58], gap="large")
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Industry Composition</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-copy">Top sectors by company count.</div>', unsafe_allow_html=True)
        st.plotly_chart(industry_donut(filtered), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Top Industries</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-copy">Highest ranking industries from the filtered view.</div>', unsafe_allow_html=True)
        industry_chart, _ = top_cities_bar(filtered), None
        st.plotly_chart(industry_donut(filtered), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Lead Quality Breakdown</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Lead Quality & Engagement</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">A clear view of category mix, engagement and priority companies.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([0.4, 0.6], gap="large")
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Lead Category Mix</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-copy">High / Medium / Low distribution.</div>', unsafe_allow_html=True)
        st.plotly_chart(industry_donut(filtered), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Industry Engagement</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-copy">Average lead score by industry.</div>', unsafe_allow_html=True)
        st.plotly_chart(engagement_bars(filtered), use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Top Leads</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Priority Companies</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Top priority companies with lead score and status.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    top_leads = filtered.nlargest(5, "lead_score")[ ["company_name", "roc_city", "industry", "lead_score", "lead_category"] ]
    for _, row in top_leads.iterrows():
        st.markdown(
            f'<div class="lead-card"><div class="lead-name">{row["company_name"]}</div><div class="lead-meta">{row["industry"]} • {row["roc_city"]}</div><div class="status-pill">Score {row["lead_score"]:.0f}% • {row["lead_category"]}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Capital Analysis</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Capital Allocation by Company Size</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Authorized and paid capital for small, medium and large companies.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    capital_summary = capital_segment_rows(filtered)
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(capital_segment_chart(capital_summary), use_container_width=True, config={"displayModeBar": False})
    st.markdown('</div>', unsafe_allow_html=True)

    capital_mini = [
        {"title": "Total Authorized Capital", "value": format_inr_crore(filtered["authorized_capital_rs"].sum()), "note": "Current filtered capital"},
        {"title": "Paid Capital", "value": format_inr_crore(filtered["paid_up_capital_rs"].sum()), "note": "Paid-up amount"},
        {"title": "Avg Lead Score", "value": f"{filtered["lead_score"].mean():.1f}%", "note": "Quality average"},
        {"title": "Largest Capital Company", "value": filtered.sort_values("authorized_capital_rs", ascending=False)["company_name"].iloc[0] if not filtered.empty else "N/A", "note": "Top capital firm"},
    ]
    render_mini_cards(capital_mini)

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Search & Company Explorer</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Search, Explore and Export</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Search companies, preview details, paginate results and export the filtered universe.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    page_size = 18
    total_rows = len(filtered)
    total_pages = max(1, math.ceil(total_rows / page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    paged = paginate_frame(filtered, page, page_size)

    st.markdown('<div class="table-card">', unsafe_allow_html=True)
    st.markdown('<div class="table-title">Company Explorer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="table-copy">Showing page {page} of {total_pages} • {total_rows:,} companies in the current filter.</div>', unsafe_allow_html=True)
    explorer_view = paged.rename(columns={
        "company_name": "Company",
        "roc_city": "City",
        "state": "State",
        "industry": "Industry",
        "sub_industry": "Sub Industry",
        "company_age_years": "Age (Years)",
        "lead_score": "Lead Score",
        "lead_category": "Lead Category",
        "authorized_capital_lakh": "Authorized Cap (Lakh)",
        "paid_up_capital_lakh": "Paid Cap (Lakh)",
    })
    st.dataframe(explorer_view, use_container_width=True, height=420)
    st.markdown('</div>', unsafe_allow_html=True)

    if not paged.empty:
        details = paged.iloc[0]
        st.markdown('<div class="final-card">', unsafe_allow_html=True)
        st.markdown('<div class="final-card-title">Company Details Preview</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="final-card-copy"><strong>{details["company_name"]}</strong><br>{details["industry"]} • {details["roc_city"]}, {details["state"]}<br>Lead Score: {details["lead_score"]:.0f}% • {details["lead_category"]}<br>Authorized Capital: {details["authorized_capital_lakh"]:,} Lakh<br>Company Age: {details["company_age_years"]} years</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    report = export_excel(filtered, pd.DataFrame({
        "Metric": ["Total Companies", "High Leads", "Industries", "Active Cities"],
        "Value": [
            f"{len(filtered):,}",
            f"{int((filtered["lead_category"] == "High").sum()):,}",
            f"{filtered["industry"].nunique():,}",
            f"{filtered["roc_city"].nunique():,}",
        ],
    }))
    st.download_button(
        label="Export Filtered Report",
        data=report,
        file_name="mca_lead_intelligence.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.markdown('<div class="section-header">', unsafe_allow_html=True)
    st.markdown('<span class="section-label">Final Insights</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Executive Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Four simple executive insights from the current view.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    final_insights = [
        {"title": "Best Expansion Region", "note": filtered["state"].mode().iloc[0] if not filtered.empty else "N/A"},
        {"title": "Highest Opportunity Sector", "note": filtered.groupby("industry")["lead_score"].mean().idxmax() if not filtered.empty else "N/A"},
        {"title": "Fastest Growing Market", "note": filtered.groupby("roc_city")["cin"].count().idxmax() if not filtered.empty else "N/A"},
        {"title": "Recommended Lead Segment", "note": "High"},
    ]
    cols = st.columns(4, gap="large")
    for item, col in zip(final_insights, cols):
        col.markdown(
            f'<div class="final-card"><div class="final-card-title">{item["title"]}</div><div class="final-card-copy">{item["note"]}</div></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
