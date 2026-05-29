"""Data preparation and lead-scoring utilities for the MCA Pulse dashboard."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "eir_May2019.xlsx"
DEFAULT_SHEET = "Indian company registered - May"
DATA_AS_OF = pd.Timestamp("2019-05-31")

CITY_COORDINATES = {
    "Ahmedabad": (23.0225, 72.5714),
    "Andaman": (11.6234, 92.7265),
    "Bangalore": (12.9716, 77.5946),
    "Chandigarh": (30.7333, 76.7794),
    "Chennai": (13.0827, 80.2707),
    "Chhattisgarh": (21.2514, 81.6296),
    "Coimbatore": (11.0168, 76.9558),
    "Cuttack": (20.4625, 85.8830),
    "Delhi": (28.6139, 77.2090),
    "Ernakulam": (9.9816, 76.2999),
    "Goa": (15.4909, 73.8278),
    "Gwalior": (26.2183, 78.1828),
    "Hyderabad": (17.3850, 78.4867),
    "Himachal Prade": (31.1048, 77.1734),
    "Jaipur": (26.9124, 75.7873),
    "Jammu": (32.7266, 74.8570),
    "Jharkhand": (23.3441, 85.3096),
    "Kanpur": (26.4499, 80.3319),
    "Kolkata": (22.5726, 88.3639),
    "Mumbai": (19.0760, 72.8777),
    "Patna": (25.5941, 85.1376),
    "Pondicherry": (11.9416, 79.8083),
    "Pune": (18.5204, 73.8567),
    "Shillong": (25.5788, 91.8933),
    "Uttarakhand": (30.3165, 78.0322),
    "Vijayawada": (16.5062, 80.6480),
}

VALID_STATES = [
    "Andaman & Nicobar",
    "Andhra Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chattisgarh",
    "Dadra & Nagar Haveli",
    "Daman and Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu & Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Orissa",
    "Pondicherry",
    "Punjab",
    "Rajasthan",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

MODEL_FEATURES = [
    "authorized_log",
    "paid_log",
    "capital_conversion",
    "registration_day",
    "recency_days",
    "company_class",
    "listed_flag",
    "roc_city",
    "state",
    "industry",
]
NUMERIC_FEATURES = [
    "authorized_log",
    "paid_log",
    "capital_conversion",
    "registration_day",
    "recency_days",
]
CATEGORICAL_FEATURES = [
    "company_class",
    "listed_flag",
    "roc_city",
    "state",
    "industry",
]


def _clean_text(series: pd.Series, fallback: str = "Unknown") -> pd.Series:
    return (
        series.fillna(fallback)
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace("", fallback)
    )


def _get_column(
    frame: pd.DataFrame, candidates: Iterable[str], fallback: object = ""
) -> pd.Series:
    for name in candidates:
        if name in frame.columns:
            return frame[name]
    return pd.Series(fallback, index=frame.index)


def load_company_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and standardize the Indian-company registration sheet."""
    source = Path(path)
    raw = pd.read_excel(source, sheet_name=DEFAULT_SHEET, engine="openpyxl")
    frame = pd.DataFrame(index=raw.index)
    frame["serial_no"] = _get_column(raw, ["S.No", "S.No."])
    frame["roc_location"] = _clean_text(_get_column(raw, ["ROC Location", "ROC"]))
    frame["cin"] = _clean_text(_get_column(raw, ["CIN", "FCIN"]))
    frame["company_name"] = _clean_text(
        _get_column(raw, ["Company Name", "COMPANY NAME"]), "Unnamed company"
    )
    frame["registration_date"] = pd.to_datetime(
        _get_column(raw, ["Date of Registration", "DATE OF REGISTRATION"]),
        errors="coerce",
        dayfirst=True,
    )
    frame["month_name"] = _clean_text(_get_column(raw, ["MONTH NAME"])).str.title()
    frame["company_class"] = _clean_text(
        _get_column(raw, ["Company Class"]), "Not specified"
    )
    frame["listed_flag"] = _clean_text(
        _get_column(raw, ["Listed Flag"]), "Not specified"
    )
    frame["state"] = _clean_text(
        _get_column(raw, ["State ", "State", "STATE"]), "Not specified"
    )
    frame["industry"] = _clean_text(
        _get_column(raw, ["Industrial Description", "ACTIVITY DESCRIPTION"]),
        "Industry not recorded",
    )
    frame["sub_industry"] = _clean_text(
        _get_column(raw, ["Company Sub-category"]), "Not specified"
    )
    frame["sub_industry_code"] = (
        pd.to_numeric(_get_column(raw, ["Industrial Activity Code"]), errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    frame["sub_industry_code"] = frame["sub_industry_code"].replace(
        "0", "Not specified"
    )
    frame["address"] = _clean_text(_get_column(raw, [" Address", "Address"]), "")
    frame["email"] = _clean_text(_get_column(raw, ["Company Email ID", "EMAIL"]), "")

    auth_col = next(
        (c for c in raw.columns if "Authorized Capital" in str(c)),
        "Authorized Capital",
    )
    paid_col = next(
        (
            c
            for c in raw.columns
            if "Paid-up Capital" in str(c) or "Paid Up Capital" in str(c)
        ),
        "Paid Up Capital",
    )
    authorized = (
        pd.to_numeric(raw.get(auth_col, 0), errors="coerce").fillna(0).clip(lower=0)
    )
    paid = pd.to_numeric(raw.get(paid_col, 0), errors="coerce").fillna(0).clip(lower=0)
    auth_scale = 1000 if "Thousand" in str(auth_col) else 1
    paid_scale = 1000 if "Thousand" in str(paid_col) else 1
    frame["authorized_capital_rs"] = authorized * auth_scale
    frame["paid_up_capital_rs"] = paid * paid_scale
    frame["authorized_capital_lakh"] = frame["authorized_capital_rs"] / 100_000
    frame["paid_up_capital_lakh"] = frame["paid_up_capital_rs"] / 100_000

    frame["roc_city"] = (
        frame["roc_location"]
        .str.replace(r"^ROC\s*-\s*", "", regex=True, case=False)
        .str.title()
    )
    frame["registration_date"] = frame["registration_date"].fillna(DATA_AS_OF)
    frame["month_name"] = frame["registration_date"].dt.strftime("%B")
    return frame


def add_model_features(
    frame: pd.DataFrame, reference_date: pd.Timestamp = DATA_AS_OF
) -> pd.DataFrame:
    result = frame.copy()
    result["authorized_log"] = np.log1p(result["authorized_capital_rs"].clip(lower=0))
    result["paid_log"] = np.log1p(result["paid_up_capital_rs"].clip(lower=0))
    result["capital_conversion"] = np.where(
        result["authorized_capital_rs"] > 0,
        (result["paid_up_capital_rs"] / result["authorized_capital_rs"]).clip(0, 5),
        0,
    )
    result["registration_day"] = result["registration_date"].dt.day.astype(int)
    result["recency_days"] = (
        pd.Timestamp(reference_date).normalize()
        - result["registration_date"].dt.normalize()
    ).dt.days.clip(lower=0)
    return result


def calculate_lead_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Create cohort-relative lead and engagement potential indicators."""
    result = add_model_features(frame)
    result["authorized_percentile"] = result["authorized_log"].rank(
        pct=True, method="average"
    )
    result["paid_percentile"] = result["paid_log"].rank(pct=True, method="average")
    result["city_density"] = result.groupby("roc_city")["cin"].transform("count")
    result["industry_density"] = result.groupby("industry")["cin"].transform("count")
    result["city_density_percentile"] = result["city_density"].rank(
        pct=True, method="average"
    )
    result["industry_density_percentile"] = result["industry_density"].rank(
        pct=True, method="average"
    )
    result["class_signal"] = np.select(
        [
            result["company_class"].str.contains("Public", case=False, na=False),
            result["company_class"].str.contains("One Person", case=False, na=False),
        ],
        [1.0, 0.35],
        default=0.60,
    )
    result["listed_signal"] = result["listed_flag"].str.contains(
        "Listed", case=False, na=False
    ) & ~result["listed_flag"].str.contains("Unlisted", case=False, na=False)
    result["recency_signal"] = (1 - (result["recency_days"] / 31)).clip(0, 1)
    result["lead_score"] = (
        (
            result["authorized_percentile"] * 0.33
            + result["paid_percentile"] * 0.22
            + result["class_signal"] * 0.12
            + result["listed_signal"].astype(float) * 0.08
            + result["recency_signal"] * 0.10
            + result["city_density_percentile"] * 0.15
        )
        * 100
    ).round(1)
    cohort_rank = result["lead_score"].rank(pct=True, method="first")
    result["lead_category"] = np.select(
        [cohort_rank >= 0.80, cohort_rank >= 0.45],
        ["High", "Medium"],
        default="Low",
    )
    result["qualified_signal"] = (result["lead_category"] == "High").astype(int)
    result["engagement_index"] = (
        (
            result["lead_score"] * 0.58
            + result["industry_density_percentile"] * 22
            + result["city_density_percentile"] * 20
        )
        .clip(0, 100)
        .round(1)
    )
    result["modelled_monthly_sessions"] = (
        (50 + result["engagement_index"] * 8).round().astype(int)
    )
    result["engagement_tier"] = pd.cut(
        result["engagement_index"],
        bins=[-1, 45, 70, 100],
        labels=["Developing", "Engaged", "High intent"],
    ).astype(str)
    return result


def make_prediction_record(
    authorized_lakh: float,
    paid_lakh: float,
    company_class: str,
    listed_flag: str,
    registration_date: object,
    roc_city: str,
    state: str,
    industry: str,
) -> pd.DataFrame:
    record = pd.DataFrame(
        {
            "authorized_capital_rs": [max(float(authorized_lakh), 0) * 100_000],
            "paid_up_capital_rs": [max(float(paid_lakh), 0) * 100_000],
            "company_class": [company_class],
            "listed_flag": [listed_flag],
            "registration_date": [pd.to_datetime(registration_date)],
            "roc_city": [roc_city],
            "state": [state],
            "industry": [industry],
        }
    )
    return add_model_features(record)


def format_inr_crore(value: float) -> str:
    return f"Rs {value / 10_000_000:,.1f} Cr"


def export_excel(frame: pd.DataFrame, summary: pd.DataFrame) -> bytes:
    output = BytesIO()
    export_columns = [
        "cin",
        "company_name",
        "registration_date",
        "roc_city",
        "state",
        "industry",
        "company_class",
        "listed_flag",
        "authorized_capital_lakh",
        "paid_up_capital_lakh",
        "lead_score",
        "lead_category",
        "lead_probability",
        "business_opportunity_score",
        "growth_potential",
        "priority_rank",
    ]
    available_columns = [col for col in export_columns if col in frame.columns]
    if not available_columns:
        available_columns = list(frame.columns)

    high_leads = pd.DataFrame(columns=available_columns)
    if "lead_category" in frame.columns:
        high_leads = frame.loc[frame["lead_category"] == "High", available_columns]

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Market Summary", index=False)
        frame[available_columns].to_excel(
            writer, sheet_name="Scored Leads", index=False
        )
        high_leads.to_excel(writer, sheet_name="High Leads", index=False)
        for sheet_name in ["Market Summary", "Scored Leads", "High Leads"]:
            sheet = writer.book[sheet_name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(
                    max(max(len(str(cell.value or "")) for cell in column) + 2, 12), 42
                )
                sheet.column_dimensions[column[0].column_letter].width = width
    return output.getvalue()
