"""District form that scores one row through the house-price API.

Run locally with the API already up:

    API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py

Compose starts this on port 8501 and points API_URL at the api service.
The browser talks only to Streamlit. Streamlit calls FastAPI server-side,
so the page does not need CORS.
"""

from __future__ import annotations

import streamlit as st

from housing_form import (
    EXAMPLE_DISTRICT,
    OCEAN_PROXIMITY_VALUES,
    ApiError,
    api_base_url,
    build_payload,
    fetch_health,
    score_district,
)

st.set_page_config(
    page_title="District house value",
    layout="centered",
)
st.markdown(
    """
    <style>
      .block-container { max-width: 860px; padding-top: 1.6rem; padding-bottom: 3rem; }
      div[data-testid="stMetricValue"] { font-size: 2.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

api_url = api_base_url()

st.title("California district house value")
st.write(
    "Estimate the median sale price for a 1990 census district. "
    "This is the middle of the district, not the price of one house. "
    "The form starts on a real Near Bay district whose recorded median was about $281,500."
)

try:
    health = fetch_health(api_url)
except ApiError as exc:
    health = None
    st.error(f"The prediction API is not ready ({api_url}). {exc}")
else:
    if health.get("status") == "ok":
        st.success(
            f"Model {health.get('artifact_version')} is loaded "
            f"({health.get('selected_feature_count')} features)."
        )
    else:
        st.warning(f"API responded, but no model is loaded yet. Status: {health.get('status')}.")

with st.form("district"):
    location, size = st.columns(2)
    with location:
        longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(EXAMPLE_DISTRICT["longitude"]),
            step=0.01,
            format="%.4f",
            help="West is negative. California sits around -124 to -114.",
        )
        latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(EXAMPLE_DISTRICT["latitude"]),
            step=0.01,
            format="%.4f",
            help="California sits around 32 to 42.",
        )
        housing_median_age = st.number_input(
            "Median housing age (years)",
            min_value=0.0,
            max_value=120.0,
            value=float(EXAMPLE_DISTRICT["housing_median_age"]),
            step=1.0,
            help="Median age of the houses in the district, not the residents.",
        )
        ocean_proximity = st.selectbox(
            "Ocean proximity",
            options=list(OCEAN_PROXIMITY_VALUES),
            index=list(OCEAN_PROXIMITY_VALUES).index(EXAMPLE_DISTRICT["ocean_proximity"]),
        )
    with size:
        total_rooms = st.number_input(
            "Total rooms",
            min_value=0.0,
            value=float(EXAMPLE_DISTRICT["total_rooms"]),
            step=1.0,
        )
        total_bedrooms = st.number_input(
            "Total bedrooms",
            min_value=0.0,
            value=float(EXAMPLE_DISTRICT["total_bedrooms"]),
            step=1.0,
            help="Leave the box below checked if this count is missing.",
        )
        bedrooms_unknown = st.checkbox(
            "Bedroom count is unknown",
            value=False,
            help="Sends null. Training imputed those gaps from the training mean.",
        )
        population = st.number_input(
            "Population",
            min_value=0.0,
            value=float(EXAMPLE_DISTRICT["population"]),
            step=1.0,
        )
        households = st.number_input(
            "Households",
            min_value=1.0,
            value=float(EXAMPLE_DISTRICT["households"]),
            step=1.0,
            help="Must be at least 1. Ratios are per household.",
        )
        median_income = st.number_input(
            "Median income (dataset units)",
            min_value=0.0,
            value=float(EXAMPLE_DISTRICT["median_income"]),
            step=0.1,
            format="%.4f",
            help="Already scaled in the census file: roughly dollars divided by 10,000. 3.2 means about $32,000.",
        )

    submitted = st.form_submit_button("Predict median value", type="primary")

if submitted:
    try:
        payload = build_payload(
            longitude=longitude,
            latitude=latitude,
            housing_median_age=housing_median_age,
            total_rooms=total_rooms,
            total_bedrooms=total_bedrooms,
            bedrooms_known=not bedrooms_unknown,
            population=population,
            households=households,
            median_income=median_income,
            ocean_proximity=ocean_proximity,
        )
        prediction = score_district(api_url, payload)
    except (ApiError, ValueError) as exc:
        st.error(str(exc))
    else:
        value = float(prediction["median_house_value"])
        st.metric("Predicted district median", f"${value:,.0f}")
        st.caption(
            f"Model {prediction.get('artifact_version', 'unknown')}. "
            "The figure is exp(model output), in dollars."
        )
        with st.expander("Request sent to /predict"):
            st.json(payload)
