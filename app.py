# app_streamlit_lga_community.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import requests
from pathlib import Path
import math

st.set_page_config(layout="wide", page_title="Malaria Predictor by LGA & Community")

# -------------------------
# CONFIG - update these
# -------------------------
MODEL_PATH = "newmalariamodel.pkl"
COMMUNITY_CSV = "lagoscomplete_dummyprevalence.csv"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# -------------------------
# Helpers / caches
# -------------------------
@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data(ttl=60*60)
def load_community_df(path):
    p = Path(path)
    if not p.exists():
        st.error(f"Community CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(p)
    col_map = {c: c.strip() for c in df.columns}
    df = df.rename(columns=col_map)
    lower = {c.lower(): c for c in df.columns}
    
    rename_map = {}
    if 'lga' in lower:
        rename_map[lower['lga']] = 'LGA'
    if 'community' in lower:
        rename_map[lower['community']] = 'Community'
    if 'latitude' in lower:
        rename_map[lower['latitude']] = 'Latitude'
    if 'longitude' in lower:
        rename_map[lower['longitude']] = 'Longitude'
    if 'populationdensity' in lower:
        rename_map[lower['populationdensity']] = 'PopulationDensity'
    
    df = df.rename(columns=rename_map)
    
    if 'LGA' not in df.columns or 'Community' not in df.columns:
        st.error("CSV must contain LGA and Community columns.")
        return pd.DataFrame()
    
    df['LGA'] = df['LGA'].astype(str).str.strip().str.title()
    df['Community'] = df['Community'].astype(str).str.strip().str.title()
    
    if 'Latitude' in df.columns:
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    if 'Longitude' in df.columns:
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    return df

def _month_start_end(year:int, month:int):
    start = f"{year:04d}-{month:02d}-01"
    if month == 12:
        last = pd.Timestamp(f"{year+1}-01-01") - pd.Timedelta(days=1)
    else:
        last = pd.Timestamp(f"{year}-{month+1:02d}-01") - pd.Timedelta(days=1)
    end = last.strftime("%Y-%m-%d")
    return start, end

@st.cache_data(ttl=60*60)
def fetch_open_meteo_monthly(lat, lon, year, month):
    start_date, end_date = _month_start_end(year, month)
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "timezone": "UTC"
    }
    try:
        r = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        daily = js.get("daily", {})
        temp = daily.get("temperature_2m_mean")
        prec = daily.get("precipitation_sum")
        rh = daily.get("relative_humidity_2m_mean")
        temp_mean = None if temp is None else float(np.nanmean(temp))
        rain_sum = None if prec is None else float(np.nansum(prec))
        rh_mean = None if rh is None else float(np.nanmean(rh))
        return {"Temperature": temp_mean, "Rainfall": rain_sum, "Humidity": rh_mean}
    except Exception as e:
        st.warning(f"Open-Meteo failed: {e}")
        return {"Temperature": None, "Rainfall": None, "Humidity": None}

# -------------------------
# Load resources
# -------------------------
model = load_model(MODEL_PATH)
df_comm = load_community_df(COMMUNITY_CSV)

# Initialize session state for predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# -------------------------
# UI
# -------------------------
st.title("Malaria Prevalence Predictor - Lagos State")
st.markdown("**Predict malaria risk using live environmental data**")

if df_comm.empty:
    st.error("Community dataframe could not be loaded. Fix COMMUNITY_CSV path and retry.")
    st.stop()

# Sidebar inputs
st.sidebar.header("Location Selection")
lgas = sorted(df_comm['LGA'].unique().tolist())
selected_lga = st.sidebar.selectbox("Select LGA", lgas)

comm_df = df_comm[df_comm['LGA'] == selected_lga].sort_values('Community')
communities = comm_df['Community'].unique().tolist()
selected_community = st.sidebar.selectbox("Select Community", communities)

st.sidebar.header("Time Period")
selected_year = st.sidebar.selectbox("Year", [2021, 2022, 2023, 2024, 2025,2026,2027,2028,2029,2030], index=2)
selected_month = st.sidebar.selectbox("Month", list(range(1, 13)), index=5, 
                                       format_func=lambda x: ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][x-1])

# Get community data
sel_row = comm_df[comm_df['Community'] == selected_community]
if sel_row.empty:
    st.error("Selected community not found.")
    st.stop()

lat = sel_row['Latitude'].values[0] if 'Latitude' in sel_row.columns else None
lon = sel_row['Longitude'].values[0] if 'Longitude' in sel_row.columns else None

if pd.isna(lat) or pd.isna(lon):
    if 'Latitude' in comm_df.columns and 'Longitude' in comm_df.columns:
        lat = comm_df['Latitude'].mean()
        lon = comm_df['Longitude'].mean()

# Population density
pop_default = 3000
if 'PopulationDensity' in sel_row.columns and not sel_row['PopulationDensity'].isna().all():
    pop_default = int(sel_row['PopulationDensity'].values[0])

st.sidebar.header("Demographics")
pop_density = st.sidebar.number_input("Population Density", value=pop_default, min_value=10, max_value=100000, step=100)

st.sidebar.header("Vegetation Index")
ndvi_manual = st.sidebar.slider("NDVI", 0.0, 1.0, 0.45, 0.01, 
                                 help="Normalized Difference Vegetation Index (0=no vegetation, 1=dense vegetation)")

st.sidebar.markdown("---")
if lat and lon:
    st.sidebar.success(f"📍 Coordinates: {lat:.4f}, {lon:.4f}")
else:
    st.sidebar.warning("⚠️ No coordinates available")

# Prediction button
if st.sidebar.button("🔮 Fetch Live Data & Predict", type="primary"):
    with st.spinner("Fetching live environmental data..."):
        if lat is not None and lon is not None:
            env = fetch_open_meteo_monthly(lat, lon, int(selected_year), int(selected_month))
            temp_val = env.get("Temperature")
            rain_val = env.get("Rainfall")
            humid_val = env.get("Humidity")
            
            # Display fetched data
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🌡️ Temperature", f"{temp_val:.1f}°C" if temp_val else "N/A")
            with col2:
                st.metric("🌧️ Rainfall", f"{rain_val:.1f}mm" if rain_val else "N/A")
            with col3:
                st.metric("💧 Humidity", f"{humid_val:.1f}%" if humid_val else "N/A")
        else:
            st.error("No coordinates available for environmental data fetch.")
            temp_val = rain_val = humid_val = None

        # Fallback to manual input if fetch failed
        if temp_val is None or rain_val is None or humid_val is None:
            st.warning("Using default environmental values (fetch failed)")
            temp_val = temp_val or 28.0
            rain_val = rain_val or 120.0
            humid_val = humid_val or 75.0

        # Build input for model
        input_df = pd.DataFrame([[
            selected_year, 
            selected_month, 
            ndvi_manual, 
            temp_val, 
            rain_val, 
            humid_val, 
            pop_density
        ]], columns=['Year', 'Month_Num', 'NDVI', 'Temperature', 'Rainfall', 'Humidity', 'PopulationDensity'])

        # Make prediction
        try:
            pred = model.predict(input_df)[0]
            predicted_value = float(pred)
            
            # Store prediction in session state
            key = f"{selected_lga}_{selected_community}"
            st.session_state.predictions[key] = predicted_value
            
            # Display prediction prominently
            st.success("✅ Prediction Complete!")
            st.metric(
                label=f"Predicted Malaria Prevalence - {selected_community}", 
                value=f"{predicted_value:.2f}",
                help="Cases per 1,000 population"
            )
            
            # Risk categorization
            if predicted_value < 10:
                st.info("🟢 **Risk Level:** Low")
            elif predicted_value < 15:
                st.warning("🟡 **Risk Level:** Moderate")
            else:
                st.error("🔴 **Risk Level:** High")
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            predicted_value = None

# -------------------------
# Map visualization
# -------------------------
st.markdown("---")
st.subheader(f"📍 Map: {selected_lga} Communities")

# Prepare map data
plot_df = comm_df.copy()

# Add predictions from session state
plot_df['predicted_prevalence'] = plot_df.apply(
    lambda row: st.session_state.predictions.get(f"{row['LGA']}_{row['Community']}", np.nan), 
    axis=1
)

# Clean data for plotting
plot_df['Latitude'] = pd.to_numeric(plot_df['Latitude'], errors='coerce')
plot_df['Longitude'] = pd.to_numeric(plot_df['Longitude'], errors='coerce')
plot_df = plot_df.dropna(subset=['Latitude', 'Longitude']).reset_index(drop=True)

# Handle cases where no predictions exist yet
if plot_df['predicted_prevalence'].notna().sum() == 0:
    plot_df['plot_val'] = 0.0
    plot_df['marker_size'] = 8
    st.info("ℹ️ No predictions yet. Click 'Fetch Live Data & Predict' to generate predictions.")
else:
    # Scale values for visualization
    plot_df['plot_val'] = plot_df['predicted_prevalence'].fillna(0.0)
    
    # Create marker sizes
    min_size = 8
    scale_factor = 4.0
    plot_df['marker_size'] = plot_df['plot_val'].apply(
        lambda x: max(min_size, x * scale_factor) if x > 0 else min_size
    )
    plot_df['marker_size'] = plot_df['marker_size'].clip(lower=min_size, upper=60)
    
    # Emphasize selected community
    is_selected = plot_df['Community'] == selected_community
    if is_selected.any():
        plot_df.loc[is_selected, 'marker_size'] = plot_df['marker_size'].max() * 1.5

# Set color range
vmin = float(plot_df['plot_val'].min())
vmax = float(plot_df['plot_val'].max())
if math.isclose(vmin, vmax):
    vmax = max(1.0, vmax * 1.1)

# Create map
fig = px.scatter_mapbox(
    plot_df,
    lat='Latitude',
    lon='Longitude',
    hover_name='Community',
    hover_data={
        'LGA': True, 
        'predicted_prevalence': ':.2f',
        'Latitude': False,
        'Longitude': False,
        'plot_val': False,
        'marker_size': False
    },
    color='plot_val',
    size='marker_size',
    size_max=60,
    color_continuous_scale='Reds',
    range_color=(vmin, vmax),
    zoom=10,
    height=600,
    labels={'plot_val': 'Prevalence', 'predicted_prevalence': 'Prevalence (per 1,000)'}
)

fig.update_layout(
    mapbox_style='open-street-map',
    coloraxis_colorbar=dict(title="Prevalence<br>(per 1,000)")
)
fig.update_traces(marker=dict(sizemode='area', opacity=0.7))

st.plotly_chart(fig, use_container_width=True)

# Summary statistics
if plot_df['predicted_prevalence'].notna().sum() > 0:
    st.markdown("---")
    st.subheader("📊 Prediction Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Communities Predicted", plot_df['predicted_prevalence'].notna().sum())
    with col2:
        st.metric("Avg Prevalence", f"{plot_df['predicted_prevalence'].mean():.2f}")
    with col3:
        st.metric("Max Prevalence", f"{plot_df['predicted_prevalence'].max():.2f}")
    with col4:
        st.metric("Min Prevalence", f"{plot_df['predicted_prevalence'].min():.2f}")

st.markdown("---")
st.caption("📌 **How to use:** Select an LGA and community, then click 'Fetch Live Data & Predict' to get real-time malaria prevalence predictions based on current environmental conditions.")
st.caption("⚠️ **Disclaimer:** Predictions are model estimates based on environmental factors and should be used alongside actual surveillance data.")