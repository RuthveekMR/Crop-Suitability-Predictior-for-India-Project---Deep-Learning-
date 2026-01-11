import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
# --- CORRECT IMPORT LINE ---
from utils.predictor import predict # Make sure this line imports 'predict'
# ---

# --- Page Configuration ---
st.set_page_config(
    page_title="Crop Suitability Predictor",
    page_icon="🌱",
    layout="wide",
)

# --- App Title and Introduction ---
st.title("🌱 Crop Suitability Predictor for India")
st.markdown(
    "Click on the map or use the sliders to select a location. A deep learning model (ResNet) will analyze climate and soil data "
    "to recommend the top 5 most suitable crops."
)
st.divider()

# --- UI Layout ---
col1, col2 = st.columns([0.4, 0.6], gap="large") # 40% for map/input, 60% for results

with col1:
    st.header("📍 Select Location")

    # --- Interactive Map using Folium ---
    if 'center' not in st.session_state:
        st.session_state.center = [20.5937, 78.9629] # Default center on India
    if 'location' not in st.session_state:
        st.session_state.location = st.session_state.center # Default location is center

    m = folium.Map(location=st.session_state.center, zoom_start=5)
    folium.Marker(
        st.session_state.location,
        popup="Selected Location",
        tooltip="Click to predict for this location"
    ).add_to(m)

    map_data = st_folium(m, center=st.session_state.center, width=700, height=500)

    # Update state if map is clicked and rerun
    if map_data and map_data['last_clicked']:
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']
        if abs(clicked_lat - st.session_state.location[0]) > 0.01 or \
           abs(clicked_lon - st.session_state.location[1]) > 0.01:
            st.session_state.location = [clicked_lat, clicked_lon]
            st.session_state.center = [clicked_lat, clicked_lon]
            st.rerun()

    # --- Manual Coordinate Input ---
    lat = st.slider("Latitude", 6.0, 38.0, st.session_state.location[0], 0.01, help="Adjust the latitude (North).")
    lon = st.slider("Longitude", 68.0, 98.0, st.session_state.location[1], 0.01, help="Adjust the longitude (East).")

    # Update state if sliders are used and rerun
    if lat != st.session_state.location[0] or lon != st.session_state.location[1]:
        st.session_state.location = [lat, lon]
        st.rerun()

    st.info(f"Predicting for Coordinates: **{st.session_state.location[0]:.2f}° N, {st.session_state.location[1]:.2f}° E**")

# --- Display Results in the Second Column ---
with col2:
    st.header("💡 Prediction Results")

    with st.spinner("Analyzing location and running deep learning model..."):
        # Call the predict function
        status, grid_info, top_5 = predict(st.session_state.location[0], st.session_state.location[1])

    # --- Handle Different Prediction Outcomes ---
    if status == "out_of_bounds":
        st.error("**(Out of Bounds)** The selected coordinates are outside the valid range for India (Lat 6-38, Lon 68-98). Please select a point within the boundaries.")
    elif status == "ocean_point":
        st.warning("**(Ocean/Invalid Point)** The selected location is too far from a known land grid point in our data. Please select a location on mainland India.")
    elif status == "prediction_error":
        st.error("**(Prediction Error)** An issue occurred during the model prediction step. Please check the logs or try again.")
    elif status == "data_error":
         st.error("**(Data Error)** Could not process data correctly (e.g., 'crop_id' missing). Check data loading and preprocessing steps.")
    elif status == "success" and top_5 is not None and not top_5.empty:
        # --- Display Grid Information ---
        st.subheader("🌍 Nearest Grid Climate Summary")
        m1, m2, m3 = st.columns(3)
        soil_type = ('Light' if grid_info.get('g_soil_light', 0)==1 else
                     'Medium' if grid_info.get('g_soil_medium', 0)==1 else
                     'Heavy' if grid_info.get('g_soil_heavy', 0)==1 else 'Unknown')

        m1.metric("Avg. Temp", f"{grid_info.get('g_temp_median', 0):.1f} °C")
        m2.metric("Avg. Rainfall", f"{grid_info.get('g_rf_median', 0):.0f} mm/yr")
        m3.metric("Soil Type", soil_type)
        st.caption(f"Using climate/soil data from nearest grid point: Lat {grid_info.get('g_lat', 0):.2f}, Lon {grid_info.get('g_lon', 0):.2f}")

        st.divider()

        # --- Display Top 5 Crops ---
        st.subheader("🌱 Top 5 Recommended Crops")
        for i, row in top_5.reset_index(drop=True).iterrows():
            with st.expander(f"**#{i+1}: {row.get('crop_name', 'N/A')}** (Score: **{row.get('suitability_score', 0.0):.3f}**)", expanded=(i==0)):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.markdown(f"**Scientific Name:** *{row.get('ScientificName', 'N/A')}*")
                    st.markdown(f"**Category:** {row.get('CAT', 'N/A')}")
                    st.markdown(f"**Life Form:** {row.get('LIFO', 'N/A')}")
                    st.markdown(f"**Growth Habit:** {row.get('HABI', 'N/A')}")
                with col_info2:
                    st.markdown(f"**Life Span:** {row.get('LISPA', 'N/A')}")
                    st.markdown(f"**Photoperiod:** {row.get('PHOTO', 'N/A')}")
                    gmin, gmax = row.get('GMIN'), row.get('GMAX')
                    st.markdown(f"**Cycle:** {int(gmin) if pd.notna(gmin) else 'N/A'} - {int(gmax) if pd.notna(gmax) else 'N/A'} days")
                    st.markdown(f"**Physical Char.:** {row.get('PHYS', 'N/A')}") # Added PHYS

    elif status == "success" and (top_5 is None or top_5.empty):
         st.warning("Prediction successful, but no suitable crops were found based on the model's analysis for this location.")
    else: # Default message
        st.info("Select a location on the map or adjust sliders to see crop recommendations.")

# --- Footer ---
st.divider()
st.caption("A Deep Learning Application for Crop Suitability Prediction in India.")