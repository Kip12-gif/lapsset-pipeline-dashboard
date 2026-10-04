import streamlit as st
import ee
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra
import folium
from streamlit_folium import st_folium
import plotly.express as px
import pandas as pd

# Page Configuration
st.set_page_config(
    page_title="LAPSSET Pipeline Optimization Portal",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ Lokichar–Mokowe Pipeline Trajectory Optimizer")
st.markdown("Dynamic Least-Cost Path (LCP) spatial analysis using Google Earth Engine and SciPy graph algorithms along the LAPSSET corridor.")

# Initialize Earth Engine using Streamlit Secrets or Environment Auth
@st.cache_resource
def init_earth_engine():
    try:
        ee.Initialize()
    except Exception:
        # Streamlit Cloud uses service account credentials stored in Secrets if needed
        ee.Authenticate()
        ee.Initialize()

init_earth_engine()

# Coordinate Definitions
LOKICHAR_COORDS = [2.3789, 35.6425]  # [Lat, Lon]
MOKOWE_COORDS = [-2.2355, 40.8522]

# Sidebar Controls for Multi-Criteria Friction Weights
st.sidebar.header("⚖️ Multi-Criteria Friction Weights")
st.sidebar.info("Adjust parameters to dynamically re-evaluate optimal pipeline trajectory.")

slope_weight = st.sidebar.slider("Terrain Slope Weight", min_value=0.1, max_value=5.0, value=1.5, step=0.1)
wc_weight = st.sidebar.slider("Land Cover Avoidance Weight", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
protected_weight = st.sidebar.slider("Protected Area Penalty", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

# Extract Cloud Spatial Rasters & Compile Friction Surface
@st.cache_data(ttl=3600, show_spinner=False)
def compute_friction_surface(s_w, wc_w, p_w):
    lokichar = ee.Geometry.Point([LOKICHAR_COORDS[1], LOKICHAR_COORDS[0]])
    mokowe = ee.Geometry.Point([MOKOWE_COORDS[1], MOKOWE_COORDS[0]])
    corridor_bounds = lokichar.buffer(100000).union(mokowe.buffer(100000)).bounds()

    # Elevation & Slope (SRTM DEM)
    dem = ee.Image("USGS/SRTM30m_NUM").clip(corridor_bounds)
    slope = ee.Terrain.slope(dem).multiply(s_w)

    # ESA WorldCover 10m
    worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(corridor_bounds)
    remap_from = [10, 20, 30, 40, 50, 60, 80, 90, 95, 100]
    remap_to   = [30, 10, 10, 15, 85, 20, 99, 90, 80, 20]
    landcover_cost = worldcover.remap(remap_from, remap_to, 10).multiply(wc_w)

    # Protected Areas (WDPA)
    protected = ee.FeatureCollection("WCMC/WDPA/current/polygons")
    protected_mask = protected.reduceToImage(properties=['REP_AREA'], reducer=ee.Reducer.first()).unmask(0).gt(0)
    protected_cost = protected_mask.multiply(100 * p_w)

    friction = slope.add(landcover_cost).add(protected_cost).rename('friction')
    
    # 4km grid sampling for fast interactive calculation on Streamlit Cloud
    dataset = friction.sampleRectangle(region=corridor_bounds, defaultValue=50)
    grid = np.array(dataset.get('friction').getInfo())
    
    dem_dataset = dem.sampleRectangle(region=corridor_bounds, defaultValue=0)
    dem_grid = np.array(dem_dataset.get('elevation').getInfo())
    
    return grid, dem_grid

with st.spinner("Streaming cloud rasters & compiling multi-criteria friction surface..."):
    friction_grid, dem_grid = compute_friction_surface(slope_weight, wc_weight, protected_weight)

# Graph Construction & Dijkstra Solver
@st.cache_data(show_spinner=False)
def solve_least_cost_path(grid):
    r, c = grid.shape
    nodes = r * c
    row_idx, col_idx, weights = [], [], []

    for i in range(r):
        for j in range(c):
            curr = i * c + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < r and 0 <= nj < c:
                        neighbor = ni * c + nj
                        dist_factor = np.sqrt(di**2 + dj**2)
                        cost = ((grid[i, j] + grid[ni, nj]) / 2.0) * dist_factor
                        row_idx.append(curr)
                        col_idx.append(neighbor)
                        weights.append(cost)

    graph = sp.csr_matrix((weights, (row_idx, col_idx)), shape=(nodes, nodes))
    start_idx = 0
    end_idx = (r - 1) * c + (c - 1)
    
    distances, predecessors = dijkstra(csgraph=graph, directed=False, indices=start_idx, return_predecessors=True)
    
    path = []
    curr = end_idx
    while curr != start_idx and curr != -9999:
        path.append(curr)
        curr = predecessors[curr]
    path.append(start_idx)
    path.reverse()
    
    coords = [(p // c, p % c) for p in path]
    return coords

path_pixels = solve_least_cost_path(friction_grid)

# Map grid indices to lat/lon bounding box interpolation
lat_lin = np.linspace(LOKICHAR_COORDS[0], MOKOWE_COORDS[0], friction_grid.shape[0])
lon_lin = np.linspace(LOKICHAR_COORDS[1], MOKOWE_COORDS[1], friction_grid.shape[1])

route_latlon = [(lat_lin[r], lon_lin[c]) for r, c in path_pixels]
elevations = [dem_grid[r, c] for r, c in path_pixels]

# Dashboard Display
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Interactive Least-Cost Trajectory Map")
    m = folium.Map(location=[0.1, 38.2], zoom_start=7, tiles="CartoDB positron")
    
    # Origin & Destination Markers
    folium.Marker(LOKICHAR_COORDS, popup="Lokichar Oil Fields", icon=folium.Icon(color='black', icon='tint')).add_to(m)
    folium.Marker(MOKOWE_COORDS, popup="Mokowe Lamu Refinery", icon=folium.Icon(color='red', icon='industry')).add_to(m)
    
    # Render Calculated Pipeline Route Line
    folium.PolyLine(route_latlon, color="#0284c7", weight=5, opacity=0.8, tooltip="Optimal Pipeline Route").add_to(m)
    
    st_folium(m, width="100%", height=520)

with col2:
    st.subheader("Elevation Profile along Trajectory")
    
    profile_df = pd.DataFrame({
        "Waypoints": range(len(elevations)),
        "Elevation (m)": elevations
    })
    
    fig = px.area(
        profile_df, 
        x="Waypoints", 
        y="Elevation (m)", 
        title="Topographic Elevation Profile (Lokichar to Mokowe)",
        labels={"Waypoints": "Pipeline Segment Step", "Elevation (m)": "Altitude (m)"}
    )
    fig.update_traces(line_color="#0d9488", fillcolor="rgba(13, 148, 136, 0.2)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Total Route Discrete Segments", f"{len(route_latlon)} steps")
    st.metric("Max Traversal Elevation", f"{max(elevations)} m")