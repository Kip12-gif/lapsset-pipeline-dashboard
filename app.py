import json
import ee
import folium
import numpy as np
import plotly.graph_objects as go
import scipy.sparse as sp
import streamlit as st
from google.oauth2.service_account import Credentials
from scipy.sparse.csgraph import dijkstra
from streamlit_folium import st_folium

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & EARTH ENGINE INITIALIZATION
# -----------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="LAPSSET Pipeline Route Optimizer",
    page_icon="🛢️",
)


@st.cache_resource
def init_earth_engine():
  """Initializes Earth Engine using Service Account credentials from Streamlit Secrets."""
  if "GEE_SERVICE_ACCOUNT" not in st.secrets or "GEE_KEY" not in st.secrets:
    st.error(
        "Missing GEE secrets! Please set GEE_SERVICE_ACCOUNT and GEE_KEY in"
        " Streamlit Settings -> Secrets."
    )
    st.stop()

  try:
    raw_key = st.secrets["GEE_KEY"]

    # Parse JSON key stored in secrets
    if isinstance(raw_key, str):
      key_dict = json.loads(raw_key)
    else:
      key_dict = dict(raw_key)

    # Load Service Account OAuth2 credentials directly
    credentials = Credentials.from_service_account_info(
        key_dict, scopes=["https://www.googleapis.com/auth/earthengine"]
    )

    # Initialize Earth Engine explicitly with service account credentials & project
    ee.Initialize(
        credentials=credentials, project=st.secrets["GEE_SERVICE_ACCOUNT"]
    )

  except Exception as e:
    st.error(
        f"Failed to initialize Earth Engine. Check your secrets formatting: {e}"
    )
    st.stop()


init_earth_engine()

# -----------------------------------------------------------------------------
# 2. CONSTANTS & BOUNDING BOX
# -----------------------------------------------------------------------------
# Route endpoints: Lokichar (Turkana) -> Mokowe (Lamu)
START_COORDS = [2.353, 35.602]  # [Lat, Lon]
END_COORDS = [2.235, 40.852]  # [Lat, Lon]

# Corridor Bounding Box
CORRIDOR_BBOX = ee.Geometry.BBox(35.0, 1.0, 41.5, 3.5)
SCALE = 2000  # Grid sampling resolution in meters (2km)


# -----------------------------------------------------------------------------
# 3. CACHED GEE RASTER RETRIEVAL
# -----------------------------------------------------------------------------
@st.cache_data(ttl=86400)
def fetch_base_rasters():
  """Fetches DEM, WorldCover, and WDPA rasters from GEE at 2km resolution."""
  # 1. Slope layer derived from SRTM DEM
  dem = ee.Image("USGS/SRTMGL1_003").clip(CORRIDOR_BBOX)
  slope = ee.Terrain.slope(dem)

  # 2. Reclassified Land Cover (ESA WorldCover 2020)
  lc = ee.Image("ESA/WorldCover/v100/2020").clip(CORRIDOR_BBOX)
  # Assign relative friction costs
  lc_cost = lc.remap(
      [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
      [1, 2, 1, 2, 50, 1, 2, 80, 90, 80, 1],
      1,
  )

  # 3. Protected Areas (WDPA)
  wdpa = ee.FeatureCollection("WCMC/WDPA/current/polygons").filterBounds(
      CORRIDOR_BBOX
  )
  protected_mask = ee.Image(0).paint(wdpa, 100).clip(CORRIDOR_BBOX)

  # Combine into a single multi-band image for efficient single API payload
  composite = ee.Image.cat([
      slope.rename("slope"),
      lc_cost.rename("landcover"),
      protected_mask.rename("protected"),
      dem.rename("dem"),
  ])

  # Retrieve sample grid directly into numpy arrays
  data_dict = composite.sampleRectangle(region=CORRIDOR_BBOX, defaultValue=0)

  slope_arr = np.array(data_dict.get("slope").getInfo(), dtype=np.float32)
  lc_arr = np.array(data_dict.get("landcover").getInfo(), dtype=np.float32)
  prot_arr = np.array(data_dict.get("protected").getInfo(), dtype=np.float32)
  dem_arr = np.array(data_dict.get("dem").getInfo(), dtype=np.float32)

  return slope_arr, lc_arr, prot_arr, dem_arr


with st.spinner("Retrieving satellite and environmental layers from GEE..."):
  slope_arr, lc_arr, prot_arr, dem_arr = fetch_base_rasters()


# -----------------------------------------------------------------------------
# 4. DIJKSTRA LEAST-COST PATH SOLVER
# -----------------------------------------------------------------------------
def solve_least_cost_path(cost_matrix):
  """Computes shortest path across the dynamic cost matrix using scipy.sparse graph."""
  rows, cols = cost_matrix.shape
  n_pixels = rows * cols

  def to_index(r, c):
    return r * cols + c

  row_idx, col_idx, weights = [], [], []

  # 8-neighbor connectivity vectors
  directions = [
      (-1, 0, 1.0),
      (1, 0, 1.0),
      (0, -1, 1.0),
      (0, 1, 1.0),
      (-1, -1, 1.414),
      (-1, 1, 1.414),
      (1, -1, 1.414),
      (1, 1, 1.414),
  ]

  for r in range(rows):
    for c in range(cols):
      u = to_index(r, c)
      c_u = cost_matrix[r, c]
      for dr, dc, dist in directions:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
          v = to_index(nr, nc)
          c_v = cost_matrix[nr, nc]
          weight = ((c_u + c_v) / 2.0) * dist
          row_idx.append(u)
          col_idx.append(v)
          weights.append(weight)

  graph = sp.csr_matrix(
      (weights, (row_idx, col_idx)), shape=(n_pixels, n_pixels)
  )

  # Map start/end node positions relative to the raster extent
  start_r, start_c = int(rows * 0.4), int(cols * 0.1)
  end_r, end_c = int(rows * 0.45), int(cols * 0.9)

  start_idx = to_index(start_r, start_c)
  end_idx = to_index(end_r, end_c)

  # Compute shortest path tree
  dist_matrix, predecessors = dijkstra(
      csgraph=graph, directed=False, indices=start_idx, return_predecessors=True
  )

  # Backtrack route nodes
  path = []
  curr = end_idx
  while curr != -9999 and curr != start_idx:
    path.append(curr)
    curr = predecessors[curr]
  if curr == start_idx:
    path.append(start_idx)
  path.reverse()

  # Project node indices back to Lat/Lon coordinates
  lons = np.linspace(35.0, 41.5, cols)
  lats = np.linspace(3.5, 1.0, rows)

  route_coords = []
  elevation_profile = []
  for idx in path:
    r = idx // cols
    c = idx % cols
    route_coords.append([lats[r], lons[c]])
    elevation_profile.append(dem_arr[r, c])

  return route_coords, elevation_profile


# -----------------------------------------------------------------------------
# 5. STREAMLIT USER INTERFACE
# -----------------------------------------------------------------------------
st.title("🛢️ LAPSSET Pipeline Route Optimizer")
st.caption(
    "Interactive least-cost path routing across Turkana to Lamu powered by"
    " Google Earth Engine."
)

# Sidebar Parameters
st.sidebar.header("🎛️ Cost Surface Parameters")
w_slope = st.sidebar.slider("Terrain Slope Weight", 0.0, 5.0, 1.5, 0.1)
w_land = st.sidebar.slider("Land Cover Avoidance", 0.0, 5.0, 2.0, 0.1)
w_prot = st.sidebar.slider("Protected Area Penalty", 0.0, 10.0, 8.0, 0.5)

# Calculate cost surface dynamically
cost_surface = (
    (slope_arr * w_slope) + (lc_arr * w_land) + (prot_arr * w_prot) + 1.0
)

# Compute path
route_coords, elevation_profile = solve_least_cost_path(cost_surface)

# Layout
col1, col2 = st.columns([3, 1])

with col1:
  m = folium.Map(location=[2.3, 38.0], zoom_start=7, tiles="OpenStreetMap")

  # Optimized Polyline
  folium.PolyLine(
      route_coords,
      color="#D9381E",
      weight=4,
      opacity=0.85,
      popup="Optimized LAPSSET Route",
  ).add_to(m)

  # Start/End Markers
  folium.Marker(
      START_COORDS,
      popup="Lokichar Oil Fields",
      icon=folium.Icon(color="green", icon="play"),
  ).add_to(m)
  folium.Marker(
      END_COORDS,
      popup="Mokowe Terminal (Lamu)",
      icon=folium.Icon(color="red", icon="stop"),
  ).add_to(m)

  st_folium(m, width="100%", height=500)

with col2:
  st.metric("Total Route Nodes", len(route_coords))
  st.metric("Grid Resolution", f"{SCALE} meters")
  st.info(
      "Adjust the sliders in the sidebar to re-route around steep terrain or"
      " protected wildlife reserves."
  )

# Elevation Profile Chart
st.subheader("⛰️ Elevation Profile along Optimized Route")
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        y=elevation_profile,
        mode="lines",
        fill="tozeroy",
        name="Elevation (m)",
        line=dict(color="#2E8B57", width=2),
    )
)
fig.update_layout(
    xaxis_title="Route Sequence (West to East)",
    yaxis_title="Elevation (meters)",
    height=260,
    margin=dict(l=20, r=20, t=20, b=20),
)
st.plotly_chart(fig, use_container_width=True)
