# 🛢️ LAPSSET Pipeline Route Optimizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Google Earth Engine](https://img.shields.io/badge/Data-Google%20Earth%20Engine-green.svg)](https://earthengine.google.com/)

An interactive web map that calculates the safest, most efficient pipeline path between the **Lokichar Oil Fields** in Turkana and the **Mokowe Refinery** in Lamu, Kenya.

Instead of drawing a simple straight line, this tool scans real-world terrain data in real time to route around steep mountains, protected wildlife areas, and difficult wetlands.

---

## 💡 What Does This Project Do?

Building major infrastructure like an oil pipeline requires balancing engineering costs with environmental protection. 

This app acts as a decision-support dashboard where users can adjust sliders to change priorities:
- **Avoid Steep Hills:** Prefers flatter terrain to keep construction costs down.
- **Protect Wildlife & Nature:** Steers the pipeline away from national parks and reserves.
- **Bypass Difficult Land:** Avoids wetlands and urban communities.

As you move the sliders on the screen, the algorithm recalculates the route live on an interactive map and shows the elevation profile along the journey.

---

## 🗺️ How It Works (The Simple Version)
1. **Streams Live Data:** It connects to satellite data in the cloud (so no massive map files need to be downloaded locally).

2. **Creates a "Cost Map":** It looks at slope, land cover, and protected areas, assigning a difficulty score to every square kilometer between Lokichar and Lamu.

3. **Finds the Best Path:** A pathfinding algorithm searches through millions of options to pick the path of least resistance.

4. **Displays Results:** Renders the calculated route on an interactive map alongside a graph showing how high or low the pipeline travels.