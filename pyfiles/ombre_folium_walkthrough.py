"""
ombre_folium_walkthrough.py

Geoplotting with Folium: a layered walkthrough.
Beginner → Intermediate → Advanced → Expert, all in one file, with a pastel ombré aesthetic.
I explain why I do each step and offer multiple approaches so you can compare and grow.

INSTALL:
    pip install folium

RUN:
    python ombre_folium_walkthrough_cazzy.py
    Then open the generated HTML files in your browser.

FILES PRODUCED:
    01_basics_pastel_map.html
    02_markers_popups_pastel_map.html
    03_choropleth_builtin_pastel_map.html
    04_choropleth_custom_pastel_map.html
    05_advanced_layers_controls_pastel_map.html

DATA NOTE:
    For choropleths, I show two paths:
      - A self-contained toy GeoJSON (runs out-of-the-box).
      - An optional path to your own GeoJSON + CSV/JSON lookups (if provided).

CITATIONS (documentation):
  - Folium user guide: https://python-visualization.github.io/folium/
  - Folium GeoJSON layer: https://python-visualization.github.io/folium/latest/user_guide/vector_layers/geojson.html
  - Folium Choropleth: https://python-visualization.github.io/folium/latest/user_guide/choropleth.html
  - Leaflet (under the hood): https://leafletjs.com/
  - branca colormaps: https://python-visualization.github.io/branca/colormap.html

Author: Cazzy Aporbo
"""

# ====== Standard Library (I keep it light and explicit) ======
import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# ====== Third-Party (one core library + its built-ins) ======
import folium
from folium import Map, Marker, Circle, GeoJson, Choropleth
from folium.features import DivIcon
from folium.plugins import MarkerCluster, MiniMap, Fullscreen, MeasureControl, Draw, FloatImage, MousePosition
from branca.colormap import LinearColormap


# -----------------------------------------------------------------------------
# 0) Shared pastel ombré palette + small helpers
# -----------------------------------------------------------------------------
def ombre_pastel_palette_cazzy() -> Dict[str, str]:
    """
    My project palette (soft pastels). I keep hexes human-readable and on-brand.
    """
    return {
        "ombre_pink": "#FFD6E8",
        "ombre_lavender": "#E6CCFF",
        "ombre_mint": "#D4FFE4",
        "ombre_teal": "#C6FFF7",
        "ombre_blue": "#A6D8FF",
        "ombre_purple_text": "#6B5B95",
        "ombre_gray_text": "#444444",
        "ombre_border_soft": "#EADFF7",
        "marker_purple": "purple",       # Folium named color for icons
        "marker_green": "green",
        "marker_blue": "blue",
        "marker_red": "red"
    }


def ombre_linear_colormap_cazzy() -> LinearColormap:
    """
    A left-to-right ombré for scalar data (pastel sweep). I set a caption later.
    """
    shades: List[str] = ["#FFD6E8", "#E6CCFF", "#C6FFF7", "#A6D8FF"]
    cmap = LinearColormap(colors=shades, vmin=0.0, vmax=1.0)
    return cmap


def write_html_and_print_path_cazzy(the_map: folium.Map, filename: str) -> None:
    """
    Write an HTML file and tell the user where it went. Simple and explicit.
    """
    the_map.save(outfile=filename)
    print(f"[saved] {filename}")


# -----------------------------------------------------------------------------
# 1) Beginner: a first, friendly map (center, zoom, tiles, labels)
# -----------------------------------------------------------------------------
def build_simple_intro_map_cazzy() -> None:
    """
    I start with the simplest mental model: a map is a canvas.
    I choose a pleasant center (San Francisco) and a readable tile set.
    I also add a basic title label that matches the pastel palette.
    """
    palette = ombre_pastel_palette_cazzy()

    # Create a soft base map (CartoDB Positron is a nice neutral background).
    # Docs: https://python-visualization.github.io/folium/latest/user_guide/quickstart.html
    map_canvas_intro_cazzy: Map = Map(
        location=[37.7749, -122.4194],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True
    )

    # Add a top-centered title using a DivIcon. This is my quick "label" pattern.
    title_html_divicon_cazzy = DivIcon(
        icon_size=(250, 36),
        icon_anchor=(0, 0),
        html=(
            f'<div style="'
            f'background:{palette["ombre_pink"]};'
            f'border:2px solid {palette["ombre_border_soft"]};'
            f'padding:6px 12px;'
            f'border-radius:10px;'
            f'font-weight:600;'
            f'color:{palette["ombre_purple_text"]};'
            f'font-family:Helvetica, Arial, sans-serif;'
            f'box-shadow:0 1px 4px rgba(0,0,0,0.08);'
            f'">Geoplotting Basics – Pastel Ombré</div>'
        )
    )
    Marker(location=[37.8049, -122.4194], icon=title_html_divicon_cazzy).add_to(map_canvas_intro_cazzy)

    # Add a quick click-to-get-latlng helper.
    # Docs: https://python-visualization.github.io/folium/latest/user_guide/quickstart.html#latitude-and-longitude-popup
    map_canvas_intro_cazzy.add_child(folium.LatLngPopup())

    write_html_and_print_path_cazzy(map_canvas_intro_cazzy, "01_basics_pastel_map.html")


# -----------------------------------------------------------------------------
# 2) Beginner → Intermediate: markers, tooltips, popups, circles
# -----------------------------------------------------------------------------
def build_markers_and_popups_map_cazzy() -> None:
    """
    Now I layer points, tooltips, and popups. I also use Circle for numeric radius.
    The intent: show different annotation types while keeping the style consistent.
    """
    palette = ombre_pastel_palette_cazzy()

    map_canvas_markers_cazzy: Map = Map(
        location=[37.7749, -122.4194],
        zoom_start=12,
        tiles="CartoDB positron"
    )

    # A small set of interesting points with varying styles.
    # I keep names unique and transparent.
    points_of_interest_cazzy: List[Tuple[str, float, float]] = [
        ("City Hall", 37.7793, -122.4193),
        ("Golden Gate Park", 37.7694, -122.4862),
        ("Ferry Building", 37.7955, -122.3937),
        ("Mission Dolores Park", 37.7596, -122.4269),
    ]

    for idx_marker_cazzy, (label_cazzy, lat_cazzy, lon_cazzy) in enumerate(points_of_interest_cazzy, start=1):
        ic = folium.Icon(color=["purple", "green", "blue", "red"][idx_marker_cazzy % 4], icon="info-sign")
        Marker(
            location=[lat_cazzy, lon_cazzy],
            tooltip=f"{label_cazzy}",
            popup=folium.Popup(
                html=(
                    f"<b>{label_cazzy}</b><br>"
                    f"<i>Lat:</i> {lat_cazzy:.4f}, <i>Lon:</i> {lon_cazzy:.4f}<br>"
                    f"<span style='color:{palette['ombre_gray_text']}'>A pastel popup with practical details.</span>"
                ),
                max_width=250
            ),
            icon=ic
        ).add_to(map_canvas_markers_cazzy)

    # Circle markers to illustrate numeric radius and fill aesthetics.
    Circle(
        location=[37.7749, -122.4194],
        radius=300,
        color=palette["ombre_lavender"],
        fill=True,
        fill_color=palette["ombre_mint"],
        fill_opacity=0.5,
        weight=2,
        tooltip="Illustrative circle (300m radius)"
    ).add_to(map_canvas_markers_cazzy)

    write_html_and_print_path_cazzy(map_canvas_markers_cazzy, "02_markers_popups_pastel_map.html")


# -----------------------------------------------------------------------------
# 3) Intermediate: choropleth with built-in Folium.Choropleth
# -----------------------------------------------------------------------------
def build_builtin_choropleth_map_cazzy(
    optional_geojson_path_cazzy: str = "",
    optional_csv_path_cazzy: str = ""
) -> None:
    """
    I demonstrate Folium's built-in Choropleth layer.
    If no files are provided, I use a tiny self-contained GeoJSON (three squares)
    plus a minimal CSV-like dict to map names → values.

    This gives you the shape of the solution without extra downloads.
    """
    palette = ombre_pastel_palette_cazzy()
    base_map_choro_cazzy: Map = Map(
        location=[37.0, -95.0],  # rough center for a generic, continental view
        zoom_start=4,
        tiles="CartoDB positron"
    )

    # Option A: real files if provided
    geojson_data_to_use_cazzy: Dict[str, Any]
    key_name_field_cazzy: str = "name"
    data_rows_for_choro_cazzy: List[Tuple[str, float]] = []

    if optional_geojson_path_cazzy and Path(optional_geojson_path_cazzy).exists():
        with open(optional_geojson_path_cazzy, "r", encoding="utf-8") as f_in:
            geojson_data_to_use_cazzy = json.load(f_in)
        # If CSV provided, read it (State, Value)
        if optional_csv_path_cazzy and Path(optional_csv_path_cazzy).exists():
            with open(optional_csv_path_cazzy, newline="", encoding="utf-8") as f_csv:
                reader = csv.DictReader(f_csv)
                for row in reader:
                    state_key = row.get("State") or row.get("name") or row.get("region")
                    val_key = row.get("Unemployment Rate") or row.get("value")
                    if state_key and val_key:
                        try:
                            data_rows_for_choro_cazzy.append((state_key, float(val_key)))
                        except ValueError:
                            pass
        else:
            # If no CSV, derive a trivial uniform mapping so the map still renders.
            # This is only a placeholder and not a real dataset.
            features = geojson_data_to_use_cazzy.get("features", [])
            for idx_uniform, feat in enumerate(features, start=1):
                nm = feat.get("properties", {}).get(key_name_field_cazzy, f"region_{idx_uniform}")
                data_rows_for_choro_cazzy.append((nm, float(idx_uniform % 10) / 10.0))
    else:
        # Option B: tiny toy GeoJSON (three rectangles).
        # Each polygon has a "name" matching the keys below.
        geojson_data_to_use_cazzy = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Alpha"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-101.0, 39.0], [-97.0, 39.0], [-97.0, 41.0], [-101.0, 41.0], [-101.0, 39.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Beta"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-96.0, 38.0], [-92.0, 38.0], [-92.0, 40.0], [-96.0, 40.0], [-96.0, 38.0]]]
                    }
                },
                {
                    "type": "Feature",
                    "properties": {"name": "Gamma"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[-95.0, 37.0], [-91.0, 37.0], [-91.0, 39.0], [-95.0, 39.0], [-95.0, 37.0]]]
                    }
                }
            ]
        }
        # Minimal values for demonstration (e.g., unemployment rate, scaled 0-1)
        data_rows_for_choro_cazzy = [
            ("Alpha", 0.2),
            ("Beta", 0.6),
            ("Gamma", 0.85)
        ]

    # Build a pastel ombré colormap.
    ombre_cmap_for_choro_cazzy = ombre_linear_colormap_cazzy()
    ombre_cmap_for_choro_cazzy.caption = "Pastel Ombré Scale (0 → 1)"

    # Use Folium's built-in Choropleth.
    # Docs: https://python-visualization.github.io/folium/latest/user_guide/choropleth.html
    Choropleth(
        geo_data=geojson_data_to_use_cazzy,
        name="choropleth_builtin_layer",
        data=data_rows_for_choro_cazzy,
        columns=[0, 1],  # tuple (name, value)
        key_on=f"feature.properties.{key_name_field_cazzy}",
        fill_color="PuRd",       # brewer scheme (purple-red) that reads pastel on Positron
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color="#f0f0f0",
        legend_name="Scalar Value (demo)"
    ).add_to(base_map_choro_cazzy)

    # Add my custom colormap as a legend-like control too.
    ombre_cmap_for_choro_cazzy.add_to(base_map_choro_cazzy)

    folium.LayerControl(collapsed=False).add_to(base_map_choro_cazzy)
    write_html_and_print_path_cazzy(base_map_choro_cazzy, "03_choropleth_builtin_pastel_map.html")


# -----------------------------------------------------------------------------
# 4) Intermediate → Advanced: a custom choropleth with GeoJson + style_function
# -----------------------------------------------------------------------------
def build_custom_choropleth_map_cazzy() -> None:
    """
    Here I avoid pandas entirely and style polygons myself with a function.
    This teaches the mental model: value lookup → color → style dict.
    """
    palette = ombre_pastel_palette_cazzy()
    map_canvas_custom_cazzy: Map = Map(location=[38.0, -96.0], zoom_start=5, tiles="CartoDB positron")

    # Reuse the toy regions so this runs out-of-the-box. Same shapes as above.
    toy_regions_geojson_cazzy = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Alpha"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-101.0, 39.0], [-97.0, 39.0], [-97.0, 41.0], [-101.0, 41.0], [-101.0, 39.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Beta"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-96.0, 38.0], [-92.0, 38.0], [-92.0, 40.0], [-96.0, 40.0], [-96.0, 38.0]]]
                }
            },
            {
                "type": "Feature",
                "properties": {"name": "Gamma"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[-95.0, 37.0], [-91.0, 37.0], [-91.0, 39.0], [-95.0, 39.0], [-95.0, 37.0]]]
                }
            }
        ]
    }

    # Here are the scalar values I'll visualize.
    toy_scalar_values_cazzy: Dict[str, float] = {"Alpha": 0.25, "Beta": 0.55, "Gamma": 0.9}

    # Build a pastel colormap and define how to style each polygon.
    pastel_scale_cazzy = ombre_linear_colormap_cazzy()
    pastel_scale_cazzy.caption = "Custom Pastel Ombré (0 → 1)"

    def style_function_cazzy(feature: Dict[str, Any]) -> Dict[str, Any]:
        region_name = feature.get("properties", {}).get("name", "")
        val = toy_scalar_values_cazzy.get(region_name, 0.0)
        color_fill = pastel_scale_cazzy(val)
        return {
            "fillColor": color_fill,
            "color": palette["ombre_border_soft"],
            "weight": 2,
            "fillOpacity": 0.75
        }

    def highlight_function_cazzy(feature: Dict[str, Any]) -> Dict[str, Any]:
        return {"weight": 3, "color": palette["ombre_purple_text"]}

    # Docs: https://python-visualization.github.io/folium/latest/user_guide/vector_layers/geojson.html
    GeoJson(
        data=toy_regions_geojson_cazzy,
        name="custom_choropleth_layer",
        style_function=style_function_cazzy,
        highlight_function=highlight_function_cazzy,
        tooltip=folium.GeoJsonTooltip(
            fields=["name"],
            aliases=["Region:"],
            localize=True,
            sticky=True
        )
    ).add_to(map_canvas_custom_cazzy)

    pastel_scale_cazzy.add_to(map_canvas_custom_cazzy)
    folium.LayerControl(collapsed=False).add_to(map_canvas_custom_cazzy)
    write_html_and_print_path_cazzy(map_canvas_custom_cazzy, "04_choropleth_custom_pastel_map.html")


# -----------------------------------------------------------------------------
# 5) Advanced → Expert: layered controls, drawing tools, measurement, clustering
# -----------------------------------------------------------------------------
def build_advanced_controls_map_cazzy() -> None:
    """
    At this level I combine multiple interaction patterns:
      - MarkerCluster for dense points
      - MiniMap for spatial bearings
      - Fullscreen toggle
      - MeasureControl for distance/area
      - Draw to sketch polygons/lines
      - MousePosition readout
    I also overlay a floating label to act like a title block.
    """
    palette = ombre_pastel_palette_cazzy()
    complex_map_cazzy: Map = Map(location=[39.5, -98.35], zoom_start=4, tiles="CartoDB positron")

    # A grid of synthetic points to make clustering obvious
    synthetic_points_cazzy: List[Tuple[float, float, str]] = []
    for lat_seed in [36.5, 37.0, 37.5, 38.0, 38.5]:
        for lon_seed in [-123.0, -122.5, -122.0, -121.5, -121.0]:
            synthetic_points_cazzy.append((lat_seed, lon_seed, f"Point {lat_seed:.1f},{lon_seed:.1f}"))

    cluster_layer_cazzy = MarkerCluster(name="Clustered Points")
    for lat_s, lon_s, label_s in synthetic_points_cazzy:
        Marker(location=[lat_s, lon_s], tooltip=label_s).add_to(cluster_layer_cazzy)
    cluster_layer_cazzy.add_to(complex_map_cazzy)

    # MiniMap: https://python-visualization.github.io/folium/latest/user_guide/plugins.html#minimap
    MiniMap(toggle_display=True, position="bottomleft").add_to(complex_map_cazzy)

    # Fullscreen: https://python-visualization.github.io/folium/latest/user_guide/plugins.html#fullscreen
    Fullscreen(position="topleft", title="Enter Fullscreen", title_cancel="Exit Fullscreen").add_to(complex_map_cazzy)

    # MeasureControl: https://python-visualization.github.io/folium/latest/user_guide/plugins.html#measurecontrol
    MeasureControl(primary_length_unit="meters", secondary_length_unit="miles").add_to(complex_map_cazzy)

    # Draw: https://python-visualization.github.io/folium/latest/user_guide/plugins.html#draw
    Draw(export=True, filename="cazzy_drawn_shapes.geojson").add_to(complex_map_cazzy)

    # MousePosition readout for quick inspections
    MousePosition(position="bottomright", prefix="Lat/Lng", separator=" | ").add_to(complex_map_cazzy)

    # Floating title using Div overlay (keeps visuals labeled).
    folium.map.CustomPane("title_pane_cazzy").add_to(complex_map_cazzy)
    title_block_html_cazzy = folium.Element(
        f"""
        <div style="
            position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
            background: linear-gradient(90deg, {palette['ombre_pink']}, {palette['ombre_lavender']}, {palette['ombre_teal']}, {palette['ombre_blue']});
            color: {palette['ombre_gray_text']};
            border: 2px solid {palette['ombre_border_soft']};
            padding: 8px 14px; border-radius: 10px; font-weight: 600;
            font-family: Helvetica, Arial, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,0.08); z-index: 9999;">
            Advanced Controls – Pastel Ombré
        </div>
        """
    )
    complex_map_cazzy.get_root().html.add_child(title_block_html_cazzy)

    folium.LayerControl(collapsed=False).add_to(complex_map_cazzy)
    write_html_and_print_path_cazzy(complex_map_cazzy, "05_advanced_layers_controls_pastel_map.html")


# -----------------------------------------------------------------------------
# Main: I run each stage so you can open and compare the HTML outputs.
# -----------------------------------------------------------------------------
def main_cazzy():
    build_simple_intro_map_cazzy()
    build_markers_and_popups_map_cazzy()
    build_builtin_choropleth_map_cazzy()       # runs using embedded toy polygons
    build_custom_choropleth_map_cazzy()        # manual styling approach
    build_advanced_controls_map_cazzy()        # plugins + interactions


if __name__ == "__main__":
    main_cazzy()
