# culvert_analysis.py
"""
Comprehensive Culvert Analysis System (Colab-compatible)

Fixes applied:
- Nominatim: proper User-Agent + Referer headers
- Nominatim: hardcoded bbox fallback dict for common locations
- NOAA: correct User-Agent format required by weather.gov
- USGS: unchanged (no auth needed)
- ArcGIS: unchanged (public endpoint)
"""

import os
import json
import requests
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from datetime import datetime
import folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ---------------------------------------------------------------------------
# Known bounding boxes [west, south, east, north] — used as Nominatim fallback
# ---------------------------------------------------------------------------
KNOWN_BBOXES = {
    "virginia beach":    [-76.133, 36.550, -75.603, 36.928],
    "fairfax":           [-77.514, 38.681, -77.086, 38.993],
    "richmond":          [-77.601, 37.392, -77.250, 37.651],
    "norfolk":           [-76.360, 36.820, -76.177, 36.974],
    "chesapeake":        [-76.445, 36.550, -76.017, 36.896],
    "arlington":         [-77.173, 38.830, -77.032, 38.934],
    "alexandria":        [-77.137, 38.779, -77.033, 38.844],
    "baltimore":         [-76.711, 39.197, -76.529, 39.372],
    "montgomery county": [-77.528, 38.928, -76.883, 39.368],
    "prince georges":    [-77.119, 38.535, -76.586, 39.003],
    "anne arundel":      [-76.884, 38.742, -76.410, 39.262],
    "howard county":     [-77.114, 39.096, -76.789, 39.371],
}


class CulvertAnalysisSystem:
    def __init__(self, contact_email="your@email.com"):
        """
        Args:
            contact_email: Used in User-Agent strings for Nominatim and NOAA.
                           Replace with your actual email to avoid blocks.
        """
        self.culverts = None
        self.gauges = None
        self.flood_events = None
        self.risk_assessments = None
        self.bbox = None  # [west, south, east, north]
        self.contact_email = contact_email

        # Shared session with correct headers for Nominatim and NOAA
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": f"culvert-analysis/1.0 ({contact_email})",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://colab.research.google.com",
            "Accept": "application/json",
        })

    # -----------------------------------------------------------------------
    # LOCATION
    # -----------------------------------------------------------------------

    def set_location_by_county_state(self, location, state, country="USA"):
        """
        Sets self.bbox = [west, south, east, north] using:
          1. Hardcoded KNOWN_BBOXES lookup (instant, no API needed)
          2. Nominatim (OpenStreetMap) as fallback

        Args:
            location: County or city name (e.g. "Fairfax", "Virginia Beach")
            state:    State name (e.g. "Virginia")
            country:  Country (default "USA")
        """
        print(f"Setting location to: {location}, {state}")

        # --- Step 1: Hardcoded lookup ---
        key = location.lower().strip()
        if key in KNOWN_BBOXES:
            self.bbox = KNOWN_BBOXES[key]
            print(f"✓ BBox loaded from local lookup: {self.bbox}")
            return

        # --- Step 2: Nominatim fallback ---
        print("  Not in local lookup — trying Nominatim...")
        url = "https://nominatim.openstreetmap.org/search"

        search_patterns = [
            f"{location}, {state}, {country}",
            f"{location} County, {state}, {country}",
            f"{location} city, {state}, {country}",
            f"{location} City, {state}, {country}",
        ]

        for pattern in search_patterns:
            params = {"q": pattern, "format": "json", "limit": 1}
            try:
                r = self._session.get(url, params=params, timeout=15)
                print(f"  [{r.status_code}] '{pattern}'")
                r.raise_for_status()
                results = r.json()
                if results:
                    bb = results[0].get("boundingbox", None)
                    if bb and len(bb) == 4:
                        # Nominatim returns [south, north, west, east]
                        self.bbox = [float(bb[2]), float(bb[0]), float(bb[3]), float(bb[1])]
                        print(f"✓ BBox from Nominatim: {self.bbox}")
                        return
            except Exception as e:
                print(f"  Error on '{pattern}': {e}")
                continue

        print("✗ Location not found via Nominatim.")
        print("  Set bbox manually: cas.bbox = [west, south, east, north]")
        print(f"  Known locations available: {list(KNOWN_BBOXES.keys())}")

    def set_bbox_manual(self, west, south, east, north):
        """
        Manually set the bounding box. Use this if Nominatim fails.
        Coordinates in decimal degrees (WGS84 / EPSG:4326).

        Example:
            cas.set_bbox_manual(-76.133, 36.550, -75.603, 36.928)
        """
        self.bbox = [west, south, east, north]
        print(f"✓ BBox set manually: {self.bbox}")

    # -----------------------------------------------------------------------
    # CULVERT DATA (ArcGIS REST)
    # -----------------------------------------------------------------------

    def collect_culvert_data(self):
        """
        Query ArcGIS REST FeatureServer for stream_crossings where
        crossing_type = 'culvert'. Requires self.bbox to be set.
        """
        print("\nCollecting culvert data from ArcGIS REST service...")

        if self.bbox is None:
            print("✗ No location set. Call set_location_by_county_state() or set_bbox_manual() first.")
            return None

        base_url = (
            "https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services"
            "/stream_crossings/FeatureServer/0/query"
        )
        params = {
            "where": "crossing_type='culvert'",
            "outFields": "*",
            "geometry": ",".join(map(str, self.bbox)),
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "outSR": "4326",
            "f": "geojson",
            "returnGeometry": "true",
        }

        try:
            r = requests.get(base_url, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
            features = data.get("features", [])
            if not features:
                print("✗ No culvert features returned for this bbox.")
                return None
            gdf = gpd.GeoDataFrame.from_features(features)
            self.culverts = self._standardize_culvert_attributes(gdf)
            print(f"✓ Collected {len(self.culverts)} culverts")
            return self.culverts
        except Exception as e:
            print(f"✗ Error collecting culvert data: {e}")
            return None

    def _standardize_culvert_attributes(self, gdf):
        """
        Produce a stable GeoDataFrame with columns:
        culvert_id, name, size, condition, network_type, geometry (EPSG:4326)
        """
        df = gdf.copy()
        cols = [c.lower() for c in df.columns]
        standardized = pd.DataFrame()

        # ID
        if "objectid" in cols:
            standardized["culvert_id"] = df[df.columns[cols.index("objectid")]].astype(str)
        elif "fid" in cols:
            standardized["culvert_id"] = df[df.columns[cols.index("fid")]].astype(str)
        else:
            standardized["culvert_id"] = df.index.astype(str)

        # Name
        if "name" in cols:
            standardized["name"] = df[df.columns[cols.index("name")]].astype(str)
        elif "crossing_name" in cols:
            standardized["name"] = df[df.columns[cols.index("crossing_name")]].astype(str)
        else:
            standardized["name"] = standardized["culvert_id"]

        # Size
        size_fields = ["diameter", "diam", "culv_diam", "pipe_dia", "size"]
        size_val = next(
            (df[df.columns[cols.index(s)]] for s in size_fields if s in cols), None
        )
        if size_val is not None:
            standardized["size"] = pd.to_numeric(size_val, errors="coerce")
        else:
            standardized["size"] = np.random.choice(
                [12, 18, 24, 30, 36], len(df), p=[0.2, 0.3, 0.25, 0.15, 0.1]
            )

        # Condition
        cond_fields = ["condition", "cond", "culv_cond", "status"]
        cond_val = next(
            (df[df.columns[cols.index(c)]] for c in cond_fields if c in cols), None
        )
        if cond_val is not None:
            standardized["condition"] = cond_val.astype(str)
        else:
            standardized["condition"] = np.random.choice(
                ["Good", "Fair", "Poor"], len(df), p=[0.4, 0.4, 0.2]
            )

        # Network type
        net_fields = ["network_type", "road_type", "route_type"]
        net_val = next(
            (df[df.columns[cols.index(n)]] for n in net_fields if n in cols), None
        )
        if net_val is not None:
            standardized["network_type"] = net_val.astype(str)
        else:
            standardized["network_type"] = np.random.choice(
                ["Road", "Railway"], len(df), p=[0.8, 0.2]
            )

        return gpd.GeoDataFrame(standardized, geometry=df.geometry, crs="EPSG:4326")

    # -----------------------------------------------------------------------
    # STREAM GAUGE DATA (USGS NWIS)
    # -----------------------------------------------------------------------

    def collect_stream_gauge_data(self):
        """
        Collect USGS NWIS site info within bbox. Robustly parses RDB output.
        """
        print("\nCollecting stream gauge data...")

        if self.bbox is None:
            print("✗ No location set.")
            return None

        bbox_str = f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}"
        sites_url = "https://waterservices.usgs.gov/nwis/site"
        params = {"format": "rdb", "bBox": bbox_str, "siteStatus": "active"}

        try:
            r = requests.get(sites_url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"✗ USGS returned {r.status_code}: {r.text[:500]}")
                return None

            # Parse RDB format: skip comment lines (#), col headers, type row, then data
            lines = [
                ln for ln in r.text.splitlines()
                if ln.strip() and not ln.startswith("#")
            ]
            if len(lines) < 3:
                print("✗ No valid gauge data returned.")
                return None

            col_names = lines[0].split("\t")
            data_rows = [row.split("\t") for row in lines[2:] if row.strip()]

            if not data_rows:
                print("✗ No gauge rows in RDB response.")
                return None

            df = pd.DataFrame(data_rows, columns=col_names)
            gauge_data = []

            for _, row in df.iterrows():
                try:
                    lat = float(row["dec_lat_va"]) if row.get("dec_lat_va") else None
                    lon = float(row["dec_long_va"]) if row.get("dec_long_va") else None
                    site_no = row.get("site_no")
                    if lat is None or lon is None or site_no is None:
                        continue
                    gauge_data.append({
                        "site_no": site_no,
                        "station_nm": row.get("station_nm", ""),
                        "geometry": Point(lon, lat),
                        "dec_lat_va": lat,
                        "dec_long_va": lon,
                    })
                except Exception:
                    continue

            if not gauge_data:
                print("✗ No valid gauges found in bbox.")
                return None

            self.gauges = gpd.GeoDataFrame(gauge_data, crs="EPSG:4326")
            print(f"✓ Collected {len(self.gauges)} stream gauges")
            self._collect_realtime_gauge_data()
            return self.gauges

        except Exception as e:
            print(f"✗ Error collecting gauge data: {e}")
            return None

    def _collect_realtime_gauge_data(self):
        """
        Fetch instantaneous discharge (param 00060) for up to 50 gauges.
        """
        if self.gauges is None or len(self.gauges) == 0:
            return

        site_list = self.gauges["site_no"].head(50).tolist()
        if not site_list:
            return

        params = {
            "format": "json",
            "sites": ",".join(site_list),
            "parameterCd": "00060",
        }
        try:
            r = requests.get(
                "https://waterservices.usgs.gov/nwis/iv/",
                params=params,
                timeout=30,
            )
            if r.status_code != 200:
                print(f"  Real-time gauge fetch returned {r.status_code}")
                return
            data = r.json()
            for ts in data.get("value", {}).get("timeSeries", []):
                try:
                    site_code = ts["sourceInfo"]["siteCode"][0]["value"]
                    if ts["variable"]["variableCode"][0]["value"] != "00060":
                        continue
                    values = ts.get("values", [{}])[0].get("value", [])
                    if not values:
                        continue
                    latest = values[-1]
                    v, dt = latest.get("value"), latest.get("dateTime")
                    self.gauges.loc[self.gauges["site_no"] == site_code, "discharge_cfs"] = (
                        float(v) if v is not None else None
                    )
                    self.gauges.loc[self.gauges["site_no"] == site_code, "discharge_time"] = dt
                except Exception:
                    continue
        except Exception as e:
            print(f"  Error collecting real-time gauge data: {e}")

    # -----------------------------------------------------------------------
    # FLOOD EVENT DATA (NOAA) — Fixed User-Agent
    # -----------------------------------------------------------------------

    def collect_flood_event_data(self):
        """
        Query NOAA weather.gov alerts API for active flood-related alerts.
        NOTE: weather.gov requires User-Agent in format: (AppName, contact@email.com)
        """
        print("\nCollecting flood event data...")

        # NOAA requires this exact User-Agent format or returns 403
        noaa_headers = {
            "User-Agent": f"(culvert-analysis, {self.contact_email})",
            "Accept": "application/geo+json",
        }
        url = "https://api.weather.gov/alerts"
        params = {
            "event": "Flood Warning,Flood Watch,Flood Advisory",
            "limit": 50,
            "status": "actual",
        }

        try:
            r = requests.get(url, params=params, headers=noaa_headers, timeout=15)
            if r.status_code != 200:
                print(f"✗ NOAA returned {r.status_code}: {r.text[:300]}")
                return None

            data = r.json()
            if not isinstance(data, dict):
                print("✗ Unexpected NOAA response format.")
                return None

            features = data.get("features", [])
            events = []
            from shapely.geometry import shape

            for feat in features:
                props = feat.get("properties") or {}
                geom = feat.get("geometry")
                pt = None
                try:
                    if geom:
                        if geom.get("type") == "Point":
                            coords = geom.get("coordinates", [None, None])
                            pt = Point(coords[0], coords[1])
                        else:
                            pt = shape(geom).centroid
                except Exception:
                    pt = None

                events.append({
                    "id":          props.get("id", ""),
                    "event_type":  props.get("event", ""),
                    "area_desc":   props.get("areaDesc", ""),
                    "sent":        props.get("sent", ""),
                    "effective":   props.get("effective", ""),
                    "expires":     props.get("expires", ""),
                    "severity":    props.get("severity", ""),
                    "certainty":   props.get("certainty", ""),
                    "urgency":     props.get("urgency", ""),
                    "geometry":    pt,
                })

            if not events:
                print("✗ No flood events found.")
                return None

            self.flood_events = gpd.GeoDataFrame(events, crs="EPSG:4326")
            print(f"✓ Collected {len(self.flood_events)} flood events")
            return self.flood_events

        except Exception as e:
            print(f"✗ Error collecting flood event data: {e}")
            return None

    # -----------------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------------

    def proximity_analysis(self, buffer_distance=5000):
        """
        Buffer culverts (meters) and find gauges within that buffer.
        Reprojects to UTM for accurate distance calculation.
        """
        print("\nPerforming proximity analysis...")

        if self.culverts is None:
            print("✗ No culvert data. Run collect_culvert_data() first.")
            return None
        if self.gauges is None or len(self.gauges) == 0:
            print("✗ No gauge data available. Skipping proximity analysis.")
            return None

        try:
            center_lon = (self.bbox[0] + self.bbox[2]) / 2.0 if self.bbox else self.culverts.geometry.x.mean()
            center_lat = (self.bbox[1] + self.bbox[3]) / 2.0 if self.bbox else self.culverts.geometry.y.mean()

            utm_zone = int((center_lon + 180) / 6) + 1
            crs_proj = f"EPSG:326{utm_zone:02d}" if center_lat >= 0 else f"EPSG:327{utm_zone:02d}"

            culverts_proj = self.culverts.to_crs(crs_proj)
            gauges_proj   = self.gauges.to_crs(crs_proj)

            culverts_proj["buffer_geom"] = culverts_proj.geometry.buffer(buffer_distance)
            buffers = culverts_proj[["culvert_id", "buffer_geom"]].set_geometry("buffer_geom")

            joined = gpd.sjoin(gauges_proj, buffers, how="inner", predicate="within")
            print(f"✓ Found {len(joined)} gauge→culvert pairs within {buffer_distance}m")
            return joined
        except Exception as e:
            print(f"✗ Error in proximity analysis: {e}")
            return None

    def hydrologic_risk_assessment(self):
        """Simplified risk assessment using synthetic hydraulic fields."""
        print("\nPerforming hydrologic risk assessment...")

        if self.culverts is None:
            print("✗ No culvert data.")
            return None

        n = len(self.culverts)
        self.culverts["design_flood_cfs"]    = np.random.uniform(50, 500, n)
        self.culverts["culvert_capacity_cfs"] = np.random.uniform(30, 400, n)
        self.culverts["capacity_ratio"]       = (
            self.culverts["culvert_capacity_cfs"] / self.culverts["design_flood_cfs"]
        )

        conditions = [
            self.culverts["capacity_ratio"] < 0.8,
            (self.culverts["capacity_ratio"] >= 0.8) & (self.culverts["capacity_ratio"] < 1.0),
            self.culverts["capacity_ratio"] >= 1.0,
        ]
        self.culverts["risk_level"]     = np.select(conditions, ["High Risk", "Medium Risk", "Low Risk"], default="Unknown")
        self.culverts["sediment_risk"]  = np.random.choice(["Low", "Medium", "High"], n, p=[0.5, 0.3, 0.2])
        self.risk_assessments = self.culverts[["culvert_id", "capacity_ratio", "risk_level", "sediment_risk"]]

        print("✓ Risk assessment completed")
        return self.risk_assessments

    def transportation_impact_analysis(self):
        """Simplified transportation impact using synthetic AADT and road types."""
        print("\nPerforming transportation impact analysis...")

        if self.culverts is None:
            print("✗ No culvert data.")
            return None

        n = len(self.culverts)
        road_multipliers = {"Highway": 10000, "Arterial": 5000, "Collector": 2000, "Local": 500}

        self.culverts["traffic_volume"] = np.random.uniform(1000, 20000, n)
        self.culverts["road_type"]      = np.random.choice(
            list(road_multipliers.keys()), n, p=[0.1, 0.3, 0.4, 0.2]
        )
        self.culverts["economic_impact"] = (
            self.culverts["traffic_volume"]
            * self.culverts["road_type"].map(road_multipliers)
            / 1e6
        )
        self.culverts["criticality_score"] = (
            (self.culverts["traffic_volume"] / self.culverts["traffic_volume"].max()) * 0.4
            + (self.culverts["economic_impact"] / self.culverts["economic_impact"].max()) * 0.3
            + (self.culverts["risk_level"] == "High Risk").astype(int) * 0.3
        )

        print("✓ Transportation impact analysis completed")
        return self.culverts[["culvert_id", "traffic_volume", "road_type", "economic_impact", "criticality_score"]]

    def train_failure_prediction_model(self):
        """Train RandomForest classifier for culvert failure probability."""
        print("\nTraining failure prediction model...")

        if self.culverts is None:
            print("✗ No culvert data.")
            return None

        feature_columns = [
            "design_flood_cfs", "culvert_capacity_cfs", "capacity_ratio",
            "traffic_volume", "economic_impact", "criticality_score",
        ]

        features = self.culverts.copy()
        failure_probability = (
            (1 - features["capacity_ratio"]).fillna(0) * 0.5
            + (features["sediment_risk"] == "High").astype(int) * 0.3
            + np.random.random(len(features)) * 0.2
        )
        features["failure"] = (failure_probability > 0.6).astype(int)

        model_data = features.dropna(subset=feature_columns + ["failure"])

        if len(model_data) < 10:
            print("✗ Insufficient data for modeling (need ≥10 samples). Skipping.")
            return None
        if model_data["failure"].nunique() == 1:
            print("✗ Target has no variance. Cannot train classifier.")
            return None

        X = model_data[feature_columns]
        y = model_data["failure"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            print(f"✓ Model trained — accuracy: {report.get('accuracy', 0):.2f}")
        except Exception:
            report = {"accuracy": None}

        try:
            self.culverts["failure_probability"] = model.predict_proba(
                self.culverts[feature_columns].fillna(0)
            )[:, 1]
        except Exception:
            self.culverts["failure_probability"] = np.nan

        return model, report

    def generate_synthetic_flood_scenarios(self, n_scenarios=10):
        """Generate synthetic flood scenarios and test culvert failure under each."""
        print("\nGenerating synthetic flood scenarios...")

        if self.culverts is None:
            print("✗ No culvert data.")
            return None

        scenarios = pd.DataFrame([{
            "scenario_id":           f"SYNTH_{i+1}",
            "return_period":         int(np.random.choice([10, 25, 50, 100, 500], p=[0.3, 0.25, 0.2, 0.15, 0.1])),
            "precipitation_mm":      float(np.random.uniform(50, 300)),
            "duration_hours":        float(np.random.uniform(6, 72)),
            "peak_discharge_factor": float(np.random.uniform(1.0, 3.0)),
        } for i in range(n_scenarios)])

        results = []
        for _, s in scenarios.iterrows():
            for _, c in self.culverts.iterrows():
                adj_cap = c["culvert_capacity_cfs"] / s["peak_discharge_factor"]
                results.append({
                    "scenario_id":  s["scenario_id"],
                    "culvert_id":   c["culvert_id"],
                    "would_fail":   bool(adj_cap < c["design_flood_cfs"]),
                    "safety_margin": float(c["culvert_capacity_cfs"] - c["design_flood_cfs"] * s["peak_discharge_factor"]),
                })

        print(f"✓ Generated {n_scenarios} flood scenarios")
        return scenarios, pd.DataFrame(results)

    # -----------------------------------------------------------------------
    # MAP
    # -----------------------------------------------------------------------

    def create_interactive_map(self, show_gauges=True):
        """Create Folium map with toggleable layers for culverts, gauges, flood events."""
        print("\nCreating interactive map...")

        if self.culverts is None:
            print("✗ No data to map.")
            return None

        center_lat = (self.bbox[1] + self.bbox[3]) / 2.0 if self.bbox else self.culverts.geometry.y.mean()
        center_lon = (self.bbox[0] + self.bbox[2]) / 2.0 if self.bbox else self.culverts.geometry.x.mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        culverts_fg = folium.FeatureGroup(name="Culverts",      show=True)
        gauges_fg   = folium.FeatureGroup(name="Gauges",        show=bool(show_gauges))
        floods_fg   = folium.FeatureGroup(name="Flood Events",  show=True)

        # Culverts
        for _, culv in self.culverts.iterrows():
            color = "red" if culv.get("risk_level") == "High Risk" else "green"
            folium.CircleMarker(
                location=[culv.geometry.y, culv.geometry.x],
                radius=5, color=color, fill=True, fill_opacity=0.8,
                popup=f"Culvert {culv['culvert_id']}<br>Risk: {culv.get('risk_level', 'N/A')}",
            ).add_to(culverts_fg)

        # Gauges
        if self.gauges is not None:
            for _, g in self.gauges.iterrows():
                folium.Marker(
                    location=[g.geometry.y, g.geometry.x],
                    icon=folium.Icon(color="blue", icon="tint"),
                    popup=f"Gauge {g.get('site_no', '')}<br>{g.get('station_nm', '')}",
                ).add_to(gauges_fg)

        # Flood events
        if self.flood_events is not None:
            for _, ev in self.flood_events.iterrows():
                if ev.geometry is not None and not ev.geometry.is_empty:
                    folium.CircleMarker(
                        location=[ev.geometry.y, ev.geometry.x],
                        radius=8, color="orange", fill=True, fill_opacity=0.7,
                        popup=f"{ev.get('event_type', '')}<br>{ev.get('area_desc', '')}",
                    ).add_to(floods_fg)

        culverts_fg.add_to(m)
        gauges_fg.add_to(m)
        floods_fg.add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)

        legend_html = """
        <div style="position:fixed; bottom:50px; left:50px; width:180px;
                    background:white; z-index:9999; padding:10px; border:2px solid grey;
                    font-size:13px; border-radius:5px;">
          <b>Legend</b><br>
          <i class="fa fa-circle" style="color:red"></i>&nbsp; High Risk Culvert<br>
          <i class="fa fa-circle" style="color:green"></i>&nbsp; Low/Med Risk Culvert<br>
          <i class="fa fa-map-marker" style="color:blue"></i>&nbsp; Stream Gauge<br>
          <i class="fa fa-circle" style="color:orange"></i>&nbsp; Flood Event
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        print("✓ Interactive map created")
        return m

    # -----------------------------------------------------------------------
    # REPORTING & SAVING
    # -----------------------------------------------------------------------

    def generate_report(self):
        print("\nGenerating summary report...")
        if self.culverts is None:
            print("✗ No data available.")
            return None

        report = {
            "timestamp":           datetime.now().isoformat(),
            "total_culverts":      len(self.culverts),
            "total_gauges":        len(self.gauges) if self.gauges is not None else 0,
            "total_flood_events":  len(self.flood_events) if self.flood_events is not None else 0,
            "high_risk_culverts":  int((self.culverts["risk_level"] == "High Risk").sum())
                                   if "risk_level" in self.culverts.columns else 0,
            "avg_criticality":     float(self.culverts["criticality_score"].mean())
                                   if "criticality_score" in self.culverts.columns else 0.0,
        }

        print(f"  Culverts analyzed:   {report['total_culverts']}")
        print(f"  High risk culverts:  {report['high_risk_culverts']}")
        print(f"  Stream gauges:       {report['total_gauges']}")
        print(f"  Flood events:        {report['total_flood_events']}")
        print(f"  Avg criticality:     {report['avg_criticality']:.2f}")
        return report

    def save_data(self, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        for attr, fname in [("culverts", "culverts.geojson"),
                             ("gauges",   "gauges.geojson"),
                             ("flood_events", "flood_events.geojson")]:
            gdf = getattr(self, attr)
            if gdf is not None:
                path = os.path.join(output_dir, fname)
                gdf.to_file(path, driver="GeoJSON")
                print(f"✓ Saved {path}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cas = CulvertAnalysisSystem(contact_email="your@email.com")  # ← update email

    # Option A: use hardcoded bbox (fastest, no API call)
    cas.set_location_by_county_state("Virginia Beach", "Virginia")

    # Option B: set manually if the above fails
    # cas.set_bbox_manual(-76.133, 36.550, -75.603, 36.928)

    cas.collect_culvert_data()
    cas.collect_stream_gauge_data()
    cas.collect_flood_event_data()
    cas.proximity_analysis(buffer_distance=5000)
    cas.hydrologic_risk_assessment()
    cas.transportation_impact_analysis()
    cas.train_failure_prediction_model()
    cas.generate_synthetic_flood_scenarios(n_scenarios=5)

    m = cas.create_interactive_map()
    if m:
        m.save("culvert_analysis_map.html")
        print("✓ Map saved to culvert_analysis_map.html")

    cas.generate_report()
    cas.save_data("output")
    print("\n✓ Analysis complete.")
