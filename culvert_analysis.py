# culvert_analysis.py
"""
Comprehensive Culvert Analysis System (updated)

- Uses ArcGIS REST service for culvert data (crossing_type = 'culvert')
- Uses OpenStreetMap/Nominatim to get a county/state bounding box
- Robust USGS NWIS site RDB parsing (bBox major filter)
- Safer NOAA flood event parsing
- CRS-safe buffering / reprojection for proximity analysis
- Better error handling and clear messages
- Fixed location handling for both counties and independent cities
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

class CulvertAnalysisSystem:
    def __init__(self):
        self.culverts = None
        self.gauges = None
        self.flood_events = None
        self.risk_assessments = None
        self.bbox = None  # [west, south, east, north]

    def set_location_by_county_state(self, location, state, country="USA"):
        """
        Determine bounding box for "Location, State" using Nominatim (OpenStreetMap).
        Works for both counties and independent cities.
        Sets self.bbox = [west, south, east, north]
        
        Args:
            location: County name (e.g., "Fairfax") or city name (e.g., "Virginia Beach")
            state: State name (e.g., "Virginia")
            country: Country name (default "USA")
        """
        print(f"Setting location to {location}, {state}")
        
        # Use Nominatim search - try multiple search patterns for better coverage
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "culvert-analysis/1.0 (email@example.com)"}  # change contact if desired
        
        # Try different search patterns in order of preference
        search_patterns = [
            f"{location}, {state}, {country}",  # General pattern (works for both counties and cities)
            f"{location} County, {state}, {country}",  # Explicit county
            f"{location} city, {state}, {country}",  # Explicit city
            f"{location} City, {state}, {country}",  # Capitalized city
        ]
        
        for pattern in search_patterns:
            params = {"q": pattern, "format": "json", "limit": 1}
            
            try:
                r = requests.get(url, params=params, headers=headers, timeout=15)
                r.raise_for_status()
                results = r.json()
                
                if results:
                    bb = results[0].get("boundingbox", None)
                    if bb and len(bb) == 4:
                        # Nominatim boundingbox format: [south_lat, north_lat, west_lon, east_lon]
                        south_lat = float(bb[0])
                        north_lat = float(bb[1])
                        west_lon = float(bb[2])
                        east_lon = float(bb[3])
                        self.bbox = [west_lon, south_lat, east_lon, north_lat]
                        print(f"Found location using pattern: '{pattern}'")
                        print(f"Bounding box set to: {self.bbox}")
                        return
                        
            except Exception as e:
                print(f"Error trying pattern '{pattern}': {e}")
                continue
        
        print("Location not found via Nominatim with any search pattern.")
        print("Please provide bbox manually or check location/state names.")

    def collect_culvert_data(self):
        """
        Query ArcGIS REST FeatureServer for stream_crossings where crossing_type = 'culvert'
        Requires self.bbox to be set.
        """
        print("Collecting culvert data from ArcGIS REST service...")

        if self.bbox is None:
            print("No location set. Please call set_location_by_county_state(location, state) first.")
            return None

        base_url = "https://services.arcgis.com/v01gqwM5QqNysAAi/arcgis/rest/services/stream_crossings/FeatureServer/0/query"
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
                print("No culvert features returned for the bbox.")
                return None
            gdf = gpd.GeoDataFrame.from_features(features)
            # standardize
            self.culverts = self._standardize_culvert_attributes(gdf)
            print(f"Collected {len(self.culverts)} culverts from ArcGIS service")
            return self.culverts
        except Exception as e:
            print(f"Error collecting culvert data: {e}")
            return None

    def _standardize_culvert_attributes(self, gdf):
        """
        Make a stable minimal culvert GeoDataFrame with columns:
        culvert_id, name, size, condition, network_type, geometry (CRS EPSG:4326)
        """
        df = gdf.copy()
        standardized = pd.DataFrame()

        cols = [c.lower() for c in df.columns]

        # ID field - try several possibilities
        if "objectid" in cols:
            standardized["culvert_id"] = df[df.columns[cols.index("objectid")]].astype(str)
        elif "fid" in cols:
            standardized["culvert_id"] = df[df.columns[cols.index("fid")]].astype(str)
        else:
            standardized["culvert_id"] = df.index.astype(str)

        # name
        if "name" in cols:
            standardized["name"] = df[df.columns[cols.index("name")]].astype(str)
        elif "crossing_name" in cols:
            standardized["name"] = df[df.columns[cols.index("crossing_name")]].astype(str)
        else:
            standardized["name"] = standardized["culvert_id"]

        # size
        size_fields = ["diameter", "diam", "culv_diam", "pipe_dia", "size"]
        size_val = None
        for s in size_fields:
            if s in cols:
                size_val = df[df.columns[cols.index(s)]]
                break
        if size_val is not None:
            standardized["size"] = pd.to_numeric(size_val, errors="coerce")
        else:
            standardized["size"] = np.random.choice([12, 18, 24, 30, 36], len(df), p=[0.2, 0.3, 0.25, 0.15, 0.1])

        # condition
        cond_fields = ["condition", "cond", "culv_cond", "status"]
        cond_val = None
        for c in cond_fields:
            if c in cols:
                cond_val = df[df.columns[cols.index(c)]]
                break
        if cond_val is not None:
            standardized["condition"] = cond_val.astype(str)
        else:
            standardized["condition"] = np.random.choice(["Good", "Fair", "Poor"], len(df), p=[0.4, 0.4, 0.2])

        # network_type
        net_fields = ["network_type", "road_type", "route_type"]
        net_val = None
        for n in net_fields:
            if n in cols:
                net_val = df[df.columns[cols.index(n)]]
                break
        if net_val is not None:
            standardized["network_type"] = net_val.astype(str)
        else:
            standardized["network_type"] = np.random.choice(["Road", "Railway"], len(df), p=[0.8, 0.2])

        # geometry + CRS
        standardized = gpd.GeoDataFrame(standardized, geometry=df.geometry, crs="EPSG:4326")
        return standardized

    def collect_stream_gauge_data(self):
        """
        Collect USGS NWIS site information within the bbox. Robustly parse RDB output.
        Falls back gracefully if nothing is returned.
        """
        print("Collecting stream gauge data...")

        if self.bbox is None:
            print("No location set. Please set location first.")
            return None

        # USGS site service expects a major filter; we'll use bBox (capital B)
        bbox_str = f"{self.bbox[0]},{self.bbox[1]},{self.bbox[2]},{self.bbox[3]}"
        sites_url = "https://waterservices.usgs.gov/nwis/site"

        params = {
            "format": "rdb",
            "bBox": bbox_str,
            # keep request minimal to avoid 400 errors
            "siteStatus": "active"
        }

        try:
            r = requests.get(sites_url, params=params, timeout=30)
            # If non-200, show body for debugging
            if r.status_code != 200:
                print(f"Failed to retrieve gauge data: {r.status_code}")
                print(f"Response: {r.text[:1000]}")
                return None

            # Parse RDB: skip lines starting with '#', then first line is column names,
            # second is data types, remaining lines are rows
            lines = [ln for ln in r.text.splitlines() if ln.strip() and not ln.startswith("#")]
            if len(lines) < 3:
                print("No valid gauge lines returned by USGS site service.")
                return None

            cols = lines[0].split("\t")
            # data rows start at index 2 (typical RDB format)
            rows = [row.split("\t") for row in lines[2:] if row.strip()]

            # if no rows, nothing to do
            if not rows:
                print("No gauge rows in RDB response.")
                return None

            df = pd.DataFrame(rows, columns=cols)
            # Required columns (if present) - find by name
            needed = {}
            for name in ["site_no", "station_nm", "dec_lat_va", "dec_long_va"]:
                if name in df.columns:
                    needed[name] = df[name]
                else:
                    needed[name] = None

            gauge_data = []
            for idx, row in df.iterrows():
                try:
                    lat = float(row["dec_lat_va"]) if row.get("dec_lat_va") else None
                    lon = float(row["dec_long_va"]) if row.get("dec_long_va") else None
                    site_no = row["site_no"] if row.get("site_no") else None
                    station = row["station_nm"] if row.get("station_nm") else None
                    if lat is None or lon is None or site_no is None:
                        continue
                    gauge = {
                        "site_no": site_no,
                        "station_nm": station,
                        "geometry": Point(lon, lat),
                        "dec_lat_va": lat,
                        "dec_long_va": lon
                    }
                    gauge_data.append(gauge)
                except Exception:
                    continue

            if not gauge_data:
                print("No valid gauge data found inside bbox.")
                return None

            self.gauges = gpd.GeoDataFrame(gauge_data, crs="EPSG:4326")
            print(f"Collected {len(self.gauges)} stream gauges")

            # Try to get instantaneous discharge for the first 50 gauges
            self._collect_realtime_gauge_data()
            return self.gauges
        except Exception as e:
            print(f"Error collecting gauge data: {e}")
            return None

    def _collect_realtime_gauge_data(self):
        """
        Query USGS instantaneous values for discharge (parameter 00060) for the gauges we found.
        """
        if self.gauges is None or len(self.gauges) == 0:
            return

        iv_url = "https://waterservices.usgs.gov/nwis/iv/"
        site_list = self.gauges["site_no"].head(50).tolist()  # limit to 50
        if not site_list:
            return

        params = {"format": "json", "sites": ",".join(site_list), "parameterCd": "00060"}
        try:
            r = requests.get(iv_url, params=params, timeout=30)
            if r.status_code != 200:
                print(f"Failed to retrieve real-time data: {r.status_code}")
                return
            data = r.json()
            timeSeries = data.get("value", {}).get("timeSeries", [])
            for ts in timeSeries:
                try:
                    site_code = ts["sourceInfo"]["siteCode"][0]["value"]
                    var_code = ts["variable"]["variableCode"][0]["value"]
                    if var_code != "00060":
                        continue
                    values = ts.get("values", [{}])[0].get("value", [])
                    if not values:
                        continue
                    latest = values[-1]
                    v = latest.get("value")
                    dt = latest.get("dateTime")
                    # update gauge row
                    self.gauges.loc[self.gauges["site_no"] == site_code, "discharge_cfs"] = float(v) if v is not None else None
                    self.gauges.loc[self.gauges["site_no"] == site_code, "discharge_time"] = dt
                except Exception:
                    continue
        except Exception as e:
            print(f"Error collecting real-time gauge data: {e}")

    def collect_flood_event_data(self):
        """
        Query NOAA /api.weather.gov alerts for flood-related alerts.
        """
        print("Collecting flood event data...")

        url = "https://api.weather.gov/alerts"
        params = {"event": "Flood Warning,Flood Watch,Flood Advisory", "limit": 50, "status": "actual"}

        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                print(f"Failed to retrieve flood data: {r.status_code}")
                return None
            data = r.json()
            if not isinstance(data, dict):
                print("Unexpected flood API response format.")
                return None

            features = data.get("features", [])
            events = []
            from shapely.geometry import shape
            for feat in features:
                props = feat.get("properties") or {}
                geom = feat.get("geometry")
                try:
                    if geom and geom.get("type") == "Point":
                        coords = geom.get("coordinates", [None, None])
                        pt = Point(coords[0], coords[1])
                    elif geom:
                        s = shape(geom)
                        pt = s.centroid
                    else:
                        pt = None
                except Exception:
                    pt = None

                event = {
                    "id": props.get("id", ""),
                    "event_type": props.get("event", ""),
                    "area_desc": props.get("areaDesc", ""),
                    "sent": props.get("sent", ""),
                    "effective": props.get("effective", ""),
                    "expires": props.get("expires", ""),
                    "severity": props.get("severity", ""),
                    "certainty": props.get("certainty", ""),
                    "urgency": props.get("urgency", ""),
                    "geometry": pt
                }
                events.append(event)

            if not events:
                print("No valid flood events found.")
                return None

            self.flood_events = gpd.GeoDataFrame(events, crs="EPSG:4326")
            print(f"Collected {len(self.flood_events)} flood events")
            return self.flood_events
        except Exception as e:
            print(f"Error collecting flood event data: {e}")
            return None

    def proximity_analysis(self, buffer_distance=5000):
        """
        Buffer culverts (in meters) and find gauges within that buffer.
        Reprojects to an appropriate UTM zone for accurate buffer distances.
        """
        print("Performing proximity analysis...")

        if self.culverts is None:
            print("No culvert data; run collect_culvert_data() first.")
            return None
        if self.gauges is None or len(self.gauges) == 0:
            print("No gauge data available; skipping proximity analysis.")
            return None

        try:
            # UTM zone based on bbox center if available
            if self.bbox:
                center_lon = (self.bbox[0] + self.bbox[2]) / 2.0
                center_lat = (self.bbox[1] + self.bbox[3]) / 2.0
            else:
                center_lon = self.culverts.geometry.x.mean()
                center_lat = self.culverts.geometry.y.mean()

            utm_zone = int((center_lon + 180) / 6) + 1
            crs_proj = f"EPSG:326{utm_zone:02d}" if center_lat >= 0 else f"EPSG:327{utm_zone:02d}"

            culverts_proj = self.culverts.to_crs(crs_proj)
            gauges_proj = self.gauges.to_crs(crs_proj)

            culverts_proj["buffer_geom"] = culverts_proj.geometry.buffer(buffer_distance)
            buffers = culverts_proj[["culvert_id", "buffer_geom"]].set_geometry("buffer_geom")

            joined = gpd.sjoin(gauges_proj, buffers, how="inner", predicate="within")
            print(f"Found {len(joined)} gauge -> culvert pairs within {buffer_distance} meters")
            return joined
        except Exception as e:
            print(f"Error in proximity analysis: {e}")
            return None

    def hydrologic_risk_assessment(self):
        """
        Simplified hydrologic risk assessment using placeholder hydraulic fields.
        """
        print("Performing hydrologic risk assessment...")

        if self.culverts is None:
            print("No culvert data available.")
            return None

        n = len(self.culverts)
        self.culverts["design_flood_cfs"] = np.random.uniform(50, 500, n)
        self.culverts["culvert_capacity_cfs"] = np.random.uniform(30, 400, n)

        self.culverts["capacity_ratio"] = self.culverts["culvert_capacity_cfs"] / self.culverts["design_flood_cfs"]
        conditions = [
            self.culverts["capacity_ratio"] < 0.8,
            (self.culverts["capacity_ratio"] >= 0.8) & (self.culverts["capacity_ratio"] < 1.0),
            self.culverts["capacity_ratio"] >= 1.0,
        ]
        choices = ["High Risk", "Medium Risk", "Low Risk"]
        self.culverts["risk_level"] = np.select(conditions, choices, default="Unknown")
        self.culverts["sediment_risk"] = np.random.choice(["Low", "Medium", "High"], n, p=[0.5, 0.3, 0.2])

        self.risk_assessments = self.culverts[["culvert_id", "capacity_ratio", "risk_level", "sediment_risk"]]
        print("Risk assessment completed")
        return self.risk_assessments

    def transportation_impact_analysis(self):
        """
        Simplified transportation impact analysis using randomized AADT and road types.
        """
        print("Performing transportation impact analysis...")

        if self.culverts is None:
            print("No culvert data available.")
            return None

        n = len(self.culverts)
        self.culverts["traffic_volume"] = np.random.uniform(1000, 20000, n)
        self.culverts["road_type"] = np.random.choice(["Highway", "Arterial", "Collector", "Local"], n, p=[0.1, 0.3, 0.4, 0.2])
        road_multipliers = {"Highway": 10000, "Arterial": 5000, "Collector": 2000, "Local": 500}
        self.culverts["economic_impact"] = self.culverts["traffic_volume"] * self.culverts["road_type"].map(road_multipliers) / 1e6
        self.culverts["criticality_score"] = (
            (self.culverts["traffic_volume"] / self.culverts["traffic_volume"].max()) * 0.4 +
            (self.culverts["economic_impact"] / self.culverts["economic_impact"].max()) * 0.3 +
            (self.culverts["risk_level"] == "High Risk").astype(int) * 0.3
        )

        print("Transportation impact analysis completed")
        return self.culverts[["culvert_id", "traffic_volume", "road_type", "economic_impact", "criticality_score"]]

    def train_failure_prediction_model(self):
        """
        Train a RandomForest classifier for failure probability if data quality permits.
        """
        print("Training failure prediction model...")

        if self.culverts is None:
            print("No culvert data available.")
            return None

        features = self.culverts.copy()
        # create synthetic target
        failure_probability = (
            (1 - features["capacity_ratio"]).fillna(0) * 0.5 +
            (features["sediment_risk"] == "High").astype(int) * 0.3 +
            np.random.random(len(features)) * 0.2
        )
        features["failure"] = (failure_probability > 0.6).astype(int)

        feature_columns = ["design_flood_cfs", "culvert_capacity_cfs", "capacity_ratio", "traffic_volume", "economic_impact", "criticality_score"]
        model_data = features.dropna(subset=feature_columns + ["failure"])

        if len(model_data) < 10:
            print("Insufficient data for reliable modeling (need >= 10 samples). Skipping ML model.")
            return None

        X = model_data[feature_columns]
        y = model_data["failure"]
        if y.nunique() == 1:
            print("Target has no variance; cannot train classification model.")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        try:
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            print("Model training completed")
            print(f"Model accuracy: {report.get('accuracy',0):.2f}")
        except Exception:
            report = {"accuracy": None}
            print("Model trained but unable to compute full metrics.")

        # add probabilities where possible
        try:
            probs = model.predict_proba(self.culverts[feature_columns].fillna(0))[:, 1]
            self.culverts["failure_probability"] = probs
        except Exception:
            self.culverts["failure_probability"] = np.nan

        return model, report

    def generate_synthetic_flood_scenarios(self, n_scenarios=10):
        """
        Generate synthetic flood scenarios and test culvert failure under them.
        """
        print("Generating synthetic flood scenarios...")

        if self.culverts is None:
            print("No culvert data available.")
            return None

        scenarios = []
        for i in range(n_scenarios):
            scenarios.append({
                "scenario_id": f"SYNTH_{i+1}",
                "return_period": int(np.random.choice([10, 25, 50, 100, 500], p=[0.3, 0.25, 0.2, 0.15, 0.1])),
                "precipitation_mm": float(np.random.uniform(50, 300)),
                "duration_hours": float(np.random.uniform(6, 72)),
                "peak_discharge_factor": float(np.random.uniform(1.0, 3.0))
            })
        scenario_df = pd.DataFrame(scenarios)

        synthetic_results = []
        for _, scenario in scenario_df.iterrows():
            for _, culvert in self.culverts.iterrows():
                adjusted_capacity = culvert["culvert_capacity_cfs"] / scenario["peak_discharge_factor"]
                would_fail = adjusted_capacity < culvert["design_flood_cfs"]
                safety_margin = culvert["culvert_capacity_cfs"] - (culvert["design_flood_cfs"] * scenario["peak_discharge_factor"])
                synthetic_results.append({
                    "scenario_id": scenario["scenario_id"],
                    "culvert_id": culvert["culvert_id"],
                    "would_fail": bool(would_fail),
                    "safety_margin": float(safety_margin)
                })

        synthetic_df = pd.DataFrame(synthetic_results)
        print(f"Generated {n_scenarios} synthetic flood scenarios")
        return scenario_df, synthetic_df

    def create_interactive_map(self, show_gauges=True):
        """
        Create folium map using bbox center or culvert centroids.

        Parameters:
            show_gauges (bool): whether the Gauges layer should be visible by default. Users
                                can toggle the layer on/off in the map layer control.
        """
        print("Creating interactive map...")

        if self.culverts is None:
            print("No data to map.")
            return None

        if self.bbox:
            center_lat = self.bbox[1] + (self.bbox[3] - self.bbox[1]) / 2.0
            center_lon = self.bbox[0] + (self.bbox[2] - self.bbox[0]) / 2.0
        else:
            center_lat = self.culverts.geometry.y.mean()
            center_lon = self.culverts.geometry.x.mean()

        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Create FeatureGroups so layers can be toggled on/off
        culverts_fg = folium.FeatureGroup(name="Culverts", show=True)
        gauges_fg = folium.FeatureGroup(name="Gauges", show=bool(show_gauges))
        floods_fg = folium.FeatureGroup(name="Flood Events", show=True)

        # culverts
        for _, culv in self.culverts.iterrows():
            color = "red" if culv.get("risk_level") == "High Risk" else "green"
            folium.CircleMarker(location=[culv.geometry.y, culv.geometry.x],
                                radius=5,
                                color=color,
                                fill=True,
                                fill_opacity=0.8,
                                popup=f"Culvert {culv['culvert_id']}<br>Risk: {culv.get('risk_level','N/A')}",
                                ).add_to(culverts_fg)

        # gauges
        if self.gauges is not None:
            for _, g in self.gauges.iterrows():
                folium.Marker(location=[g.geometry.y, g.geometry.x],
                              icon=folium.Icon(color="blue", icon="tint"),
                              popup=f"Gauge {g.get('site_no','')}\n{g.get('station_nm','')}",
                              ).add_to(gauges_fg)

        # flood events
        if self.flood_events is not None:
            for _, ev in self.flood_events.iterrows():
                if ev.geometry is not None and not ev.geometry.is_empty:
                    folium.CircleMarker(location=[ev.geometry.y, ev.geometry.x],
                                        radius=8,
                                        color="orange",
                                        fill=True,
                                        fill_opacity=0.7,
                                        popup=f"{ev.get('event_type','')}\n{ev.get('area_desc','')}",
                                        ).add_to(floods_fg)

        # add feature groups to map
        culverts_fg.add_to(m)
        gauges_fg.add_to(m)
        floods_fg.add_to(m)

        # Add layer control so users can toggle layers including turning gauges off
        folium.LayerControl(collapsed=False).add_to(m)

        # Add a simple legend (fixed-position HTML)
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 160px; height: 140px;
                    background-color: white; z-index:9999; padding: 10px; border:2px solid grey;">
        <b>Map legend</b><br>
        &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp;High Risk Culvert<br>
        &nbsp;<i class="fa fa-circle" style="color:green"></i>&nbsp;Low/Medium Risk Culvert<br>
        &nbsp;<i class="fa fa-circle" style="color:blue"></i>&nbsp;Stream Gauge<br>
        &nbsp;<i class="fa fa-circle" style="color:orange"></i>&nbsp;Flood Event
        </div>
        '''
        from folium import IFrame
        m.get_root().html.add_child(folium.Element(legend_html))

        print("Interactive map created")
        return m

    def generate_report(self):
        print("Generating summary report...")
        if self.culverts is None:
            print("No data available for report.")
            return None

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_culverts": len(self.culverts),
            "total_gauges": len(self.gauges) if self.gauges is not None else 0,
            "total_flood_events": len(self.flood_events) if self.flood_events is not None else 0,
            "high_risk_culverts": int((self.culverts["risk_level"] == "High Risk").sum()) if "risk_level" in self.culverts.columns else 0,
            "avg_criticality_score": float(self.culverts["criticality_score"].mean()) if "criticality_score" in self.culverts.columns else 0.0
        }

        print("Summary Report:")
        print(f"  Total Culverts Analyzed: {report['total_culverts']}")
        print(f"  High Risk Culverts: {report['high_risk_culverts']}")
        print(f"  Stream Gauges Collected: {report['total_gauges']}")
        print(f"  Flood Events Recorded: {report['total_flood_events']}")
        print(f"  Average Criticality Score: {report['avg_criticality_score']:.2f}")

        return report

    def save_data(self, output_dir="output"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.culverts is not None:
            self.culverts.to_file(os.path.join(output_dir, "culverts.geojson"), driver="GeoJSON")
            print(f"Saved culverts to {output_dir}/culverts.geojson")
        if self.gauges is not None:
            self.gauges.to_file(os.path.join(output_dir, "gauges.geojson"), driver="GeoJSON")
            print(f"Saved gauges to {output_dir}/gauges.geojson")
        if self.flood_events is not None:
            self.flood_events.to_file(os.path.join(output_dir, "flood_events.geojson"), driver="GeoJSON")
            print(f"Saved flood events to {output_dir}/flood_events.geojson")


# Example run guard for direct execution (keeps module import safe)
if __name__ == "__main__":
    cas = CulvertAnalysisSystem()
    # Example: Now works for both counties and independent cities
    cas.set_location_by_county_state("Virginia Beach", "Virginia")
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
    cas.generate_report()
    cas.save_data("output")
    print("Analysis complete.")
