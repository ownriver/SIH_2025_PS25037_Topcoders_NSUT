import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
from utils import get_indian_coastal_regions, get_grain_size_color, get_beach_type_icon, format_coordinates, format_confidence

def render_coastal_map(images_data):
    """Render interactive coastal map with sediment data points"""
    
    # Create base map centered on India
    india_center = [20.5937, 78.9629]
    m = folium.Map(
        location=india_center,
        zoom_start=5,
        tiles='OpenStreetMap'
    )
    
    # Add coastal regions as markers
    coastal_regions = get_indian_coastal_regions()
    
    # Create a feature group for regional centers
    region_group = folium.FeatureGroup(name="Coastal Regions")
    
    for region, data in coastal_regions.items():
        folium.Marker(
            location=data['center'],
            popup=f"<b>{region}</b><br>{data['description']}",
            tooltip=region,
            icon=folium.Icon(color='blue', icon='map-marker')
        ).add_to(region_group)
    
    region_group.add_to(m)
    
    # Process and add image data points
    if images_data:
        # Create feature groups for different grain sizes
        fine_group = folium.FeatureGroup(name="Fine Sediments")
        medium_group = folium.FeatureGroup(name="Medium Sediments")
        coarse_group = folium.FeatureGroup(name="Coarse Sediments")
        unknown_group = folium.FeatureGroup(name="Unknown Classification")
        
        for img in images_data:
            lat = img.get('latitude')
            lon = img.get('longitude')
            
            if lat is None or lon is None:
                continue
            
            # Get prediction data
            prediction = img.get('prediction', {})
            grain_size_class = img.get('grain_size_class') or prediction.get('grain_size_class', 'Unknown')
            beach_type = img.get('beach_type') or prediction.get('beach_type', 'Unknown')
            confidence = img.get('confidence') or prediction.get('confidence', 0)
            
            # Determine which group to add to
            if grain_size_class == 'fine':
                target_group = fine_group
                color = 'blue'
            elif grain_size_class == 'medium':
                target_group = medium_group
                color = 'orange'
            elif grain_size_class == 'coarse':
                target_group = coarse_group
                color = 'red'
            else:
                target_group = unknown_group
                color = 'gray'
            
            # Create popup content
            popup_content = f"""
            <div style="width: 200px;">
                <h4>Sediment Sample</h4>
                <p><b>Location:</b> {format_coordinates(lat, lon)}</p>
                <p><b>Grain Size:</b> {grain_size_class.title()}</p>
                <p><b>Beach Type:</b> {beach_type.title()}</p>
                <p><b>Confidence:</b> {format_confidence(confidence)}</p>
                <p><b>Uploaded:</b> {img.get('uploaded_at', 'Unknown')[:10]}</p>
            </div>
            """
            
            # Add marker
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                popup=folium.Popup(popup_content, max_width=250),
                tooltip=f"{grain_size_class.title()} - {format_confidence(confidence)}",
                color='white',
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(target_group)
        
        # Add all groups to map
        fine_group.add_to(m)
        medium_group.add_to(m)
        coarse_group.add_to(m)
        unknown_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a custom legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4>Grain Size Legend</h4>
    <p><i class="fa fa-circle" style="color:blue"></i> Fine (< 0.5mm)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Medium (0.5-1.0mm)</p>
    <p><i class="fa fa-circle" style="color:red"></i> Coarse (> 1.0mm)</p>
    <p><i class="fa fa-circle" style="color:gray"></i> Unknown</p>
    </div>
    '''
    # Add legend using folium Element
    try:
        m.get_root().html.add_child(folium.Element(legend_html))  # type: ignore
    except:
        pass  # Skip legend if error occurs
    
    # Display map in Streamlit
    map_data = st_folium(m, width=700, height=500)
    
    # Handle map interactions
    if map_data.get('last_object_clicked_popup'):
        st.info("Click on markers to see detailed sediment information!")
    
    return map_data

def render_heatmap_overlay(images_data):
    """Render heatmap overlay for erosion-prone areas"""
    
    if not images_data:
        st.warning("No data available for heatmap")
        return
    
    # Create base map
    india_center = [20.5937, 78.9629]
    m = folium.Map(location=india_center, zoom_start=5)
    
    # Prepare data for heatmap
    heat_data = []
    
    for img in images_data:
        lat = img.get('latitude')
        lon = img.get('longitude')
        
        if lat is None or lon is None:
            continue
        
        # Use grain size as intensity (finer sediments might indicate more erosion)
        prediction = img.get('prediction', {})
        grain_size_class = img.get('grain_size_class') or prediction.get('grain_size_class', 'medium')
        
        # Assign weights (higher for fine sediments - potential erosion indicators)
        if grain_size_class == 'fine':
            weight = 1.0
        elif grain_size_class == 'medium':
            weight = 0.5
        else:  # coarse
            weight = 0.2
        
        heat_data.append([lat, lon, weight])
    
    if heat_data:
        from folium.plugins import HeatMap
        HeatMap(heat_data, radius=15, blur=15, gradient={
            0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'
        }).add_to(m)
        
        # Display heatmap
        st.subheader("Erosion Risk Heatmap")
        st.caption("Red areas indicate higher concentration of fine sediments (potential erosion indicators)")
        
        st_folium(m, width=700, height=400)
    else:
        st.warning("Insufficient data for heatmap generation")

def render_region_summary_map(images_data):
    """Render map with region summaries"""
    
    # Create base map
    india_center = [20.5937, 78.9629]
    m = folium.Map(location=india_center, zoom_start=5)
    
    if not images_data:
        st.warning("No data available for region summary")
        return
    
    # Group data by region
    df = pd.DataFrame(images_data)
    
    # Get region from prediction data
    df['region'] = df.apply(lambda row: 
        row.get('prediction', {}).get('region', 'Unknown') if row.get('prediction') else 'Unknown', 
        axis=1
    )
    
    region_summary = df.groupby('region').agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'grain_size_class': lambda x: str(x.value_counts().index[0]) if len(x) > 0 else 'Unknown',
        'confidence': 'mean'
    }).reset_index()
    
    coastal_regions = get_indian_coastal_regions()
    
    for _, row in region_summary.iterrows():
        region = str(row['region'])
        if region in coastal_regions:
            center_lat, center_lon = coastal_regions[region]['center']
            
            # Count samples in region
            region_mask = (df['region'] == region).values
            sample_count = int(region_mask.sum())
            # Get confidence safely
            conf_value = row['confidence']
            # Use scalar check to avoid Series ambiguity
            is_valid = bool(pd.notna(conf_value).all() if hasattr(pd.notna(conf_value), 'all') else pd.notna(conf_value))
            if is_valid:
                avg_confidence = float(conf_value)
            else:
                avg_confidence = 0.0
            dominant_grain_size = str(row['grain_size_class'])
            
            # Create summary popup
            popup_content = f"""
            <div style="width: 200px;">
                <h4>{region}</h4>
                <p><b>Total Samples:</b> {sample_count}</p>
                <p><b>Dominant Grain Size:</b> {dominant_grain_size}</p>
                <p><b>Avg Confidence:</b> {format_confidence(avg_confidence)}</p>
                <p><b>Description:</b> {coastal_regions[region]['description']}</p>
            </div>
            """
            
            # Determine marker color based on dominant grain size
            color = get_grain_size_color(dominant_grain_size)
            
            folium.Marker(
                location=[center_lat, center_lon],
                popup=folium.Popup(popup_content, max_width=250),
                tooltip=f"{region}: {sample_count} samples",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
    
    st.subheader("Regional Summary Map")
    st_folium(m, width=700, height=400)
