import numpy as np
import pandas as pd
from datetime import datetime
import streamlit as st

def get_indian_coastal_regions():
    """Get information about Indian coastal regions"""
    return {
        'Goa': {
            'center': [15.4, 73.95],
            'zoom': 10,
            'description': 'Western coast known for sandy beaches and tourism'
        },
        'Chennai': {
            'center': [13.0, 80.2],
            'zoom': 10,
            'description': 'Eastern coast with long sandy coastline'
        },
        'Vizag': {
            'center': [17.7, 83.3],
            'zoom': 10,
            'description': 'Eastern coast with rocky and sandy beaches'
        },
        'Gujarat': {
            'center': [21.5, 70.0],
            'zoom': 8,
            'description': 'Western coast with diverse coastal features'
        },
        'Kerala': {
            'center': [10.5, 76.0],
            'zoom': 9,
            'description': 'Western coast with backwaters and beaches'
        },
        'Odisha': {
            'center': [20.5, 86.0],
            'zoom': 9,
            'description': 'Eastern coast with temples and beaches'
        },
        'Andaman': {
            'center': [12.0, 93.0],
            'zoom': 8,
            'description': 'Island territory with pristine beaches'
        }
    }

def classify_grain_size(grain_size):
    """Classify grain size into categories"""
    if grain_size < 0.5:
        return 'fine'
    elif grain_size < 1.0:
        return 'medium'
    else:
        return 'coarse'

def get_grain_size_color(grain_size_class):
    """Get color for grain size visualization"""
    colors = {
        'fine': '#3498db',     # Blue
        'medium': '#f39c12',   # Orange
        'coarse': '#e74c3c'    # Red
    }
    return colors.get(grain_size_class, '#95a5a6')

def get_beach_type_icon(beach_type):
    """Get icon for beach type"""
    icons = {
        'berm': '🏖️',
        'dune': '🏔️',
        'intertidal': '🌊'
    }
    return icons.get(beach_type, '📍')

def format_confidence(confidence):
    """Format confidence score for display"""
    if isinstance(confidence, (int, float)):
        return f"{confidence:.1%}"
    return "N/A"

def format_coordinates(lat, lon):
    """Format coordinates for display"""
    lat_dir = 'N' if lat >= 0 else 'S'
    lon_dir = 'E' if lon >= 0 else 'W'
    return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r

def get_region_from_coordinates(lat, lon):
    """Determine region based on coordinates"""
    coastal_regions = get_indian_coastal_regions()
    
    min_distance = float('inf')
    closest_region = 'Unknown'
    
    for region, data in coastal_regions.items():
        center_lat, center_lon = data['center']
        distance = calculate_distance(lat, lon, center_lat, center_lon)
        
        if distance < min_distance:
            min_distance = distance
            closest_region = region
    
    return closest_region

def validate_coordinates(lat, lon):
    """Validate if coordinates are within Indian coastal bounds"""
    # Indian coastal bounds (approximate)
    min_lat, max_lat = 6.0, 25.0
    min_lon, max_lon = 68.0, 98.0
    
    if not (min_lat <= lat <= max_lat):
        return False, "Latitude must be between 6.0 and 25.0 degrees"
    
    if not (min_lon <= lon <= max_lon):
        return False, "Longitude must be between 68.0 and 98.0 degrees"
    
    return True, "Valid coordinates"

def generate_sample_image_url(sample_id, region, beach_type):
    """Generate a placeholder image URL"""
    base_url = "https://picsum.photos/400/300"
    return f"{base_url}?random={sample_id}"

def process_upload_data(uploaded_file, latitude, longitude):
    """Process uploaded file and metadata"""
    if uploaded_file is None:
        return None, "No file uploaded"
    
    # Validate coordinates
    is_valid, message = validate_coordinates(latitude, longitude)
    if not is_valid:
        return None, message
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"upload_{timestamp}_{uploaded_file.name}"
    
    return {
        'filename': filename,
        'latitude': latitude,
        'longitude': longitude,
        'region': get_region_from_coordinates(latitude, longitude),
        'upload_time': datetime.now().isoformat(),
        'file_size': len(uploaded_file.getvalue()) if hasattr(uploaded_file, 'getvalue') else 0
    }, "Valid upload data"

def create_summary_stats(data):
    """Create summary statistics from data"""
    if not data:
        return {}
    
    df = pd.DataFrame(data)
    
    stats = {
        'total_samples': len(df),
        'unique_regions': df['prediction'].apply(lambda x: x.get('region', 'Unknown') if x else 'Unknown').nunique() if 'prediction' in df.columns else 0,
        'date_range': {
            'start': df['uploaded_at'].min() if 'uploaded_at' in df.columns else None,
            'end': df['uploaded_at'].max() if 'uploaded_at' in df.columns else None
        }
    }
    
    # Grain size distribution
    if 'grain_size_class' in df.columns:
        grain_size_dist = df['grain_size_class'].value_counts().to_dict()
        stats['grain_size_distribution'] = grain_size_dist
    
    # Beach type distribution
    if 'beach_type' in df.columns:
        beach_type_dist = df['beach_type'].value_counts().to_dict()
        stats['beach_type_distribution'] = beach_type_dist
    
    # Average confidence
    if 'confidence' in df.columns:
        avg_confidence = df['confidence'].mean()
        stats['average_confidence'] = avg_confidence
    
    return stats

@st.cache_data
def load_cached_data(data_source):
    """Cache data loading for better performance"""
    # This function can be extended to cache expensive operations
    return data_source

def format_timestamp(timestamp_str):
    """Format timestamp for display"""
    try:
        if isinstance(timestamp_str, str):
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        return str(timestamp_str)
    except:
        return "Invalid timestamp"

def export_data_to_csv(data, filename="export.csv"):
    """Export data to CSV format"""
    try:
        df = pd.DataFrame(data)
        csv_data = df.to_csv(index=False)
        return csv_data
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        return None
