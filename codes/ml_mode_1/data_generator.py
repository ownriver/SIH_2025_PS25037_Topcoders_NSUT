import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker()

def get_indian_coastal_coordinates():
    """Get predefined coordinates for Indian coastal regions"""
    coastal_regions = {
        'Goa': {
            'lat_range': (15.0, 15.8),
            'lon_range': (73.7, 74.2),
            'beaches': ['Baga Beach', 'Calangute Beach', 'Anjuna Beach', 'Palolem Beach']
        },
        'Chennai': {
            'lat_range': (12.8, 13.2),
            'lon_range': (80.1, 80.3),
            'beaches': ['Marina Beach', 'Elliot Beach', 'Covelong Beach']
        },
        'Vizag': {
            'lat_range': (17.6, 17.8),
            'lon_range': (83.2, 83.4),
            'beaches': ['RK Beach', 'Rushikonda Beach', 'Yarada Beach']
        },
        'Gujarat': {
            'lat_range': (20.0, 23.0),
            'lon_range': (68.0, 72.5),
            'beaches': ['Mandvi Beach', 'Chorwad Beach', 'Somnath Beach']
        },
        'Kerala': {
            'lat_range': (8.2, 12.8),
            'lon_range': (74.8, 77.1),
            'beaches': ['Kovalam Beach', 'Varkala Beach', 'Marari Beach']
        },
        'Odisha': {
            'lat_range': (19.3, 21.9),
            'lon_range': (85.0, 87.5),
            'beaches': ['Puri Beach', 'Chandrabhaga Beach', 'Gopalpur Beach']
        },
        'Andaman': {
            'lat_range': (10.5, 13.7),
            'lon_range': (92.2, 93.9),
            'beaches': ['Radhanagar Beach', 'Elephant Beach', 'Kalapathar Beach']
        }
    }
    return coastal_regions

def generate_synthetic_dataset(num_samples=2000):
    """Generate synthetic coastal sediment dataset"""
    
    coastal_regions = get_indian_coastal_coordinates()
    beach_types = ['berm', 'dune', 'intertidal']
    
    data = []
    
    for i in range(num_samples):
        # Select random region
        region = random.choice(list(coastal_regions.keys()))
        region_data = coastal_regions[region]
        
        # Generate coordinates within region bounds
        latitude = round(random.uniform(*region_data['lat_range']), 6)
        longitude = round(random.uniform(*region_data['lon_range']), 6)
        
        # Generate grain size (0.1 to 2.0 mm)
        grain_size = round(random.uniform(0.1, 2.0), 2)
        
        # Classify grain size
        if grain_size < 0.5:
            grain_size_class = 'fine'
        elif grain_size < 1.0:
            grain_size_class = 'medium'
        else:
            grain_size_class = 'coarse'
        
        # Select beach type
        beach_type = random.choice(beach_types)
        
        # Generate placeholder image path
        image_path = f"images/sample_{i+1:04d}_{region.lower()}_{beach_type}.jpg"
        
        # Generate timestamp (last 6 months)
        start_date = datetime.now() - timedelta(days=180)
        random_date = start_date + timedelta(
            seconds=random.randint(0, int((datetime.now() - start_date).total_seconds()))
        )
        
        data.append({
            'sample_id': f"SAMPLE_{i+1:04d}",
            'latitude': latitude,
            'longitude': longitude,
            'grain_size': grain_size,
            'grain_size_class': grain_size_class,
            'beach_type': beach_type,
            'region': region,
            'image_path': image_path,
            'timestamp': random_date.isoformat(),
            'beach_name': random.choice(region_data['beaches'])
        })
    
    return pd.DataFrame(data)

def generate_features_for_ml(df):
    """Generate additional features for ML model training"""
    
    # Encode categorical variables
    beach_type_mapping = {'berm': 0, 'dune': 1, 'intertidal': 2}
    region_mapping = {region: i for i, region in enumerate(df['region'].unique())}
    
    df['beach_type_encoded'] = df['beach_type'].map(beach_type_mapping)
    df['region_encoded'] = df['region'].map(region_mapping)
    
    # Add derived features
    df['distance_from_equator'] = abs(df['latitude'])
    df['coastal_position'] = df['longitude'] - df['longitude'].min()
    
    # Add seasonal features
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.month
    df['season'] = df['month'].apply(lambda x: 
        'winter' if x in [12, 1, 2] else
        'summer' if x in [3, 4, 5] else
        'monsoon' if x in [6, 7, 8, 9] else
        'post_monsoon'
    )
    
    season_mapping = {'winter': 0, 'summer': 1, 'monsoon': 2, 'post_monsoon': 3}
    df['season_encoded'] = df['season'].map(season_mapping)
    
    return df

if __name__ == "__main__":
    # Generate sample dataset for testing
    dataset = generate_synthetic_dataset(100)
    print("Sample dataset generated:")
    print(dataset.head())
    print(f"\nDataset shape: {dataset.shape}")
    print(f"\nGrain size distribution:")
    print(dataset['grain_size_class'].value_counts())
