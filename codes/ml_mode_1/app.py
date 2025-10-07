import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Import custom components
from components.map_component import render_coastal_map
from components.analytics import render_analytics_dashboard
from components.upload_interface import render_upload_interface
from supabase_client import SupabaseClient
from ml_model import SedimentClassifier
from data_generator import generate_synthetic_dataset
from utils import get_indian_coastal_regions

# Page configuration
st.set_page_config(
    page_title="SamundraManthan - Coastal Sediment Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

def main():
    # Header
    st.title("🌊 SamundraManthan")
    st.markdown("### Coastal Sediment Monitoring Platform")
    
    # Initialize clients
    try:
        supabase_client = SupabaseClient()
        ml_classifier = SedimentClassifier()
    except Exception as e:
        st.error(f"Failed to initialize services: {str(e)}")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Dashboard", "Upload Interface", "Data Management", "Model Training", "Live Feed"]
    )
    
    # Main content based on selected page
    if page == "Dashboard":
        render_dashboard(supabase_client, ml_classifier)
    elif page == "Upload Interface":
        render_upload_page(supabase_client, ml_classifier)
    elif page == "Data Management":
        render_data_management(supabase_client)
    elif page == "Model Training":
        render_model_training(ml_classifier)
    elif page == "Live Feed":
        render_live_feed(supabase_client)

def render_dashboard(supabase_client, ml_classifier):
    """Render the main dashboard with map and analytics"""
    st.header("Coastal Sediment Analytics Dashboard")
    
    # Get data from Supabase
    try:
        images_data = supabase_client.get_all_images_with_predictions()
        
        if not images_data:
            st.info("No data available. Please upload some images or generate synthetic data first.")
            return
        
        # Create two columns for layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Interactive Coastal Map")
            render_coastal_map(images_data)
        
        with col2:
            st.subheader("Quick Stats")
            total_samples = len(images_data)
            st.metric("Total Samples", total_samples)
            
            if total_samples > 0:
                # Calculate grain size distribution
                grain_sizes = [img.get('prediction', {}).get('grain_size_class', 'Unknown') for img in images_data]
                grain_size_counts = pd.Series(grain_sizes).value_counts()
                
                if len(grain_size_counts) > 0:
                    most_common = str(grain_size_counts.index[0])
                    st.metric("Most Common Grain Size", most_common)
        
        # Analytics section
        st.subheader("Analytics")
        render_analytics_dashboard(images_data)
        
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")

def render_upload_page(supabase_client, ml_classifier):
    """Render the image upload interface"""
    st.header("Image Upload Interface")
    st.markdown("Simulate ESP device uploads with automatic ML predictions")
    
    render_upload_interface(supabase_client, ml_classifier)

def render_data_management(supabase_client):
    """Render data management interface"""
    st.header("Data Management")
    
    # Generate synthetic data section
    st.subheader("Generate Synthetic Dataset")
    
    if st.button("Generate 2000 Synthetic Samples"):
        with st.spinner("Generating synthetic dataset..."):
            try:
                dataset = generate_synthetic_dataset(num_samples=2000)
                
                # Save dataset locally
                dataset.to_csv('synthetic_dataset.csv', index=False)
                
                # Insert into Supabase
                success_count = 0
                for _, row in dataset.iterrows():
                    try:
                        result = supabase_client.insert_image(
                            image_url=row['image_path'],
                            latitude=row['latitude'],
                            longitude=row['longitude'],
                            prediction={
                                'grain_size': row['grain_size'],
                                'beach_type': row['beach_type'],
                                'region': row['region']
                            }
                        )
                        if result:
                            success_count += 1
                    except Exception as e:
                        st.error(f"Error inserting row: {str(e)}")
                        continue
                
                st.session_state.data_generated = True
                st.success(f"Generated and inserted {success_count} samples into database!")
                
                # Show preview
                st.subheader("Dataset Preview")
                st.dataframe(dataset.head(10))
                
            except Exception as e:
                st.error(f"Error generating dataset: {str(e)}")
    
    # Display existing data
    st.subheader("Existing Data")
    try:
        images_data = supabase_client.get_all_images_with_predictions()
        if images_data:
            df = pd.DataFrame(images_data)
            st.dataframe(df)
            st.info(f"Total records: {len(images_data)}")
        else:
            st.info("No data found in database.")
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")

def render_model_training(ml_classifier):
    """Render model training interface"""
    st.header("ML Model Training")
    
    st.subheader("Train Sediment Classification Model")
    
    if st.button("Train RandomForest Model"):
        with st.spinner("Training model..."):
            try:
                # Load synthetic dataset if exists
                if os.path.exists('synthetic_dataset.csv'):
                    dataset = pd.read_csv('synthetic_dataset.csv')
                else:
                    st.warning("No dataset found. Generating synthetic data first...")
                    dataset = generate_synthetic_dataset(num_samples=2000)
                    dataset.to_csv('synthetic_dataset.csv', index=False)
                
                # Train model
                accuracy, report = ml_classifier.train_model(dataset)
                
                st.session_state.model_trained = True
                st.success(f"Model trained successfully! Accuracy: {accuracy:.3f}")
                
                # Show classification report
                st.subheader("Classification Report")
                st.text(report)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Model status
    st.subheader("Model Status")
    if os.path.exists('sediment_model.pkl'):
        st.success("✅ Model is trained and ready for predictions")
        
        # Show model info
        try:
            model_info = ml_classifier.get_model_info()
            if model_info:
                st.json(model_info)
        except:
            pass
    else:
        st.warning("⚠️ No trained model found. Please train the model first.")

def render_live_feed(supabase_client):
    """Render live feed of recent uploads"""
    st.header("Live Feed")
    st.markdown("Latest uploads with predictions")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    try:
        # Get recent images (last 24 hours)
        recent_images = supabase_client.get_recent_images(hours=24)
        
        if not recent_images:
            st.info("No recent uploads found.")
            return
        
        st.subheader(f"Recent Uploads ({len(recent_images)} items)")
        
        # Display recent uploads
        for img in recent_images[:10]:  # Show latest 10
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col1:
                    st.write(f"📍 **Location:**")
                    st.write(f"Lat: {img.get('latitude', 'N/A'):.4f}")
                    st.write(f"Lon: {img.get('longitude', 'N/A'):.4f}")
                
                with col2:
                    prediction = img.get('prediction', {})
                    st.write(f"🔬 **Prediction:**")
                    if prediction:
                        st.write(f"Grain Size: {prediction.get('grain_size', 'N/A')}")
                        st.write(f"Beach Type: {prediction.get('beach_type', 'N/A')}")
                        st.write(f"Region: {prediction.get('region', 'N/A')}")
                    else:
                        st.write("No prediction available")
                
                with col3:
                    uploaded_at = img.get('uploaded_at', '')
                    st.write(f"⏰ **Uploaded:**")
                    st.write(uploaded_at[:19] if uploaded_at else 'N/A')
                
                st.divider()
    
    except Exception as e:
        st.error(f"Error loading live feed: {str(e)}")

if __name__ == "__main__":
    main()
