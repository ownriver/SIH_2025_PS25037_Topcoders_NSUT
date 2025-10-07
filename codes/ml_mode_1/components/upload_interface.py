import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from utils import validate_coordinates, get_region_from_coordinates, format_coordinates
from PIL import Image
import io

def render_upload_interface(supabase_client, ml_classifier):
    """Render the image upload interface with ML prediction"""
    
    st.markdown("### Upload Coastal Sediment Sample")
    st.markdown("Simulate ESP device uploads with automatic sediment classification")
    
    # Create two columns for upload form
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Image Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo of coastal sediment sample"
        )
        
        # Display uploaded image
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.caption(f"File: {uploaded_file.name}")
                st.caption(f"Size: {len(uploaded_file.getvalue())} bytes")
                
            except Exception as e:
                st.error(f"Error displaying image: {str(e)}")
    
    with col2:
        st.subheader("Location Data")
        
        # Coordinate inputs
        latitude = st.number_input(
            "Latitude (°N)",
            min_value=6.0,
            max_value=25.0,
            value=15.5,
            step=0.0001,
            format="%.4f",
            help="Latitude in decimal degrees (Indian coastal range: 6.0 - 25.0)"
        )
        
        longitude = st.number_input(
            "Longitude (°E)",
            min_value=68.0,
            max_value=98.0,
            value=73.9,
            step=0.0001,
            format="%.4f",
            help="Longitude in decimal degrees (Indian coastal range: 68.0 - 98.0)"
        )
        
        # Validate coordinates
        is_valid, coord_message = validate_coordinates(latitude, longitude)
        
        if is_valid:
            st.success(f"📍 {format_coordinates(latitude, longitude)}")
            region = get_region_from_coordinates(latitude, longitude)
            st.info(f"Detected Region: **{region}**")
        else:
            st.error(coord_message)
    
    # Additional metadata
    st.subheader("Additional Information")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Manual inputs (optional)
        use_manual_grain = st.checkbox("Provide manual grain size measurement", value=False)
        if use_manual_grain:
            manual_grain_size = st.number_input(
                "Grain Size (mm)",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Manual measurement of grain size (0.1-2.0 mm)"
            )
        else:
            manual_grain_size = None
        
        manual_beach_type = st.selectbox(
            "Beach Type - Optional",
            options=[None, "berm", "dune", "intertidal"],
            index=0,
            help="Select the type of beach environment"
        )
    
    with col4:
        # Device simulation
        device_id = st.text_input(
            "Device ID",
            value=f"ESP32_COASTAL_{np.random.randint(1000, 9999)}",
            help="Simulated ESP device identifier"
        )
        
        # Timestamp
        upload_timestamp = st.selectbox(
            "Upload Time",
            options=["Current Time", "Custom Time"],
            index=0
        )
        
        if upload_timestamp == "Custom Time":
            custom_time = st.date_input(
                "Custom Upload Date",
                value=datetime.now()
            )
            custom_time = datetime.combine(custom_time, datetime.now().time())
        else:
            custom_time = datetime.now()
    
    # Upload button
    st.markdown("---")
    
    if st.button("🚀 Upload and Predict", type="primary", use_container_width=True):
        
        # Validation
        if not is_valid:
            st.error("Please provide valid coordinates")
            return
        
        if uploaded_file is None:
            st.warning("Please upload an image file")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Process upload
            status_text.text("Processing upload...")
            progress_bar.progress(25)
            
            # Generate image URL (placeholder since we can't actually store the file)
            timestamp_str = custom_time.strftime("%Y%m%d_%H%M%S")
            image_url = f"uploads/{device_id}_{timestamp_str}_{uploaded_file.name}"
            
            # Step 2: Insert into database
            status_text.text("Storing in database...")
            progress_bar.progress(50)
            
            # Prepare initial prediction data
            initial_prediction = {
                'device_id': device_id,
                'upload_timestamp': custom_time.isoformat(),
                'region': get_region_from_coordinates(latitude, longitude),
                'manual_grain_size': manual_grain_size,
                'manual_beach_type': manual_beach_type
            }
            
            image_id = supabase_client.insert_image(
                image_url=image_url,
                latitude=latitude,
                longitude=longitude,
                prediction=initial_prediction
            )
            
            # Step 3: Run ML prediction
            status_text.text("Running ML prediction...")
            progress_bar.progress(75)
            
            # Prepare features for ML model
            features = {
                'latitude': latitude,
                'longitude': longitude,
                'grain_size': manual_grain_size if manual_grain_size else 0.8,  # Default for prediction
                'beach_type': manual_beach_type if manual_beach_type else 'berm',  # Default
                'region': get_region_from_coordinates(latitude, longitude),
                'timestamp': custom_time.isoformat()
            }
            
            try:
                ml_result = ml_classifier.predict(features)
                
                # Update prediction in database
                updated_prediction = {
                    **initial_prediction,
                    'ml_prediction': ml_result,
                    'grain_size_class': ml_result.get('grain_size_class'),
                    'confidence': ml_result.get('confidence')
                }
                
                supabase_client.update_image_prediction(image_id, updated_prediction)
                
                # Insert into predictions table
                supabase_client.insert_prediction(
                    image_id=image_id,
                    grain_size_class=ml_result.get('grain_size_class'),
                    beach_type=manual_beach_type or 'berm',
                    confidence=ml_result.get('confidence', 0)
                )
                
            except Exception as ml_error:
                st.warning(f"ML prediction failed: {str(ml_error)}")
                ml_result = {
                    'grain_size_class': 'unknown',
                    'confidence': 0.0,
                    'error': str(ml_error)
                }
            
            # Step 4: Complete
            status_text.text("Upload complete!")
            progress_bar.progress(100)
            
            # Show results
            st.success("✅ Upload successful!")
            
            # Display prediction results
            st.subheader("🔬 Prediction Results")
            
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                grain_size_class = ml_result.get('grain_size_class', 'Unknown')
                st.metric(
                    "Predicted Grain Size", 
                    grain_size_class.title(),
                    help="AI-predicted sediment grain size classification"
                )
            
            with result_col2:
                confidence = ml_result.get('confidence', 0)
                st.metric(
                    "Confidence", 
                    f"{confidence:.1%}",
                    help="Model confidence in the prediction"
                )
            
            with result_col3:
                st.metric(
                    "Sample ID", 
                    image_id[:8],
                    help="Unique identifier for this sample"
                )
            
            # Detailed results
            with st.expander("📊 Detailed Results"):
                st.json({
                    'image_id': image_id,
                    'location': {
                        'latitude': latitude,
                        'longitude': longitude,
                        'region': get_region_from_coordinates(latitude, longitude)
                    },
                    'upload_info': {
                        'device_id': device_id,
                        'timestamp': custom_time.isoformat(),
                        'image_url': image_url
                    },
                    'prediction': ml_result,
                    'manual_inputs': {
                        'grain_size': manual_grain_size,
                        'beach_type': manual_beach_type
                    }
                })
            
            # Auto-refresh suggestion
            st.info("💡 Visit the Dashboard or Live Feed to see your upload on the map!")
            
        except Exception as e:
            progress_bar.progress(0)
            status_text.text("")
            st.error(f"Upload failed: {str(e)}")
    
    # Bulk upload section
    st.markdown("---")
    render_bulk_upload_section(supabase_client, ml_classifier)

def render_bulk_upload_section(supabase_client, ml_classifier):
    """Render bulk upload interface for CSV data"""
    
    st.subheader("📁 Bulk Upload (CSV)")
    st.markdown("Upload multiple samples from a CSV file")
    
    # CSV format info
    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
        Your CSV file should contain the following columns:
        - `latitude`: Decimal degrees (6.0 - 25.0)
        - `longitude`: Decimal degrees (68.0 - 98.0)
        - `image_url`: Path or URL to image file
        - `grain_size`: Optional - grain size in mm (0.1 - 2.0)
        - `beach_type`: Optional - berm, dune, or intertidal
        - `device_id`: Optional - device identifier
        """)
        
        # Sample CSV
        sample_data = pd.DataFrame({
            'latitude': [15.5, 13.0, 17.7],
            'longitude': [73.9, 80.2, 83.3],
            'image_url': ['sample1.jpg', 'sample2.jpg', 'sample3.jpg'],
            'grain_size': [0.8, 1.2, 0.4],
            'beach_type': ['berm', 'dune', 'intertidal'],
            'device_id': ['ESP32_001', 'ESP32_002', 'ESP32_003']
        })
        
        st.dataframe(sample_data)
    
    # File uploader for CSV
    csv_file = st.file_uploader(
        "Choose CSV file",
        type=['csv'],
        help="Upload a CSV file with multiple sample data"
    )
    
    if csv_file is not None:
        try:
            # Read CSV
            bulk_data = pd.read_csv(csv_file)
            
            st.success(f"📊 CSV loaded successfully: {len(bulk_data)} records")
            
            # Preview data
            st.subheader("Data Preview")
            st.dataframe(bulk_data.head())
            
            # Validation
            required_cols = ['latitude', 'longitude', 'image_url']
            missing_cols = [col for col in required_cols if col not in bulk_data.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                return
            
            # Coordinate validation
            invalid_coords = bulk_data[
                (bulk_data['latitude'] < 6.0) | (bulk_data['latitude'] > 25.0) |
                (bulk_data['longitude'] < 68.0) | (bulk_data['longitude'] > 98.0)
            ]
            
            if not invalid_coords.empty:
                st.warning(f"⚠️ {len(invalid_coords)} records have invalid coordinates")
                st.dataframe(invalid_coords[['latitude', 'longitude']])
            
            # Upload button
            if st.button("🚀 Process Bulk Upload", type="primary"):
                
                success_count = 0
                error_count = 0
                
                progress_bar = st.progress(0)
                status_container = st.container()
                
                for i, (_, row) in enumerate(bulk_data.iterrows()):
                    try:
                        # Validate coordinates
                        lat, lon = row['latitude'], row['longitude']
                        is_valid, _ = validate_coordinates(lat, lon)
                        
                        if not is_valid:
                            error_count += 1
                            continue
                        
                        # Insert image
                        image_id = supabase_client.insert_image(
                            image_url=row['image_url'],
                            latitude=lat,
                            longitude=lon,
                            prediction={
                                'device_id': row.get('device_id', f'BULK_{i}'),
                                'region': get_region_from_coordinates(lat, lon),
                                'bulk_upload': True
                            }
                        )
                        
                        # Run ML prediction if model is available
                        try:
                            features = {
                                'latitude': lat,
                                'longitude': lon,
                                'grain_size': row.get('grain_size', 0.8),
                                'beach_type': row.get('beach_type', 'berm'),
                                'region': get_region_from_coordinates(lat, lon),
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            ml_result = ml_classifier.predict(features)
                            
                            # Insert prediction
                            supabase_client.insert_prediction(
                                image_id=image_id,
                                grain_size_class=ml_result.get('grain_size_class'),
                                beach_type=row.get('beach_type', 'berm'),
                                confidence=ml_result.get('confidence', 0)
                            )
                            
                        except Exception as ml_error:
                            # Continue even if ML prediction fails
                            pass
                        
                        success_count += 1
                        
                    except Exception as e:
                        error_count += 1
                        continue
                    
                    # Update progress
                    progress = float(i + 1) / len(bulk_data)
                    progress_bar.progress(progress)
                
                # Show results
                progress_bar.progress(1.0)
                
                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count} records")
                
                if error_count > 0:
                    st.warning(f"⚠️ Failed to process {error_count} records")
                
                st.info("💡 Check the Dashboard to see your uploaded data on the map!")
                
        except Exception as e:
            st.error(f"Error processing CSV file: {str(e)}")

def render_device_simulation(supabase_client, ml_classifier):
    """Render ESP device simulation interface"""
    
    st.subheader("🤖 ESP Device Simulation")
    st.markdown("Simulate automated uploads from ESP32 coastal monitoring devices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Device configuration
        num_devices = st.slider("Number of Devices", min_value=1, max_value=10, value=3)
        upload_interval = st.slider("Upload Interval (seconds)", min_value=5, max_value=60, value=10)
        
    with col2:
        # Region selection
        regions = ['Goa', 'Chennai', 'Vizag', 'Gujarat', 'Kerala', 'Odisha', 'Andaman']
        selected_regions = st.multiselect(
            "Active Regions",
            options=regions,
            default=['Goa', 'Chennai', 'Kerala']
        )
    
    # Simulation controls
    col3, col4 = st.columns(2)
    
    with col3:
        start_simulation = st.button("▶️ Start Simulation", type="primary")
    
    with col4:
        stop_simulation = st.button("⏹️ Stop Simulation")
    
    # Simulation status
    if 'simulation_running' not in st.session_state:
        st.session_state.simulation_running = False
    
    if start_simulation:
        st.session_state.simulation_running = True
        st.success("🚀 Device simulation started!")
        st.info("This would normally run continuous uploads. For demo purposes, use the Upload Interface above.")
    
    if stop_simulation:
        st.session_state.simulation_running = False
        st.warning("⏹️ Device simulation stopped!")
