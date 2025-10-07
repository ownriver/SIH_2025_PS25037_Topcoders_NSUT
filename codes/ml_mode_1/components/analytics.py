import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import get_grain_size_color, create_summary_stats

def render_analytics_dashboard(images_data):
    """Render comprehensive analytics dashboard"""
    
    if not images_data:
        st.warning("No data available for analytics")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(images_data)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        render_grain_size_pie_chart(df)
        render_confidence_distribution(df)
    
    with col2:
        render_beach_type_distribution(df)
        render_temporal_analysis(df)
    
    # Full-width charts
    st.subheader("Temporal Trends")
    render_time_series_analysis(df)
    
    st.subheader("Regional Analysis")
    render_regional_comparison(df)

def render_grain_size_pie_chart(df):
    """Render pie chart for grain size distribution"""
    st.subheader("Grain Size Distribution")
    
    # Get grain size data
    grain_sizes = []
    for _, row in df.iterrows():
        prediction = row.get('prediction', {})
        grain_size_class = row.get('grain_size_class') or prediction.get('grain_size_class')
        if grain_size_class:
            grain_sizes.append(grain_size_class)
    
    if not grain_sizes:
        st.warning("No grain size data available")
        return
    
    # Count occurrences
    grain_size_counts = pd.Series(grain_sizes).value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=grain_size_counts.values,
        names=grain_size_counts.index,
        title="Sediment Grain Size Distribution",
        color_discrete_map={
            'fine': '#3498db',
            'medium': '#f39c12',
            'coarse': '#e74c3c'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def render_beach_type_distribution(df):
    """Render beach type distribution chart"""
    st.subheader("Beach Type Distribution")
    
    # Get beach type data
    beach_types = []
    for _, row in df.iterrows():
        prediction = row.get('prediction', {})
        beach_type = row.get('beach_type') or prediction.get('beach_type')
        if beach_type:
            beach_types.append(beach_type)
    
    if not beach_types:
        st.warning("No beach type data available")
        return
    
    # Count occurrences
    beach_type_counts = pd.Series(beach_types).value_counts()
    
    # Create bar chart
    fig = px.bar(
        x=beach_type_counts.index,
        y=beach_type_counts.values,
        title="Beach Type Distribution",
        labels={'x': 'Beach Type', 'y': 'Count'},
        color=beach_type_counts.index,
        color_discrete_map={
            'berm': '#2ecc71',
            'dune': '#f1c40f',
            'intertidal': '#3498db'
        }
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

def render_confidence_distribution(df):
    """Render confidence score distribution"""
    st.subheader("Prediction Confidence")
    
    # Get confidence data
    confidences = []
    for _, row in df.iterrows():
        confidence = row.get('confidence')
        if confidence is not None:
            confidences.append(confidence)
    
    if not confidences:
        st.warning("No confidence data available")
        return
    
    # Create histogram
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Prediction Confidence Distribution",
        labels={'x': 'Confidence Score', 'y': 'Frequency'},
        color_discrete_sequence=['#9b59b6']
    )
    
    # Add average line
    avg_confidence = np.mean(confidences)
    fig.add_vline(
        x=avg_confidence,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Avg: {avg_confidence:.2f}"
    )
    
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def render_temporal_analysis(df):
    """Render temporal analysis of uploads"""
    st.subheader("Upload Timeline")
    
    # Convert uploaded_at to datetime
    df_temp = df.copy()
    df_temp['uploaded_at'] = pd.to_datetime(df_temp['uploaded_at'])
    
    if df_temp['uploaded_at'].isnull().all():
        st.warning("No timestamp data available")
        return
    
    # Group by date
    daily_uploads = df_temp.groupby(df_temp['uploaded_at'].dt.date).size()
    
    # Create line chart
    fig = px.line(
        x=daily_uploads.index,
        y=daily_uploads.values,
        title="Daily Upload Trends",
        labels={'x': 'Date', 'y': 'Number of Uploads'}
    )
    
    fig.update_layout(height=300)
    
    st.plotly_chart(fig, use_container_width=True)

def render_time_series_analysis(df):
    """Render detailed time series analysis"""
    
    # Convert uploaded_at to datetime
    df_temp = df.copy()
    df_temp['uploaded_at'] = pd.to_datetime(df_temp['uploaded_at'])
    
    if df_temp['uploaded_at'].isnull().all():
        st.warning("No timestamp data available for time series")
        return
    
    # Get grain size for each record
    df_temp['grain_size_class'] = df_temp.apply(lambda row: 
        row.get('grain_size_class') or 
        (row.get('prediction', {}).get('grain_size_class') if row.get('prediction') else None), 
        axis=1
    )
    
    # Group by date and grain size
    df_temp['date'] = df_temp['uploaded_at'].dt.date
    time_grain_data = df_temp.groupby(['date', 'grain_size_class']).size().reset_index(name='count')
    
    if time_grain_data.empty:
        st.warning("Insufficient data for time series analysis")
        return
    
    # Create stacked area chart
    fig = px.area(
        time_grain_data,
        x='date',
        y='count',
        color='grain_size_class',
        title="Sediment Classification Trends Over Time",
        color_discrete_map={
            'fine': '#3498db',
            'medium': '#f39c12',
            'coarse': '#e74c3c'
        }
    )
    
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def render_regional_comparison(df):
    """Render regional comparison charts"""
    
    # Get region data
    df_temp = df.copy()
    df_temp['region'] = df_temp.apply(lambda row: 
        row.get('prediction', {}).get('region', 'Unknown') if row.get('prediction') else 'Unknown', 
        axis=1
    )
    
    df_temp['grain_size_class'] = df_temp.apply(lambda row: 
        row.get('grain_size_class') or 
        (row.get('prediction', {}).get('grain_size_class') if row.get('prediction') else None), 
        axis=1
    )
    
    # Filter out unknown regions and grain sizes
    df_temp = df_temp[
        (df_temp['region'] != 'Unknown') & 
        (df_temp['grain_size_class'].notna())
    ]
    
    if df_temp.empty:
        st.warning("Insufficient regional data for comparison")
        return
    
    # Create regional comparison
    regional_data = df_temp.groupby(['region', 'grain_size_class']).size().reset_index(name='count')
    
    # Create grouped bar chart
    fig = px.bar(
        regional_data,
        x='region',
        y='count',
        color='grain_size_class',
        title="Grain Size Distribution by Region",
        labels={'count': 'Number of Samples'},
        color_discrete_map={
            'fine': '#3498db',
            'medium': '#f39c12',
            'coarse': '#e74c3c'
        }
    )
    
    fig.update_layout(height=400)
    
    st.plotly_chart(fig, use_container_width=True)

def render_summary_metrics(df):
    """Render summary metrics cards"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_samples = len(df)
        st.metric("Total Samples", total_samples)
    
    with col2:
        unique_regions = df.apply(lambda row: 
            row.get('prediction', {}).get('region', 'Unknown') if row.get('prediction') else 'Unknown', 
            axis=1
        ).nunique()
        st.metric("Regions Covered", unique_regions)
    
    with col3:
        # Calculate average confidence
        confidences = [row.get('confidence') for _, row in df.iterrows() if row.get('confidence') is not None]
        avg_confidence = np.mean(confidences) if confidences else 0
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col4:
        # Get date range
        dates = pd.to_datetime(df['uploaded_at'], errors='coerce').dropna()
        if not dates.empty:
            date_range = (dates.max() - dates.min()).days
            st.metric("Data Span (days)", date_range)
        else:
            st.metric("Data Span", "N/A")

def render_advanced_analytics(df):
    """Render advanced analytics and insights"""
    
    st.subheader("Advanced Analytics")
    
    # Correlation analysis
    try:
        # Create numerical features for correlation
        df_numeric = df.copy()
        
        # Convert categorical to numerical
        grain_size_map = {'fine': 1, 'medium': 2, 'coarse': 3}
        beach_type_map = {'berm': 1, 'dune': 2, 'intertidal': 3}
        
        df_numeric['grain_size_numeric'] = df_numeric.apply(lambda row: 
            grain_size_map.get(
                row.get('grain_size_class') or 
                (row.get('prediction', {}).get('grain_size_class') if isinstance(row.get('prediction'), dict) else None) or 'unknown',
                0
            ), 
            axis=1
        )
        
        df_numeric['beach_type_numeric'] = df_numeric.apply(lambda row: 
            beach_type_map.get(
                row.get('beach_type') or 
                (row.get('prediction', {}).get('beach_type') if isinstance(row.get('prediction'), dict) else None) or 'unknown',
                0
            ), 
            axis=1
        )
        
        # Select numeric columns
        numeric_cols = ['latitude', 'longitude', 'confidence', 'grain_size_numeric', 'beach_type_numeric']
        df_corr = df_numeric[numeric_cols].dropna()
        
        if not df_corr.empty:
            correlation_matrix = df_corr.corr()
            
            # Create heatmap
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.warning(f"Could not generate correlation analysis: {str(e)}")
