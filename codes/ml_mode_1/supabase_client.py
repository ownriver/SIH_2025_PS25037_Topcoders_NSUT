import os
import json
import uuid
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

class SupabaseClient:
    def __init__(self):
        # Get Supabase credentials from environment
        self.supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "https://cekqlmicdxqmuaobpdse.supabase.co")
        self.supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNla3FsbWljZHhxbXVhb2JwZHNlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkxOTA3ODYsImV4cCI6MjA3NDc2Njc4Nn0.ZRbfx6JZU0YwGB8LPpJxJc3EpnhxBTEVe_RChh0e9Zk")
        
        # Create database connection string
        # Extract project ref from URL
        project_ref = self.supabase_url.split("//")[1].split(".")[0]
        
        # Try to get database URL from environment first
        db_url = os.getenv("DATABASE_URL")
        
        if db_url:
            # Use provided DATABASE_URL (PostgreSQL)
            try:
                self.engine = create_engine(db_url)
                self.is_sqlite = False
                self._create_tables_if_not_exist()
            except Exception as e:
                st.warning(f"Database connection failed: {str(e)}. Using local SQLite fallback.")
                # Fallback to SQLite for local development
                self.engine = create_engine('sqlite:///coastal_sediment.db')
                self.is_sqlite = True
                self._create_tables_if_not_exist()
        else:
            # Fallback to SQLite for local development
            st.info("Using local SQLite database for development")
            self.engine = create_engine('sqlite:///coastal_sediment.db')
            self.is_sqlite = True
            self._create_tables_if_not_exist()
    
    def _create_tables_if_not_exist(self):
        """Create tables if they don't exist"""
        try:
            with self.engine.connect() as conn:
                if not self.is_sqlite:
                    # Enable UUID extension for PostgreSQL (gen_random_uuid requires pgcrypto)
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))
                    
                    # Create images table (PostgreSQL)
                    images_table = """
                    CREATE TABLE IF NOT EXISTS images (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        uploaded_at TIMESTAMP DEFAULT NOW(),
                        image_url TEXT NOT NULL,
                        latitude FLOAT8 NOT NULL,
                        longitude FLOAT8 NOT NULL,
                        prediction JSONB,
                        researcher_id UUID
                    );
                    """
                    conn.execute(text(images_table))
                    
                    # Create predictions table (PostgreSQL)
                    predictions_table = """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        image_id UUID REFERENCES images(id),
                        grain_size_class TEXT,
                        beach_type TEXT,
                        confidence FLOAT8,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                    """
                    conn.execute(text(predictions_table))
                else:
                    # Create images table (SQLite)
                    images_table = """
                    CREATE TABLE IF NOT EXISTS images (
                        id TEXT PRIMARY KEY,
                        uploaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        image_url TEXT NOT NULL,
                        latitude REAL NOT NULL,
                        longitude REAL NOT NULL,
                        prediction TEXT,
                        researcher_id TEXT
                    );
                    """
                    conn.execute(text(images_table))
                    
                    # Create predictions table (SQLite)
                    predictions_table = """
                    CREATE TABLE IF NOT EXISTS predictions (
                        id TEXT PRIMARY KEY,
                        image_id TEXT REFERENCES images(id),
                        grain_size_class TEXT,
                        beach_type TEXT,
                        confidence REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                    conn.execute(text(predictions_table))
                
                conn.commit()
                
        except Exception as e:
            print(f"Error creating tables: {str(e)}")
    
    def insert_image(self, image_url, latitude, longitude, prediction=None):
        """Insert a new image record"""
        try:
            with self.engine.connect() as conn:
                if self.is_sqlite:
                    # SQLite version - generate UUID in Python
                    image_id = str(uuid.uuid4())
                    query = text("""
                        INSERT INTO images (id, image_url, latitude, longitude, prediction)
                        VALUES (:id, :image_url, :latitude, :longitude, :prediction)
                    """)
                    
                    conn.execute(query, {
                        'id': image_id,
                        'image_url': image_url,
                        'latitude': latitude,
                        'longitude': longitude,
                        'prediction': json.dumps(prediction) if prediction else None
                    })
                else:
                    # PostgreSQL version - use RETURNING
                    query = text("""
                        INSERT INTO images (image_url, latitude, longitude, prediction)
                        VALUES (:image_url, :latitude, :longitude, :prediction)
                        RETURNING id
                    """)
                    
                    result = conn.execute(query, {
                        'image_url': image_url,
                        'latitude': latitude,
                        'longitude': longitude,
                        'prediction': json.dumps(prediction) if prediction else None
                    })
                    
                    row = result.fetchone()
                    image_id = str(row[0]) if row else None
                
                conn.commit()
                return image_id
                
        except Exception as e:
            raise Exception(f"Error inserting image: {str(e)}")
    
    def insert_prediction(self, image_id, grain_size_class, beach_type, confidence):
        """Insert a prediction record"""
        try:
            with self.engine.connect() as conn:
                if self.is_sqlite:
                    # SQLite version - generate UUID in Python
                    prediction_id = str(uuid.uuid4())
                    query = text("""
                        INSERT INTO predictions (id, image_id, grain_size_class, beach_type, confidence)
                        VALUES (:id, :image_id, :grain_size_class, :beach_type, :confidence)
                    """)
                    
                    conn.execute(query, {
                        'id': prediction_id,
                        'image_id': image_id,
                        'grain_size_class': grain_size_class,
                        'beach_type': beach_type,
                        'confidence': confidence
                    })
                else:
                    # PostgreSQL version - use RETURNING
                    query = text("""
                        INSERT INTO predictions (image_id, grain_size_class, beach_type, confidence)
                        VALUES (:image_id, :grain_size_class, :beach_type, :confidence)
                        RETURNING id
                    """)
                    
                    result = conn.execute(query, {
                        'image_id': image_id,
                        'grain_size_class': grain_size_class,
                        'beach_type': beach_type,
                        'confidence': confidence
                    })
                    
                    row = result.fetchone()
                    prediction_id = str(row[0]) if row else None
                
                conn.commit()
                return prediction_id
                
        except Exception as e:
            raise Exception(f"Error inserting prediction: {str(e)}")
    
    def get_all_images_with_predictions(self):
        """Get all images with their predictions"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    SELECT 
                        i.id,
                        i.uploaded_at,
                        i.image_url,
                        i.latitude,
                        i.longitude,
                        i.prediction,
                        p.grain_size_class,
                        p.beach_type,
                        p.confidence,
                        p.created_at as prediction_created_at
                    FROM images i
                    LEFT JOIN predictions p ON i.id = p.image_id
                    ORDER BY i.uploaded_at DESC
                """)
                
                result = conn.execute(query)
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                data = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # Parse JSON prediction if available
                    if row_dict['prediction']:
                        try:
                            if isinstance(row_dict['prediction'], str):
                                row_dict['prediction'] = json.loads(row_dict['prediction'])
                        except:
                            pass
                    
                    # Convert timestamps to string
                    if row_dict['uploaded_at']:
                        row_dict['uploaded_at'] = row_dict['uploaded_at'].isoformat()
                    if row_dict['prediction_created_at']:
                        row_dict['prediction_created_at'] = row_dict['prediction_created_at'].isoformat()
                    
                    data.append(row_dict)
                
                return data
                
        except Exception as e:
            raise Exception(f"Error fetching images: {str(e)}")
    
    def get_recent_images(self, hours=24):
        """Get images uploaded in the last N hours"""
        try:
            with self.engine.connect() as conn:
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                query = text("""
                    SELECT 
                        i.id,
                        i.uploaded_at,
                        i.image_url,
                        i.latitude,
                        i.longitude,
                        i.prediction,
                        p.grain_size_class,
                        p.beach_type,
                        p.confidence
                    FROM images i
                    LEFT JOIN predictions p ON i.id = p.image_id
                    WHERE i.uploaded_at >= :cutoff_time
                    ORDER BY i.uploaded_at DESC
                """)
                
                result = conn.execute(query, {'cutoff_time': cutoff_time})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                data = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    
                    # Parse JSON prediction if available
                    if row_dict['prediction']:
                        try:
                            if isinstance(row_dict['prediction'], str):
                                row_dict['prediction'] = json.loads(row_dict['prediction'])
                        except:
                            pass
                    
                    # Convert timestamps to string
                    if row_dict['uploaded_at']:
                        row_dict['uploaded_at'] = row_dict['uploaded_at'].isoformat()
                    
                    data.append(row_dict)
                
                return data
                
        except Exception as e:
            raise Exception(f"Error fetching recent images: {str(e)}")
    
    def update_image_prediction(self, image_id, prediction_data):
        """Update prediction data for an image"""
        try:
            with self.engine.connect() as conn:
                query = text("""
                    UPDATE images 
                    SET prediction = :prediction
                    WHERE id = :image_id
                """)
                
                conn.execute(query, {
                    'image_id': image_id,
                    'prediction': json.dumps(prediction_data)
                })
                
                conn.commit()
                return True
                
        except Exception as e:
            raise Exception(f"Error updating prediction: {str(e)}")
    
    def get_statistics(self):
        """Get database statistics"""
        try:
            with self.engine.connect() as conn:
                # Total images
                total_images_query = text("SELECT COUNT(*) FROM images")
                result = conn.execute(total_images_query).fetchone()
                total_images = result[0] if result else 0
                
                # Total predictions
                total_predictions_query = text("SELECT COUNT(*) FROM predictions")
                result = conn.execute(total_predictions_query).fetchone()
                total_predictions = result[0] if result else 0
                
                # Recent uploads (last 24 hours)
                recent_cutoff = datetime.now() - timedelta(hours=24)
                recent_query = text("SELECT COUNT(*) FROM images WHERE uploaded_at >= :cutoff")
                result = conn.execute(recent_query, {'cutoff': recent_cutoff}).fetchone()
                recent_uploads = result[0] if result else 0
                
                return {
                    'total_images': total_images,
                    'total_predictions': total_predictions,
                    'recent_uploads': recent_uploads
                }
                
        except Exception as e:
            raise Exception(f"Error fetching statistics: {str(e)}")

if __name__ == "__main__":
    # Test the client
    client = SupabaseClient()
    
    # Test insertion
    image_id = client.insert_image(
        image_url="test_image.jpg",
        latitude=15.5,
        longitude=73.9,
        prediction={'grain_size': 0.8, 'beach_type': 'berm'}
    )
    print(f"Inserted image with ID: {image_id}")
    
    # Test fetching
    images = client.get_all_images_with_predictions()
    print(f"Total images: {len(images)}")
