# app.py - Flask Application untuk Deteksi Alzheimer

from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU (Railway hanya CPU)
import cv2
from PIL import Image
import io
import base64

app = Flask(__name__)

# Konfigurasi
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Buat folder uploads jika belum ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model CNN
MODEL_PATH = 'models/alzheimer_model.h5'
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Kelas Alzheimer (sesuaikan dengan model Anda)
CLASS_NAMES = [

    'Moderate Demented',
    'Mild Demented',
    'Non Demented',
    'Very Mild Demented'
]

# Deskripsi untuk setiap kelas
CLASS_DESCRIPTIONS = {
    'Moderate Demented': 'Demensia sedang. Memerlukan bantuan untuk aktivitas sehari-hari.',
    'Mild Demented': 'Demensia ringan. Kesulitan dengan tugas-tugas kompleks dan memori jangka pendek.',
    'Non Demented': 'Tidak terdeteksi tanda-tanda demensia. Fungsi kognitif normal.',
    'Very Mild Demented': 'Demensia sangat ringan. Mungkin mengalami sedikit kehilangan memori.'   
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    """
    Preprocessing gambar MRI untuk prediksi
    Sesuaikan dengan preprocessing yang digunakan saat training
    """
    try:
        # Load image
        img = image.load_img(img_path, target_size=(224, 224))
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Normalize (sesuaikan dengan normalisasi saat training)
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_prediction(img_path):
    """
    Melakukan prediksi pada gambar
    Returns: dictionary dengan hasil prediksi
    """
    if model is None:
        return None
    
    try:
        # Preprocess image
        processed_img = preprocess_image(img_path)
        
        if processed_img is None:
            return None
        
        # Predict
        predictions = model.predict(processed_img)
        
        # Get class with highest probability
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[0][predicted_class_idx]) * 100
        
        # Get all probabilities
        all_probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i]) * 100 
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort by probability
        sorted_probabilities = dict(sorted(
            all_probabilities.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': sorted_probabilities,
            'description': CLASS_DESCRIPTIONS.get(predicted_class, 'No description available')
        }
        
        return result
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

@app.route('/')
def index():
    """Homepage"""
    return render_template('index.html')

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400
        
        try:
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get prediction
            result = get_prediction(filepath)
            
            if result is None:
                return jsonify({'error': 'Error processing image'}), 500
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Clean up - delete uploaded file
            os.remove(filepath)
            
            # Return result
            return jsonify({
                'success': True,
                'result': result,
                'image': img_data
            })
            
        except Exception as e:
            print(f"Error: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = get_prediction(filepath)
        
        os.remove(filepath)
        
        if result is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    """404 error handler"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """500 error handler"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Development server
    app.run(debug=True, host='0.0.0.0', port=5000)