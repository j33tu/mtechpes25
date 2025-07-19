from flask import Flask, render_template, request, redirect, url_for, session, flash
import mysql.connector
import os
from datetime import datetime
import cv2
import uuid
import numpy as np

from huggingface_hub import hf_hub_download

from ultralytics import YOLO
from PIL import Image
from paddleocr import PaddleOCR

app = Flask(__name__)
app.secret_key = 'Qazwsx@123' 

print("Downloading YOLOv11 model...")

repo_id = "morsetechlab/yolov11-license-plate-detection"
filename = "license-plate-finetune-v1n.pt" 

try:
    downloaded_model_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Model downloaded to: {downloaded_model_path}")
except Exception as e:
    print(f"FATAL: Could not download model. Error: {e}")
    exit()

print("Loading YOLOv11 model...")
YOLO_MODEL = YOLO(downloaded_model_path)
print("Plate detector loaded.")

print("Loading PaddleOCR...")
OCR_READER = PaddleOCR(use_textline_orientation=True, lang='en',enable_mkldnn=True)
print("PaddleOCR loaded.")

def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="dirtyvehicleplate_2025"
    )
    return connection

def preprocess_full_frame(frame):
    """
    Enhance the full frame before plate detection to improve both detection and OCR accuracy.
    """
     
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
     
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    
    enhanced_frame = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return enhanced_frame

def preprocess_plate_image(image):
    """
    Light preprocessing for cropped plate images - your original simple method works well.
    """
    
    upscaled_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

   
    gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    
     
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
   
    processed_image = cv2.adaptiveThreshold(blurred, 255, 
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    
    
    processed_image_3_channel = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
    
    return processed_image_3_channel

os.makedirs('uploads', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
        user = cursor.fetchone()
        
        cursor.close()
        connection.close()
        
        if user:
            session['logged_in'] = True
            session['user_id'] = user['id']
            session['username'] = user['username']
            
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                flash('Username already exists. Please choose a different one.', 'danger')
                return render_template('register.html')

            cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
            if cursor.fetchone():
                flash('Email address already registered. Please use a different one.', 'danger')
                return render_template('register.html')
            
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, password)
            )
            connection.commit()
            
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'danger')
        finally:
            cursor.close()
            connection.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file selected', 'danger')
            return redirect(request.url)

        video = request.files['video']
        if video.filename == '':
            flash('No video file selected', 'danger')
            return redirect(request.url)

        user_id = session['user_id']
        run_id = str(uuid.uuid4())  
        user_run_upload_dir = os.path.join('static/uploads', f'user_{user_id}', run_id)
        os.makedirs(user_run_upload_dir, exist_ok=True)
        video_path = os.path.join(user_run_upload_dir, video.filename)
        video.save(video_path)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        found_plates_in_session = set()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 5 == 0:
                print(f"\n--- [FRAME {frame_count}] ---")
                
                
                print("[DEBUG] Preprocessing full frame...")
                enhanced_frame = preprocess_full_frame(frame)
                
                print("[DEBUG] Running YOLOv11 plate detection on enhanced frame...")
                detection_results = YOLO_MODEL(enhanced_frame, verbose=False)

                for r in detection_results:
                    if not r.boxes:
                        print("[DEBUG] No bounding boxes found in this result object.")
                        continue

                    print(f"[DEBUG] Found {len(r.boxes)} potential bounding box(es).")
                    for idx, box in enumerate(r.boxes):
                        detection_confidence = box.conf[0]
                        print(f"[DEBUG]  - Box {idx} has detection confidence: {detection_confidence:.4f}")
                        
                        if detection_confidence > 0.6:
                            print(f"[DEBUG]    - Confidence > 0.6. ACCEPTED.")
                            xyxy = box.xyxy[0]
                            x1, y1, x2, y2 = map(int, xyxy)
                            
                            
                            plate_img = frame[y1:y2, x1:x2]
                            
                            if plate_img.size == 0:
                                print("[DEBUG]    - Cropped image was empty. Skipping.")
                                continue
                            
                            plate_number_to_save = "READ_FAILED"

                            print("[DEBUG]    - Running OCR on cropped plate...")
                            
                            
                            preprocessed_plate_img = preprocess_plate_image(plate_img)
                            ocr_result = OCR_READER.predict(preprocessed_plate_img)

                            if ocr_result and ocr_result[0]:
                                result_dict = ocr_result[0]
                                rec_texts = result_dict.get('rec_texts', [])
                                rec_scores = result_dict.get('rec_scores', [])

                                high_conf_texts = []
                                ocr_confidence_threshold = 0.60

                                for text, score in zip(rec_texts, rec_scores):
                                    print(f"[DEBUG]      - Found text: '{text}' with OCR confidence: {score:.2f}")
                                    if score > ocr_confidence_threshold:
                                        high_conf_texts.append(text)
                                
                                if high_conf_texts:
                                    clean_plate_number = "".join(high_conf_texts).strip().upper()
                                    plate_number_to_save = ''.join(filter(str.isalnum, clean_plate_number))
                                    print(f"[SUCCESS]  - OCR read successful. Plate number: '{plate_number_to_save}'")
                                else:
                                    print("[DEBUG]    - No OCR text passed the confidence threshold.")
                            else:
                                print("[DEBUG]    - PaddleOCR returned an empty or invalid result.")

                             
                            saved_image_path = os.path.join(user_run_upload_dir, f'plate_{frame_count}_{idx}.jpg')
                            cv2.imwrite(saved_image_path, plate_img)
                            db_path = os.path.join('uploads', f'user_{user_id}', run_id, f'plate_{frame_count}_{idx}.jpg').replace(os.sep, '/')

                             
                            if plate_number_to_save != "READ_FAILED" and plate_number_to_save not in found_plates_in_session:
                                found_plates_in_session.add(plate_number_to_save)
                                print(f"[UNIQUE]   - Plate '{plate_number_to_save}' is new. SAVING TO DATABASE.")
                                
                                connection = get_db_connection()
                                cursor = connection.cursor()
                                cursor.execute(
                                    "INSERT INTO predictions (user_id, video_path, plate_image_path, plate_number, run_id) VALUES (%s, %s, %s, %s, %s)",
                                    (user_id, video_path, db_path, plate_number_to_save, run_id)
                                )
                                connection.commit()
                                cursor.close()
                                connection.close()
                            elif plate_number_to_save == "READ_FAILED":
                                print(f"[FAILED]   - OCR failed for plate. Image saved to: {db_path}. Not saving to database.")
                            else:
                                print(f"[DUPLICATE]- Plate '{plate_number_to_save}' already saved. SKIPPING DATABASE SAVE.")

                        else:
                            print(f"[DEBUG]  - Confidence <= 0.6. REJECTED.")

            frame_count += 1
        cap.release()

        flash('Video processed successfully!', 'success')
        return redirect(url_for('history'))

    return render_template('upload.html')

@app.route('/history')
def history():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    user_id = session['user_id']
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM predictions WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
    predictions = cursor.fetchall()
    cursor.close()
    connection.close()

    
    grouped_predictions = {}
    for p in predictions:
        run_id = p.get('run_id', 'unknown_run') 
        if run_id not in grouped_predictions:
            grouped_predictions[run_id] = {
                'run_id': run_id,
                'video_path': p['video_path'],
                'created_at': p['created_at'],
                'plates': []
            }
        grouped_predictions[run_id]['plates'].append(p)
    
    
    sorted_runs = sorted(grouped_predictions.values(), key=lambda x: x['created_at'], reverse=True)

    return render_template('history.html', runs=sorted_runs)

if __name__ == '__main__':
    app.run(debug=True)