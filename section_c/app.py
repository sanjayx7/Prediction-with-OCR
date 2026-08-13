import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import io
import pytesseract
from PIL import Image
import shutil
import pypdf

# Add project root to path to allow importing section_b parser
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from section_b.extract_text import parse_ocr_text

def get_tesseract_path():
    path = shutil.which("tesseract")
    if path:
        return path
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Tesseract-OCR\tesseract.exe"),
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None

tesseract_cmd = get_tesseract_path()
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

app = FastAPI(title="Insurance CRM Machine Learning Suite", version="1.0")

# Setup directory paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Mount static files to serve images and outputs
app.mount("/static", StaticFiles(directory=OUTPUT_DIR), name="static")
app.mount("/data", StaticFiles(directory=os.path.join(BASE_DIR, 'data')), name="data")

# Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'section_c', 'templates'))

# Load models and metadata on startup
models_total = {}
models_max = {}
scalers = {}
dataset_metadata = {}

try:
    # Load model files
    for name in ['svr', 'randomforest', 'xgboost']:
        # Percentage of Total models
        model_tot_path = os.path.join(MODELS_DIR, f'{name}_model_total.joblib')
        if os.path.exists(model_tot_path):
            models_total[name] = joblib.load(model_tot_path)
            
        # Percentage of Max models
        model_max_path = os.path.join(MODELS_DIR, f'{name}_model_max.joblib')
        if os.path.exists(model_max_path):
            models_max[name] = joblib.load(model_max_path)
            
        # SVR Scaler
        scaler_tot_path = os.path.join(MODELS_DIR, f'{name}_scaler_total.joblib')
        if os.path.exists(scaler_tot_path):
            scalers[f'{name}_total'] = joblib.load(scaler_tot_path)
            
        scaler_max_path = os.path.join(MODELS_DIR, f'{name}_scaler_max.joblib')
        if os.path.exists(scaler_max_path):
            scalers[f'{name}_max'] = joblib.load(scaler_max_path)
            
    # Load metadata (like total sum and max value of actual premiums)
    import json
    metrics_path = os.path.join(OUTPUT_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            dataset_metadata = json.load(f)
            
    print("FastAPI Models and Metadata loaded successfully!")
except Exception as e:
    print(f"Error loading models or metadata: {e}")

class OCRRequest(BaseModel):
    text: str

class PredictionRequest(BaseModel):
    date_str: str  # YYYY-MM-DD
    model_name: str = 'XGBoost' # SVR, RandomForest, XGBoost

def get_date_features(date_obj):
    year = date_obj.year
    month = date_obj.month
    
    # Year Fraction
    year_fraction = year + (month - 1) / 12.0
    
    # Cyclical month features
    month_sin = np.sin(2 * np.pi * month / 12.0)
    month_cos = np.cos(2 * np.pi * month / 12.0)
    
    # Months elapsed from 2023-03-01 (start of dataset)
    months_elapsed = (year - 2023) * 12 + (month - 3)
    
    return np.array([[year_fraction, month_sin, month_cos, months_elapsed]])

@app.get("/", response_class=HTMLResponse)
async def home_dashboard(request: Request):
    # Prepare model metrics summary for display
    metrics_summary = {}
    if dataset_metadata and 'targets' in dataset_metadata:
        metrics_summary = dataset_metadata['targets']
        
    return templates.TemplateResponse("index.html", {
        "request": request,
        "metrics": metrics_summary,
        "total_premium": dataset_metadata.get('total_premium', 0),
        "max_premium": dataset_metadata.get('max_premium', 0)
    })

@app.post("/predict")
async def predict_premium(req: PredictionRequest):
    try:
        date_obj = datetime.strptime(req.date_str, "%Y-%m-%d")
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid date format. Use YYYY-MM-DD."})
        
    model_key = req.model_name.lower().replace(' ', '')
    
    if model_key not in models_total:
        return JSONResponse(status_code=400, content={"error": f"Model '{req.model_name}' not available."})
        
    features = get_date_features(date_obj)
    
    # Predict % of Total
    if model_key == 'svr':
        scaler = scalers.get('svr_total')
        features_scaled = scaler.transform(features)
        pred_pct_total = float(models_total['svr'].predict(features_scaled)[0])
    else:
        pred_pct_total = float(models_total[model_key].predict(features)[0])
        
    # Predict % of Max
    if model_key == 'svr':
        scaler = scalers.get('svr_max')
        features_scaled = scaler.transform(features)
        pred_pct_max = float(models_max['svr'].predict(features_scaled)[0])
    else:
        pred_pct_max = float(models_max[model_key].predict(features)[0])
        
    # Calculate approximate absolute values
    total_sum = dataset_metadata.get('total_premium', 0.0)
    max_val = dataset_metadata.get('max_premium', 0.0)
    
    absolute_premium_est_total = (pred_pct_total / 100.0) * total_sum
    absolute_premium_est_max = (pred_pct_max / 100.0) * max_val
    
    # Bound check (can't have negative premium)
    pred_pct_total = max(0.0, pred_pct_total)
    pred_pct_max = max(0.0, pred_pct_max)
    absolute_premium_est_total = max(0.0, absolute_premium_est_total)
    absolute_premium_est_max = max(0.0, absolute_premium_est_max)
    
    return {
        "date": req.date_str,
        "model_used": req.model_name,
        "prediction_percentage_of_total": round(pred_pct_total, 4),
        "predicted_amount_via_total_pct": round(absolute_premium_est_total, 2),
        "prediction_percentage_of_max": round(pred_pct_max, 4),
        "predicted_amount_via_max_pct": round(absolute_premium_est_max, 2)
    }

@app.post("/extract")
async def extract_ocr_entities(req: OCRRequest):
    # Save the input text to a temporary file, then use parse_ocr_text
    temp_path = os.path.join(BASE_DIR, 'output', 'temp_ocr_input.txt')
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(req.text)
            
        parsed_results = parse_ocr_text(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "raw_text": req.text,
            "parsed_profiles": parsed_results
        }
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return JSONResponse(status_code=500, content={"error": f"Extraction failed: {str(e)}"})

@app.post("/extract-file")
async def extract_ocr_file(file: UploadFile = File(...)):
    filename = file.filename.lower()
    try:
        contents = await file.read()
        text = ""
        
        # 1. Text File (.txt)
        if filename.endswith('.txt'):
            text = contents.decode('utf-8-sig', errors='ignore')
            
        # 2. PDF Document (.pdf)
        elif filename.endswith('.pdf'):
            pdf_file = io.BytesIO(contents)
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
                
        # 3. Image Document (.png, .jpg, .jpeg)
        elif filename.endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(io.BytesIO(contents))
            text = pytesseract.image_to_string(image)
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file format. Supported: PDF, TXT, PNG, JPG, JPEG"})
            
        if not text.strip():
            return JSONResponse(status_code=422, content={"error": f"No text could be extracted from the file '{file.filename}'."})
            
        # Save the extracted text to a temp file and parse it
        temp_path = os.path.join(BASE_DIR, 'output', 'temp_file_ocr.txt')
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        parsed_results = parse_ocr_text(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return {
            "raw_text": text,
            "parsed_profiles": parsed_results
        }
        
    except Exception as e:
        error_msg = str(e)
        if "tesseract" in error_msg.lower() or "tesseractnotfound" in error_msg.lower():
            return JSONResponse(
                status_code=422,
                content={
                    "error": "Tesseract OCR engine binary was not found on this system.\n\n"
                             "To test image uploads:\n"
                             "1. Download our sample cards (e.g. Ramesh Kumar, Priya Sharma) and upload them.\n"
                             "2. Install Tesseract-OCR and configure its path to use custom images."
                }
            )
        return JSONResponse(status_code=500, content={"error": f"File extraction failed: {error_msg}"})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
