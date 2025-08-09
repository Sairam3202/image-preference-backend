# ==============================================================================
# FINAL main.py FOR DEPLOYMENT
# ==============================================================================

from fastapi.responses import StreamingResponse
import io
from io import BytesIO
import psycopg2
import logging,requests
import os, random, torch, matplotlib ,shutil
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from fastapi import FastAPI, Form , UploadFile, File
import boto3
from fastapi.responses import JSONResponse
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
matplotlib.use('Agg')

# --- Logging / Initial Config ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App & Global Variables ---
app = FastAPI(title="Image Preference API", version="1.0.0")

# These will hold our loaded models and managers
preference_model = None
classification_model = None
story_manager = None

# This is the PyTorch transform for the preference model
image_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
#  PATHS
# ==============================================================================

HF_REPO_ID = "firesquad/master_dataset"
HF_TOKEN = os.getenv("HF_TOKEN") 

DISK_PATH = Path("/mnt/data")
DISK_PATH.mkdir(exist_ok=True)
TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)

# Define where your models and initial config files will be stored on the server
MODEL_PATH = DISK_PATH / "best_model.pth"
LSTM_PATH = DISK_PATH / "lstm_model.pth"
PREFERENCE_MODEL_PATH = DISK_PATH / "preference_model.pth"
CACHE_PATH = DISK_PATH / "class_cache.json"
MAX_ID_FILE = DISK_PATH / "max_id.txt"
INITIAL_DATASET_DIR = DISK_PATH / "master_dataset"

# --- Boto3 Client for Cloudflare R2 ---
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL") # The public URL of your bucket

s3_client = boto3.client(
    service_name='s3',
    endpoint_url=R2_ENDPOINT_URL,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto' # Important for R2
)

# ==============================================================================
#  DATABASE & STORAGE HELPERS
# ==============================================================================

def initialize_database():
    """Connects to the database and creates all necessary tables if they don't exist."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("FATAL: DATABASE_URL environment variable not found.")
        return

    # SQL commands to create tables
    commands = (
        """
        CREATE TABLE IF NOT EXISTS images (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255) NOT NULL UNIQUE,
            r2_url VARCHAR(1024) NOT NULL,
            class_id INTEGER,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS preferences (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            category VARCHAR(255),
            step INTEGER,
            image1_filename VARCHAR(255),
            image2_filename VARCHAR(255),
            user_choice VARCHAR(255),
            model_preference VARCHAR(255)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS app_config (
            key VARCHAR(255) PRIMARY KEY,
            value TEXT
        );
        """
    )

    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        for command in commands:
            cur.execute(command)

        # Insert initial config values if they don't exist
        cur.execute("INSERT INTO app_config (key, value) VALUES ('max_id', '0') ON CONFLICT (key) DO NOTHING;")

        cur.close()
        conn.commit()
        logger.info("Database tables initialized successfully.")

    except (Exception, psycopg2.DatabaseError) as error:
        logger.error(f"Error while connecting to PostgreSQL: {error}")

    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed.")

def get_config_value_from_db(key: str, default_value: str = None) -> str:
    """Fetches a configuration value from the 'app_config' table."""
    db_url = os.getenv("DATABASE_URL")
    sql = "SELECT value FROM app_config WHERE key = %s;"
    
    value = default_value
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql, (key,))
        result = cur.fetchone() # Fetch one result
        if result:
            value = result[0]
        cur.close()
    except Exception as e:
        logger.error(f"Database error getting config for key '{key}': {e}")
    finally:
        if conn is not None:
            conn.close()
    return value

def set_config_value_in_db(key: str, value: str):
    """Inserts or updates a configuration value in the 'app_config' table."""
    db_url = os.getenv("DATABASE_URL")
    # This SQL command will UPDATE the key if it exists, or INSERT it if it doesn't.
    sql = """
    INSERT INTO app_config (key, value) VALUES (%s, %s)
    ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
    """
    
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql, (key, value))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Database error setting config for key '{key}': {e}")
    finally:
        if conn is not None:
            conn.close()

def get_image_url_from_db(filename: str) -> str:
    """Fetches a single image's public R2 URL from the database."""
    db_url = os.getenv("DATABASE_URL")
    sql = "SELECT r2_url FROM images WHERE filename = %s;"
    
    url = None
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql, (filename,))
        result = cur.fetchone()
        if result:
            url = result[0]
        cur.close()
    except Exception as e:
        logger.error(f"Database error getting URL for {filename}: {e}")
    finally:
        if conn is not None:
            conn.close()
    return url

def log_preference_to_db(category, step, image1_filename, image2_filename, user_choice, model_preference):
    """Inserts a new preference log into the 'preferences' table."""
    db_url = os.getenv("DATABASE_URL")
    sql = """
    INSERT INTO preferences (category, step, image1_filename, image2_filename, user_choice, model_preference)
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql, (category, step, image1_filename, image2_filename, user_choice, model_preference))
        conn.commit()
        cur.close()
        logger.info(f"Logged preference for step {step}.")
    except Exception as e:
        logger.error(f"Database error logging preference: {e}")
    finally:
        if conn is not None:
            conn.close()

def insert_image_record_to_db(filename, r2_url, class_id):
    """Inserts a new record into the 'images' table in the database."""
    db_url = os.getenv("DATABASE_URL")
    sql = "INSERT INTO images (filename, r2_url, class_id) VALUES (%s, %s, %s) ON CONFLICT (filename) DO NOTHING;"
    
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql, (filename, r2_url, class_id))
        conn.commit()
        cur.close()
    except Exception as e:
        logger.error(f"Database error inserting {filename}: {e}")
    finally:
        if conn is not None:
            conn.close()

def upload_to_r2(local_file_path, object_name):
    """Uploads a file to your R2 bucket and returns its public URL."""
    if not all([R2_BUCKET_NAME, R2_PUBLIC_URL]):
        logger.error("R2 environment variables are not set.")
        return None
    
    try:
        s3_client.upload_file(str(local_file_path), R2_BUCKET_NAME, object_name)
        public_url = f"{R2_PUBLIC_URL}/{object_name}"
        logger.info(f"Successfully uploaded {object_name} to {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Failed to upload {object_name} to R2: {e}")
        return None


# ==============================================================================
#  MODEL & LOGIC CLASSES/FUNCTIONS
# ==============================================================================

# === Preference Model ===
class PreferenceNet(nn.Module):
    def __init__(self):
        super(PreferenceNet, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img1, img2):
        f1 = self.feature_extractor(img1).squeeze(-1).squeeze(-1)
        f2 = self.feature_extractor(img2).squeeze(-1).squeeze(-1)
        diff = torch.abs(f1 - f2)
        return self.fc(diff)
    
# === LSTM Sequence Model ===
class LSTMSeqModel(nn.Module):
    def __init__(self, input_dim=100, hidden_dim=256, num_layers=2, num_classes=100):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, 64, padding_idx=0)
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,dropout=0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# === Sequence Prediction ===
class SequencePredictor:
    def __init__(self, lstm_model_path):
        # Note: We removed the 'classifier' argument
        # We will load the model onto the CPU, which is correct for Render's free tier
        device = torch.device("cpu")
        self.model = LSTMSeqModel().to(device)
        self.model.load_state_dict(torch.load(lstm_model_path, map_location=device))
        self.model.eval()

    def predict_sequence(self, start_classes, steps=9, max_class=99):
        sequence = start_classes[:]
        device = torch.device("cpu") # Ensure tensors are on the CPU
        for _ in range(steps):
            inp = torch.tensor([sequence], dtype=torch.long).to(device)
            with torch.no_grad():
                output = self.model(inp)
                # Get the class ID with the highest probability
                predicted_class = torch.argmax(output, dim=1).item()
                # Ensure the predicted class is within a valid range
                predicted_class = max(0, min(predicted_class, max_class))
                sequence.append(predicted_class)
        return sequence

# === Story Management ===
class StoryManager:
    def __init__(self, predictor: SequencePredictor):
        # The manager now only needs the predictor
        self.predictor = predictor
        self.current_story = []
        self.step_index = 0

    def generate_new_story(self):
        """
        Generates a new story by predicting a sequence of classes and
        then fetching corresponding image pairs from the database.
        """
        # 1. Get a sequence of class IDs from the LSTM model
        initial_class = random.choice([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
        predicted_class_sequence = self.predictor.predict_sequence([initial_class])

        # 2. For each class ID, fetch a pair of image filenames from the database
        image_pairs = []
        for class_id in predicted_class_sequence:
            image1, image2 = get_image_pair_from_db(class_id)
            if image1 and image2:
                image_pairs.append((image1, image2))
        
        self.current_story = image_pairs
        self.step_index = 0
        logger.info(f"Generated a new story with {len(self.current_story)} steps.")

    def get_next_image_pair(self):
        """
        Gets the next pair of images from the current story. If the story is
        finished or doesn't exist, it generates a new one.
        """
        if not self.current_story or self.step_index >= len(self.current_story):
            self.generate_new_story()
        
        # If the story is still empty after trying to generate (e.g., DB error), return None
        if not self.current_story:
            logger.error("Could not retrieve a story.")
            return None

        pair = self.current_story[self.step_index]
        self.step_index += 1
        return pair

    def reset(self):
        self.current_story = []
        self.step_index = 0
        
def classify_single_image(model, image_path):
    """Classifies a single image using the provided model."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        return int(predicted.item())

def get_image_pair_from_db(class_id):
    """
    Connects to the database and fetches two random image URLs for a given class ID.
    """
    db_url = os.getenv("DATABASE_URL")
    # SQL to get all images for a specific class
    sql = "SELECT filename, r2_url FROM images WHERE class_id = %s;"
    
    images = []
    conn = None
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(sql, (class_id,))
        # Fetch all results as a list of tuples
        images = cur.fetchall()
        cur.close()
    except Exception as e:
        logger.error(f"Database error fetching images for class {class_id}: {e}")
    finally:
        if conn is not None:
            conn.close()
            
    # If the database connection fails, images will be empty
    if not images:
        return None, None 

# ==============================================================================
#  FASTAPI STARTUP EVENT
# ==============================================================================

@app.on_event("startup")
def startup_event():
    global preference_model, story_manager, classification_model
    logger.info("--- Running startup tasks ---")
    
    # 1. Initialize database tables
    initialize_database()

    # 2. Download models from Hugging Face if they don't exist on the disk
    def download_from_hf_if_needed(filename: str, local_path: Path):
        """
        Checks if a file exists on the local disk. If not, it downloads it
        from the HF_REPO_ID.
        """
        if not local_path.exists():
            logger.info(f"'{filename}' not found on disk. Downloading from Hugging Face...")
            try:
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    local_dir=DISK_PATH,
                    local_dir_use_symlinks=False, # Important for Render Disks
                    token=HF_TOKEN
                )
                logger.info(f"Successfully downloaded '{filename}'.")
            except Exception as e:
                logger.error(f"Failed to download '{filename}' from Hugging Face: {e}")
        else:
            logger.info(f"Found '{filename}' on disk. Skipping download.")

    download_from_hf_if_needed("best_model.pth", MODEL_PATH)
    download_from_hf_if_needed("lstm_model.pth", LSTM_PATH)
    download_from_hf_if_needed("preference_model.pth", PREFERENCE_MODEL_PATH)
    
    # 3. Load models into memory
    try:
        logger.info("Loading models...")
        # Load Preference Model
        preference_model = PreferenceNet()
        preference_model.load_state_dict(torch.load(PREFERENCE_MODEL_PATH, map_location="cpu"))
        preference_model.eval()

        # Load Classification Model
        classification_model = models.convnext_base(pretrained=False)
        classification_model.classifier[2] = nn.Linear(classification_model.classifier[2].in_features, 100)
        classification_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        classification_model.eval()

        # Create Predictor and Story Manager
        sequence_predictor = SequencePredictor(lstm_model_path=LSTM_PATH)
        story_manager = StoryManager(predictor=sequence_predictor)
        
        logger.info("All models and managers loaded successfully.")
    except Exception as e:
        logger.error(f"Error during model loading: {e}")

    logger.info("--- Startup tasks complete. Server is ready. ---")

# ==============================================================================
#  API ENDPOINTS
# ==============================================================================

@app.get("/ping")
def ping():
    """A simple endpoint to check if the server is alive."""
    return {"message": "pong"}

@app.get("/get-image-pair")
def get_image_pair():
    """
    Gets the next pair of images for the story, including their full public URLs.
    """
    if not story_manager:
        return JSONResponse(status_code=503, content={"error": "Story manager not initialized."})

    # 1. Get a pair of image filenames from the story manager
    image_pair = story_manager.get_next_image_pair()
    if not image_pair:
        return JSONResponse(status_code=500, content={"error": "Could not get an image pair."})
    
    filename1, filename2 = image_pair
    
    # 2. Get the public URLs for these images from our database
    # (This uses the get_image_url_from_db helper we wrote earlier)
    url1 = get_image_url_from_db(filename1)
    url2 = get_image_url_from_db(filename2)

    if not url1 or not url2:
        return JSONResponse(status_code=404, content={"error": "Could not find image URL in database."})
    
    # 3. Return the full URLs to the frontend
    return {
        "image1": {
            "filename": filename1,
            "url": url1
        },
        "image2": {
            "filename": filename2,
            "url": url2
        }
    }

@app.post("/log-choice")
def log_choice(
    category: str = Form(...),
    step: int = Form(...),
    image1_filename: str = Form(...),
    image2_filename: str = Form(...),
    user_choice_image: str = Form(...) # e.g., "image1" or "image2"
):
    """
    Logs the user's choice, runs the preference model, and saves to the database.
    """
    if not preference_model:
        return JSONResponse(status_code=503, content={"error": "Model not initialized."})

    # 1. Get image URLs from the database
    url1 = get_image_url_from_db(image1_filename)
    url2 = get_image_url_from_db(image2_filename)

    if not url1 or not url2:
        return JSONResponse(status_code=404, content={"error": "One or more images not found in database."})

    try:
        # 2. Download images from R2 and transform them
        response1 = requests.get(url1)
        response1.raise_for_status()
        img1_tensor = image_transform(Image.open(BytesIO(response1.content)).convert("RGB")).unsqueeze(0)

        response2 = requests.get(url2)
        response2.raise_for_status()
        img2_tensor = image_transform(Image.open(BytesIO(response2.content)).convert("RGB")).unsqueeze(0)

        # 3. Run preference model to determine its choice
        with torch.no_grad():
            output = preference_model(img1_tensor, img2_tensor).squeeze()
            prob = torch.sigmoid(output).item()
            model_pref_image = "image2" if prob > 0.5 else "image1"

        # 4. Log the result to our Neon database
        log_preference_to_db(
            category=category,
            step=step,
            image1_filename=image1_filename,
            image2_filename=image2_filename,
            user_choice=user_choice_image,
            model_preference=model_pref_image
        )

        # 5. Send a success response back to the frontend
        return {
            "message": "Preference logged successfully",
            "model_choice": 2 if model_pref_image == "image2" else 1
        }

    except Exception as e:
        logger.error(f"An error occurred in /log-choice: {e}")
        return JSONResponse(status_code=500, content={"error": "An internal server error occurred."})


@app.get("/preference-distribution")
def preference_distribution():
    """
    Fetches the last 10 preferences from the database and returns a
    KDE plot as a direct image response.
    """
    db_url = os.getenv("DATABASE_URL")
    # SQL to get the last 10 preference logs, ordered by time
    sql = "SELECT user_choice, model_preference FROM preferences ORDER BY timestamp DESC LIMIT 10;"
    
    conn = None
    try:
        # 1. Fetch data from the database
        conn = psycopg2.connect(db_url)
        # Use pandas to directly read the SQL query results into a DataFrame
        df = pd.read_sql_query(sql, conn)
        
        if df.empty:
            return JSONResponse(status_code=404, content={"error": "No preference data found."})

        # 2. Convert string choices ('image1', 'image2') to numbers (1, 2)
        df['user_numeric'] = df['user_choice'].str[-1].astype(int)
        df['model_numeric'] = df['model_preference'].str[-1].astype(int)

        # 3. Create the plot in memory
        plt.figure(figsize=(8, 5))
        sns.kdeplot(df['user_numeric'], label='User', fill=True, color='blue')
        sns.kdeplot(df['model_numeric'], label='Model', fill=True, color='green')
        plt.xticks([1, 2], ['Image 1', 'Image 2'])
        plt.title("KDE Distribution: User vs Model Preferences")
        plt.xlabel("Preferred Image")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        
        # 4. Save the plot to an in-memory buffer
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        plt.close() # Close the plot to free up memory
        img_buffer.seek(0) # Move the "cursor" to the beginning of the buffer

        # 5. Stream the image directly as the HTTP response
        return StreamingResponse(img_buffer, media_type="image/png")

    except Exception as e:
        logger.error(f"Error generating preference distribution plot: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to generate plot."})
    
    finally:
        if conn is not None:
            conn.close()

@app.post("/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Handles multiple image uploads, saves them to R2, classifies them,
    and records their metadata in the database.
    """
    # Note: Ensure your classification model is loaded into a global variable,
    # similar to how we loaded the preference_model.
    # For this example, let's assume it's called 'classification_model'.
    if not classification_model:
        return JSONResponse(status_code=503, content={"error": "Classification model not initialized."})

    # 1. Get the current max_id from the database
    try:
        current_max_id = int(get_config_value_from_db(key='max_id', default_value='0'))
    except (ValueError, TypeError):
        current_max_id = 0

    uploaded_files_info = []

    # 2. Process each uploaded file
    for i, file in enumerate(files):
        temp_save_path = None
        try:
            # Generate a new unique filename
            new_id = current_max_id + i + 1
            # Ensure the filename has a valid extension
            ext = os.path.splitext(file.filename)[1].lower() if os.path.splitext(file.filename)[1] else ".jpg"
            new_filename = f"{str(new_id).zfill(5)}{ext}" # zfill pads with zeros, e.g., 00001.jpg
            
            # 3. Save to a temporary local file
            temp_save_path = TEMP_DIR / new_filename
            with open(temp_save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # 4. Upload the temporary file to Cloudflare R2
            r2_url = upload_to_r2(temp_save_path, new_filename)
            if not r2_url:
                # Skip this file if upload fails
                continue

            # 5. Classify the image
            class_id = classify_single_image(classification_model, temp_save_path)

            # 6. Save the new image's metadata to the database
            insert_image_record_to_db(new_filename, r2_url, class_id)
            
            uploaded_files_info.append({"filename": new_filename, "class": class_id})

        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {e}")
        
        finally:
            # 7. Clean up the temporary file
            if temp_save_path and temp_save_path.exists():
                os.remove(temp_save_path)

    # 8. Update the max_id in the database for the next batch of uploads
    if uploaded_files_info:
        new_max_id = current_max_id + len(files)
        set_config_value_in_db('max_id', str(new_max_id))

    return {
        "message": f"{len(uploaded_files_info)} images processed successfully!",
        "files": uploaded_files_info
    }

def seed_database_and_storage():
    """
    ONE-TIME SCRIPT.
    - Classifies all images from the Hugging Face dataset.
    - Uploads each image to Cloudflare R2.
    - Saves the image metadata (filename, R2 URL, class) to the database.
    
    
    """
    logger.info("--- Starting database seeding process ---")
    
    # 1. Load the classification model
    # Note: Ensure MODEL_PATH points to the correct downloaded path on your Render Disk
    try:
        classifier_model = models.convnext_base(pretrained=False)
        classifier_model.classifier[2] = nn.Linear(classifier_model.classifier[2].in_features, 100)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        classifier_model.load_state_dict(state_dict)
        logger.info("Classification model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load classification model: {e}")
        return

    # 2. Get list of all initial image files from Hugging Face
    try:
        hf_api = HfApi()
        all_files = hf_api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        image_files = [f for f in all_files if f.startswith("master_dataset/") and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logger.info(f"Found {len(image_files)} initial images in Hugging Face repo.")
    except Exception as e:
        logger.error(f"Could not list files from Hugging Face: {e}")
        return

    # 3. Loop through, classify, upload, and save to DB
    for hf_path in tqdm(image_files, desc="Seeding Database"):
        image_filename = os.path.basename(hf_path)
        try:
            # Download the image file temporarily from Hugging Face
            temp_image_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=hf_path,
                token=HF_TOKEN
            )
            
            # Classify the image to get its class_id
            class_id = classify_single_image(classifier_model, temp_image_path)

            # Upload the image from the temp path to Cloudflare R2
            r2_url = upload_to_r2(temp_image_path, image_filename)

            # Insert the final record into our Neon database
            if r2_url:
                insert_image_record_to_db(image_filename, r2_url, class_id)
            
            # Clean up the temporary file from Hugging Face cache
            # (huggingface_hub manages its own cache, so manual deletion is complex.
            # We can ignore this for now as it won't affect the Render server.)

        except Exception as e:
            logger.error(f"Failed to process {hf_path}: {e}")
            
    logger.info("--- Database seeding process complete ---")
    
@app.get("/run-seed/{secret_key}")
def run_seed_endpoint(secret_key: str):
    """
    A secret endpoint to trigger the one-time database seeding process.
    """
    # Use a simple secret key from your environment variables to protect this
    SEED_SECRET = os.getenv("SEED_SECRET", "default_secret")
    
    if secret_key != SEED_SECRET:
        return JSONResponse(status_code=403, content={"error": "Not authorized"})
    
    # Run the seeding in the background (optional but good practice)
    # from threading import Thread
    # thread = Thread(target=seed_database_and_storage)
    # thread.start()

    # For simplicity, we can also run it directly
    seed_database_and_storage()
    
    return {"message": "Database seeding process has been started."}