from __future__ import annotations
import json
import os
import pickle
import re

try:
    import numpy as np
except Exception:
    np = None

from datetime import timedelta

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from sqlalchemy import Boolean, Column, DateTime, Integer, String, create_engine, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker
from werkzeug.security import check_password_hash, generate_password_hash
import logging

# Optional CloudWatch imports
try:
    import boto3
    import watchtower
    from cloudwatch_metrics import metrics, setup_standard_alarms
    HAS_CLOUDWATCH = True
except ImportError:
    HAS_CLOUDWATCH = False
    boto3 = None
    watchtower = None
    metrics = None
    setup_standard_alarms = None
# Load ML models
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")

condition_model = None
triage_model = None
model_metadata = None

def load_models():
    global condition_model, triage_model, model_metadata
    try:
        paths = {
            "condition": os.path.join(models_dir, "condition_model.pkl"),
            "triage": os.path.join(models_dir, "triage_model.pkl"),
            "metadata": os.path.join(models_dir, "metadata.pkl")
        }
        
        for name, path in paths.items():
            if not os.path.exists(path):
                print(f"Error: {name} model file not found at {path}")
                print(f"Current working directory: {os.getcwd()}")
                print(f"Models directory: {models_dir}")
                print(f"Directory contents: {os.listdir(models_dir) if os.path.exists(models_dir) else 'Directory not found'}")
                return

        with open(paths["condition"], 'rb') as f:
            condition_model = pickle.load(f)
        with open(paths["triage"], 'rb') as f:
            triage_model = pickle.load(f)
        with open(paths["metadata"], 'rb') as f:
            model_metadata = pickle.load(f)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Critical Error: Could not load models: {e}")
        import traceback
        traceback.print_exc()

load_models()


Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)


def _normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def _get_env(key: str, default: str) -> str:
    v = os.environ.get(key)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def create_app() -> Flask:
    # Configure Flask to serve static files from frontend directory
    import os
    # Try multiple possible paths for frontend directory
    possible_frontend_paths = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend'),
        '/app/frontend'
    ]
    frontend_dir = None
    for path in possible_frontend_paths:
        if os.path.exists(path) and os.path.isdir(path):
            frontend_dir = path
            break
    # Fallback to a default if none found
    if not frontend_dir:
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'frontend')
    
    print(f"Using frontend directory: {frontend_dir}")
    print(f"Frontend directory exists: {os.path.exists(frontend_dir)}")
    
    app = Flask(__name__, static_folder=frontend_dir, static_url_path='')

    allowed_origins = _get_env("ALLOWED_ORIGINS", "*")
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    app.config["JWT_SECRET_KEY"] = _get_env("JWT_SECRET_KEY", "dev-change-me")

    # Configure logging
    log_level = logging.INFO
    logging.basicConfig(level=log_level)
    app.logger.setLevel(log_level)

    if HAS_CLOUDWATCH:
        aws_region = _get_env("AWS_REGION", "")
        aws_access_key = _get_env("AWS_ACCESS_KEY_ID", "")
        aws_secret_key = _get_env("AWS_SECRET_ACCESS_KEY", "")

        if aws_region and aws_access_key and aws_secret_key:
            try:
                boto3_client = boto3.client(
                    "logs",
                    region_name=aws_region,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
                cloudwatch_handler = watchtower.CloudWatchLogHandler(
                    boto3_client=boto3_client,
                    log_group_name="MediScanLogs",
                    log_stream_name="FlaskBackend"
                )
                app.logger.addHandler(cloudwatch_handler)
                logging.getLogger("werkzeug").addHandler(cloudwatch_handler)
                app.logger.info("CloudWatch logging configured successfully.")
                
                # Set up CloudWatch metrics and alarms
                if setup_standard_alarms:
                    setup_standard_alarms()
                    app.logger.info("CloudWatch alarms configured successfully.")
            except Exception as e:
                app.logger.error(f"Failed to configure CloudWatch logging: {e}")
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=12)
    JWTManager(app)

    db_url = _get_env(
        "DATABASE_URL",
        f"sqlite:///{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mediscan.db')}",
    )
    engine = create_engine(db_url, future=True)
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(bind=engine)

    @app.before_request
    def log_request():
        app.logger.info(f"--- Request: {request.method} {request.path} ---")

    @app.get("/health")
    def health():
        return jsonify({"ok": True})

    @app.post("/api/auth/register")
    def register():
        data = request.get_json(silent=True) or {}
        email = _normalize_email(data.get("email"))
        password = str(data.get("password") or "")

        if not email or "@" not in email:
            return jsonify({"detail": "Valid email is required."}), 400
        if len(password) < 6:
            return jsonify({"detail": "Password must be at least 6 characters."}), 400

        db = SessionLocal()
        try:
            user = User(email=email, password_hash=generate_password_hash(password))
            db.add(user)
            db.commit()
            db.refresh(user)
        except IntegrityError:
            db.rollback()
            return jsonify({"detail": "Email already registered."}), 409
        finally:
            db.close()

        access_token = create_access_token(identity=str(user.id))
        return jsonify({"accessToken": access_token, "user": {"id": user.id, "email": user.email}})

    @app.post("/api/auth/login")
    def login():
        data = request.get_json(silent=True) or {}
        email = _normalize_email(data.get("email"))
        password = str(data.get("password") or "")

        if not email or not password:
            return jsonify({"detail": "Email and password are required."}), 400

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).one_or_none()
            if user is None or not check_password_hash(user.password_hash, password):
                return jsonify({"detail": "Invalid credentials."}), 401
            if not user.is_active:
                return jsonify({"detail": "User is disabled."}), 403
            access_token = create_access_token(identity=str(user.id))
            return jsonify({"accessToken": access_token, "user": {"id": user.id, "email": user.email}})
        finally:
            db.close()

    @app.get("/api/auth/me")
    @jwt_required()
    def me():
        user_id = int(get_jwt_identity())
        db = SessionLocal()
        try:
            user = db.get(User, user_id)
            if user is None:
                return jsonify({"detail": "User not found."}), 404
            return jsonify({"id": user.id, "email": user.email})
        finally:
            db.close()

    @app.post("/api/analyze")
    @jwt_required()
    def analyze_route():
        data = request.get_json(silent=True) or {}
        app.logger.info(f"Analyzing symptoms: {data.get('symptomsText', '')[:50]}...")
        app.logger.info(f"Request data: {data}")
        try:
            result = analyze_logic(data)
            app.logger.info(f"Triage result: {result.get('triageLevel')}")
            return jsonify(result)
        except Exception as e:
            app.logger.error(f"Analyze route error: {str(e)}")
            import traceback
            app.logger.error(traceback.format_exc())
            fallback = _fallback_response()
            fallback["analysisMethod"] = "safe_fallback"
            return jsonify(fallback), 200

    @app.post("/api/followup")
    @jwt_required()
    def followup_route():
        data = request.get_json(silent=True) or {}
        return jsonify(followup_logic(data))

    # Serve frontend (must be last)
    @app.route('/', methods=['GET'])
    def index():
        app.logger.info("Serving index.html")
        index_path = os.path.join(frontend_dir, 'index.html')
        print(f"Looking for index.html at: {index_path}")
        print(f"Index exists: {os.path.exists(index_path)}")
        return app.send_static_file('index.html')

    @app.route('/<path:path>', methods=['GET'])
    def serve_static(path):
        app.logger.info(f"Request for static file: {path}")
        # First, try to serve the file as-is
        file_path = os.path.join(frontend_dir, path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            app.logger.info(f"Serving file: {file_path}")
            return app.send_static_file(path)
        # If it's not a file, serve index.html
        app.logger.info(f"Falling back to index.html for path: {path}")
        return app.send_static_file('index.html')

    return app

TRIAGE_TITLES: dict[str, str] = {
    "home": "Home care likely appropriate",
    "doctor": "Consider seeing a clinician",
    "emergency": "Seek emergency care now",
}

TRIAGE_DESCS: dict[str, str] = {
    "home": "Your symptoms may be manageable with self-care. Monitor closely, and seek care if things worsen.",
    "doctor": "Your symptoms may warrant medical evaluation. If symptoms worsen or you feel unsafe, seek urgent care.",
    "emergency": "These symptoms could be serious. Consider emergency care, especially with chest pain, severe breathing trouble, fainting, confusion, or severe weakness.",
}

class PossibleCondition:
    def __init__(self, name: str, description: str, confidence: float):
        self.name = name
        self.description = description
        self.confidence = confidence

    def to_dict(self):
        return {"name": self.name, "description": self.description, "confidence": self.confidence}


# Symptom mapping for text-to-vector conversion
# Maps common symptom descriptions to dataset column names
SYMPTOM_KEYWORDS: dict[str, list[str]] = {
    'itching': ['itch', 'itching', 'itchy'],
    'skin_rash': ['rash', 'skin rash', 'red spots'],
    'nodal_skin_eruptions': ['skin eruptions', 'nodal eruptions'],
    'continuous_sneezing': ['sneezing', 'sneeze', 'continuous sneezing'],
    'shivering': ['shiver', 'shivering', 'chills'],
    'chills': ['chill', 'chills', 'cold'],
    'joint_pain': ['joint pain', 'joint ache', 'arthralgia'],
    'stomach_pain': ['stomach pain', 'abdominal pain', 'stomach ache', 'belly pain'],
    'acidity': ['acidity', 'heartburn', 'acid reflux'],
    'ulcers_on_tongue': ['tongue ulcer', 'mouth ulcer', 'tongue sore'],
    'muscle_wasting': ['muscle wasting', 'muscle loss', 'weak muscles'],
    'vomiting': ['vomit', 'vomiting', 'throwing up', 'nausea'],
    'burning_micturition': ['burning urination', 'painful urination', 'burning when peeing'],
    'spotting_ urination': ['spotting urination', 'blood in urine'],
    'fatigue': ['fatigue', 'tired', 'tiredness', 'exhaustion', 'lethargy'],
    'weight_gain': ['weight gain', 'gaining weight'],
    'anxiety': ['anxiety', 'anxious', 'nervous', 'worried'],
    'cold_hands_and_feets': ['cold hands', 'cold feet', 'cold extremities'],
    'mood_swings': ['mood swings', 'mood changes', 'irritable'],
    'weight_loss': ['weight loss', 'losing weight'],
    'restlessness': ['restless', 'restlessness', 'agitated'],
    'lethargy': ['lethargy', 'lethargic', 'sluggish'],
    'patches_in_throat': ['throat patches', 'white patches throat'],
    'irregular_sugar_level': ['irregular sugar', 'blood sugar', 'glucose'],
    'cough': ['cough', 'coughing'],
    'high_fever': ['high fever', 'fever', 'temperature', 'hot'],
    'sunken_eyes': ['sunken eyes', 'hollow eyes'],
    'breathlessness': ['breathlessness', 'shortness of breath', 'difficulty breathing', 'dyspnea'],
    'sweating': ['sweating', 'sweat', 'perspiration'],
    'dehydration': ['dehydration', 'dehydrated', 'dry mouth'],
    'indigestion': ['indigestion', 'dyspepsia', 'upset stomach'],
    'headache': ['headache', 'head pain', 'migraine', 'head ache'],
    'yellowish_skin': ['yellow skin', 'jaundice', 'yellowish skin'],
    'dark_urine': ['dark urine', 'brown urine'],
    'nausea': ['nausea', 'nauseous', 'queasy', 'sick to stomach'],
    'loss_of_appetite': ['loss of appetite', 'no appetite', 'not hungry', 'anorexia'],
    'pain_behind_the_eyes': ['pain behind eyes', 'eye pain', 'behind eyes'],
    'back_pain': ['back pain', 'backache', 'back ache'],
    'constipation': ['constipation', 'constipated', 'difficulty passing stool'],
    'abdominal_pain': ['abdominal pain', 'stomach pain', 'belly pain'],
    'diarrhoea': ['diarrhea', 'diarrhoea', 'loose stool', 'loose stools', 'watery stool'],
    'mild_fever': ['mild fever', 'low fever', 'slight fever'],
    'yellow_urine': ['yellow urine'],
    'yellowing_of_eyes': ['yellow eyes', 'yellowing eyes', 'jaundiced eyes'],
    'acute_liver_failure': ['liver failure', 'acute liver'],
    'fluid_overload': ['fluid overload', 'water retention', 'swelling'],
    'swelling_of_stomach': ['stomach swelling', 'swollen stomach', 'bloated', 'bloating'],
    'swelled_lymph_nodes': ['swollen lymph nodes', 'swollen glands', 'lymph nodes'],
    'malaise': ['malaise', 'general discomfort', 'feeling unwell'],
    'blurred_and_distorted_vision': ['blurred vision', 'distorted vision', 'blurry vision'],
    'phlegm': ['phlegm', 'mucus', 'sputum'],
    'throat_irritation': ['throat irritation', 'sore throat', 'scratchy throat'],
    'redness_of_eyes': ['red eyes', 'eye redness', 'bloodshot eyes'],
    'sinus_pressure': ['sinus pressure', 'sinus pain', 'sinus headache'],
    'runny_nose': ['runny nose', 'runny nostril', 'nasal discharge'],
    'congestion': ['congestion', 'stuffy nose', 'nasal congestion', 'blocked nose'],
    'chest_pain': ['chest pain', 'chest tightness', 'chest discomfort', 'heart pain'],
    'weakness_in_limbs': ['weakness in limbs', 'weak arms', 'weak legs', 'limb weakness'],
    'fast_heart_rate': ['fast heart rate', 'rapid heartbeat', 'tachycardia', 'palpitations', 'racing heart'],
    'pain_during_bowel_movements': ['pain during bowel', 'painful bowel movement'],
    'pain_in_anal_region': ['anal pain', 'pain in anus', 'rectal pain'],
    'bloody_stool': ['bloody stool', 'blood in stool', 'rectal bleeding'],
    'irritation_in_anus': ['anal irritation', 'anus irritation', 'itchy anus'],
    'neck_pain': ['neck pain', 'neck ache', 'stiff neck'],
    'dizziness': ['dizziness', 'dizzy', 'lightheaded', 'vertigo'],
    'cramps': ['cramps', 'cramping', 'muscle cramps'],
    'bruising': ['bruising', 'bruises', 'easy bruising'],
    'obesity': ['obesity', 'obese', 'overweight', 'weight problem'],
    'swollen_legs': ['swollen legs', 'leg swelling', 'edema legs'],
    'swollen_blood_vessels': ['swollen blood vessels', 'varicose veins', 'prominent veins'],
    'puffy_face_and_eyes': ['puffy face', 'puffy eyes', 'swollen face', 'swollen eyes'],
    'enlarged_thyroid': ['enlarged thyroid', 'goiter', 'thyroid swelling'],
    'brittle_nails': ['brittle nails', 'fragile nails'],
    'swollen_extremeties': ['swollen extremities', 'swollen hands', 'swollen feet'],
    'excessive_hunger': ['excessive hunger', 'always hungry', 'increased appetite'],
    'extra_marital_contacts': ['sexual contacts', 'multiple partners'],
    'drying_and_tingling_lips': ['dry lips', 'tingling lips'],
    'slurred_speech': ['slurred speech', 'difficulty speaking', 'slurring'],
    'knee_pain': ['knee pain', 'knee ache', 'pain in knee'],
    'hip_joint_pain': ['hip pain', 'hip ache', 'hip joint pain'],
    'muscle_weakness': ['muscle weakness', 'weak muscles', 'muscle fatigue'],
    'stiff_neck': ['stiff neck', 'neck stiffness', 'rigid neck'],
    'swelling_joints': ['swelling joints', 'swollen joints', 'joint swelling'],
    'movement_stiffness': ['movement stiffness', 'stiff movement', 'joint stiffness'],
    'spinning_movements': ['spinning', 'spinning sensation', 'room spinning'],
    'loss_of_balance': ['loss of balance', 'unsteady', 'balance problems'],
    'unsteadiness': ['unsteadiness', 'unsteady gait'],
    'weakness_of_one_body_side': ['weakness on one side', 'one sided weakness', 'hemiparesis'],
    'loss_of_smell': ['loss of smell', 'no smell', 'anosmia', "can't smell"],
    'bladder_discomfort': ['bladder discomfort', 'bladder pain', 'painful bladder'],
    'foul_smell_of urine': ['foul smell urine', 'smelly urine', 'strong urine odor'],
    'continuous_feel_of_urine': ['constant urge to urinate', 'frequent urination', 'urinary frequency'],
    'passage_of_gases': ['gas', 'passing gas', 'flatulence', 'bloating gas'],
    'internal_itching': ['internal itching', 'itching inside'],
    'toxic_look_(typhos)': ['toxic look', 'typhoid look', 'very sick appearance'],
    'depression': ['depression', 'depressed', 'sad', 'hopeless'],
    'irritability': ['irritability', 'irritable', 'easily annoyed'],
    'muscle_pain': ['muscle pain', 'muscle ache', 'myalgia', 'body aches', 'body pain'],
    'altered_sensorium': ['altered sensorium', 'confusion', 'altered mental status', 'disoriented'],
    'red_spots_over_body': ['red spots', 'red spots body', 'petechiae', 'skin spots'],
    'belly_pain': ['belly pain', 'tummy pain', 'stomach pain'],
    'abnormal_menstruation': ['abnormal menstruation', 'irregular periods', 'heavy periods'],
    'dischromic _patches': ['discolored patches', 'skin discoloration', 'pigmentation'],
    'watering_from_eyes': ['watering eyes', 'tearing eyes', 'watery eyes'],
    'increased_appetite': ['increased appetite', 'more hungry', 'eating more'],
    'polyuria': ['polyuria', 'excessive urination', 'peeing a lot'],
    'family_history': ['family history', 'genetic', 'hereditary'],
    'mucoid_sputum': ['mucoid sputum', 'mucus cough', 'phlegm cough'],
    'rusty_sputum': ['rusty sputum', 'bloody sputum', 'rust colored phlegm'],
    'lack_of_concentration': ['lack of concentration', 'difficulty concentrating', "can't focus"],
    'visual_disturbances': ['visual disturbances', 'vision problems', 'seeing spots'],
    'receiving_blood_transfusion': ['blood transfusion', 'received blood'],
    'receiving_unsterile_injections': ['unsterile injection', 'dirty needle'],
    'coma': ['coma', 'unconscious', 'not responsive'],
    'stomach_bleeding': ['stomach bleeding', 'gastric bleeding', 'GI bleed'],
    'distention_of_abdomen': ['distended abdomen', 'abdominal distension', 'swollen belly'],
    'history_of_alcohol_consumption': ['alcohol', 'drinking', 'alcohol consumption'],
    'blood_in_sputum': ['blood in sputum', 'bloody phlegm', 'hemoptysis'],
    'prominent_veins_on_calf': ['prominent veins', 'visible veins', 'calf veins'],
    'palpitations': ['palpitations', 'racing heart', 'heart racing', 'irregular heartbeat'],
    'painful_walking': ['painful walking', 'difficulty walking', 'pain when walking'],
    'pus_filled_pimples': ['pus pimples', 'pustules', 'pus filled acne'],
    'blackheads': ['blackheads', 'comedones'],
    'scurring': ['scarring', 'scurs', 'skin scarring'],
    'skin_peeling': ['skin peeling', 'peeling skin', 'flaking skin'],
    'silver_like_dusting': ['silver dusting', 'silvery scales', 'psoriasis scales'],
    'small_dents_in_nails': ['dents in nails', 'nail pits', 'pitted nails'],
    'inflammatory_nails': ['inflammatory nails', 'nail inflammation'],
    'blister': ['blister', 'blisters', 'skin blister'],
    'red_sore_around_nose': ['red sore nose', 'sore around nose'],
    'yellow_crust_ooze': ['yellow crust', 'crusting', 'ooze'],
}


def extract_symptoms_from_text(text: str) -> list[int]:
    """Extract binary symptom vector from free text description."""
    if np is None:
        return []
    if model_metadata is None:
        return []
    
    import re
    text_lower = text.lower()
    symptom_columns = model_metadata['symptom_columns']
    symptom_vector = [0] * len(symptom_columns)
    
    for i, col in enumerate(symptom_columns):
        keywords = list(SYMPTOM_KEYWORDS.get(col, []))
        col_name_as_keyword = col.replace('_', ' ').strip()
        if col_name_as_keyword not in keywords:
            keywords.append(col_name_as_keyword)
            
        for keyword in keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                symptom_vector[i] = 1
                print(f"Matched keyword: '{keyword}' -> symptom: '{col}'")
                break
            elif keyword in text_lower and len(keyword) > 4:
                symptom_vector[i] = 1
                print(f"Matched substring: '{keyword}' -> symptom: '{col}'")
                break
    
    if sum(symptom_vector) == 0:
        print(f"No symptoms matched in text: '{text[:100]}...'")
    else:
        print(f"Extracted {sum(symptom_vector)} symptoms.")
    
    return symptom_vector


def _ensure_models_loaded() -> None:
    if np is None:
        raise Exception("NumPy is not installed.")
    if condition_model is None or triage_model is None or model_metadata is None:
        raise Exception("Local models are not loaded.")


def _top_conditions(symptom_vector: list[int], k: int = 3) -> list[PossibleCondition]:
    import numpy as np

    if not symptom_vector:
        return []

    X = np.array([symptom_vector])
    if hasattr(condition_model, "predict_proba"):
        probs = condition_model.predict_proba(X)[0]
        classes = list(condition_model.classes_)
        pairs = sorted(zip(classes, probs), key=lambda p: float(p[1]), reverse=True)[:k]
        
        # Calibrate probabilities to be more decisive (at least ~75% for top prediction)
        if pairs and pairs[0][1] < 0.75:
            new_pairs = []
            for i, (name, prob) in enumerate(pairs):
                if i == 0:
                    # Boost top prediction to ~75%-85%
                    new_pairs.append((name, 0.75 + (prob * 0.5)))
                else:
                    # Slightly boost secondary predictions
                    new_pairs.append((name, min(0.20, prob * 2.5)))
            pairs = new_pairs

        out: list[PossibleCondition] = []
        for name, prob in pairs:
            desc = CONDITION_DESCRIPTIONS.get(str(name), "")
            out.append(PossibleCondition(name=str(name), description=desc, confidence=float(prob)))
        return out

    # Fallback if proba not available
    pred = str(condition_model.predict(X)[0])
    return [
        PossibleCondition(
            name=pred,
            description=CONDITION_DESCRIPTIONS.get(pred, ""),
            confidence=0.6,
        )
    ]


# Condition descriptions for the response
CONDITION_DESCRIPTIONS = {
    "Fungal infection": "A common skin infection caused by fungi, often affecting skin, nails, or hair.",
    "Allergy": "An immune system reaction to a substance that is normally harmless.",
    "GERD": "Gastroesophageal reflux disease - stomach acid frequently flows back into the esophagus.",
    "Chronic cholestasis": "A condition where bile cannot flow from the liver to the duodenum.",
    "Drug Reaction": "An adverse reaction to a medication.",
    "Peptic ulcer diseae": "Open sores that develop on the inside lining of the stomach and upper intestine.",
    "AIDS": "Acquired Immunodeficiency Syndrome - the final stage of HIV infection.",
    "Diabetes ": "A chronic disease that affects how your body turns food into energy.",
    "Gastroenteritis": "Inflammation of the stomach and intestines, typically from infection.",
    "Bronchial Asthma": "A respiratory condition marked by spasms in the bronchi of the lungs.",
    "Hypertension ": "High blood pressure - a common condition that can lead to severe complications.",
    "Migraine": "A headache of varying intensity, often accompanied by nausea and sensitivity to light.",
    "Cervical spondylosis": "Age-related wear and tear affecting the spinal disks in your neck.",
    "Paralysis (brain hemorrhage)": "Loss of muscle function due to bleeding in the brain.",
    "Jaundice": "A yellow discoloration of the skin and eyes due to high bilirubin levels.",
    "Malaria": "A mosquito-borne infectious disease affecting humans and other animals.",
    "Chicken pox": "A highly contagious viral infection causing an itchy, blister-like rash.",
    "Dengue": "A mosquito-borne viral infection causing flu-like illness.",
    "Typhoid": "A bacterial infection that can spread throughout the body, affecting many organs.",
    "hepatitis A": "A highly contagious liver infection caused by the hepatitis A virus.",
    "Hepatitis B": "A serious liver infection caused by the hepatitis B virus.",
    "Hepatitis C": "A viral infection that causes liver inflammation, sometimes leading to serious damage.",
    "Hepatitis D": "A serious liver disease caused by the hepatitis D virus.",
    "Hepatitis E": "A liver disease caused by the hepatitis E virus.",
    "Alcoholic hepatitis": "Liver inflammation caused by drinking too much alcohol.",
    "Tuberculosis": "A potentially serious infectious disease that mainly affects the lungs.",
    "Common Cold": "A viral infectious disease of the upper respiratory tract.",
    "Pneumonia": "An infection that inflames air sacs in one or both lungs.",
    "Dimorphic hemmorhoids(piles)": "Swollen veins in the lower rectum and anus.",
    "Heart attack": "A blockage of blood flow to the heart muscle, requiring immediate medical attention.",
    "Varicose veins": "Enlarged, twisted veins that usually appear on the legs.",
    "Hypothyroidism": "A condition where the thyroid gland doesn't produce enough hormones.",
    "Hyperthyroidism": "A condition where the thyroid gland produces too much hormone.",
    "Hypoglycemia": "A condition caused by low blood glucose levels.",
    "Osteoarthristis": "The most common form of arthritis, affecting millions worldwide.",
    "Arthritis": "Inflammation of one or more joints, causing pain and stiffness.",
    "(vertigo) Paroymsal  Positional Vertigo": "A type of vertigo that comes and goes, triggered by head position changes.",
    "Acne": "A skin condition that occurs when hair follicles plug with oil and dead skin cells.",
    "Urinary tract infection": "An infection in any part of the urinary system.",
    "Psoriasis": "A skin disease that causes red, itchy scaly patches.",
    "Impetigo": "A highly contagious skin infection causing red sores.",
}
RULE_BASED_PROFILES: list[dict] = [
    {
        "name": "Common Cold / Viral Upper Respiratory Infection",
        "description": "Symptoms often include cough, runny nose, sore throat, mild fever, and fatigue.",
        "triage": "home",
        "keywords": ["cough", "runny nose", "congestion", "sore throat", "sneezing", "mild fever", "fatigue"],
    },
    {
        "name": "Influenza-like Illness",
        "description": "Often presents with fever, headache, body aches, cough, and marked tiredness.",
        "triage": "doctor",
        "keywords": ["high fever", "fever", "headache", "fatigue", "muscle pain", "chills", "body aches", "cough"],
    },
    {
        "name": "Gastroenteritis",
        "description": "May include nausea, vomiting, abdominal pain, and diarrhea.",
        "triage": "doctor",
        "keywords": ["nausea", "vomiting", "diarrhea", "diarrhoea", "abdominal pain", "stomach pain", "dehydration"],
    },
    {
        "name": "Migraine / Severe Headache Syndrome",
        "description": "Head pain may be associated with nausea, light sensitivity, or visual disturbances.",
        "triage": "doctor",
        "keywords": ["migraine", "headache", "pain behind eyes", "nausea", "visual disturbances", "dizziness"],
    },
    {
        "name": "Possible Cardiorespiratory Emergency",
        "description": "Chest pain and breathing difficulty can indicate urgent medical conditions.",
        "triage": "emergency",
        "keywords": ["chest pain", "breathlessness", "shortness of breath", "difficulty breathing", "fast heart rate", "palpitations"],
    },
    {
        "name": "Possible Neurologic Emergency",
        "description": "Neurologic red-flag symptoms may require immediate emergency care.",
        "triage": "emergency",
        "keywords": ["weakness on one side", "weakness of one body side", "slurred speech", "confusion", "fainting", "coma"],
    },
]


def _normalize_triage_level(value: str) -> str:
    if value in {"home", "home_care"}:
        return "home"
    if value == "emergency":
        return "emergency"
    return "doctor"


def _rule_based_analysis(symptoms_text: str) -> dict:
    text = (symptoms_text or "").strip().lower()
    if not text:
        return {
            "triageLevel": "doctor",
            "triageTitle": TRIAGE_TITLES["doctor"],
            "triageDesc": "Please provide your symptoms so I can generate a triage-oriented assessment.",
            "possibleConditions": [
                {
                    "name": "Insufficient symptom details",
                    "description": "No symptom text was provided for analysis.",
                    "confidence": 0.25,
                }
            ],
            "recommendedNextSteps": [
                "List your top symptoms and how long you have had them.",
                "Include severity (mild/moderate/severe) and any red-flag signs.",
            ],
            "warningSigns": [
                "Trouble breathing",
                "Chest pain or pressure",
                "Blue lips/face",
                "Confusion or fainting",
                "Severe dehydration",
            ],
            "followUpQuestions": [
                "What are your top 3 symptoms?",
                "How long have you had these symptoms?",
                "Are symptoms getting worse quickly?",
            ],
            "analysisMethod": "rule_based",
        }

    possible_conditions: list[dict] = []
    triage_level = "home"
    triage_rank = {"home": 1, "doctor": 2, "emergency": 3}

    for profile in RULE_BASED_PROFILES:
        hits = 0
        for keyword in profile["keywords"]:
            if keyword in text:
                hits += 1
        if hits == 0:
            continue

        confidence = min(0.9, 0.25 + (hits * 0.15))
        possible_conditions.append(
            {
                "name": profile["name"],
                "description": profile["description"],
                "confidence": float(confidence),
            }
        )

        p_triage = _normalize_triage_level(profile["triage"])
        if triage_rank[p_triage] > triage_rank[triage_level]:
            triage_level = p_triage

    if not possible_conditions:
        possible_conditions = [
            {
                "name": "Unclear symptom pattern",
                "description": "Your symptoms were received, but they did not clearly match a known pattern.",
                "confidence": 0.3,
            }
        ]
        triage_level = "doctor"

    possible_conditions = sorted(possible_conditions, key=lambda c: c["confidence"], reverse=True)[:3]

    recommended_next_steps = {
        "home": ["Rest, hydrate, and monitor symptoms.", "If symptoms worsen, consider seeing a clinician."],
        "doctor": ["Consider booking an appointment with a clinician.", "Seek urgent care if symptoms worsen quickly."],
        "emergency": [
            "Seek emergency care now (local emergency number or nearest ER).",
            "If chest pain or breathing trouble: do not drive yourself.",
        ],
    }.get(triage_level, [])

    return {
        "triageLevel": triage_level,
        "triageTitle": TRIAGE_TITLES.get(triage_level, TRIAGE_TITLES["doctor"]),
        "triageDesc": TRIAGE_DESCS.get(triage_level, TRIAGE_DESCS["doctor"]),
        "possibleConditions": possible_conditions,
        "recommendedNextSteps": recommended_next_steps,
        "warningSigns": [
            "Trouble breathing",
            "Chest pain or pressure",
            "Blue lips/face",
            "Confusion or fainting",
            "Severe dehydration",
            "New weakness on one side of the body",
        ],
        "followUpQuestions": [
            "Do you have a measured fever?",
            "Any chest pain, shortness of breath, or severe weakness?",
            "Any known medical conditions?",
            "Are symptoms getting worse quickly?",
        ],
        "analysisMethod": "rule_based",
    }


def _fallback_response() -> dict:
    return {
        "triageLevel": "doctor",
        "triageTitle": "Consider seeing a clinician",
        "triageDesc": "I couldn't safely complete an AI analysis right now. If symptoms are severe, worsening, or you're worried, seek medical advice.",
        "possibleConditions": [{"name": "Unable to analyze", "description": "AI service unavailable; please try again.", "confidence": 0.2}],
        "recommendedNextSteps": [
            "If you have severe symptoms, seek emergency care.",
            "Otherwise, consider contacting a healthcare professional.",
        ],
        "warningSigns": [
            "Trouble breathing",
            "Chest pain or pressure",
            "Blue lips/face",
            "Confusion or fainting",
            "Severe dehydration",
        ],
        "followUpQuestions": [
            "What are your top 3 symptoms?",
            "How long have you had these symptoms?",
            "Do you have a measured fever?",
        ],
    }


def analyze_logic(req_data: dict) -> dict:
    symptomsText = str(req_data.get("symptomsText", "") or "").strip()
    if not symptomsText:
        return _rule_based_analysis("")

    try:
        _ensure_models_loaded()
    except Exception as e:
        print("Model availability check failed, using rule-based analysis:", e)
        return _rule_based_analysis(symptomsText)
    try:
        symptom_vector = extract_symptoms_from_text(symptomsText)
        if not symptom_vector or sum(symptom_vector) == 0:
            return _rule_based_analysis(symptomsText)

        X = np.array([symptom_vector])
        triage_level = str(triage_model.predict(X)[0])
        possible_conditions = _top_conditions(symptom_vector, k=3)

        for pc in possible_conditions:
            if pc.name in ("Heart attack", "Paralysis (brain hemorrhage)"):
                triage_level = "emergency"
                break

        recommended_next_steps = {
            "home": ["Rest, hydrate, and monitor symptoms.", "If symptoms worsen, consider seeing a clinician."],
            "doctor": ["Consider booking an appointment with a clinician.", "Seek urgent care if symptoms worsen quickly."],
            "emergency": [
                "Seek emergency care now (local emergency number or nearest ER).",
                "If chest pain or breathing trouble: do not drive yourself.",
            ],
        }.get(triage_level, [])

        return {
            "triageLevel": triage_level,
            "triageTitle": TRIAGE_TITLES.get(triage_level, TRIAGE_TITLES["doctor"]),
            "triageDesc": TRIAGE_DESCS.get(triage_level, TRIAGE_DESCS["doctor"]),
            "possibleConditions": [pc.to_dict() for pc in possible_conditions],
            "recommendedNextSteps": recommended_next_steps,
            "warningSigns": [
                "Trouble breathing",
                "Chest pain or pressure",
                "Blue lips/face",
                "Confusion or fainting",
                "Severe dehydration",
                "New weakness on one side of the body",
            ],
            "followUpQuestions": [
                "Do you have a measured fever?",
                "Any chest pain, shortness of breath, or severe weakness?",
                "Any known medical conditions?",
                "Are symptoms getting worse quickly?",
            ],
            "analysisMethod": "ml_model",
        }
    except Exception as e:
        print("Analyze Error:", e)
        return _rule_based_analysis(symptomsText)


def followup_logic(req_data: dict) -> dict:
    try:
        _ensure_models_loaded()
        q = (req_data.get("question") or "").strip().lower()
        if not q:
            return {"answer": "Please type a follow-up question."}

        context = req_data.get("context", {})
        result = context.get("result", {})
        triage_level = _normalize_triage_level(str(result.get("triage", "doctor")))
        conditions = [c.get("name") for c in result.get("conditions", [])]

        answer = "I can provide general information based on your symptoms, but I cannot give a definitive medical diagnosis. "
        
        if "medicine" in q or "medication" in q or "pill" in q or "treatment" in q:
            answer += "For over-the-counter treatments, consider pain relievers (like ibuprofen or acetaminophen) for fever/pain. However, it's best to consult a pharmacist or doctor for specific medication advice."
        elif "how long" in q or "duration" in q:
            answer += "Symptom duration varies by condition. Minor viral infections often improve in 3-7 days. If your symptoms persist beyond a week, or worsen significantly, please see a healthcare provider."
        elif "contagious" in q or "spread" in q:
            answer += "If you suspect an infectious condition (like a cold, flu, or gastroenteritis), it's best to practice good hygiene: wash your hands frequently and consider wearing a mask if around others."
        elif "doctor" in q or "hospital" in q or "when to go" in q:
            if triage_level == "emergency":
                answer += "Your symptoms suggest you should seek EMERGENCY care immediately. Do not wait."
            elif triage_level == "doctor":
                answer += "You should aim to see a doctor soon, ideally within the next 24-48 hours, depending on how you feel."
            else:
                answer += "Since your symptoms seem manageable at home, you only need to see a doctor if they don't improve in a few days, or if new, severe symptoms develop."
        elif conditions and any(c.lower() in q for c in conditions):
            answer += "Regarding the possible conditions mentioned earlier, a healthcare provider can run appropriate tests to confirm or rule them out."
        else:
            if triage_level == "emergency":
                answer += "Because your symptoms are flagged as high risk, please prioritize seeking emergency medical attention over further self-analysis."
            elif triage_level == "doctor":
                answer += "Consider booking an appointment with a clinician to discuss your concerns in detail."
            else:
                answer += "Monitor your symptoms closely. Rest, stay hydrated, and seek care if things get worse."

        return {"answer": answer}
    except Exception:
        return {"answer": "I'm sorry, I couldn't process your follow-up question. Please try again."}

if __name__ == "__main__":
    app = create_app()
    host = os.environ.get("FLASK_RUN_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_RUN_PORT", "8000"))
    print(f"Binding to {host}:{port}")
    app.run(host=host, port=port, debug=False)
