# finance_streamlit.py
import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
from datetime import datetime, timedelta
import os
import re
import nltk
import time
import pymongo
from pymongo.errors import ConnectionFailure, OperationFailure
from bson.objectid import ObjectId
import io
import xlsxwriter
import plotly.express as px

# --- Streamlit Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="centered", page_title="Complaint Classification App")

# --- Database Configuration ---
MONGO_URI = "mongodb+srv://sharmaakhil944:EKNvoBILJxmorU5X@financialcluster0.432qo5e.mongodb.net/?retryWrites=true&w=majority&appName=FinancialCluster0"
DB_NAME = "complaint_system"
MAIN_COLLECTION_NAME = "complaints"

# Define the main departments for display and their sanitized collection names
DEPARTMENT_COLLECTIONS = {
    'Credit card / Prepaid card': 'credit_card_complaints',
    'Bank account services': 'bank_account_complaints',
    'Theft/Dispute reporting': 'theft_dispute_complaints',
    'Mortgages/loans': 'mortgages_loans_complaints',
    'Others': 'others_complaints'
}

# --- Department multilingual descriptions ---
def get_department_descriptions():
    return {
        'Credit card / Prepaid card': {
            'English': "Handles issues related to credit and prepaid cards — billing disputes, lost/stolen cards, charges, rewards and statements.",
            'Hindi': "क्रेडिट/प्रीपेड कार्ड से संबंधित समस्याएँ — बिल विवाद, खोया/चोरी हुआ कार्ड, चार्ज, रिवॉर्ड और स्टेटमेंट।",
            'Marathi': "क्रेडिट/प्रीपेड कार्डशी संबंधित समस्या — बिल वाद, हरवले/चोरलेले कार्ड, चार्ज, रिवॉर्ड आणि स्टेटमेंट.",
            'Punjabi': "ਕ੍ਰੈਡਿਟ/ਪ੍ਰੀਪੇਡ ਕਾਰਡ ਸਬੰਧੀ ਮੁੱਦੇ — ਬਿੱਲ ਵਿਵਾਦ, ਗੁੰਮ/ਚੋਰੀ ਕਾਰਡ, ਚਾਰਜ, ਇਨਾਮ ਅਤੇ ਸਟੇਟਮੈਂਟ।",
            'Kannada': "ಕ್ರೆಡಿಟ್/ಪ್ರೀಪೇಯ್ಡ್ ಕಾರ್ಡ್ ಸಂಬಂಧಿತ ಸಮಸ್ಯೆಗಳು — ಬಿಲ್ಲಿಂಗ್ ವಿವಾದಗಳು, ಕಳೆದುಕೊಂಡ/ದೋಚಲ್ಪಟ್ಟ ಕಾರ್ಡ್, ಶುಲ್ಕಗಳು, ರಿವಾರ್ಡ್ಸ್ ಮತ್ತು ಸ್ಟೇಟ್ಮೆಂಟ್.",
            'Malayalam': "ക്രെഡിറ്റ്/പ്രീപെയ്ഡ് കാർഡ് സംബന്ധിച്ച പ്രശ്നങ്ങൾ — ബില്ലിംഗ് വിക്ഷേപങ്ങൾ, നഷ്ടമോ മോഷ്ടിക്കപ്പെട്ട കാർഡുകൾ, ചാർജുകൾ, റിവാർഡുകൾ, സ്റ്റേറ്റ്മെന്റുകൾ."
        },
        'Bank account services': {
            'English': "Account access, deposits, withdrawals, transfers, login issues, and other account management queries.",
            'Hindi': "खाता एक्सेस, जमा, निकासी, ट्रांसफर, लॉगिन समस्याएँ और अन्य खाता प्रबंधन प्रश्न।",
            'Marathi': "खाते प्रवेश, जमा, निकासी, ट्रान्सफर, लॉगिन समस्या आणि इतर खाते व्यवस्थापन प्रश्न.",
            'Punjabi': "ਅਕਾਊਂਟ ਐਕਸੈਸ, ਜਮ੍ਹਾਂ, ਨਿਕਾਸ, ਟ੍ਰਾਂਸਫਰ, ਲੌਗਇਨ ਸਮੱਸਿਆਵਾਂ ਅਤੇ ਹੋਰ ਅਕਾਊਂਟ ਪ੍ਰਬੰਧਨ ਪ੍ਰਸ਼ਨ।",
            'Kannada': "ಖಾತೆ ಪ್ರವೇಶ, ಠೇವಣಿ, ಹಿಂಪಡೆಯುವಿಕೆ, ವರ್ಗಾಯನೆ, ಲಾಗಿನ್ ಸಮಸ್ಯೆಗಳು ಮತ್ತು ಇತರೆ ಖಾತೆ ನಿರ್ವಹಣಾ ಪ್ರಶ್ನೆಗಳು.",
            'Malayalam': "അക്കൗണ്ട് ആക്‌സസ്, നിക്ഷേപം, പിൻവിലപ്പുകൾ, കൈമാറ്റങ്ങൾ, ലോഗിൻ പ്രശ്നങ്ങൾ എന്നിവയുമായി ബന്ധപ്പെട്ട കാര്യങ്ങൾ."
        },
        'Theft/Dispute reporting': {
            'English': "Fraud, unauthorized transactions, identity theft, disputes and security incidents. Priority handling recommended.",
            'Hindi': "धोखाधड़ी, अनधिकृत लेनदेन, पहचान की चोरी, विवाद और सुरक्षा घटनाएँ। प्राथमिकता से संभालने की सलाह।",
            'Marathi': "फraud, अनधिकृत व्यवहार, ओळख चोरी, वाद आणि सुरक्षा घटना. प्राधान्याने हाताळण्याचा सल्ला.",
            'Punjabi': "ਧੋਖਾਧੜੀ, ਬਿਨਾਂ ਆਗਿਆ ਵਾਲੇ ਲੈਣ-ਦੇਣ, ਪਛਾਣ ਚੋਰੀ, ਵਿਵਾਦ ਅਤੇ ਸੁਰੱਖਿਆ ਘਟਨਾਵਾਂ। ਤਰਜੀਹੀ ਸੰਭਾਲ ਦੀ ਸਿਫਾਰਿਸ਼।",
            'Kannada': "ವಂಚನೆ, ನಿರಾಕೃತ ಲೆನದेन, ಗುರುತು ಕಳವರು, ವಾದಗಳು ಮತ್ತು ಭದ್ರತಾ ಘಟನೆಗಳು. ಪ್ರಾಥಮ್ಯತೆಯಿಂದ ನಿರ್ವಹಿಸುವ ಶಿಫಾರಸು.",
            'Malayalam': "ഫ്രൗഡ്, അനധികൃത ഇടപാട്, തിരിച്ചറിയൽ മോഷണം, വിവാദങ്ങൾ, സുരക്ഷാ സംഭവങ്ങൾ — പ്രാഥമിക കൈകര്യം നിർദ്ദേശിക്കുന്നു."
        },
        'Mortgages/loans': {
            'English': "Mortgage and loan enquiries — EMI, interest rates, repayment schedules, loan approval, refinancing queries.",
            'Hindi': "ब्याज और ऋण पूछताछ — EMI, ब्याज दरें, पुनर्भुगतान अनुसूची, ऋण स्वीकृति, रिफाइनेंसिंग प्रश्न।",
            'Marathi': "गृहकर्ज/कर्ज संबंधित प्रश्न — EMI, व्याज दर, परतफेड वेळापत्रक, कर्ज मंजुरी, रिफायनान्सिंग प्रश्न.",
            'Punjabi': "ਮੋਰਟਗੇਜ/ਲੋਨ ਸਵਾਲ — EMI, ਬਿਆਜ ਦਰਾਂ, ਵਾਪਸੀ ਸਮਾਂ-ਸੂਚੀ, ਲੋਨ ਮਨਜ਼ੂਰੀ, ਰੀਫਾਇਨੈਂਸਿੰਗ ਪ੍ਰਸ਼ਨ।",
            'Kannada': "ಮಾರ್ಗೇಜ್/ಉಧಾರ ವಿಚಾರಣೆ — EMI, ಬಡ್ಡಿದರ, ಮರುಪಾವತಿ ವೇಳಾಪಟ್ಟಿ, ಸಾಲ ಅಂಗೀಕಾರ, ಮರುನಿವೇಶನ ಪ್ರಶ್ನೆಗಳು.",
            'Malayalam': "മോർട്ട്ഗേജ്/ഋണം സംബന്ധിച്ച അന്വേഷണം — EMI, പലിശനിരക്കുകൾ, പുനരായത് പദ്ധതി, വായ്പ അംഗീകാരം, റിഫൈനാൻസിങ്."
        },
        'Others': {
            'English': "Miscellaneous complaints that don't fit primary categories — will be routed to manual review.",
            'Hindi': "अन्य शिकायतें जो मुख्य श्रेणियों में फिट नहीं होतीं — मैन्युअल समीक्षा के लिए रूट की जाएँगी।",
            'Marathi': "इतर तक्रारी ज्या मुख्य श्रेणीत बसत नाहीत — मैन्युअल पुनरावलोकनासाठी मार्गदर्शित केल्या जातील.",
            'Punjabi': "ਹੋਰ ਸ਼ਿਕਾਇਤਾਂ ਜੋ ਪ੍ਰਮੁੱਖ ਵਰਗਾਂ ਵਿੱਚ ਫਿੱਟ ਨਹੀਂ ਹੁੰਦੀਆਂ — ਮੈਨੁਅਲ ਸਮੀਖਿਆ ਲਈ ਰੂਟ ਕੀਤੀਆਂ ਜਾਣਗੀਆਂ।",
            'Kannada': "ಇತರೆ ಕೊಡಂಬರಿ ಜತೆಗಿನ ದೂರುಗಳು — ಮುಖ್ಯ ವರ್ಗಗಳಿಗೆ ಹೊಂದದವು — ಕೈಯಿಂದ ಪರಿಶೀಲನೆಗೆ ರೂಟ್ ಮಾಡಲಾಗುವುದು.",
            'Malayalam': "പ്രധാന വിഭാഗങ്ങളിലേക്ക് പെട്ടില്ലാത്ത വിവിധ പരാതികൾ — മാനുവൽ റിവ്യൂവിന് റൂട്ടുചെയ്യുന്നു."
        }
    }

DEPT_DESCRIPTIONS = get_department_descriptions()

# --- small sample complaints per department (for quick demo insertion) ---
DEPT_SAMPLES = {
    'Credit card / Prepaid card': "My credit card was double charged for the same purchase and the statement shows an extra fee.",
    'Bank account services': "I can't access my bank account through the mobile app; login keeps failing despite correct credentials.",
    'Theft/Dispute reporting': "There are unauthorized transactions on my account that I did not make — looks like fraud.",
    'Mortgages/loans': "My loan EMI amount is incorrect after the refinance; please check interest rate calculations.",
    'Others': "I have a general product feedback regarding the customer portal layout and accessibility."
}

# --- Database Connection and Table Creation (Cached) ---
@st.cache_resource
def get_mongo_connection():
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        if MAIN_COLLECTION_NAME not in db.list_collection_names():
            db.create_collection(MAIN_COLLECTION_NAME)
        for dept_name, collection_name in DEPARTMENT_COLLECTIONS.items():
            if collection_name not in db.list_collection_names():
                db.create_collection(collection_name)
        return db
    except ConnectionFailure:
        st.error("Error: Could not connect to MongoDB Atlas. Please check your internet connection and URI.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.stop()

db = get_mongo_connection()

# --- Functions for Database Operations ---
@st.cache_data(ttl=5)
def get_all_complaints_from_db():
    try:
        complaints_collection = db[MAIN_COLLECTION_NAME]
        records = list(complaints_collection.find().sort("timestamp", pymongo.DESCENDING))
        df_complaints = pd.DataFrame(records)
        if not df_complaints.empty:
            df_complaints['_id'] = df_complaints['_id'].astype(str)
            df_complaints['timestamp'] = pd.to_datetime(df_complaints['timestamp'])
        return df_complaints
    except OperationFailure as e:
        st.error(f"Error fetching complaints from MongoDB: {e}")
        return pd.DataFrame()

def update_checked_twice_status(complaint_id, status):
    try:
        main_collection = db[MAIN_COLLECTION_NAME]
        # Wrap ObjectId conversion defensively
        try:
            oid = ObjectId(complaint_id)
            doc = main_collection.find_one({"_id": oid})
        except Exception:
            doc = main_collection.find_one({"_id": complaint_id})
        if not doc:
            return False
        predicted_department = doc.get("predicted_department")
        dept_collection_name = DEPARTMENT_COLLECTIONS.get(predicted_department)
        if isinstance(doc.get("_id"), ObjectId):
            query_id = ObjectId(complaint_id)
        else:
            query_id = complaint_id
        result_main = main_collection.update_one({"_id": query_id}, {"$set": {"checked_twice": status}})
        if dept_collection_name:
            dept_collection = db[dept_collection_name]
            dept_collection.update_one({"_id": query_id}, {"$set": {"checked_twice": status}})
        return result_main.modified_count > 0
    except Exception as e:
        st.error(f"Error updating complaint ID {complaint_id}: {e}")
        return False

def delete_complaint(complaint_id):
    try:
        main_collection = db[MAIN_COLLECTION_NAME]
        try:
            oid = ObjectId(complaint_id)
            doc = main_collection.find_one({"_id": oid})
        except Exception:
            doc = main_collection.find_one({"_id": complaint_id})
        if not doc:
            st.warning(f"Complaint ID {complaint_id} not found in the main log. No deletion performed.")
            return False
        predicted_department = doc.get("predicted_department")
        dept_collection_name = DEPARTMENT_COLLECTIONS.get(predicted_department)
        if isinstance(doc.get("_id"), ObjectId):
            query_id = ObjectId(complaint_id)
        else:
            query_id = complaint_id
        result_main = main_collection.delete_one({"_id": query_id})
        if dept_collection_name:
            dept_collection = db[dept_collection_name]
            dept_collection.delete_one({"_id": query_id})
        return result_main.deleted_count > 0
    except Exception as e:
        st.error(f"Error deleting complaint ID {complaint_id}: {e}")
        return False

def log_to_database(data):
    try:
        main_collection = db[MAIN_COLLECTION_NAME]
        timestamp_dt = datetime.strptime(data["Timestamp"], "%Y-%m-%d %H:%M:%S")
        normalized_complaint = data["Complaint"].strip()
        time_window = timestamp_dt - timedelta(seconds=60)
        duplicate_count = main_collection.count_documents({
            "complaint": normalized_complaint,
            "timestamp": {"$gte": time_window}
        })
        if duplicate_count > 0:
            st.warning(" यह शिकायत हाल ही में दोबारा सबमिट की गई है। इसे दोबारा सेव नहीं किया जा रहा है। (This complaint appears to be a duplicate submitted recently. Not logging again.)")
            return False
        complaint_document = {
            "complaint": data["Complaint"],
            "sentiment": data["Sentiment"],
            "score": data["Score"],
            "predicted_department": data["Predicted Department"],
            "checked_twice": data["Checked Twice"],
            "timestamp": timestamp_dt
        }
        result = main_collection.insert_one(complaint_document)
        inserted_id = result.inserted_id
        predicted_department_key = data["Predicted Department"]
        dept_collection_name = DEPARTMENT_COLLECTIONS.get(predicted_department_key)
        if dept_collection_name:
            dept_collection = db[dept_collection_name]
            dept_complaint_document = complaint_document.copy()
            dept_complaint_document['_id'] = inserted_id
            dept_collection.insert_one(dept_complaint_document)
        return True
    except Exception as e:
        st.error(f"Error logging complaint to database: {e}")
        return False

# --- NLTK Downloads (Cached to run only once) ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon', quiet=True)
download_nltk_data()

# --- Initialize VADER sentiment analyzer (Cached) ---
@st.cache_resource
def get_sentiment_analyzer():
    return SentimentIntensityAnalyzer()

# --- Load Model and Vectorizer (Cached) ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        model = joblib.load('ticket_classifier_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Error: Model or vectorizer files not found. "
                 "Please ensure 'ticket_classifier_model.pkl' and 'vectorizer.pkl' "
                 "are in the same directory as this script, and run your Jupyter notebook "
                 "to generate them first.")
        st.stop()
model, vectorizer = load_model_and_vectorizer()

# --- Text Preprocessing & Classifier (unchanged) ---
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_complaint(text):
    analyzer = get_sentiment_analyzer()
    text_lower = text.lower()
    vs = analyzer.polarity_scores(text)
    vader_compound_score = vs['compound']
    negative_keywords_strong = [
        'fraud', 'stole', 'unauthorized', 'scam', 'stolen', 'misleading', 'deceptive',
        'incorrect', 'error', 'issue', 'unacceptable', 'frustrating', 'dispute',
        'complaint', 'problem', 'wrong', 'never received', 'overcharged', 'lost', 'failed',
        'difficult', 'unable', 'bad', 'poor', 'missing', 'denied', 'refused'
    ]
    positive_keywords_strong = [
        'excellent', 'helpful', 'resolved', 'happy', 'satisfied', 'smooth', 'efficient',
        'great', 'thank you', 'appreciate', 'best service', 'easy to use', 'good',
        'quick', 'fast', 'smooth', 'seamless', 'pleased', 'impressed', 'love'
    ]
    sentiment = "Neutral"
    if any(keyword in text_lower for keyword in negative_keywords_strong):
        sentiment = "Negative"
    elif any(keyword in text_lower for keyword in positive_keywords_strong):
        sentiment = "Positive"
    else:
        if vader_compound_score >= 0.05:
            sentiment = "Positive"
        elif vader_compound_score <= -0.05:
            sentiment = "Negative"
    predicted_department = 'Others'
    if any(keyword in text_lower for keyword in [
        'fraud', 'stole', 'unauthorized', 'suspicious email', 'scam', 'stolen',
        'dispute', 'report fraud', 'identity theft', 'phishing', 'data breach',
        'compromised account', 'unrecognized transaction', 'hacked', 'security issue',
        'debt collection', 'collect debt', 'collection agency', 'owed money',
        'bill collector', 'harassment', 'credit bureau', 'debt dispute', 'collection call',
        'credit report', 'credit score', 'credit history', 'dispute credit',
        'reporting error', 'fico', 'equifax', 'transunion', 'experian', 'credit inquiry'
    ]):
        predicted_department = 'Theft/Dispute reporting'
        sentiment = 'Negative'
    elif any(keyword in text_lower for keyword in [
        'credit card', 'prepaid card', 'double charged', 'transaction error',
        'card payment', 'billing issue', 'lost card', 'stolen card', 'card balance',
        'annual fee', 'statement', 'rewards', 'card dispute', 'credit limit', 'apr'
    ]):
        predicted_department = 'Credit card / Prepaid card'
    elif any(keyword in text_lower for keyword in [
        'mortgage', 'loan', 'personal loan', 'home loan', 'auto loan',
        'interest rate', 'refinance', 'payment plan', 'student loan', 'debt',
        'loan application', 'mortgage payment', 'vehicle loan', 'payday loan', 'title loan'
    ]):
        predicted_department = 'Mortgages/loans'
    elif any(keyword in text_lower for keyword in [
        'bank account', 'savings account', 'checking account', 'online banking',
        'mobile app', 'account access', 'deposit', 'withdrawal', 'transfer funds',
        'account balance', 'atm', 'branch', 'login', 'passcode', 'routing number',
        'direct deposit', 'account closed', 'new account', 'money transfer', 'virtual currency', 'money service', 'wire transfer',
        'cryptocurrency', 'send money', 'receive money', 'payment app', 'remittance'
    ]):
        predicted_department = 'Bank account services'
    else:
        try:
            processed_text_for_ml = preprocess_text(text)
            vec = vectorizer.transform([processed_text_for_ml])
            ml_prediction = model.predict(vec)[0]
            if 'credit card' in ml_prediction.lower() or 'prepaid card' in ml_prediction.lower():
                predicted_department = 'Credit card / Prepaid card'
            elif 'bank account' in ml_prediction.lower() or 'checking' in ml_prediction.lower() or 'savings' in ml_prediction.lower() or 'money transfer' in ml_prediction.lower():
                predicted_department = 'Bank account services'
            elif 'mortgage' in ml_prediction.lower() or 'loan' in ml_prediction.lower():
                predicted_department = 'Mortgages/loans'
            elif 'debt collection' in ml_prediction.lower() or 'credit reporting' in ml_prediction.lower() or 'consumer reports' in ml_prediction.lower():
                predicted_department = 'Theft/Dispute reporting'
            else:
                predicted_department = 'Others'
        except Exception as e:
            st.warning(f"Warning: Error using ML model for classification: {e}. Defaulting to 'Others'.")
            predicted_department = 'Others'
    return {
        "Complaint": text,
        "Sentiment": sentiment,
        "Score": round(vader_compound_score, 4),
        "Predicted Department": predicted_department,
        "Checked Twice": "Pending Review",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# --- Excel Export Function ---
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df_for_excel = df.rename(columns={'_id': 'ID', 'complaint': 'Complaint', 'sentiment': 'Sentiment',
                                     'score': 'Score', 'predicted_department': 'Department',
                                     'checked_twice': 'Status', 'timestamp': 'Timestamp'})
    df_for_excel.to_excel(writer, index=False, sheet_name='Complaints')
    writer.close()
    return output.getvalue()

# --- Styling (compact card UI) ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    .mini-card {
        background: linear-gradient(180deg,#f7f8ff 0%,#eef2ff 100%);
        border-radius:10px;
        padding:10px 12px;
        text-align:center;
        font-weight:700;
        color:#2b0b5a;
        border:1px solid #e6e9ff;
        box-shadow:0 6px 18px rgba(99,102,241,0.06);
    }
    .mini-card-title { font-size:0.95rem; margin-bottom:6px; }
    .mini-card-note { font-size:0.82rem; color:#6b7280; margin-top:6px; }
    .info-row { display:flex; justify-content:space-between; align-items:center; gap:8px; }
    .compact-btn { padding:6px 10px; font-size:0.9rem; }
    </style>
    """, unsafe_allow_html=True
)

# --- Session State Initialization ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ''
if 'last_result' not in st.session_state: st.session_state.last_result = None
if 'complaint_input_key' not in st.session_state: st.session_state.complaint_input_key = 0
if 'is_processing' not in st.session_state: st.session_state.is_processing = False
if 'current_complaint_text' not in st.session_state: st.session_state.current_complaint_text = ""
if 'last_logged_complaint_text' not in st.session_state: st.session_state.last_logged_complaint_text = ""
if 'last_logged_complaint_timestamp' not in st.session_state: st.session_state.last_logged_complaint_timestamp = None
if 'active_tab_index' not in st.session_state or not isinstance(st.session_state.active_tab_index, int):
    st.session_state.active_tab_index = 0

# --- Sidebar Login ---
st.sidebar.title("Login / Support")
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
        st.session_state.active_tab_index = 0
        st.success("Logged out successfully.")
        st.rerun()
else:
    with st.sidebar.form("login_form"):
        st.write("### Company Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_button = st.form_submit_button("Login")
        if login_button:
            with st.sidebar.empty():
                st.markdown("""<div style="text-align:center"><div class="loader"></div><div class="loader-text">Logging in...</div></div>""", unsafe_allow_html=True)
                time.sleep(1.0)
            if (username == "Sharma.akhil" and password == "123456789") or (username == "Bhalu_ka_pati" and password == "Bhalu_loves_me"):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success("Login successful!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")

# --- Header & Compact Categories Board ---
cols = st.columns([1, 3, 1])
with cols[1]:
    try:
        st.image("Gemini_Generated_Image_sc8m3ysc8m3ysc8m.png", width=120)
    except Exception:
        pass
    st.markdown('<h2 style="text-align:center;color:#4B0082;margin:0.2rem 0">Customer Complaint Classification</h2>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h4 style='margin-bottom:0.4rem;'>Explore Our Complaint Categories — click Info to learn more</h4>", unsafe_allow_html=True)

# Compact 3-column layout of department mini-cards with Info button that opens a modal
dept_keys = list(DEPARTMENT_COLLECTIONS.keys())
cols = st.columns(3)
for idx, dept in enumerate(dept_keys):
    col = cols[idx % 3]
    with col:
        st.markdown(f'<div class="mini-card"><div class="mini-card-title">{dept}</div><div class="mini-card-note">Tap Info for details & sample</div></div>', unsafe_allow_html=True)
        if st.button("Info", key=f"info_btn_{idx}", help="Open details dialog"):
            # open a modal with multilingual description and sample insertion
            modal_title = f"{dept} — Details"
            try:
                with st.modal(modal_title):
                    # top row: language selector + close (close provided by builtin modal)
                    languages = list(DEPT_DESCRIPTIONS.get(dept, {}).keys())
                    selected_lang = st.selectbox("Language", languages, index=0)
                    st.markdown("---")
                    st.markdown(f"**{selected_lang}**")
                    st.write(DEPT_DESCRIPTIONS.get(dept, {}).get(selected_lang, "No description available."))
                    st.markdown("---")
                    st.write("**Sample complaint (one-click insert):**")
                    st.code(DEPT_SAMPLES.get(dept, ""))
                    st.write("")
                    insert_col1, insert_col2 = st.columns([2,1])
                    with insert_col1:
                        if st.button("Insert sample into complaint box", key=f"insert_sample_{idx}"):
                            # Populate the current complaint text area for the current input key
                            input_key = f"complaint_input_{st.session_state.complaint_input_key}"
                            st.session_state[input_key] = DEPT_SAMPLES.get(dept, "")
                            st.success("Sample inserted into complaint box.")
                    with insert_col2:
                        if st.button("Close", key=f"close_modal_btn_{idx}"):
                            # just exit modal; nothing else required
                            pass
            except Exception:
                # Fallback to expanders if st.modal not available in the environment
                st.warning("Modal not supported in this Streamlit runtime. Showing inline details instead.")
                st.expander(f"{dept} details (fallback)")
                st.markdown(f"**English:** {DEPT_DESCRIPTIONS[dept].get('English','')}")
                st.write("")
st.markdown("---")

# --- Complaint input & submit (unchanged) ---
complaint_key = f"complaint_input_{st.session_state.complaint_input_key}"
complaint_text = st.text_area(
    "Write Your Complaint:",
    height=150,
    placeholder="Describe your issue here...",
    key=complaint_key,
    value="" if st.session_state.is_processing else st.session_state.get(complaint_key, "")
)
loader_placeholder = st.empty()
submit_button = st.button("Submit Complaint", key="main_submit_complaint_button")
if submit_button:
    if complaint_text:
        now = datetime.now()
        normalized_current_complaint = complaint_text.strip()
        if (st.session_state.last_logged_complaint_text == normalized_current_complaint and
            st.session_state.last_logged_complaint_timestamp and
            (now - st.session_state.last_logged_complaint_timestamp).total_seconds() < 60):
            st.warning(" यह शिकायत हाल ही में दोबारा सबमिट की गई है। कृपया कुछ देर बाद कोशिश करें या एक नई शिकायत दर्ज करें। (This complaint has been submitted recently. Please try again later or submit a new complaint.)")
            st.session_state.is_processing = False
            st.session_state.last_result = None
        else:
            st.session_state.current_complaint_text = complaint_text
            st.session_state.is_processing = True
            st.session_state.complaint_input_key += 1
            st.session_state.last_result = None
            st.rerun()
    else:
        st.warning("कृपया अपनी शिकायत दर्ज करें। (Please enter your complaint before submitting.)")

if st.session_state.is_processing:
    loader_placeholder.markdown(
        """
        <div style="text-align:center">
            <div class="loader"></div>
            <div class="loader-text">Analyzing complaint and routing...</div>
        </div>
        """, unsafe_allow_html=True
    )
    time.sleep(2.0)
    result = classify_complaint(st.session_state.current_complaint_text)
    if log_to_database(result):
        st.session_state.last_result = result
        st.session_state.last_logged_complaint_text = result["Complaint"].strip()
        st.session_state.last_logged_complaint_timestamp = datetime.now()
        loader_placeholder.empty()
        st.success(f" शिकायत सबमिट हो गई है और **{result['Predicted Department']}** विभाग को वर्गीकृत कर दी गई है। (Complaint submitted and classified to **{result['Predicted Department']}**.)")
    else:
        st.session_state.is_processing = False
        loader_placeholder.empty()
        st.error("शिकायत को डेटाबेस में लॉग करने में विफलता। कृपया पुन: प्रयास करें।")
    st.session_state.is_processing = False

# Show the last classification result
if st.session_state.last_result and not st.session_state.is_processing:
    st.markdown("---")
    st.markdown('<h3 style="margin-bottom:0.4rem;">Classification Result</h3>', unsafe_allow_html=True)
    rd = st.session_state.last_result
    st.markdown(f"""
    <div style="display:flex;gap:12px;flex-wrap:wrap;">
        <div style="background:#F8F8FF;padding:12px;border-radius:10px;min-width:140px;">
            <strong>Sentiment</strong><div style="font-size:1.05rem">{rd['Sentiment']}</div>
        </div>
        <div style="background:#F8F8FF;padding:12px;border-radius:10px;min-width:140px;">
            <strong>Score</strong><div style="font-size:1.05rem">{rd['Score']:.4f}</div>
        </div>
        <div style="background:#F8F8FF;padding:12px;border-radius:10px;min-width:140px;">
            <strong>Department</strong><div style="font-size:1.05rem">{rd['Predicted Department']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.info(f"**Original Complaint:** {rd['Complaint']}")

# --- About / Help ---
st.markdown("---")
with st.expander("About This App / Help", expanded=False):
    st.markdown("""
    This application classifies customer complaints into departmental categories and analyzes sentiment.
    Hybrid approach: rule-based keywords + VADER + ML fallback model.
    Use the compact Info buttons above to open modal dialogs (language selector + sample insertion).
    """)

# --- Admin Portal (visible after login) ---
if st.session_state.logged_in:
    st.markdown("---")
    st.markdown('<h2>Company Support Portal: Overview</h2>', unsafe_allow_html=True)
    st.write(f"Welcome, **{st.session_state.username}**! Here's an overview of customer complaints and tools to manage them.")
    if st.button("Refresh All Portal Data", key="support_refresh_all_button"):
        st.cache_data.clear()
        st.rerun()
    all_complaints_df = get_all_complaints_from_db()

    # Use radio to emulate tabs so active tab can be controlled programmatically
    TAB_LABELS = ["📊 Dashboard & Visualizations", "📝 Manage & Update Complaints"]
    selected_tab_label = st.radio("", TAB_LABELS, index=st.session_state.active_tab_index, horizontal=True, key="main_tabs_radio")
    st.session_state.active_tab_index = TAB_LABELS.index(selected_tab_label)

    if selected_tab_label == TAB_LABELS[0]:
        if not all_complaints_df.empty:
            st.markdown("<h3>Complaint Analytics</h3>", unsafe_allow_html=True)
            total_complaints = len(all_complaints_df)
            negative_count = all_complaints_df[all_complaints_df['sentiment'] == 'Negative'].shape[0]
            pending_review_count = all_complaints_df[all_complaints_df['checked_twice'] == 'Pending Review'].shape[0]
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Complaints", total_complaints)
            with col2: st.metric("Negative Complaints", negative_count)
            with col3: st.metric("Pending Review", pending_review_count)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                department_counts = all_complaints_df['predicted_department'].value_counts().reset_index()
                department_counts.columns = ['Department', 'Count']
                fig_dept = px.bar(department_counts, x='Department', y='Count', color='Department', template='plotly_white')
                st.plotly_chart(fig_dept, use_container_width=True)
            with chart_col2:
                sentiment_counts = all_complaints_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig_sent = px.pie(sentiment_counts, values='Count', names='Sentiment', template='plotly_white')
                st.plotly_chart(fig_sent, use_container_width=True)
            daily_counts = all_complaints_df.groupby(all_complaints_df['timestamp'].dt.date).size().reset_index(name='Count')
            daily_counts.columns = ['Date', 'Count']
            fig_time = px.line(daily_counts, x='Date', y='Count', template='plotly_white')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No data available for analytics. Submit some complaints first!")
    else:
        # Manage & Update Complaints
        st.markdown("<h3>Search & Filter Complaints</h3>", unsafe_allow_html=True)
        with st.expander("Filter Options", expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1:
                search_query = st.text_input("Keyword Search in Complaint", key="search_keyword_filter_tab")
            with f2:
                selected_department = st.selectbox("Filter by Department", ["All"] + list(DEPARTMENT_COLLECTIONS.keys()), key="filter_department_tab")
            with f3:
                selected_sentiment = st.multiselect("Filter by Sentiment", ["Positive", "Negative", "Neutral"], default=[], key="filter_sentiment_tab")
            date_c1, date_c2 = st.columns(2)
            if not all_complaints_df.empty:
                min_date = all_complaints_df['timestamp'].min().date()
                max_date = all_complaints_df['timestamp'].max().date()
            else:
                min_date = datetime.today().date()
                max_date = datetime.today().date()
            with date_c1:
                start_date = st.date_input("Start Date", value=min_date, key="filter_start_date_tab")
            with date_c2:
                end_date = st.date_input("End Date", value=max_date, key="filter_end_date_tab")
        filtered_df_for_tab = all_complaints_df.copy()
        if search_query:
            filtered_df_for_tab = filtered_df_for_tab[filtered_df_for_tab['complaint'].str.contains(search_query, case=False, na=False)]
        if selected_department != "All":
            filtered_df_for_tab = filtered_df_for_tab[filtered_df_for_tab['predicted_department'] == selected_department]
        if selected_sentiment:
            filtered_df_for_tab = filtered_df_for_tab[filtered_df_for_tab['sentiment'].isin(selected_sentiment)]
        if not filtered_df_for_tab.empty:
            filtered_df_for_tab = filtered_df_for_tab[(filtered_df_for_tab['timestamp'].dt.date >= start_date) & (filtered_df_for_tab['timestamp'].dt.date <= end_date)]
        st.write(f"Displaying {len(filtered_df_for_tab)} complaints after filtering in this tab:")
        st.dataframe(filtered_df_for_tab.rename(columns={'_id': 'ID'}).drop(columns=['_id'], errors='ignore'), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("<h3>Update Complaint Status</h3>", unsafe_allow_html=True)
        available_ids_for_update = filtered_df_for_tab['_id'].tolist() if not filtered_df_for_tab.empty else []
        selected_complaint_id_str_update = st.selectbox("Select Complaint ID to Update (from table above)", [""] + available_ids_for_update, key="select_complaint_id_to_update_tab")
        if selected_complaint_id_str_update:
            current_row = all_complaints_df[all_complaints_df['_id'] == selected_complaint_id_str_update]
            if not current_row.empty:
                row = current_row.iloc[0]
                st.info(f"**Complaint ID {selected_complaint_id_str_update}:** {row['complaint']}")
                st.write(f"**Current Status:** {row['checked_twice']}")
                status_options = ["Pending Review", "Reviewed - Action Taken", "Reviewed - No Action Needed", "Resolved"]
                try:
                    current_index = status_options.index(row['checked_twice'])
                except ValueError:
                    current_index = 0
                new_status = st.radio("Set New Status:", status_options, index=current_index, key=f"status_radio_tab_{selected_complaint_id_str_update}")
                if st.button(f"Update Status for ID {selected_complaint_id_str_update}", key=f"update_button_tab_{selected_complaint_id_str_update}"):
                    if update_checked_twice_status(selected_complaint_id_str_update, new_status):
                        st.session_state.active_tab_index = 1
                        st.success(f"Status for Complaint ID {selected_complaint_id_str_update} updated to '{new_status}'.")
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.session_state.active_tab_index = 1
                        st.error("Failed to update status.")
                        st.rerun()
        st.markdown("---")
        st.markdown("<h3>Delete Complaint</h3>", unsafe_allow_html=True)
        available_ids_for_delete = filtered_df_for_tab['_id'].tolist() if not filtered_df_for_tab.empty else []
        selected_complaint_id_str_delete = st.selectbox("Select Complaint to Delete", [""] + available_ids_for_delete, key="select_complaint_id_to_delete_tab")
        if selected_complaint_id_str_delete:
            complaint_row = all_complaints_df[all_complaints_df['_id'] == selected_complaint_id_str_delete]
            if not complaint_row.empty:
                st.error(f"You are about to DELETE Complaint ID {selected_complaint_id_str_delete}: \"{complaint_row.iloc[0]['complaint'][:80]}...\"")
                confirm_delete = st.checkbox(f"I understand this action cannot be undone and wish to delete Complaint ID {selected_complaint_id_str_delete}.", key=f"confirm_delete_{selected_complaint_id_str_delete}")
                if confirm_delete:
                    if st.button(f"CONFIRM DELETE Complaint ID {selected_complaint_id_str_delete}", key=f"delete_button_final_{selected_complaint_id_str_delete}"):
                        if delete_complaint(selected_complaint_id_str_delete):
                            st.session_state.active_tab_index = 1
                            st.success("Complaint deleted successfully!")
                            st.cache_data.clear()
                            st.rerun()
                        else:
                            st.session_state.active_tab_index = 1
                            st.error("Deletion failed.")
                            st.rerun()
