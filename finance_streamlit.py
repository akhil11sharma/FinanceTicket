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
from fpdf import FPDF
from base64 import b64encode

# For charts (install if you don't have it: pip install plotly)
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

# --- Database Connection and Table Creation (Cached) ---
@st.cache_resource
def get_mongo_connection():
    """
    Establishes and returns a MongoDB client connection.
    Ensures the main and department-specific collections exist.
    """
    try:
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        
        # Ensure main collection exists
        if MAIN_COLLECTION_NAME not in db.list_collection_names():
            db.create_collection(MAIN_COLLECTION_NAME)
        
        # Ensure department-specific collections exist
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

# Initialize DB connection (this will run once at app start)
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
    # This function now needs to update both the main collection and the department-specific collection
    try:
        main_collection = db[MAIN_COLLECTION_NAME]
        
        # First, find the document to get the predicted_department
        doc = main_collection.find_one({"_id": ObjectId(complaint_id)})
        if not doc:
            return False

        predicted_department = doc.get("predicted_department")
        dept_collection_name = DEPARTMENT_COLLECTIONS.get(predicted_department)

        # Update main collection
        result_main = main_collection.update_one(
            {"_id": ObjectId(complaint_id)},
            {"$set": {"checked_twice": status}}
        )
        
        # Update department-specific collection
        if dept_collection_name:
            dept_collection = db[dept_collection_name]
            result_dept = dept_collection.update_one(
                {"_id": ObjectId(complaint_id)},
                {"$set": {"checked_twice": status}}
            )

        return result_main.modified_count > 0
    except Exception as e:
        st.error(f"Error updating complaint ID {complaint_id}: {e}")
        return False

def delete_complaint(complaint_id):
    # This function now needs to delete from both the main and department-specific collections
    try:
        main_collection = db[MAIN_COLLECTION_NAME]
        
        # First, find the document to get the predicted_department
        doc = main_collection.find_one({"_id": ObjectId(complaint_id)})
        if not doc:
            st.warning(f"Complaint ID {complaint_id} not found in the main log. No deletion performed.")
            return False

        predicted_department = doc.get("predicted_department")
        dept_collection_name = DEPARTMENT_COLLECTIONS.get(predicted_department)

        # Delete from main collection
        result_main = main_collection.delete_one({"_id": ObjectId(complaint_id)})

        # Delete from department-specific collection
        if dept_collection_name:
            dept_collection = db[dept_collection_name]
            dept_collection.delete_one({"_id": ObjectId(complaint_id)})
        
        return result_main.deleted_count > 0
    except Exception as e:
        st.error(f"Error deleting complaint ID {complaint_id}: {e}")
        return False

def log_to_database(data):
    try:
        main_collection = db[MAIN_COLLECTION_NAME]
        timestamp_dt = datetime.strptime(data["Timestamp"], "%Y-%m-%d %H:%M:%S")

        # --- DUPLICATE CHECK ---
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
        
        # 1. Insert into main collection
        result = main_collection.insert_one(complaint_document)
        inserted_id = result.inserted_id

        # 2. Insert into department-specific collection
        predicted_department_key = data["Predicted Department"]
        dept_collection_name = DEPARTMENT_COLLECTIONS.get(predicted_department_key)

        if dept_collection_name:
            dept_collection = db[dept_collection_name]
            # Create a copy of the document and insert it.
            dept_complaint_document = complaint_document.copy()
            # It's important to use the same _id to link the documents
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
        nltk.download('averaged_perceptron_percept_tagger', quiet=True)
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

# --- Text Preprocessing ---
def preprocess_text(text):
    """Performs basic text preprocessing."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Classifier Function ---
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

# --- PDF Export Function ---
def to_pdf(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Adding a title
    pdf.cell(200, 10, txt="Customer Complaints Report", ln=True, align='C')
    pdf.cell(200, 10, txt="---", ln=True, align='C')

    # Convert DataFrame to a list of lists (including headers) for the table
    df_for_pdf = df.rename(columns={'_id': 'ID', 'complaint': 'Complaint', 'sentiment': 'Sentiment',
                                     'score': 'Score', 'predicted_department': 'Department',
                                     'checked_twice': 'Status', 'timestamp': 'Timestamp'})
    data = [df_for_pdf.columns.to_list()] + df_for_pdf.values.tolist()

    # Create table
    col_widths = [20, 80, 20, 15, 25, 20] # Define column widths to fit A4
    for row in data:
        # Manually wrap long complaint text
        complaint_text = row[1]
        lines = pdf.multi_cell(col_widths[1], 10, str(complaint_text), align='L', border=1, ln=True, split_only=True)
        
        if len(lines) > 1:
            # Handle wrapped lines
            first_line_text = lines[0]
            for i, line in enumerate(lines):
                # Print other columns only for the first line
                if i == 0:
                    pdf.cell(col_widths[0], 10, str(row[0]), border=1) # ID
                    pdf.cell(col_widths[1], 10, first_line_text, border=1) # Complaint (first line)
                    pdf.cell(col_widths[2], 10, str(row[2]), border=1) # Sentiment
                    pdf.cell(col_widths[3], 10, str(row[3]), border=1) # Score
                    pdf.cell(col_widths[4], 10, str(row[4]), border=1) # Department
                    pdf.cell(col_widths[5], 10, str(row[5]), border=1, ln=True) # Status
                else:
                    pdf.cell(col_widths[0], 10, "", border=1) # Empty ID cell
                    pdf.cell(col_widths[1], 10, line, border=1) # Subsequent complaint lines
                    pdf.cell(col_widths[2], 10, "", border=1) # Empty cells
                    pdf.cell(col_widths[3], 10, "", border=1)
                    pdf.cell(col_widths[4], 10, "", border=1)
                    pdf.cell(col_widths[5], 10, "", border=1, ln=True)
        else:
            # Single line, print normally
            for i, item in enumerate(row):
                pdf.cell(col_widths[i], 10, str(item), border=1)
            pdf.ln()

    # Save the PDF to a buffer and return
    pdf_output = pdf.output(dest='S').encode('latin-1')
    return pdf_output


# --- Excel Export Function ---
def to_excel(df):
    output = io.BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    # Rename columns for a cleaner Excel sheet
    df_for_excel = df.rename(columns={'_id': 'ID', 'complaint': 'Complaint', 'sentiment': 'Sentiment',
                                     'score': 'Score', 'predicted_department': 'Department',
                                     'checked_twice': 'Status', 'timestamp': 'Timestamp'})
    df_for_excel.to_excel(writer, index=False, sheet_name='Complaints')
    writer.close()
    processed_data = output.getvalue()
    return processed_data


# --- Streamlit UI ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 3.5em;
        color: #4B0082;
        text-align: center;
        margin-bottom: 1.5em;
        font-weight: 700;
        letter-spacing: -1px;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        animation: header-fade-in 1.5s ease-out;
    }
    @keyframes header-fade-in {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .department-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin-bottom: 20px;
    }
    .department-box {
        background-color: #EEF2FF;
        color: #4B0082;
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        flex: 1 1 auto;
        min-width: 180px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
        border: 1px solid #C7D2FE;
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInSlideUp 0.6s forwards ease-out;
    }
    .department-box:nth-child(1) { animation-delay: 0.1s; }
    .department-box:nth-child(2) { animation-delay: 0.2s; }
    .department-box:nth-child(3) { animation-delay: 0.3s; }
    .department-box:nth-child(4) { animation-delay: 0.4s; }
    .department-box:nth-child(5) { animation-delay: 0.5s; }
    .department-box:hover {
        background-color: #E0E7FF;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        transform: translateY(-5px);
    }
    .department-box.highlighted {
        background-color: #6366F1;
        color: white;
        box-shadow: 0 0 15px 5px rgba(99, 102, 241, 0.7),
                        0 0 25px 10px rgba(79, 70, 229, 0.5),
                        0 0 35px 15px rgba(59, 130, 246, 0.3);
        transform: scale(1.05);
        border: 2px solid #4F46E5;
        animation: pulse-glow 1.5s infinite alternate;
    }
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 5px 2px rgba(99, 102, 241, 0.4), 0 0 10px 5px rgba(79, 70, 229, 0.2); }
        100% { box-shadow: 0 0 15px 5px rgba(99, 102, 241, 0.7), 0 0 25px 10px rgba(79, 70, 229, 0.5), 0 0 35px 15px rgba(59, 130, 246, 0.3); }
    }
    .stTextArea textarea {
        background-color: #EFEFEF !important;
        border-radius: 10px !important;
        border: 1px solid #D1D5DB !important;
        padding: 12px !important;
        font-size: 1rem !important;
        color: #333333 !important;
        transition: all 0.2s ease-in-out;
    }
    .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        outline: none;
    }
    .stButton > button {
        width: 100%;
        background-color: #6366F1;
        color: white;
        font-weight: 700;
        padding: 12px 20px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #4F46E5;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stAlert { border-radius: 10px; }
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stDataFrame > div { border-radius: 10px; }
    .stDataFrame table { border-collapse: collapse; width: 100%; }
    .stDataFrame th {
        background-color: #EEF2FF;
        color: #4B0082;
        font-weight: 600;
        padding: 10px;
        text-align: left;
        border-bottom: 2px solid #C7D2FE;
    }
    .stDataFrame td {
        padding: 10px;
        border-bottom: 1px solid #E0E7FF;
    }
    .stDataFrame tbody tr:nth-child(even) { background-color: #F9FAFB; }
    .stDataFrame tbody tr:hover { background-color: #F3F4F6; }
    .stMarkdown h2 {
        color: #4B0082;
        text-align: center;
        margin-top: 2em;
        margin-bottom: 1em;
        font-weight: 600;
        font-size: 1.8em;
    }
    .stMetric {
        background-color: #F0F4F8;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stMetric label {
        font-weight: 500;
        color: #6B7280;
    }
    .stMetric .css-1q8dd3e {
        font-size: 1.8em;
        font-weight: 700;
        --sentiment-positive-color: #10B981;
        --sentiment-negative-color: #EF4444;
        --sentiment-neutral-color: #F59E0B;
    }
    .stMetric .css-1q8dd3e[data-sentiment="Positive"] { color: var(--sentiment-positive-color); }
    .stMetric .css-1q8dd3e[data-sentiment="Negative"] { color: var(--sentiment-negative-color); }
    .stMetric .css-1q8dd3e[data-sentiment="Neutral"] { color: var(--sentiment-neutral-color); }
    .pendulum-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px auto;
        min-height: 100px;
    }
    .pendulum_box {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100px;
        height: 30px;
    }
    .ball {
        width: 15px;
        height: 15px;
        background-color: #6366F1;
        border-radius: 50%;
        margin: 0 2px;
        animation: pendulum_swing 2s infinite ease-in-out;
    }
    .ball.first { transform-origin: 100% center; }
    .ball.last { transform-origin: 0% center; }
    .ball:nth-child(2) { animation-delay: 0.1s; }
    .ball:nth-child(3) { animation-delay: 0.2s; }
    .ball:nth-child(4) { animation-delay: 0.3s; }
    .ball.last { animation-delay: 0.4s; }
    @keyframes pendulum_swing {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-15px) rotate(-45deg); }
        75% { transform: translateX(15px) rotate(45deg); }
    }
    .loader-text {
        font-size: 1.1em;
        font-weight: 600;
        color: #4B0082;
        text-align: center;
        margin-top: 10px;
    }
    .stSidebar .stForm {
        background-color: #EEF2FF;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 6px 15px rgba(0,0,0,0.15);
        margin-top: 25px;
        transition: all 0.3s ease-in-out;
        border: 1px solid #C7D2FE;
    }
    .stSidebar .stForm:hover { box-shadow: 0 8px 20px rgba(0,0,0,0.2); }
    .stSidebar .stForm label {
        color: #4B0082;
        font-weight: 600;
        margin-bottom: 8px;
        display: block;
    }
    .stSidebar .stTextInput > div > div > input {
        background-color: #FFFFFF;
        border-radius: 8px !important;
        border: 1px solid #D1D5DB !important;
        padding: 10px 12px !important;
        color: #333333 !important;
        transition: all 0.2s ease-in-out;
    }
    .stSidebar .stTextInput > div > div > input:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.3) !important;
        outline: none;
    }
    .stSidebar .stForm .stButton > button {
        width: 100%;
        background-color: #6366F1;
        color: white;
        font-weight: 700;
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-top: 20px;
        letter-spacing: 0.5px;
    }
    .stSidebar .stForm .stButton > button:hover {
        background-color: #4F46E5;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }
    .stSidebar .stForm .stButton > button:active {
        background-color: #3B34AC;
        transform: translateY(1px) scale(0.98);
        box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    }
    .stMetric > div[data-testid="stMetricValue"] { font-size: 2em; }
    .stExpander span[data-testid="stExpanderToggleIcon"] { color: #4B0082; }
    .stExpander button[data-testid="stExpanderToggle"] {
        background-color: #EEF2FF;
        border-radius: 10px;
        border: 1px solid #C7D2FE;
        padding: 10px 15px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #4B0082;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .stExpander button[data-testid="stExpanderToggle"]:hover {
        background-color: #E0E7FF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .result-card-container {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
        margin-top: 20px;
    }
    .result-card {
        background-color: #F8F8FF;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        flex: 1 1 calc(33% - 20px);
        min-width: 250px;
        max-width: 350px;
        transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
        opacity: 0;
        transform: translateY(20px);
        animation: fadeInSlideUp 0.6s forwards ease-out;
    }
    .result-card:nth-child(1) { animation-delay: 0.1s; }
    .result-card:nth-child(2) { animation-delay: 0.2s; }
    .result-card:nth-child(3) { animation-delay: 0.3s; }
    @keyframes fadeInSlideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-card h4 {
        color: #4B0082;
        margin-bottom: 10px;
        font-weight: 600;
        font-size: 1.2em;
    }
    .result-value {
        font-size: 2.2em;
        font-weight: 800;
        color: #6366F1;
        word-break: break-word;
    }
    .result-value.positive { color: #10B981; }
    .result-value.negative { color: #EF4444; }
    .result-value.neutral { color: #F59E0B; }
    .result-card small {
        display: block;
        margin-top: 10px;
        color: #6B7280;
        font-size: 0.85em;
    }
    .stSubmitButton > button { animation: pulseButton 2s infinite ease-in-out; }
    @keyframes pulseButton {
        0% { box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 0 0 0 rgba(99, 102, 241, 0.4); }
        70% { box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 0 0 10px rgba(99, 102, 241, 0); }
        100% { box-shadow: 0 4px 6px rgba(0,0,0,0.1), 0 0 0 0 rgba(99, 102, 241, 0); }
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: nowrap;
        border-radius: 10px 10px 0 0;
        margin-bottom: -3px;
        background-color: #EEF2FF;
        border: 1px solid #C7D2FE;
        border-bottom: none;
        color: #4B0082;
        font-weight: 600;
        font-size: 1.1em;
        transition: all 0.2s ease-in-out;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E0E7FF;
        color: #3B34AC;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1;
        color: white;
        border: 1px solid #6366F1;
        border-bottom: none;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    }
    .stTabs [data-baseweb="tab-panel"] {
        border: 1px solid #C7D2FE;
        border-radius: 0 0 10px 10px;
        padding: 20px;
        background-color: #F8F8FF;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# --- Session State Initialization for Login ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'complaint_input_key' not in st.session_state:
    st.session_state.complaint_input_key = 0
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'current_complaint_text' not in st.session_state:
    st.session_state.current_complaint_text = ""
if 'last_logged_complaint_text' not in st.session_state:
    st.session_state.last_logged_complaint_text = ""
if 'last_logged_complaint_timestamp' not in st.session_state:
    st.session_state.last_logged_complaint_timestamp = None

# --- Login/Logout Functionality in Sidebar ---
st.sidebar.title("Login / Support")
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ''
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
                st.markdown(
                    """
                    <div class="pendulum-container">
                        <div class="pendulum_box">
                            <div class="ball first"></div>
                            <div class="ball"></div>
                            <div class="ball"></div>
                            <div class="ball"></div>
                            <div class="ball last"></div>
                        </div>
                        <div class="loader-text">Logging in...</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(1)
            # --- UPDATED LOGIN LOGIC TO INCLUDE A SECOND USER ---
            if (username == "Sharma.akhil" and password == "123456789") or \
               (username == "Bhalu_ka_pati" and password == "Bhalu_loves_me"):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success("Login successful!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")
# --- Main App Content ---
st.markdown('<p class="main-header">Customer Complaint Classification</p>', unsafe_allow_html=True)
# --- Department Categories ---
st.markdown("<h3>Explore Our Complaint Categories</h3>", unsafe_allow_html=True)
st.markdown('<div class="department-container">', unsafe_allow_html=True)
for dept in DEPARTMENT_COLLECTIONS.keys():
    st.markdown(f'<div class="department-box">{dept}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
# Complaint Input
complaint_text = st.text_area(
    "Write Your Complaint:",
    height=150,
    placeholder="Describe your issue here...",
    key=f"complaint_input_{st.session_state.complaint_input_key}",
    value="" if st.session_state.is_processing else st.session_state.get(f"complaint_input_{st.session_state.complaint_input_key}", "")
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
# --- Conditional Processing Block (runs only when is_processing is True) ---
if st.session_state.is_processing:
    loader_placeholder.markdown(
        """
        <div class="pendulum-container">
            <div class="pendulum_box">
                <div class="ball first"></div>
                <div class="ball"></div>
                <div class="ball"></div>
                <div class="ball"></div>
                <div class="ball last"></div>
            </div>
            <div class="loader-text">Analyzing complaint and routing...</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    result = classify_complaint(st.session_state.current_complaint_text)
    if log_to_database(result):
        st.session_state.last_result = result
        st.session_state.last_logged_complaint_text = result["Complaint"].strip()
        st.session_state.last_logged_complaint_timestamp = datetime.now()
        loader_placeholder.empty()
        st.success(f" शिकायत सबमिट हो गई है और **{result['Predicted Department']}** विभाग को वर्गीकृत कर दी गई है। आपके समय के लिए धन्यवाद! (Complaint submitted and classified to **{result['Predicted Department']}** department. Thank you for your time!)")
    else:
        st.session_state.is_processing = False
        loader_placeholder.empty()
        st.error("शिकायत को डेटाबेस में लॉग करने में विफलता। कृपया पुन: प्रयास करें। (Failed to log complaint to database. Please try again.)")
    st.session_state.is_processing = False
if st.session_state.last_result and not st.session_state.is_processing:
    st.markdown("---")
    st.markdown('<h2>Classification Result</h2>', unsafe_allow_html=True)
    result_data = st.session_state.last_result
    sentiment_class = result_data['Sentiment'].lower()
    st.markdown('<div class="result-card-container">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
        <h4>Sentiment</h4>
        <div class="result-value {sentiment_class}">{result_data['Sentiment']}</div>
        <small>Overall emotional tone of the complaint.</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
        <h4>Sentiment Score</h4>
        <div class="result-value">{result_data['Score']:.4f}</div>
        <small>VADER compound score (-1.0 to 1.0).</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="result-card">
        <h4>Predicted Department</h4>
        <div class="result-value">{result_data['Predicted Department']}</div>
        <small>Automatically routed to this department.</small>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"**Original Complaint:** {result_data['Complaint']}")
# --- New: About/Help Section (Accessible to all users) ---
st.markdown("---")
st.markdown('<h2>About This App / Help</h2>', unsafe_allow_html=True)
with st.expander("Learn more about the Complaint Classification App", expanded=False):
    st.markdown("""
    **Purpose of the App:**
    This application is designed to efficiently classify customer complaints into predefined departmental categories and analyze their sentiment (Positive, Negative, Neutral). It helps businesses quickly route customer feedback to the correct team for action and gain insights into overall customer sentiment.
    **How Classification Works:**
    Our system employs a hybrid approach for complaint classification:
    1.  **Rule-Based Keywords:** A primary layer uses a curated list of keywords to identify critical complaints (e.g., 'fraud', 'stolen', 'unauthorized') and assign them directly to the 'Theft/Dispute reporting' department, often overriding other classifications and marking sentiment as 'Negative'.
    2.  **VADER Sentiment Analysis:** The VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon is used to determine the sentiment (positive, negative, or neutral) of the complaint text. It's particularly effective for social media text and general short texts.
    3.  **Machine Learning Model:** For complaints that don't strongly match predefined categories or specific keywords, a pre-trained Machine Learning model (trained on historical complaint data) analyzes the text and predicts the most suitable department from our 5 core categories. This model leverages patterns and features learned from past complaints to make intelligent routing decisions.
    The combination of rules, sentiment, and machine learning ensures a robust and accurate classification process.
    **Frequently Asked Questions (FAQ):**
    * **Q: How accurate is the classification?**
        * A: While highly effective, no automated system is 100% accurate. The model is continuously improved with more data. The 'Checked Twice' status in the support portal allows manual verification.
    * **Q: What if a complaint doesn't fit any main department?**
        * A: Complaints that don't strongly match predefined categories or specific keywords are routed to the 'Others' department for manual review.
    * **Q: Can I pre-select a department?**
        * A: The categories displayed are for informational purposes. The system automatically classifies the complaint once submitted.
    """)
# --- Company Support Portal (after login) ---
if st.session_state.logged_in:
    st.markdown("---")
    st.markdown('<h2>Company Support Portal: Overview</h2>', unsafe_allow_html=True)
    st.write(f"Welcome, **{st.session_state.username}**! Here's an overview of customer complaints and tools to manage them.")
    refresh_button_placeholder = st.empty()
    with refresh_button_placeholder.container():
        if st.button("Refresh All Portal Data", key="support_refresh_all_button", help="Reloads all data in the portal from the database."):
            with st.empty():
                st.markdown(
                    """
                    <div class="pendulum-container">
                        <div class="pendulum_box">
                            <div class="ball first"></div>
                            <div class="ball"></div>
                            <div class="ball"></div>
                            <div class="ball"></div>
                            <div class="ball last"></div>
                        </div>
                        <div class="loader-text">Refreshing data...</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.cache_data.clear()
                time.sleep(1)
            st.rerun()
    st.markdown("<br>", unsafe_allow_html=True)
    all_complaints_df = get_all_complaints_from_db()
    
    # New control for "Last 10" or "All" complaints
    st.markdown("---")
    filter_option = st.radio(
        "Display Complaints",
        ("Last 10", "All"),
        index=0,
        key="last_or_all_filter",
        horizontal=True
    )
    
    if filter_option == "Last 10":
        display_df_main = all_complaints_df.head(10).copy()
    else:
        display_df_main = all_complaints_df.copy()
        
    if not display_df_main.empty:
        st.markdown("<h3>All Complaints Log</h3>", unsafe_allow_html=True)
        # Rename _id to ID for display, then drop it from the display
        display_df_main = display_df_main.rename(columns={'_id': 'ID'})
        st.dataframe(display_df_main.drop(columns=['_id'], errors='ignore'), use_container_width=True, hide_index=True)
        
        # New Download Section
        st.markdown("---")
        st.markdown("<h3>Download Complaints Data</h3>", unsafe_allow_html=True)
        
        # Prepare data for download
        df_for_download = display_df_main.rename(columns={'ID': '_id'})
        
        col_excel, col_pdf = st.columns(2)
        
        with col_excel:
            excel_data = to_excel(df_for_download)
            st.download_button(
                label="Export to Excel",
                data=excel_data,
                file_name="complaints_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the current table as an Excel file."
            )
            
        with col_pdf:
            pdf_data = to_pdf(df_for_download)
            pdf_base64 = b64encode(pdf_data).decode('latin-1')
            st.markdown(
                f'<a href="data:application/pdf;base64,{pdf_base64}" download="complaints_report.pdf" class="stButton" style="text-decoration:none;"><button style="width: 100%; border-radius: 10px; background-color: #6366F1; color: white; font-weight: 700; padding: 12px 20px; border: none;">Export to PDF</button></a>',
                unsafe_allow_html=True
            )
            st.write("") # Just to add some spacing below the button

    else:
        st.info("No complaints found in the database to display in the main log.")
    st.markdown("---")
    tab1, tab2 = st.tabs(["📊 Dashboard & Visualizations", "📝 Manage & Update Complaints"])
    with tab1:
        if not all_complaints_df.empty:
            st.markdown("<h3>Complaint Analytics</h3>", unsafe_allow_html=True)
            total_complaints = len(all_complaints_df)
            negative_count = all_complaints_df[all_complaints_df['sentiment'] == 'Negative'].shape[0]
            pending_review_count = all_complaints_df[all_complaints_df['checked_twice'] == 'Pending Review'].shape[0]
            col_met1, col_met2, col_met3 = st.columns(3)
            with col_met1:
                st.metric("Total Complaints", total_complaints)
            with col_met2:
                st.metric("Negative Complaints", negative_count)
            with col_met3:
                st.metric("Pending Review", pending_review_count)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<h3>Visual Trends</h3>", unsafe_allow_html=True)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("Complaints by Department")
                department_counts = all_complaints_df['predicted_department'].value_counts().reset_index()
                department_counts.columns = ['Department', 'Count']
                fig_dept = px.bar(department_counts, x='Department', y='Count',
                                 color='Department',
                                 title='Distribution of Complaints by Department',
                                 labels={'Count': 'Number of Complaints'},
                                 template='plotly_white')
                st.plotly_chart(fig_dept, use_container_width=True)
            with chart_col2:
                st.subheader("Overall Sentiment Distribution")
                sentiment_counts = all_complaints_df['sentiment'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                fig_sent = px.pie(sentiment_counts, values='Count', names='Sentiment',
                                 title='Sentiment Breakdown',
                                 color='Sentiment',
                                 color_discrete_map={'Positive': '#10B981', 'Negative': '#EF4444', 'Neutral': '#F59E0B'},
                                 template='plotly_white')
                st.plotly_chart(fig_sent, use_container_width=True)
            st.subheader("Complaint Volume Over Time")
            daily_counts = all_complaints_df.groupby(all_complaints_df['timestamp'].dt.date).size().reset_index(name='Count')
            daily_counts.columns = ['Date', 'Count']
            fig_time = px.line(daily_counts, x='Date', y='Count',
                               title='Daily Complaint Volume',
                               labels={'Count': 'Number of Complaints', 'Date': 'Date'},
                               template='plotly_white')
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No data available for analytics. Submit some complaints first!")
    with tab2:
        st.markdown("<h3>Search & Filter Complaints</h3>", unsafe_allow_html=True)
        with st.expander("Filter Options", expanded=True):
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                search_query = st.text_input(
                    "Keyword Search in Complaint",
                    key="search_keyword_filter_tab",
                    help="Type keywords to filter complaints (e.g., 'refund', 'delay', 'fraud'). Case-insensitive."
                )
            with filter_col2:
                selected_department = st.selectbox(
                    "Filter by Department",
                    ["All"] + list(DEPARTMENT_COLLECTIONS.keys()),
                    key="filter_department_tab",
                    help="Select a specific department to view complaints routed there."
                )
            with filter_col3:
                selected_sentiment = st.multiselect(
                    "Filter by Sentiment",
                    ["Positive", "Negative", "Neutral"],
                    default=[],
                    key="filter_sentiment_tab",
                    help="Choose one or more sentiment types to display."
                )
            date_col1, date_col2 = st.columns(2)
            if not all_complaints_df.empty:
                min_date = all_complaints_df['timestamp'].min().date()
                max_date = all_complaints_df['timestamp'].max().date()
            else:
                min_date = datetime.today().date()
                max_date = datetime.today().date()
            with date_col1:
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    key="filter_start_date_tab",
                    help="Set the earliest date for complaints to display."
                )
            with date_col2:
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    key="filter_end_date_tab",
                    help="Set the latest date for complaints to display."
                )
            st.info("Adjust the filters above, and the table below will update automatically.")
        filtered_df_for_tab = all_complaints_df.copy()
        if search_query:
            filtered_df_for_tab = filtered_df_for_tab[filtered_df_for_tab['complaint'].str.contains(search_query, case=False, na=False)]
        if selected_department != "All":
            filtered_df_for_tab = filtered_df_for_tab[filtered_df_for_tab['predicted_department'] == selected_department]
        if selected_sentiment:
            filtered_df_for_tab = filtered_df_for_tab[filtered_df_for_tab['sentiment'].isin(selected_sentiment)]
        if not filtered_df_for_tab.empty:
            filtered_df_for_tab = filtered_df_for_tab[
                (filtered_df_for_tab['timestamp'].dt.date >= start_date) &
                (filtered_df_for_tab['timestamp'].dt.date <= end_date)
            ]
        st.write(f"Displaying {len(filtered_df_for_tab)} complaints after filtering in this tab:")
        display_df_tab = filtered_df_for_tab.rename(columns={'_id': 'ID'})
        st.dataframe(display_df_tab.drop(columns=['_id'], errors='ignore'), use_container_width=True, hide_index=True)
        st.markdown("---")
        st.markdown("<h3>Update Complaint Status</h3>", unsafe_allow_html=True)
        
        # Create a more informative list for the selectbox
        if not filtered_df_for_tab.empty:
            display_options = [f"ID: {row['ID']} / {row['Complaint'][:70]}..." for index, row in display_df_tab.iterrows()]
            id_to_display_map = {f"ID: {row['ID']} / {row['Complaint'][:70]}...": row['ID'] for index, row in display_df_tab.iterrows()}
        else:
            display_options = []
            id_to_display_map = {}
        
        selected_display_string = st.selectbox(
            "Select Complaint to Update",
            [""] + display_options,
            key="select_complaint_to_update_tab",
            help="Choose a complaint from the table above by its ID and a snippet of its text."
        )

        if selected_display_string and selected_display_string != "":
            selected_complaint_id_update = id_to_display_map[selected_display_string]
            current_complaint_data_row_update = all_complaints_df[all_complaints_df['_id'] == selected_complaint_id_update]
            if not current_complaint_data_row_update.empty:
                current_complaint_data_update = current_complaint_data_row_update.iloc[0]
                st.info(f"**Complaint ID {selected_complaint_id_update}:** {current_complaint_data_update['complaint']}")
                st.write(f"**Current Status:** {current_complaint_data_update['checked_twice']}")
                status_options = ["Pending Review", "Reviewed - Action Taken", "Reviewed - No Action Needed", "Resolved"]
                try:
                    current_status_index_update = status_options.index(current_complaint_data_update['checked_twice'])
                except ValueError:
                    current_status_index_update = 0
                new_status = st.radio(
                    "Set New Status:",
                    status_options,
                    index=current_status_index_update,
                    key=f"status_radio_tab_{selected_complaint_id_update}",
                    help="Select the appropriate status for this complaint after review."
                )
                if st.button(f"Update Status for ID {selected_complaint_id_update}", key=f"update_button_tab_{selected_complaint_id_update}"):
                    with st.empty():
                        st.markdown(
                            """
                            <div class="pendulum-container">
                                <div class="pendulum_box">
                                    <div class="ball first"></div>
                                    <div class="ball"></div>
                                    <div class="ball"></div>
                                    <div class="ball"></div>
                                    <div class="ball last"></div>
                                </div>
                                <div class="loader-text">Updating status...</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        if update_checked_twice_status(selected_complaint_id_update, new_status):
                            st.success(f"Status for Complaint ID {selected_complaint_id_update} updated to '{new_status}'.")
                        else:
                            st.error("Failed to update status.")
                        time.sleep(0.5)
                    st.cache_data.clear()
                    st.rerun()
            else:
                st.warning("Selected Complaint ID not found.")

        st.markdown("---")
        st.markdown("<h3>Delete Complaint</h3>", unsafe_allow_html=True)

        if not filtered_df_for_tab.empty:
            display_options_delete = [f"ID: {row['ID']} / {row['Complaint'][:70]}..." for index, row in display_df_tab.iterrows()]
            id_to_display_map_delete = {f"ID: {row['ID']} / {row['Complaint'][:70]}...": row['ID'] for index, row in display_df_tab.iterrows()}
        else:
            display_options_delete = []
            id_to_display_map_delete = {}
            
        selected_display_string_delete = st.selectbox(
            "Select Complaint to Delete",
            [""] + display_options_delete,
            key="select_complaint_to_delete_tab",
            help="**WARNING:** Deleting a complaint is irreversible. Select carefully."
        )

        if selected_display_string_delete and selected_display_string_delete != "":
            selected_complaint_id_delete = id_to_display_map_delete[selected_display_string_delete]
            complaint_to_delete_row = all_complaints_df[all_complaints_df['_id'] == selected_complaint_id_delete]
            
            if not complaint_to_delete_row.empty:
                st.error(f"You are about to DELETE Complaint ID {selected_complaint_id_delete}: \"{complaint_to_delete_row.iloc[0]['complaint'][:50]}...\"")
                confirm_delete = st.checkbox(f"I understand this action cannot be undone and wish to delete Complaint ID {selected_complaint_id_delete}.", key=f"confirm_delete_{selected_complaint_id_delete}")
                
                if confirm_delete:
                    if st.button(f"CONFIRM DELETE Complaint ID {selected_complaint_id_delete}", key=f"delete_button_final_{selected_complaint_id_delete}"):
                        with st.empty():
                            st.markdown(
                                """
                                <div class="pendulum-container">
                                    <div class="pendulum_box">
                                        <div class="ball first"></div>
                                        <div class="ball"></div>
                                        <div class="ball"></div>
                                        <div class="ball"></div>
                                        <div class="ball last"></div>
                                    </div>
                                    <div class="loader-text">Deleting complaint...</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            if delete_complaint(selected_complaint_id_delete):
                                st.success("Complaint deleted successfully!")
                            else:
                                st.error("Deletion failed.")
                            time.sleep(0.5)
                        st.cache_data.clear()
                        st.rerun()
            else:
                st.warning("Selected Complaint ID for deletion not found.")
else:
    st.info("No complaints found in the database to manage. Submit some complaints first!")
