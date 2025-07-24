import streamlit as st
import pandas as pd
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import joblib
from datetime import datetime
import os
import re
import nltk
import time # Import time for a small delay for visual effect

# --- Streamlit Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(layout="centered", page_title="Complaint Classification App")

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
analyzer = get_sentiment_analyzer()

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
        st.stop() # Stop the app if files are missing
model, vectorizer = load_model_and_vectorizer()

# --- Text Preprocessing ---
def preprocess_text(text):
    """Performs basic text preprocessing."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Classifier Function (Updated to use preprocess_text for ML model) ---
def classify_complaint(text):
    text_lower = text.lower()

    # --- Enhanced Sentiment Analysis with VADER ---
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

    # --- Robust Department Classification with Priority Rules (Strict 5 Categories) ---
    predicted_department = 'Others' # Default ultimate fallback

    # Priority 1: Theft/Dispute reporting
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
        sentiment = 'Negative' # Override sentiment for critical cases
    # Priority 2: Credit card / Prepaid card
    elif any(keyword in text_lower for keyword in [
        'credit card', 'prepaid card', 'double charged', 'transaction error',
        'card payment', 'billing issue', 'lost card', 'stolen card', 'card balance',
        'annual fee', 'statement', 'rewards', 'card dispute', 'credit limit', 'apr'
    ]):
        predicted_department = 'Credit card / Prepaid card'
    # Priority 3: Mortgages/loans
    elif any(keyword in text_lower for keyword in [
        'mortgage', 'loan', 'personal loan', 'home loan', 'auto loan',
        'interest rate', 'refinance', 'payment plan', 'student loan', 'debt',
        'loan application', 'mortgage payment', 'vehicle loan', 'payday loan', 'title loan'
    ]):
        predicted_department = 'Mortgages/loans'
    # Priority 4: Bank account services
    elif any(keyword in text_lower for keyword in [
        'bank account', 'savings account', 'checking account', 'online banking',
        'mobile app', 'account access', 'deposit', 'withdrawal', 'transfer funds',
        'account balance', 'atm', 'branch', 'login', 'passcode', 'routing number',
        'direct deposit', 'account closed', 'new account', 'money transfer', 'virtual currency', 'money service', 'wire transfer',
        'cryptocurrency', 'send money', 'receive money', 'payment app', 'remittance'
    ]):
        predicted_department = 'Bank account services'
    # Fallback to ML Model for less specific cases, or assign 'Others'
    else:
        try:
            # Preprocess text before feeding to the ML model
            processed_text_for_ml = preprocess_text(text)
            vec = vectorizer.transform([processed_text_for_ml])
            ml_prediction = model.predict(vec)[0]

            # Map ML prediction to one of the 5 main categories
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

# --- Save to Excel ---
def log_to_excel(data, filename="complaints_received.xlsx"):
    df_new = pd.DataFrame([data])
    if os.path.exists(filename):
        try:
            df_old = pd.read_excel(filename)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading or concatenating with existing Excel file: {e}. Creating a new file.")
            df_combined = df_new
    else:
        df_combined = df_new
    df_combined.to_excel(filename, index=False)

# --- New Function: Log to Department-Specific Excel ---
def log_to_department_excel(data):
    predicted_department = data["Predicted Department"]
    # Sanitize department name for filename (replace problematic chars with underscore)
    sanitized_dept_name = re.sub(r'[^\w\s-]', '', predicted_department).replace(' ', '_').lower()
    department_filename = f"{sanitized_dept_name}_complaints.xlsx"

    df_new = pd.DataFrame([data])
    if os.path.exists(department_filename):
        try:
            df_old = pd.read_excel(department_filename)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            st.error(f"Error reading or concatenating with existing department Excel file '{department_filename}': {e}. Creating a new file.")
            df_combined = df_new
    else:
        df_combined = df_new
    df_combined.to_excel(department_filename, index=False)
    st.info(f"Complaint also logged to '{department_filename}' for the {predicted_department} department.")


# --- Streamlit UI ---

# Custom CSS for styling (similar to React app, with glow animation)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5em;
        color: #4B0082; /* Indigo-like color */
        text-align: center;
        margin-bottom: 1.5em;
        font-weight: 700;
    }
    .department-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        margin-bottom: 20px;
    }
    .department-box {
        background-color: #EEF2FF; /* Light blue-gray */
        color: #4B0082; /* Darker indigo */
        padding: 12px 20px;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        flex: 1 1 auto; /* Allow items to grow and shrink */
        min-width: 180px; /* Minimum width for responsiveness */
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: all 0.3s ease-in-out;
    }
    .department-box.highlighted {
        background-color: #6366F1; /* Indigo-500 */
        color: white;
        /* Stronger glow effect */
        box-shadow: 0 0 15px 5px rgba(99, 102, 241, 0.7),
                    0 0 25px 10px rgba(79, 70, 229, 0.5),
                    0 0 35px 15px rgba(59, 130, 246, 0.3); /* Added more layers for glow */
        transform: scale(1.05); /* Slightly larger scale */
        border: 2px solid #4F46E5;
        animation: pulse-glow 1.5s infinite alternate; /* Add pulse animation */
    }

    @keyframes pulse-glow {
        0% {
            box-shadow: 0 0 5px 2px rgba(99, 102, 241, 0.4), 0 0 10px 5px rgba(79, 70, 229, 0.2);
        }
        100% {
            box-shadow: 0 0 15px 5px rgba(99, 102, 241, 0.7), 0 0 25px 10px rgba(79, 70, 229, 0.5), 0 0 35px 15px rgba(59, 130, 246, 0.3);
        }
    }

    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1px solid #D1D5DB !important;
        padding: 12px !important;
        font-size: 1rem !important;
    }
    .stButton > button {
        width: 100%;
        background-color: #6366F1; /* Indigo-500 */
        color: white;
        font-weight: 700;
        padding: 12px 20px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background-color: #4F46E5; /* Indigo-700 */
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .stAlert {
        border-radius: 10px;
    }
    /* Styling for st.dataframe */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden; /* Ensures rounded corners apply to content */
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stDataFrame > div {
        border-radius: 10px;
    }
    .stDataFrame table {
        border-collapse: collapse;
        width: 100%;
    }
    .stDataFrame th {
        background-color: #EEF2FF; /* Light indigo for header */
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
    .stDataFrame tbody tr:nth-child(even) {
        background-color: #F9FAFB; /* Light stripe */
    }
    .stDataFrame tbody tr:hover {
        background-color: #F3F4F6; /* Hover effect */
    }
    .stMarkdown h2 {
        color: #4B0082;
        text-align: center;
        margin-top: 2em;
        margin-bottom: 1em;
        font-weight: 600;
        font-size: 1.8em;
    }
    .stMetric {
        background-color: #F0F4F8; /* Light gray-blue for metrics */
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stMetric label {
        font-weight: 500;
        color: #6B7280; /* Gray-500 */
    }
    .stMetric .css-1q8dd3e { /* Target the metric value specifically */
        font-size: 1.8em;
        font-weight: 700;
        /* Conditional colors for sentiment metric */
        --sentiment-positive-color: #10B981; /* Green-500 */
        --sentiment-negative-color: #EF4444; /* Red-500 */
        --sentiment-neutral-color: #F59E0B; /* Amber-500 */
    }
    .stMetric .css-1q8dd3e[data-sentiment="Positive"] {
        color: var(--sentiment-positive-color);
    }
    .stMetric .css-1q8dd3e[data-sentiment="Negative"] {
        color: var(--sentiment-negative-color);
    }
    .stMetric .css-1q8dd3e[data-sentiment="Neutral"] {
        color: var(--sentiment-neutral-color);
    }

    /* Custom loader styles */
    .custom-loader-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 20px auto; /* Center the loader */
        min-height: 100px; /* Give it some space */
    }

    .loader {
        width: 28px;
        aspect-ratio: 1;
        border-radius: 50%;
        background: #F10C49; /* Red color from your CSS */
        animation: l2 1.5s infinite;
        margin-bottom: 10px; /* Space between loader and text */
    }
    @keyframes l2 {
        0%, 100%{transform:translate(-35px);box-shadow: 0 0 #F4DD51, 0 0 #E3AAD6}
        40% {transform:translate( 35px);box-shadow: -15px 0 #F4DD51,-30px 0 #E3AAD6}
        50% {transform:translate( 35px);box-shadow: 0 0 #F4DD51, 0 0 #E3AAD6}
        90% {transform:translate(-35px);box-shadow: 15px 0 #F4DD51, 30px 0 #E3AAD6}
    }
    .loader-text {
        font-size: 1.1em;
        font-weight: 600;
        color: #4B0082; /* Dark indigo for text */
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="main-header">Customer Complaint Classification</p>', unsafe_allow_html=True)

# Define the 5 main departments for display
display_departments = [
    'Credit card / Prepaid card',
    'Bank account services',
    'Theft/Dispute reporting',
    'Mortgages/loans',
    'Others'
]

# State for highlighting department
if 'highlighted_dept' not in st.session_state:
    st.session_state.highlighted_dept = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'complaint_input_key' not in st.session_state:
    st.session_state.complaint_input_key = 0 # Used to clear text area
if 'is_processing' not in st.session_state: # New state for controlling loader
    st.session_state.is_processing = False
if 'current_complaint_text' not in st.session_state: # To hold text across reruns for processing
    st.session_state.current_complaint_text = ""


# Display departments with dynamic highlighting
st.markdown('<div class="department-container">', unsafe_allow_html=True)
for dept in display_departments:
    is_highlighted = st.session_state.highlighted_dept == dept
    highlight_class = " highlighted" if is_highlighted else ""
    st.markdown(f'<div class="department-box{highlight_class}">{dept}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Complaint Input
# Use a key to enable clearing the text area
complaint_text = st.text_area(
    "Write Your Complaint:",
    height=150,
    placeholder="Describe your issue here...",
    key=f"complaint_input_{st.session_state.complaint_input_key}",
    value="" if st.session_state.is_processing else st.session_state.get(f"complaint_input_{st.session_state.complaint_input_key}", "")
)

# Placeholder for custom loader (always present, content changes)
loader_placeholder = st.empty()

# --- Handle Button Click and Processing Logic ---
if st.button("Submit Complaint"):
    if complaint_text:
        st.session_state.current_complaint_text = complaint_text # Store text for next rerun
        st.session_state.is_processing = True # Set flag to show loader
        st.session_state.complaint_input_key += 1 # Increment key to clear input on next rerun
        st.rerun() # Rerun immediately to show loader
    else:
        st.warning("Please enter your complaint before submitting.")

# --- Conditional Processing Block (runs only when is_processing is True) ---
if st.session_state.is_processing:
    # Display custom loader and text
    loader_placeholder.markdown(
        """
        <div class="custom-loader-container">
            <div class="loader"></div>
            <div class="loader-text">Analyzing complaint and routing...</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Perform the heavy computation
    result = classify_complaint(st.session_state.current_complaint_text)
    log_to_excel(result) # Log to main Excel
    log_to_department_excel(result) # Log to department-specific Excel

    st.session_state.last_result = result # Store for display
    st.session_state.highlighted_dept = result["Predicted Department"] # Set for highlighting

    # Clear the custom loader
    loader_placeholder.empty()

    st.success("Complaint submitted and classified!")

    # Pause for visual effect, then clear highlight state and processing flag
    time.sleep(1.5) # Pause for 1.5 seconds to let the glow be seen
    st.session_state.highlighted_dept = None # Clear highlight state
    st.session_state.is_processing = False # Reset processing flag
    st.rerun() # Rerun again to clear the highlight and reset state

# Display last classification result prominently
if st.session_state.last_result:
    st.markdown("---")
    st.markdown('<h2>Last Classification Result</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        # Pass sentiment to CSS for conditional coloring
        st.markdown(f'<div class="stMetric"><label>Sentiment</label><div class="css-1q8dd3e" data-sentiment="{st.session_state.last_result["Sentiment"]}">{st.session_state.last_result["Sentiment"]}</div></div>', unsafe_allow_html=True)
    with col2:
        st.metric(label="Score", value=f"{st.session_state.last_result['Score']:.4f}")
    with col3:
        st.metric(label="Predicted Department", value=st.session_state.last_result["Predicted Department"])
    st.write(f"**Complaint:** {st.session_state.last_result['Complaint']}")

# Display Recent Complaints Log
st.markdown("---")
st.markdown('<h2>Recent Complaints Log</h2>', unsafe_allow_html=True)

# Function to load recent complaints for display (limited to 10)
@st.cache_data(ttl=5) # Cache data for 5 seconds to avoid constant re-reading
def get_recent_complaints(filename="complaints_received.xlsx", num_complaints=10):
    if os.path.exists(filename):
        try:
            df_complaints = pd.read_excel(filename)
            # Ensure Timestamp is datetime and sort
            df_complaints['Timestamp'] = pd.to_datetime(df_complaints['Timestamp'])
            df_complaints = df_complaints.sort_values(by='Timestamp', ascending=False)
            return df_complaints.head(num_complaints) # Limit to top N complaints
        except Exception as e:
            st.error(f"Error loading recent complaints: {e}")
            return pd.DataFrame() # Return empty DataFrame on error
    return pd.DataFrame() # Return empty DataFrame if file doesn't exist

# We want 9 recent complaints + the one just submitted (if any)
recent_complaints_df = get_recent_complaints(num_complaints=9)

if not recent_complaints_df.empty:
    # Display only relevant columns for the UI
    display_df = recent_complaints_df[[
        'Complaint',
        'Sentiment',
        'Score',
        'Predicted Department',
        'Checked Twice',
        'Timestamp'
    ]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No complaints submitted yet.")

# Optional: Add a refresh button for the complaints log (useful for multi-user scenarios)
if st.button("Refresh Complaints Log", key="refresh_log_button"):
    st.cache_data.clear() # Clear cache to force reload
    st.rerun() # Rerun the app to refresh data
