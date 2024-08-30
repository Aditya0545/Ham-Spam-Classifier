import streamlit as st
import joblib
import nltk
nltk.download('punkt')
nltk.download('stopwords')  # Add this line to download stopwords
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load the vectorizer and model
tfidf = joblib.load('vectorizer.pkl')
spam_model = joblib.load('spam_classifier_model.pkl')

# Initialize the PorterStemmer
ps = PorterStemmer()

# Text transformation function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [word for word in text if word.isalnum()]
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text = [ps.stem(word) for word in text]
    return " ".join(text)

# Streamlit sidebar navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Spam Ham Model", "About"])

# Home page
if option == "Home":
    st.title("Welcome to the Email/SMS Spam Classifier App")
    
    # Display an image
    st.image('hacker-activity.jpg', caption='Your Image Caption', use_column_width=True)
    
    st.write("""
        This app allows you to classify messages as spam or not spam.
        Use the navigation menu on the left to switch between pages.
    """)

# Spam Ham Model page
elif option == "Spam Ham Model":
    st.title("Email/SMS Spam Classifier")
    
    # Choice between inputting a message or using predefined messages
    choice = st.selectbox(
        "Select an option",
        ["Enter your own message", "Choose a predefined message"]
    )
    
    # Dictionary of predefined messages
    predefined_messages = {
        "Spam": [
            "Congratulations! You've won a $1000 gift card. Click here to claim your prize.",
            "Call now to get your free trial subscription for our exclusive service.",
            "You have been selected for a $1000 cash prize! Claim now!",
            "Claim your free gift card now! Limited time offer.",
            "Win big with our casino games! Click here to start winning."
        ],
        "Not Spam": [
            "Hi John, just wanted to confirm our meeting tomorrow at 10 AM.",
            "The team has completed the initial draft of the report.",
            "Can you send me the report by end of day? Thanks!",
            "I’m looking forward to our call next week. Let’s discuss the project.",
            "Reminder: Your appointment with Dr. Smith is tomorrow at 2 PM."
        ]
    }
    
    if choice == "Enter your own message":
        # Input box for entering the message
        input_sms = st.text_area("Enter the message")

    elif choice == "Choose a predefined message":
        # Dropdown to select a predefined message
        selected_message = st.selectbox(
            "Select a message",
            [f"{label}: {msg}" for label, msgs in predefined_messages.items() for msg in msgs]
        )
        # Extract the message text
        input_sms = selected_message.split(": ", 1)[1]
    
    # Predict button
    if st.button('Verify'):
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. Predict
        result = spam_model.predict(vector_input)[0]
        # 4. Display the result
        if result == 1:
            st.markdown("<div style='background-color: #800000; color: #ffffff; padding: 10px;'>This is a Spam message.</div>", unsafe_allow_html=True)
        else:
            st.success("This is not a Spam message.")

# About page
elif option == "About":
    st.title("About")
    st.write("""
        **Email/SMS Spam Classifier App**: This application is designed to classify messages as either spam or not spam using machine learning.
        
        **Features**:
        - **Spam Ham Model**: Enter a message and classify it as spam or not spam, or choose from predefined messages.

        **Developed by Aditya Kumar**.

        **Technologies Used**:
        - Python
        - Streamlit
        - Scikit-learn
        - NLTK
        - Joblib
    """)
