import os
import nltk
import ssl
import streamlit as st
import random
import json
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure SSL and NLTK
ssl._create_default_https_context = ssl._create_unverified_context

# Check if 'punkt' is already downloaded, if not, download it
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r", encoding="utf-8") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents["intents"]:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Extract test patterns and tags from the intents.json file
test_patterns = []
test_tags = []
for intent in intents["intents"]:
    for pattern in intent['patterns']:
        test_patterns.append(pattern)
        test_tags.append(intent['tag'])

# Transform test patterns and predict their tags
test_x = vectorizer.transform(test_patterns)
predicted_tags = clf.predict(test_x)

# Calculate accuracy
accuracy = accuracy_score(test_tags, predicted_tags)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Save chat history to CSV
def save_chat_history(chat_history):
    chat_history_df = pd.DataFrame(chat_history, columns=["sender", "message"])
    chat_history_df.to_csv("chat_history.csv", index=False)

# Save feedback to CSV
def save_feedback(feedback_text):
    feedback_df = pd.DataFrame([{"Feedback": feedback_text}])
    if os.path.exists("feedback.csv"):
        feedback_df.to_csv("feedback.csv", mode="a", header=False, index=False)
    else:
        feedback_df.to_csv("feedback.csv", index=False)

# Chatbot logic
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents["intents"]:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response

# Handle user input and chat history
def handle_user_input():
    user_input = st.session_state["user_input"]
    if user_input.strip():
        # Append user input to chat history
        st.session_state["chat_history"].append(("User", user_input))
        # Get chatbot response
        bot_reply = chatbot(user_input)
        st.session_state["chat_history"].append(("Chatbot", bot_reply))

        # Save the chat history to CSV
        save_chat_history(st.session_state["chat_history"])

        # Clear the input box after processing
        st.session_state["user_input"] = ""

# Main Streamlit interface
def main():
    st.set_page_config(page_title="Gardening Assistance", page_icon="🌱", layout="wide")

    # Sidebar navigation
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Go to:", ["Home", "Feedback", "About Us"])

    # Initialize session state variables
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Styling for the header
    if menu == "Home":
        st.markdown(
            """
            <div style="background-color:#2c5530; padding:10px; border-radius:10px; text-align:center;">
                <h1 style="color:#f9a620; font-family: 'Trebuchet MS', sans-serif;">Smart Gardenia Assistant</h1>
                <p style="color:#e8ffb7; font-style:italic; font-size:20px;">Cultivate a sustainable future, one plant at a time..</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # CSS for chat bubbles
        st.markdown(
            """
            <style>
            .chat-bubble {
                display: flex;
                align-items: center;
                margin: 10px 0;
            }
            .chat-bubble.user {
                justify-content: flex-end;
            }
            .chat-bubble.bot {
                justify-content: flex-start;
            }
            .chat-icon {
                width: 45px;
                height: 45px;
                display: flex;
                align-items: center;
                justify-content: center;
                background-color: #f9a620;
                border-radius: 50%;
                margin: 0 10px;
                font-size: 20px;
                color: white;
                font-weight: bold;
            }
            .chat-message {
                max-width: 70%;
                padding: 10px 15px;
                border-radius: 10px;
                font-size: 16px;
            }
            .chat-message.user {
                background-color: #8EC388;
                color: #050c05;
            }
            .chat-message.bot {
                background-color:#CBCCCC;
                color:black;
            }
            </style>
            """, unsafe_allow_html=True,
        )

        # Display chat history
        for sender, message in st.session_state["chat_history"]:
            if sender == "User":
                st.markdown(
                    f"""
                    <div class="chat-bubble user">
                        <div class="chat-message user">{message}</div>
                        <div class="chat-icon">👩</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif sender == "Chatbot":
                st.markdown(
                    f"""
                    <div class="chat-bubble bot">
                        <div class="chat-icon">🤖</div>
                        <div class="chat-message bot">{message}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Input box
        st.text_input(
            "Type your message here:",
            key="user_input",
            on_change=handle_user_input,
            placeholder="Type your message...",
            help="Ask me anything about gardening!",
        )

    elif menu == "Feedback":
          st.markdown("""<h1 style="font-family: 'Trebuchet MS', sans-serif;color:#f9a620; font-weight: bold;">📝 Feedback</h1>""", unsafe_allow_html=True)
          st.markdown("""<label style="font-size: 1.2em; color: #e8ffb7;"> We value your feedback! Please share your thoughts:</label>""", unsafe_allow_html=True)
          feedback = st.text_area("", key="feedback_textarea")

          if st.button("Submit Feedback"):
            if feedback.strip():
                save_feedback(feedback)
                st.success("Thank you for your feedback!")
            else:
                st.warning("Feedback cannot be empty.")

    elif menu == "About Us":
        st.markdown("""
            <style>
    .about-text {
        text-align: center;
        font-size: 1.3em; /* Slightly larger for readability */
        color:#FFFDEC; /* A rich sea green for contrast */
        line-height: 1.4; /* Improved line spacing */
        margin: 20px;
    }
</style> """, unsafe_allow_html=True,)
        st.markdown("""
            <div style="background-color:#2c5530; padding:10px;margin-bottom:20px; border-radius:10px; text-align:center;">
                <h1 style="color:#f9a620; font-family: 'Trebuchet MS', sans-serif;"> Welcome to Smart Gardenia</h1></div> """, unsafe_allow_html=True,)
        st.markdown("<div class='about-text'> At Smart Gardenia, our mission is simple yet impactful: to make gardening smarter, more sustainable, and eco-friendly. Whether you're a beginner just starting your green journey or a seasoned gardener looking to optimize your efforts, our intent based gardening assistant is here to help. With a focus on sustainable gardening practices</div>", unsafe_allow_html=True)
        st.markdown("<div class='about-text'>This project isn’t just about gardening—it’s about contributing to a sustainable future. By nurturing your garden, you’re also nurturing the planet. Together, let’s grow smarter and greener.</div>", unsafe_allow_html=True)
        st.markdown("<div class='about-text 'style='color: #e8ffb7; margin-top: 40px; margin-bottom: 30px; font-style: oblique 70deg;'>Here’s how Smart Gardenia can help you:</div>", unsafe_allow_html=True)
        st.markdown("""
<ul style='
    color: #FFFDED;
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
    list-style-type: none;
    font-size: 1.3em;
    line-height: 1.2;'>
    <li>» What are some eco-friendly gardening practices?</li>
    <li>» How do I start gardening at home?</li>
    <li>» What are the easiest plants to grow for beginners?</li>
    <li>» How do I combine aesthetics with sustainability in my garden?</li>
    <li>» How can gardening help the environment?</li>
    <li>» How do I protect my plants from pests naturally?</li>
    <li>» How do I create a zero-waste garden?</li>
    <li>» How can I start composting at home?</li>
    <li>» What are the best low-maintenance plants for a home garden?</li>
    <li>» What is rainwater harvesting, and how do I use it in my garden?</li>
    <li>» Can you suggest ways to conserve water in my garden?</li>
    <li>» What are some tips for small-space gardening?</li>
    <li>» What plants are best for hot climates?</li>
    <li>» How do I garden indoors or in a balcony?</li>
    <li>» How can I make my garden soil more fertile?</li>
    <li>» How can I make my garden pollinator-friendly?</li>
</ul>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()