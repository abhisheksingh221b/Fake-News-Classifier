import streamlit as st
from PIL import Image
import base64
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

vectorizer = TfidfVectorizer()
vector_form_path = 'vector1.pkl'
model_path = 'model2.pkl'

vector_form = pickle.load(open(vector_form_path, 'rb'))
loaded_model = pickle.load(open(model_path, 'rb'))

lemmatizer = WordNetLemmatizer()

def lemmatization(content):
    # Remove non-alphabetic characters, convert to lowercase, and tokenize
    tokenized_content = word_tokenize(re.sub('[^a-zA-Z]', ' ', content.lower()))

    # Lemmatize each word, excluding stopwords
    lemmatized_content = [lemmatizer.lemmatize(word) for word in tokenized_content if word not in stopwords.words('english')]

    # Join the lemmatized words into a string
    lemmatized_content = ' '.join(lemmatized_content)

    return lemmatized_content

def fake_news(news):
    news = lemmatization(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = loaded_model.predict(vector_form1)
    return prediction

def set_background_image(image_path):
    encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url('data:image/jpg;base64,{encoded_image}');
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    set_background_image('yang-xia-aett4u0y8Qk-unsplash.jpg')  # Adjust the file path as needed
    
    st.title('Fake News Identifier')
    st.subheader("Input the news content below")

    # Input text
    sentence = st.text_area("", height=200)
    predict_btn = st.button("Predict")
    if predict_btn:
        prediction_class = fake_news(sentence)
        if prediction_class == [0]:
            st.success('Reliable')
        elif prediction_class == [1]:
            st.warning('Unreliable')

if __name__ == '__main__':
    main()




