import streamlit as st
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

def set_background_image(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.title('Fake News Identifier')
    st.subheader("Input the news content below")

    # Set background image using a URL
    background_image_url = "https://unsplash.com/photos/man-sitting-on-chair-holding-newspaper-on-fire-FPNnKfjcbNU"  
    set_background_image(background_image_url)

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


