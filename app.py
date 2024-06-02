import streamlit as st
from PIL import Image
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

def main():
    st.title('Fake News Identifier')
    st.subheader("Input the news content below")
    
    # Load background image
    background_image = Image.open('yang-xia-aett4u0y8Qk-unsplash.jpg')
    
    # Display background image
    st.image(background_image, use_column_width=True)
    
    sentence = st.text_area("Enter the news content:", height=200)
    predict_btn = st.button("Predict")
    if predict_btn:
        prediction_class = fake_news(sentence)
        if prediction_class == [0]:
            st.success('Reliable')
        elif prediction_class == [1]:
            st.warning('Unreliable')

if __name__ == '__main__':
    main()


