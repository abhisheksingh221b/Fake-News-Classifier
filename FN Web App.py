import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from PIL import Image
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

vectorizer = TfidfVectorizer()
vector_form = pickle.load(open('C:/Users/Asus/Documents/FN Web app/Fake News/vector1.pkl', 'rb'))
loaded_model = pickle.load(open('C:/Users/Asus/Documents/FN Web app/Fake News 2/model2.pkl', 'rb'))

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
    news=lemmatization(news)
    input_data=[news]
    vector_form1=vector_form.transform(input_data)
    prediction = loaded_model.predict(vector_form1)
    return prediction

def main():
    st.title('Fake News Classification app ')
    st.subheader("Input the News content below")
    sentence = st.text_area( "",height=200)        
    predict_btt = st.button("predict") 
    if predict_btt:
        prediction_class=fake_news(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Reliable')
        if prediction_class == [1]:
            st.warning('Unreliable')
            
            
if __name__ == '__main__':
     main()
           
            
            
            
            
            

            
            

            