from flask import Flask,request,render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import numpy as np

with open('tokenizer.pkl','rb') as f:
    tokenizer = pickle.load(f)

model = load_model('sentiment_model.h5')

app = Flask(__name__)

porter = PorterStemmer()
def clean_str(text):
    corpus = []
    table = text.maketrans('','',string.punctuation)
    txt = text.translate(table)
    review = re.sub('[^a-zA-Z]',' ',txt)
    review = review.lower()
    review = review.split()
    review = [porter.stem(word) for word in review if word not in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    return corpus

    

@app.route('/',methods=['GET','POST'])
def main():
    if request.method == "POST":
        tweet = request.form.get('tweet')
        cleaned_text = clean_str(str(tweet))
        sequence = tokenizer.texts_to_sequences(cleaned_text)
        padded = pad_sequences(sequence,maxlen=62)
        pred =  model.predict(padded.reshape(1,62,1))
        print(np.argmax(pred))
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
