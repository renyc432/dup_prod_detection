import os

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

lemma = WordNetLemmatizer()

def text_preprocessing(text):
    text = re.sub('[^a-zA-Z0-9 @ . , : - _]', '', str(text))
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = text.lower().split()
    tokens_stop = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemma.lemmatize(word) for word in tokens_stop]
    text_processed = ' '.join(tokens)
    return text_processed
    
def get_all_img_path(directory):
    img_path = []
    for dirname,_,filenames in os.walk(directory):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            img_path.append(fullpath)
    return img_path
    