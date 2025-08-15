import numpy as np
import pandas as pd
import streamlit as st
import math
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from string import punctuation
from lexicalrichness import LexicalRichness
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.util import ngrams
from collections import Counter
from nltk.lm.preprocessing import flatten
from nltk.tokenize import word_tokenize, sent_tokenize
import unicodedata
import re
from textstat.textstat import textstatistics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from nltk.corpus import stopwords
from dotenv import load_dotenv
import os
import string
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
load_dotenv()
api_key = os.getenv("API_KEY")

pc = Pinecone(api_key=api_key)
model = SentenceTransformer('all-MiniLM-L6-v2')
index = pc.Index('aidetection')
st.header('Detect AI Generated Text 🌎')

humanData = pd.read_csv('linguisticHuman.csv')
aiData = pd.read_csv('linguisticAI.csv')

textQuery = st.text_area('Insert your text!')
analyzeBtn = st.button('Analyze')
if analyzeBtn:
    with st.status("Inserting Data", expanded=True):
        time.sleep(1)
        st.write("Generating Vector Embeddings")
        time.sleep(1)
        st.write("Detecting")
        time.sleep(1)

# Main Functions

from chroma import calculate_perplexity, calculate_burstiness
period = "."
space = " "
#Function to split text into sentences
def break_sentence(text):
    return len(text.split(period)) + 1

#Function to get avg sentence length
def get_avg_sntnc_len(text):
    return round((len(text.split(space)) + 1)/break_sentence(text),0)

#Function to remove punctuation
def remove_punctuation(text, operation):
  if(type(text)==float):
    return text
  ans=""

  if(operation=="count_punc"):
    ans = 0
    for j in text:
      if j in punctuation:
        ans+=1
  else:
    for i in text:
      if i not in punctuation:
        ans+=i.lower()

  return ans


# Function to remove accentred characters
def remove_accented_chars_func(text):
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

# Function to remove special characters and numbers
def remove_irr_char_func(text):
    return re.sub(r'[^a-zA-Z]', ' ', text)

# Function to remove extra white spaces
def remove_extra_whitespaces_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()

# Function to remove stop words
def remove_stop_words_func(text):
    return re.sub(r'^\s*|\s\s*', ' ', text).strip()


# Function to count words
def word_count(text):
    words = 0
    words += len(text.split())
    return words

# Function to count syllables in words
def syllables_count(word):
    return textstatistics().syllable_count(word)

#Function to get avg syllables per word
def avg_syllables_per_word(text):
    syllable = syllables_count(text)
    words = word_count(text)
    ASPW = float(syllable) / float(words)
    return round(ASPW, 1)


#Function to get Flesch Reading Ease Score
def get_FRE(text):
    FRE = 206.835 - float(1.015 * get_avg_sntnc_len(text)) - float(84.6 * avg_syllables_per_word(text))
    return round(FRE, 2)

#Function to get lexical corrected token to text ratio
def get_lex_cttr(text):
  lex = LexicalRichness(text)
  return lex.cttr

#Function to get lexical mean segmented token to text ratio
def get_lex_msttr(text):
  lex = LexicalRichness(text)
  return lex.msttr(segment_window=25)


#Function to remove stop words
stop_words = set(stopwords.words('english'))

def remove_stop_words_func(text):
  word_tokens = word_tokenize(text)
  filtered_words = [w.lower() for w in word_tokens if not w.lower() in stop_words]
  new_clean_text = ' '.join(filtered_words)
  return new_clean_text

#Function to count functional words
def count_stop_words_func(text):
    word_tokens = word_tokenize(text)
    stop_words_list = [w.lower() for w in word_tokens if w.lower() in stop_words]
    #unique_stop_words = set(stop_words_list)
    return len(stop_words_list)

#Function to convert text into a vector representation where each element of the vector corresponds to the frequency of a word in the text
WORD = re.compile(r"\w+")
def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)

#Function to calculate the cosine distance
def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

#Yule's characteristic K
def calculate_yules_characteristic_k(paragraph):
    words = paragraph.split()
    total_words = len(words)

    # Count the frequency of each word
    word_frequency = {}
    for word in words:
        word = word.lower()  # Convert to lowercase for case-insensitivity
        word_frequency[word] = word_frequency.get(word, 0) + 1

    # Count the frequency of frequencies
    frequency_of_frequencies = {}
    for count in word_frequency.values():
        frequency_of_frequencies[count] = frequency_of_frequencies.get(count, 0) + 1

    # Calculate Yule's Characteristic K
    sum_squared_frequencies = sum(v ** 2 for v in frequency_of_frequencies.values())
    k = 10**4 * (sum_squared_frequencies - total_words) / (total_words**2)

    return k

# Funtion to Calculate Mean Segmental Type-Token Ratio for the entire text
def calculate_ttr(segment):
    # Calculate Type-Token Ratio for a segment
    words = segment.split()
    if len(words) == 0:
        return 0
    types = set(words)
    ttr = len(types) / len(words)
    return ttr

def calculate_msttr(text, segment_size=100):
    segments = [text[i:i + segment_size] for i in range(0, len(text), segment_size)]
    ttrs = [calculate_ttr(segment) for segment in segments if segment.strip()]  # Skip empty segments
    if not ttrs:  # If there are no non-empty segments
        return 0
    msttr = sum(ttrs) / len(ttrs)
    return msttr

def add_features_to_test_data(text):

  count_punc = remove_punctuation(text,"count_punc")
  count_func_words = count_stop_words_func(text)
  avg_sntnc_len = get_avg_sntnc_len(text)
  lex_msttr = calculate_msttr(text)
  yule_characteristic = calculate_yules_characteristic_k(text)
  fre = get_FRE(text)

  col_labels = ["count_punc", "count_func_words","avg_sntnc_len","lex_msttr","yule_characteristic","fre"]
  data = [(count_punc, count_func_words, avg_sntnc_len,lex_msttr,yule_characteristic,fre)]

  X_new = pd.DataFrame(data, columns=col_labels)
  print(type(X_new))
  return X_new

tab1, tab2, tab3, tab4 = st.tabs(["Linguistic Features","Perplexity & Burstiness", "Similarity", "Classification - Machine Learning"])

with tab1:
   st.header("Linguistic Features")
   col1, col2 = st.columns(2)
   with col1:
       st.subheader('AI Statistics')
       col3, col4 = st.columns(2)
       with col3:
           st.metric(label="Flesch Reading Ease", value=round(np.median(aiData['flesch_reading_ease']),2), delta="+")
           st.metric(label="Gunning Fog Index", value=round(np.median(aiData['gunning_fog_index']),2), delta="-")
           st.metric(label="Herdan's C", value=round(np.median(aiData['herdans_c']),2), delta="-")
           st.metric(label="Mean Segmental TTR", value=round(np.median(aiData['msttr']),2), delta="+")
       with col4:
            st.metric(label="Flesch Grade Level", value=round(np.median(aiData['flesch_grade_level']),2), delta="-")
            st.metric(label="Yule Characteristic", value=round(np.median(aiData['yule_characteristic']),2), delta="-")
            st.metric(label="Functional Word Count", value=round(np.median(aiData['functional_word_count']),2), delta="-")
            st.metric(label="Punctuation Count", value=round(np.median(aiData['punctuation_count']),), delta="-")

   with col2:
       st.subheader('Human Statistics')
       col3, col4 = st.columns(2)
       with col3:
           st.metric(label="Flesch Reading Ease", value=round(np.median(humanData['flesch_reading_ease']),2), delta="-")
           st.metric(label="Gunning Fog Index", value=round(np.median(humanData['gunning_fog_index']),2), delta="+")
           st.metric(label="Herdan's C", value=round(np.median(humanData['herdans_c']),2), delta="+")
           st.metric(label="Mean Segmental TTR", value=round(np.median(humanData['msttr']),2), delta="-")
       with col4:
           st.metric(label="Flesch Grade Level", value=round(np.median(humanData['flesch_grade_level']),2), delta="+")
           st.metric(label="Yule Characteristic", value=round(np.median(humanData['yule_characteristic']),2), delta="+")
           st.metric(label="Functional Word Count", value=round(np.median(humanData['functional_word_count']),2), delta="+")
           st.metric(label="Punctuation Count", value=round(np.median(humanData['punctuation_count']),2), delta="+")

with tab2:
   col1, col2 = st.columns(2)
   with col1:
       st.subheader("Perplexity")
       st.caption(
           "Perplexity is a measure of how well a language model predicts a sample of text. It quantifies how 'surprised' the model is by a given input, based on the data it has been trained on. A lower perplexity indicates that the model is less surprised and better at predicting the input. For example, a coherent and grammatically correct text would typically have lower perplexity than a jumbled, nonsensical one")
       if textQuery:
           perplexity = calculate_perplexity(textQuery)
           st.write("Perplexity:", perplexity)

   with col2:
       st.subheader("Burstiness")
       st.caption(
           "Burstiness refers to the tendency of certain words or phrases to appear in clusters or bursts rather than being evenly distributed throughout a text. It measures how much the perplexity varies over an entire document. In writing, burstiness manifests as variations in sentence length and complexity, with some sentences being short and simple while others are longer and more complex")
       if textQuery:
           burstiness_score = calculate_burstiness(textQuery)
           st.write("Burstiness Score:", burstiness_score)

   if textQuery:
       if perplexity > 28000:
           st.error("Likely to be AI generated content")
       else:
           st.success("Not Likely to be AI generated content")

with tab3:
   st.header("Similarity between Vector Embeddings")
   col1, col2 = st.columns(2)
   with col1:
       st.metric(label = "Model", value = "BERT: all-MiniLM-L6-v2")
   with col2:
       st.metric(label = "Dimensions", value="384")
   pineconeBtn = st.button('Query')

   if pineconeBtn:
       xq = model.encode(textQuery).tolist()
       xc = index.query(vector=xq, top_k=5, include_metadata=True)
       simData = []
       for response in xc.matches:
           simData.append({'id': response['id'], 'score': response['score']})

       simData = pd.DataFrame(simData)
       st.dataframe(simData, use_container_width=True)
       simData['id'] = simData['id'].astype(int)
       count_ai = sum(value >= 150000 for value in simData['id'])
       count_human = sum(value < 150000 for value in simData['id'])
       if count_ai > count_human:
           st.error('Likely to be AI Generated')
       else:
           st.success('Likely to be Human Written')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


lg = LogisticRegression(penalty='l1', solver='liblinear')

mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
knn = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators=1000, criterion= "gini", max_features = "sqrt", max_depth = None, random_state=42)


classifiers = {
    'Logistic Regression': lg,
    'Multinomial Naive Bayes': mnb,
    'KNN': knn,
    'Random Forest Classifier': rfc
}


from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler()
import joblib
import statsmodels.api as sm

with tab4:
   st.header("Classifiers")
   model = st.selectbox('Select your Classifier', classifiers.keys())
   # st.write(model)
   X_train_norm = pd.read_csv('./data/X_train_norm.csv')
   y_train_norm = pd.read_csv('./data/y_train_norm.csv')
   modelBtn = st.button('Predict')
   if modelBtn:
       X_new = add_features_to_test_data(textQuery)
       test_df = pd.DataFrame(scaler.fit_transform(X_new), columns=X_new.columns)
       if model == 'Logistic Regression':

           loaded_logreg_model = joblib.load('./pkl_files/trained_model_logistic_regression.pkl')
           predictions = loaded_logreg_model.predict(X_new)
           col1, col2 = st.columns(2)
           with col1:
               st.metric('Accuracy', '79.13%')
           with col2:
               st.metric('Predicted Probability', round(predictions[0],4))
           if predictions[0] < 0.5:
               st.success('This text is Human written.')
           else:
               st.error('This text is AI written.')
       if model == 'Multinomial Naive Bayes':

           # loaded_naivebayes_model = joblib.load('trained_model_naive_bayes_classifier.pkl')
           loaded_naivebayes_model = mnb
           loaded_naivebayes_model.fit(X_train_norm, y_train_norm)
           predictions = loaded_naivebayes_model.predict_proba(test_df)
           col1, col2, col3 = st.columns(3)
           with col1:
               st.metric('Accuracy', '73.00%')
           with col2:
               st.metric('Predicted Probability Human', round(predictions[0, 0], 4))
           with col3:
               st.metric('Predicted Probability AI', round(predictions[0, 1], 4))
           if predictions[0, 0] > 0.5:
               st.success('This text is Human written.')
           else:
               st.error('This text is AI written.')
       if model == 'KNN':

           loaded_knn_model = joblib.load('./pkl_files/trained_model_knn.pkl')
           loaded_knn_model.fit(X_train_norm, y_train_norm)
           predictions = loaded_knn_model.predict_proba(test_df)
           col1, col2, col3 = st.columns(3)
           with col1:
               st.metric('Accuracy', '83.88%')
           with col2:
               st.metric('Predicted Probability Human', round(predictions[0,0], 4))
           with col3:
               st.metric('Predicted Probability AI', round(predictions[0,1], 4))

           if predictions[0, 0] > 0.5:
               st.success('This text is Human written.')
           else:
               st.error('This text is AI written.')
       if model == 'Random Forest Classifier':

           # loaded_rf_model = joblib.load('./pkl_files/trained_model_random_forest.pkl')
           r = rfc.fit(X_train_norm, y_train_norm)
           predictions = r.predict(X_new)
           col1, col2 = st.columns(2)
           with col1:
               st.metric('Accuracy', '83.06%')
           with col2:
               st.metric('Predicted Class', predictions[0])
           if predictions == 0:
               st.success('This text is Human written.')
           else:
               st.error('This text is AI written.')






