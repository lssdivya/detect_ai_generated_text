import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import textstat
import numpy as np
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


nltk.download('stopwords')
nltk.download('punkt')

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

data = pd.read_csv('data/data.csv')

dataHuman = data[data['isAI'] == 0]
dataAI = data[data['isAI'] == 1]


def average_word_length(paragraph):
    words = paragraph.split()
    total_length = sum(len(word) for word in words)
    average_length = total_length / len(words) if len(words) > 0 else 0
    return average_length

dataHuman['average_word_length'] = dataHuman['context'].apply(average_word_length)
dataAI['average_word_length'] = dataAI['context'].apply(average_word_length)

def count_punctuation(paragraph):
    punctuation_chars = set(string.punctuation)
    punctuation_count = sum(1 for char in paragraph if char in punctuation_chars)
    return punctuation_count

# Apply the function to each row and create a new column
dataHuman['punctuation_count'] = dataHuman['context'].apply(count_punctuation)
dataAI['punctuation_count'] = dataAI['context'].apply(count_punctuation)

def count_functional_words(paragraph):
    words = word_tokenize(paragraph.lower())
    stop_words = set(stopwords.words('english'))
    functional_word_count = sum(1 for word in words if word in stop_words)
    return functional_word_count

dataHuman['functional_word_count'] = dataHuman['context'].apply(count_functional_words)
dataAI['functional_word_count'] = dataAI['context'].apply(count_functional_words)


# Second Set of Lingustic Features:
# 1- Flesch Reading Ease
# 2- Flesch - Kincaid Grade Level
# 3- Gunning Fog Index

def calculate_flesch_reading_ease(paragraph):
    flesch_reading_ease = textstat.flesch_reading_ease(paragraph)
    return flesch_reading_ease

dataHuman['flesch_reading_ease'] = dataHuman['context'].apply(calculate_flesch_reading_ease)
dataAI['flesch_reading_ease'] = dataAI['context'].apply(calculate_flesch_reading_ease)

def calculate_flesch_kincaid_grade_level(paragraph):
    # Calculate the Flesch-Kincaid Grade Level value for the paragraph
    flesch_kincaid_grade_level = textstat.flesch_kincaid_grade(paragraph)
    return flesch_kincaid_grade_level

dataHuman['flesch_grade_level'] = dataHuman['context'].apply(calculate_flesch_kincaid_grade_level)
dataAI['flesch_grade_level'] = dataAI['context'].apply(calculate_flesch_kincaid_grade_level)


def calculate_gunning_fog_index(paragraph):
    # Calculate the Gunning Fog Index for the paragraph
    gunning_fog_index = textstat.gunning_fog(paragraph)
    return gunning_fog_index

dataHuman['gunning_fog_index'] = dataHuman['context'].apply(calculate_gunning_fog_index)
dataAI['gunning_fog_index'] = dataAI['context'].apply(calculate_gunning_fog_index)

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

dataHuman['yule_characteristic'] = dataHuman['context'].apply(calculate_yules_characteristic_k)
dataAI['yule_characteristic'] = dataAI['context'].apply(calculate_yules_characteristic_k)

def calculate_herdans_c(paragraph):
    words = paragraph.split()
    total_words = len(words)
    # Count the frequency of each word
    word_frequency = {}
    for word in words:
        word = word.lower()  # Convert to lowercase for case-insensitivity
        word_frequency[word] = word_frequency.get(word, 0) + 1
    # Count the number of distinct word forms
    m1 = len(word_frequency)
    # Calculate Herdan's C
    c = m1 / (total_words ** 0.5)
    return c

dataHuman['herdans_c'] = dataHuman['context'].apply(calculate_herdans_c)
dataAI['herdans_c'] = dataAI['context'].apply(calculate_herdans_c)

def calculate_ttr(segment):
    # Calculate Type-Token Ratio for a segment
    words = segment.split()
    if len(words) == 0:
        return 0
    types = set(words)
    ttr = len(types) / len(words)
    return ttr
def calculate_msttr(text, segment_size=100):
    # Calculate Mean Segmental Type-Token Ratio for the entire text
    segments = [text[i:i + segment_size] for i in range(0, len(text), segment_size)]
    ttrs = [calculate_ttr(segment) for segment in segments if segment.strip()]  # Skip empty segments
    if not ttrs:  # If there are no non-empty segments
        return 0
    msttr = sum(ttrs) / len(ttrs)
    return msttr

dataHuman['msttr'] = dataHuman['context'].apply(calculate_msttr)
dataAI['msttr'] = dataAI['context'].apply(calculate_msttr)

dataHuman.to_csv('linguisticHuman.csv')
dataAI.to_csv('linguisticAI.csv')

np.mean(dataHuman['yule_characteristic'])
np.mean(dataAI['yule_characteristic'])


humanData = pd.read_csv('linguisticHuman.csv')
aiData = pd.read_csv('linguisticAI.csv')

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
def calculatePerplexity(text):
    encodings = tokenizer(text, return_tensors='pt')
    loss = model(encodings['input_ids'], labels=encodings['input_ids']).loss
    perplexity = 2 ** loss.item()
    return perplexity

humanData = humanData[:100]
aiData = aiData[:100]
humanData['perplex_score'] = humanData['context'].apply(calculatePerplexity)
aiData['perplex_score'] = humanData['context'].apply(calculatePerplexity)

def getStats(data):
    stat = calculate_flesch_reading_ease(data)
    return stat