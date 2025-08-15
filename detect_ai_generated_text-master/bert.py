import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer,  AutoModelForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('GPT-wiki-intro.csv')
data.columns

dataHuman = data[['wiki_intro']]
dataHuman = dataHuman.rename(columns={'wiki_intro': 'context'})
dataHuman['isAI'] = 0
dataAI = data[['generated_intro']]
dataAI = dataAI.rename(columns={'generated_intro': 'context'})
dataAI['isAI'] = 1

dataAI.head()
dataHuman.head()

modelPath = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(modelPath,
                                          do_lower_case=True)
model = AutoModelForSequenceClassification.from_pretrained(modelPath,
                                                          output_attentions=False,
                                                          output_hidden_states=True)

def createVectorFromText(tokenizer, model, text, MAX_LEN=510):
    input_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
    )
    results = pad_sequences([input_ids], maxlen=MAX_LEN, dtype="long",
                            truncating="post", padding="post")
    # Remove the outer list.
    input_ids = results[0]
    # Create attention masks
    attention_mask = [int(i > 0) for i in input_ids]
    # Convert to tensors.
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    # Add an extra dimension for the "batch" (even though there is only one
    # input in this batch.)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()
    # Run the text through BERT, and collect all of the hidden states produced
    # from all 12 layers.
    with torch.no_grad():
        logits, encoded_layers = model(
            input_ids=input_ids,
            token_type_ids=None,
            attention_mask=attention_mask,
            return_dict=False)

    layer_i = 12  # The last BERT layer before the classifier.
    batch_i = 0  # Only one input in the batch.
    token_i = 0  # The first token, corresponding to [CLS]
    # Extract the embedding.
    vector = encoded_layers[layer_i][batch_i][token_i]
    # Move to the CPU and convert to numpy ndarray.
    vector = vector.detach().cpu().numpy()
    return (vector)


def createVectorDatabase(data):
    vectors = []
    sourceData = data.context.values
    # Loop over all the comment and get the embeddings
    for text in tqdm(sourceData):
        vector = createVectorFromText(tokenizer, model, text)
        vectors.append(vector)

    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb))
    data["vectors"] = data["vectors"].apply(lambda emb: emb.reshape(1, -1))
    return data

dataAITrimmed = dataAI[:5000]
dataHumanTrimmed = dataHuman[:5000]

vectorizedHumanDB = createVectorDatabase(dataHumanTrimmed)
vectorizedAIDB = createVectorDatabase(dataAITrimmed)

vectorizedHumanDB = createVectorDatabase(dataHuman)
vectorizedAIDB = createVectorDatabase(dataAI)

vectorizedAIDB.to_csv('aiVector.csv')
vectorizedHumanDB.to_csv('humanVector.csv')

def processDocument(text):
    """
    Create a vector for given text and adjust it for cosine similarity search
    """
    text_vect = createVectorFromText(tokenizer, model, text)
    text_vect = np.array(text_vect)
    text_vect = text_vect.reshape(1, -1)
    return text_vect

def isPlagiarism(similarity_score, plagiarism_threshold):
  is_plagiarism = False
  if(similarity_score >= plagiarism_threshold):
    is_plagiarism = True

  return is_plagiarism

# def semanticSimilarity():
#
#

def runPlagiarismAnalysis(query_text, database, plagiarism_threshold=0.8):
    top_N = 2
    query_vect = processDocument(query_text)

    # Run similarity Search
    database["similarity"] = database["vectors"].apply(lambda x: cosine_similarity(query_vect, x))
    database["similarity"] = database["similarity"].apply(lambda x: x[0][0])

    similar_articles = database.sort_values(by='similarity', ascending=False)[0:top_N + 1]
    formated_result = similar_articles[["context", "isAI", "similarity"]].reset_index(drop=True)

    similarity_score = formated_result.iloc[0]["similarity"]
    most_similar_article = formated_result.iloc[0]["context"]
    is_plagiarism_bool = isPlagiarism(similarity_score, plagiarism_threshold)

    plagiarism_decision = {'similarity_score': similarity_score,
                           'is_plagiarism': is_plagiarism_bool,
                           'most_similar_article': most_similar_article,
                           'article_submitted': query_text
                           }
    return plagiarism_decision


data01 = "Carrots are not just crunchy, orange vegetables; they're also packed with nutrients and have a fascinating history. From their origins in Central Asia to their widespread cultivation around the world, carrots have been a dietary staple for centuries. In this introduction, we'll delve into the nutritional benefits of carrots, their culinary versatility, and some fun facts about this beloved root vegetable."
data = "Mobile phones have become an integral part of our daily lives, revolutionizing the way we communicate, work, and stay connected. From their humble beginnings as bulky devices used primarily for making calls, they have evolved into powerful pocket-sized computers, capable of performing a wide range of tasks. In this introduction, we will explore the history of mobile phones, their impact on society, and the latest trends shaping the future of mobile technology."
resultAI = runPlagiarismAnalysis(data, vectorizedAIDB, plagiarism_threshold=0.8)
resultHuman = runPlagiarismAnalysis(data, vectorizedHumanDB, plagiarism_threshold=0.8)
