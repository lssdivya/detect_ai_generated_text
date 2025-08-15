import numpy as np
import ast
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors


model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

data = pd.read_csv('data.csv')
print(data.shape)
print(data)


dataHuman = data[data['isAI'] == 0]
dataAI = data[data['isAI'] == 1]

# d = pd.DataFrame()
def createVectorDatabase(data):
    # global d
    vectors = []
    sourceData = data.context.values

    for text in tqdm(sourceData):
        vector = model.encode(text)
        vectors.append(vector)

    data["vectors"] = vectors
    data["vectors"] = data["vectors"].apply(lambda emb: np.array(emb))
    data["vectors"] = data["vectors"].apply(lambda emb: emb.reshape(1, -1))

    # d = data['vectors'].copy()
    # print(len(d[0]))
    return data



dataAITrimmed = dataAI[:100]
dataHumanTrimmed = dataHuman[:100]
vectorizedHumanDB = createVectorDatabase(dataHumanTrimmed)
vectorizedAIDB = createVectorDatabase(dataAITrimmed)

# Testing Area
flattened_vectors = [item for sublist in vectorizedHumanDB['vectors'] for item in sublist]
X = np.array(flattened_vectors)
y = vectorizedHumanDB['isAI']



knnHuman = KNeighborsClassifier(n_neighbors=3)
knnHuman.fit(X, y)


inputQuery = "testint testing teasing"
inputVector = model.encode(inputQuery)
inputVector = inputVector.reshape(1, -1)

knnHuman = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = knnHuman.kneighbors(inputVector)

z =knnHuman.predict(inputVector.reshape(1, -1))

def knnAnalysis(inputQuery):
    inputVector = model.encode(inputQuery)
    distAI, indicesAI = knnAI.kneighbors(inputVector)
    distHuman, indicesHuman = knnHuman.kneighbors(inputVector)
    return distAI, distHuman
    if np.mean(distAI[0]) < np.mean(distHuman[0]):
        print('AI')
    else:
        print('Human')


distAI, indicesAI = knnAI.kneighbors(inputVector)
distHuman, indicesHuman = knnHuman.kneighbors(inputVector)
