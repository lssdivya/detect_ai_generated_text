import numpy as np
import pandas as pd

# Data Wrangling
data = pd.read_csv('GPT-wiki-intro.csv')
data.columns

# data01.loc[data01['generated_intro'] != '', 'isAi'] = 1
# dataStacked = data01.stack().reset_index(drop=True).to_frame(name='data02')

data01 = data[['wiki_intro']]
data01 = data01.rename(columns={'wiki_intro': 'context'})
data01['isAI'] = 0
data02 = data[['generated_intro']]
data02 = data02.rename(columns={'generated_intro': 'context'})
data02['isAI'] = 1
# isAI = 0, Human Written
result = pd.concat([data01, data02], ignore_index = True)
result.to_csv('data.csv')
result['isAI'].value_counts()

# Data Preprocessing
X = result['context']
y = result['isAI']

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
X_test_vector = pd.DataFrame.sparse.from_spmatrix(X_test_tfidf)

# Modelling

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

lg = LogisticRegression(penalty='l1', solver='liblinear')
sv = SVC(kernel='sigmoid', gamma=1.0)
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
knn = KNeighborsClassifier()
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bg = BaggingClassifier(n_estimators=50, random_state=2)
gbc = GradientBoostingClassifier(n_estimators=50, random_state=2)

def modelling(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    acc_score = metrics.accuracy_score(y_test, prediction)
    return acc_score

acc_score = {}
classifiers = {
    'LR': lg,
    'MNB': mnb,
    'SVM': sv,
    'DTC': dtc,
    'KNN': knn,
    'RFC': rfc,
    'ABC': abc,
    'BG': bg,
    'GBC': gbc,
}

for name, i in classifiers.items():
    acc_score[name] = modelling(i, X_train_tfidf, X_test_tfidf, y_train, y_test)

for name, acc in acc_score.items():
    print(f'Accuracy for {name}: {acc}')

lg.fit(X_train_tfidf,y_train)
y_pred = lg.predict(X_test_tfidf)
y_preddf = pd.DataFrame(y_pred)
y_predProb = lg.predict_proba(X_test_tfidf)
y_predProbdf = pd.DataFrame(y_predProb)


x_testdf=pd.DataFrame(X_test)
y_testdf=pd.DataFrame(y_test)
x_testdf['id'] = range(1, len(x_testdf) + 1)
y_testdf['id'] = range(1, len(y_testdf) + 1)
y_preddf['id'] = range(1, len(y_preddf) + 1)
y_predProbdf['id'] = range(1, len(y_predProbdf) + 1)
join1=y_testdf.merge(x_testdf,on = 'id', how = 'inner' ,indicator=False)
join2=join1.merge(y_preddf,on = 'id', how = 'inner' ,indicator=False)
join_df = join2.merge(y_predProbdf,on = 'id', how = 'inner' ,indicator=False)
finalData = join_df.rename(columns = {'0_x': 'isAI: Predicted','0_y': 'Probability: isNotAI', 1: 'Probability: isAI' })
finalData.to_csv('prediction.csv')

# Trying the predictions
input = ['Pythagoras theorem, a fundamental principle in mathematics, relates to the lengths of the sides of a right-angled triangle. According to the theorem, in a right triangle, the square of the length of the hypotenuse (the side opposite the right angle) is equal to the sum of the squares of the lengths of the other two sides. This theorem, attributed to the ancient Greek mathematician Pythagoras, has profound implications in various fields, including geometry, physics, and engineering. Trigonometry, a branch of mathematics that deals with the relationships between the angles and sides of triangles, is closely related to Pythagoras theorem. By using trigonometric functions such as sine, cosine, and tangent, one can calculate the lengths of the sides of a triangle and the measures of its angles. These functions are based on the ratios of the sides of a right triangle and are essential in solving problems involving angles, distances, and heights. Together, Pythagoras theorem and trigonometry provide powerful tools for solving a wide range of mathematical problems, from simple geometry to complex calculus. Their applications extend beyond mathematics to physics, engineering, architecture, and many other disciplines, making them foundational concepts in the study of the natural world.']
vectorInput = vectorizer.transform(input)
humanInput = ['Pythagoras is a theorem used in different sort of math techniques, it lets you evaluate the angles and sides of a triangle']
vectorHumanInput = vectorizer.transform(humanInput)
a = lg.predict(vectorInput)

# Embeddings

#USE (Unviersal Sentence Encoder) -  Google
import tensorflow as tf
import tensorflow_hub as hub
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded" % module_url)

sentence_embeddings = model(result['context'])
query = "I had pizza and pasta"
query_vec = model([query])[0]

# BERT


