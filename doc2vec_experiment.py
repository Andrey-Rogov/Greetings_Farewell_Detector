import numpy as np
from gensim.models import Word2Vec
from time import time
import re
import gensim
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

with open('stop_words_russian.txt', 'r', encoding='utf-8') as f:
    stop_words = f.read().splitlines()


def preprocessing(text):
    pattern = r'[^А-Яа-я\']'
    text = [[word.lower() for word in re.split(r'[ /]', sentence)] for sentence in text]
    for sentence in range(len(text)):
        for word in range(len(text[sentence])):
            if re.search("&#039;", text[sentence][word]):
                text[sentence][word] = re.sub("&#039;", "'", text[sentence][word])
            if re.search(pattern, text[sentence][word]):
                text[sentence][word] = re.sub(pattern, '', text[sentence][word])
        while '' in text[sentence]:
            useless = text[sentence].index('')
            text[sentence] = text[sentence][:useless] + text[sentence][useless + 1:]
    text = [[word for word in sentence if word not in stop_words] for sentence in text]
    return text


data = pd.read_csv('test_data.csv')
phrases = list(data['text'])
phrases = [i if i else np.NAN for i in preprocessing(phrases)]

data['text_vectors'] = pd.Series(data=phrases)
data.dropna(inplace=True)  # 480 - 371 rows deleted
data.to_csv('working_data.csv', index=False)

data = pd.read_csv('working_data.csv')
phrases = data['text_vectors']
tagged_train_data = [TaggedDocument(d, [i]) for i, d in enumerate(phrases)]
model = gensim.models.Doc2Vec(vector_size=50, window=2, min_count=1, workers=4)
model.build_vocab(tagged_train_data)
model.train(tagged_train_data, total_examples=model.corpus_count, epochs=100)
model.save('d2v_gf_model')

# model = gensim.models.Doc2Vec.load('d2v_gf_model')
X = []
for sent in tagged_train_data:
    X.append(*model.dv[sent[1]])
y = pd.Series(data['greetings'])
X = pd.Series(data=X)
model = xgb.XGBClassifier(n_estimators=50, n_jobs=4, booster='gbtree', max_depth=20,
                          max_leaves=100, use_label_encoder=False, eval_metric='mlogloss')
kf = KFold(n_splits=5)
accuracies_test = []
accuracies_train = []
split_num = 1
le = LabelEncoder()
for train_index, test_index in kf.split(X, y):
    s = time()
    X_train, X_test = list(X[train_index]), list(X[test_index])
    y_train, y_test = list(y[train_index]), list(y[test_index])
    y_train = le.fit_transform(y_train)
    y_test = le.fit_transform(y_test)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    accuracies_test.append(accuracy_score(y_test, y_pred_test))
    accuracies_train.append(accuracy_score(y_train, y_pred_train))

    print(f'{split_num} split is done for {time() - s} seconds')
    split_num += 1
print(f'{accuracies_test} --> mean {np.mean(accuracies_test)}')
print(f'{accuracies_train} --> mean {np.mean(accuracies_train)}')
