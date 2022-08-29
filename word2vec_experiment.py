from bs4 import BeautifulSoup
import requests
import re
from gensim.models import Word2Vec
import pandas as pd
from copy import deepcopy

url = "https://ladyeve.ru/vsyakoe/vezhlivye-slova-spisok-vezhlivyh-slov-dlya-detej-i-vzroslyh.html?ysclid=l7a4lshpqx412178542"


def preprocess(data):
    for li in range(len(data)):
        curr = list(data[li].text)
        if '(' in curr:
            new = curr[:curr.index('(')]
            data[li] = ''.join(new)
            data[li] = re.sub(r'\.!\?', '', data[li])
        else:
            data[li] = re.sub('<li>', '', data[li].text)
            data[li] = re.sub(r'\.!\?', '', data[li])
    data = [i.lower() for i in data if i]
    return data


def parse(id, counter):
    req = requests.get(url)
    bs = BeautifulSoup(req.text, "html.parser")
    all_data = bs.find_all()
    for tag in all_data:
        if tag.name == 'h2' and tag.find('span', id=id):
            c = 0
            for i in tag.find('span', id=id).next_elements:
                if c < counter:
                    if i.name == 'ul':
                        list_of_smth = i.find_all('li')
                    c += 1
                else:
                    break
    return preprocess(list_of_smth)


hello = [[i] for i in set(parse('i', 3))]
# bye = [[i] for i in set(parse('i-5', 6))]

model = Word2Vec(vector_size=50, epochs=50, min_count=1, workers=4)
model.build_vocab(hello)
model.train(hello, total_examples=model.corpus_count, epochs=model.epochs)

data = pd.read_csv('test_data.csv')
phrases = list(data['text'])
phrases = list(map(lambda x: x.lower().split(), phrases))

for sent in range(len(phrases)):
    for word in range(len(phrases[sent])):
        curr_model = deepcopy(model)
        if curr_model.wv.has_index_for(phrases[sent][word]):
            curr_word = [phrases[sent][word], curr_model.wv.get_vector(phrases[sent][word])]
            print(curr_word[0], 'in model  --> this is greeting')
        else:
            curr_model.build_vocab([[phrases[sent][word]]], update=True)
            curr_model.train([[phrases[sent][word]]], total_examples=1, total_words=1, epochs=50)
            curr_word = [phrases[sent][word], curr_model.wv.get_vector(phrases[sent][word])]
            most_sim = curr_model.wv.most_similar(curr_word, topn=1)[0]
            if most_sim[1] >= 0.5:
                print(curr_word[0], '-->', most_sim)
