# Greetings_Farewell_Detector
The task was to identify the sentences in which there was a greeting or a farewell.
My first decision was to train the Word2Vec model on a dataset of different greetings/farewells, then add
a single word from a sentence to this model and compare the vectors of this word and my dataset. 
So if some vector from my model will have more than 70-80% of simmilarity with given word, that will mean
that current word is probably a greeting/farewell too.
In case of lack of examples and pretrained models of greetings detection, I decided to create my own dataset
by searching the web looking for a website from where I can parse greetings/farewells examples.
I found such website (https://ladyeve.ru/vsyakoe/vezhlivye-slova-spisok-vezhlivyh-slov-dlya-detej-i-vzroslyh.html?ysclid=l7a4lshpqx412178542) 
and wrote a parser which helped me to create my small dataset.
After that I trained Word2Vec model and compared the vectors of each word from given examples. But unfortunatelly,
those vectors were not really representative and all vectors didn't have more than 30-40% of simmilarity.
My model was detecting greeting only if this word was absolutely the same as in my dataset.

After my first decision fails, I tried to use Doc2Vec instead to represent a whole sentences as vectors 
and make this task looks like classic classification task. I managed to label all the given sentences. 
In the copy of given csv file I added binary columns 'greetings' and 'farewell'. The value 1 means that in this 
sentence there is a greeting/farewell and 0 in the opposite. After that I preprocessed all the sentences 
with a little bit of regular expressions and deleting stop-words. Then I trained Doc2Vec model on that 
sentences and got their vector representations.
For classification I used XGBooster classificator, because gradient boosting algorithms are really strong and give very good accuracies,
moreover I worked with it not so long ago and it was not really hard for me. To make predictions more objective, 
I used KFold cross-validation with 5 folds. Average accuracy on train set is 100%, on test set is 96.7%.
Training the model took 0.073 seconds and 0.009 seconds for prediction. 

To sum up, we got a nice classification model with high accuracy. In the future this project can be improved 
by retraining this model on bigger and more representative dataset.
