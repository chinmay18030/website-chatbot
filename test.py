# import nltk
# import numpy as np
# import json
# import random
# from nltk.stem import WordNetLemmatizer
# from keras.models import Sequential , load_model
# from keras.layers import Dense, Activation, Dropout, Flatten
# import pickle
# import time
#
#
# lemmatizer = WordNetLemmatizer()
# data = json.load(open("static/intents.json",))
#
# words = []
# classes = []
# documents = []
# ignore_letters = ["?","!",",","."]
#
# for intent in data["intents"]:
#     for pattern in intent["patterns"]:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append([word_list, intent["tag"]])
#     if intent["tag"] not in classes:
#         classes.append(intent["tag"])
#
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
# words = sorted(set(words))
# print(words)
#
# classes = sorted(set(classes))
# training = []
# output_empty = [0] * len(classes)
#
# for document in documents:
#     bag = []
#
#     word_patterns = document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     print(word_patterns)
#
#     for word in words:
#
#         if word in word_patterns:
#             bag.append(1)
#         else:
#             bag.append(0)
#
#     output_row = list(output_empty)
#     output_row[classes.index(document[1])] = 1
#
#     training.append([bag, output_row])
#
# random.shuffle(training)
# training = np.array(training)
#
# train_x = list(training[:,0])
# train_y = list(training[:,1])
#
#
# x = np.array(train_x)
# y = np.array(train_y)
#
#
# model = load_model("Chatbot.model")
# def clean_sentence(sentence):
#     sentence_word = nltk.word_tokenize(sentence)
#     sentence_word = [lemmatizer.lemmatize(word) for word in sentence_word]
#     return sentence_word
#
# def  bag_of_words(sentence):
#     sw = clean_sentence(sentence)
#     bag  = [0]*len(words)
#     for w in sw:
#         for i,word in enumerate(words):
#             if word==w:
#                 bag[i]=1
#     return np.array(bag)
# print(bag_of_words("Hello I am Chinmay"))
#
# def predict_class(model,sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     results = [[i,r] for i, r in enumerate(res) if r>0.25]
#     results.sort(key = lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({"intent":classes[r[0]], "probability":str(r[1])})
#     return return_list
# print(predict_class(model,"what is your name ? "))
#
# def get_response(intent_list, intent_json):
#     tag = intent_list[0]["intent"]
#     list_of_intents = intent_json["intents"]
#     for i in list_of_intents:
#         if i["tag"]==tag:
#             result = random.choice(i["responses"])
#             break
#     return result
#
#
# while True:
#     print(
#         "_____________________________________________________________________________________________________________________________________________________________________________")
#     asi = input("Message:")
#     ints = predict_class(model, asi.lower())
#     res = get_response(ints, data)
#     val_loss, val_acc = model.evaluate(x, y)
#     h = "_____________________________________________________________________________________________________________________________________________________________________________"
#     for i in h:
#         print(i, end="")
#         time.sleep(0.001)
#     print("\n Accuracy:", val_acc)
#     print("Loss:", val_loss)
#     print("Probability:", ints[0]["probability"])
#     h = "_____________________________________________________________________________________________________________________________________________________________________________"
#     for i in h:
#         print(i, end="")
#         time.sleep(0.001)
#
#     if float(ints[0]["probability"]) > 0.40:
#         print(" \nBy chatbot : " + res)
#     else:
#         print("\nSorry I did not understand what you said")
#
#     if ints[0]["intent"] == "jokes":
#         import jokobot
#     if ints[0]["intent"] == "goodbye":
#         break
#     if ints[0]["intent"] == "nothing":
#         break
#
#
#
#





import chat

data = chat.load_data("static/intents.json")
x1,y1,words,documents,classes = chat.data_processing(data["intents"],"patterns","tag")
model = chat.loadmodel("Chatbot.model")

response = chat.predict_class(model, "hi", words, classes)
resp = chat.get_response(response,data)

print(resp)