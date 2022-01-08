#!/opt/anaconda3/bin/python

import numpy as np
import csv
import os
import pickle

#Elexicon data
words = []
prob = []
sum_prob = 0
with open("../Datasets/Items.csv", 'r') as file:
    reader = csv.reader(file)
    first = True
    for row in reader:
        if first:
            first = False
            continue
        if len(row[0]) > 20:
            continue
        words.append([row[0], 'word'])
        row[2] = row[2].replace(',', '')
        prob.append(int(row[2]))
        sum_prob += int(row[2])


for i in range(len(prob)):
    prob[i] /= sum_prob
print(words[:20])



#Use API to find meaningful words
import requests


URL = 'https://api.dictionaryapi.dev/api/v2/entries/en/'

res = []
meaningful_words = []

for w, t in words:
    # sending get request and saving the response as response object
    new_URL = URL + w
    r = requests.get(url = new_URL)
    try:
        # extracting data in json format
        data = r.json()
        if type(data) == list:
            data = data[0]
        if 'word' in data:
            meaningful_words.append(w)
    except:
        print(r)


print(meaningful_words[:10])
## Create New Dataset for Meaningful words with Frequency
freq = {}
with open("../Datasets/Items.csv", 'r') as file:
        reader = csv.reader(file)
        first = True
        for row in reader:
                if first:
                    first = False
                    continue

                if row[0] in meaningful_words:
                        if len(row[0]) > 20:
                                continue
                        row[2] = row[2].replace(',', '')
                        freq[row[0]] = int(row[2])

print(list(freq.keys())[:10])
print(list(freq.items())[:10])

outfile = open("../check_points/freq.dic", "wb")
pickle.dump(freq, outfile)
outfile.close()

with open("../Datasets/meaningful_words_with_frq.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for k in range(len(meaningful_words)):
                writer.writerow([meaningful_words[k], freq[meaningful_words[k]]])

