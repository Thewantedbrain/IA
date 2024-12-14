from gensim.models import Doc2Vec
from collections import namedtuple
import csv
import re
import string
import random
import numpy as np

# Leer el archivo CSV
with open("boletinesprueba.csv", newline='', encoding='UTF-8', errors='ignore') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    data = ''
    count = 0
    for row in reader:
        count += 1
        if count > 55:
            break
        # Verificar que la fila tenga al menos dos columnas
        if len(row) > 1:
            data += row[1]  # Concatenar solo la columna de texto
        else:
            print(f"Fila {count} no tiene suficientes columnas: {row}")

# Configurar una expresión regular para dividir el párrafo en oraciones.
sentenceEnders = re.compile('[.?!]')
data_list = sentenceEnders.split(data)

# Crear un namedtuple con words y tags
LabelDoc = namedtuple('LabelDoc', 'words tags')
exclude = set(string.punctuation)
all_docs = []
count = 0
for sen in data_list:
    word_list = sen.split()
    if len(word_list) < 3:
        continue
    tag = ['SEN_' + str(count)]
    count += 1
    sen = ''.join(ch for ch in sen if ch not in exclude)
    all_docs.append(LabelDoc(sen.split(), tag))

print(all_docs[0:10])

# Entrenar el modelo Doc2Vec
model = Doc2Vec(alpha=0.025, min_alpha=0.025)  # usar tasa de aprendizaje fija
model.build_vocab(all_docs)
for epoch in range(10):
    model.train(all_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002  # disminuir la tasa de aprendizaje
    model.min_alpha = model.alpha  # fijar la tasa de aprendizaje, sin decaimiento    

model.save('boletinesprueba.doc2vec')

# Obtener un documento aleatorio
doc_id = np.random.randint(model.dv.vectors.shape[0]) 
print(doc_id)

sims = model.dv.most_similar(doc_id, topn=model.dv.vectors.shape[0])
print('TARGET', all_docs[doc_id].words)

count = 0
for i in sims:
    if count > 8:
        break
    pid = int(i[0].replace("SEN_", ""))
    print(i[0], ": ", all_docs[pid].words)
    count += 1