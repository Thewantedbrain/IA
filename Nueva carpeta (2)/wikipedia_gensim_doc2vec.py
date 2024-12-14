from gensim.models import Doc2Vec
from collections import namedtuple
import re
import string
import random
import numpy as np

# Leer archivo y tomar una muestra aleatoria de líneas
file_path = "wikipedia.txt"
sample_size = 5000
with open(file_path, "r", encoding="UTF-8") as file:
    lines = file.readlines()  # Leer todas las líneas
    sampled_lines = random.sample(lines, min(len(lines), sample_size))  # Seleccionar muestra aleatoria

# Preprocesar las líneas seleccionadas
sentenceEnders = re.compile(r'[.!?]\s+')
exclude = set(string.punctuation)
LabelDoc = namedtuple('LabelDoc', 'words tags')
all_docs = []
for idx, line in enumerate(sampled_lines):
    sentences = sentenceEnders.split(line)  # Dividir por signos de puntuación
    for sentence in sentences:
        sentence = ''.join(ch for ch in sentence if ch not in exclude)  # Quitar puntuación
        words = sentence.split()  # Dividir en palabras
        if len(words) < 3:  # Ignorar frases muy cortas
            continue
        tag = [f"SEN_{idx}"]  # Etiqueta única
        all_docs.append(LabelDoc(words, tag))

print(f"Total de documentos procesados: {len(all_docs)}")
print(f"Primeros 3 documentos: {all_docs[:3]}")

# Entrenar el modelo Doc2Vec
model = Doc2Vec(vector_size=50, min_count=2, epochs=10, alpha=0.025, min_alpha=0.025)
model.build_vocab(all_docs)

print("Entrenando el modelo...")
for epoch in range(10):
    print(f"Época {epoch+1}")
    model.train(all_docs, total_examples=model.corpus_count, epochs=model.epochs)
    model.alpha -= 0.002  # Reducir la tasa de aprendizaje
    model.min_alpha = model.alpha

# Guardar el modelo
model.save('wikipedia_sample.doc2vec')

# Análisis de similitud
doc_id = np.random.randint(len(all_docs))  # Seleccionar un documento aleatorio
print(f"Documento objetivo (ID={doc_id}): {all_docs[doc_id].words}")

sims = model.dv.most_similar(f"SEN_{doc_id}", topn=10)  # Documentos más similares
print("\nDocumentos similares:")
for sim_id, similarity in sims:
    similar_idx = int(sim_id.replace("SEN_", ""))
    print(f"{sim_id} (similaridad: {similarity:.4f}): {all_docs[similar_idx].words}")