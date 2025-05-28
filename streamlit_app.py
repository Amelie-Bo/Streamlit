import streamlit as st
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Dense, GlobalAveragePooling1D
import numpy as np
from sklearn.preprocessing import Normalizer

st.title("Modèle Word2Vec")

# ─── 1. Charger le Tokenizer et vocab_size ─────────────────────────────────────
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# tokenizer.num_words contient la taille du vocabulaire défini en Colab
vocab_size = tokenizer.num_words

# Pour les fonctions d'inférence
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

# ─── 2. Reconstruire le modèle et charger les poids ────────────────────────────
embedding_dim = 300
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    GlobalAveragePooling1D(),
    Dense(vocab_size, activation='softmax')
])
model.load_weights("word2vec.h5")
vectors = model.layers[0].get_weights()[0]  # shape = (vocab_size, embedding_dim)

# ─── 3. Vos fonctions de similarité ────────────────────────────────────────────
def dot_product(vec1, vec2):
    return np.sum(vec1 * vec2)

def cosine_similarity(vec1, vec2):
    return dot_product(vec1, vec2) / np.sqrt(dot_product(vec1, vec1) * dot_product(vec2, vec2))

def find_closest(word_index, vectors, number_closest):
    sims = []
    query = vectors[word_index]
    for i, vec in enumerate(vectors):
        if i != word_index:
            sims.append((cosine_similarity(query, vec), i))
    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:number_closest]

def print_closest(word, number=10):
    idx = word2idx.get(word)
    if idx is None or idx >= vocab_size:
        st.write(f"Le mot « {word} » n’est pas dans le vocabulaire.")
        return
    for sim, i in find_closest(idx, vectors, number):
        st.write(f"{idx2word[i]} — {sim:.4f}")

# ─── 4. Interface utilisateur ─────────────────────────────────────────────────
mot = st.text_input("Entrez un mot :")
k    = st.slider("Nombre de voisins à afficher", 1, 50, 10)

if mot:
    st.subheader(f"Les {k} mots les plus proches de « {mot} »")
    print_closest(mot, k)
