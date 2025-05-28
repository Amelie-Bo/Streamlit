import streamlit as st
import pickle
import numpy as np
import re, unicodedata
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from sklearn.preprocessing import Normalizer

st.title("Modèle Word2Vec")

# ─── 1. Téléchargement des stopwords NLTK ───────────────────────────────────────
nltk.download('stopwords')
nltk.download('punkt')

# ─── 2. Chargement du tokenizer et du vocab_size ──────────────────────────────
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('vocab_size.pkl', 'rb') as f:
    vocab_size = pickle.load(f)

word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

# ─── 3. Reconstruction du modèle ───────────────────────────────────────────────
embedding_dim = 300
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim),
    GlobalAveragePooling1D(),
    Dense(vocab_size, activation='softmax')
])

# ─── 4. Chargement des poids ──────────────────────────────────────────────────
model.load_weights("word2vec.h5")
vectors = model.layers[0].get_weights()[0]  # shape = (vocab_size, embedding_dim)

# ─── 5. Pré‐traitement pour de nouvelles requêtes ─────────────────────────────
stop_words = set(stopwords.words('english'))

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z?.!]+", " ", w)
    w = re.sub(r'\b\w{0,2}\b', '', w)
    tokens = word_tokenize(w.strip())
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# ─── 6. Fonctions de similarité ────────────────────────────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))

def find_closest(idx, n=10):
    query = vectors[idx]
    sims = [(cosine_similarity(query, vec), i) 
            for i, vec in enumerate(vectors) if i != idx]
    sims.sort(reverse=True, key=lambda x: x[0])
    return sims[:n]

def print_closest(word, number=10):
    idx = word2idx.get(word)
    if idx is None or idx >= vocab_size:
        st.write(f"Le mot « {word} » n’est pas dans le vocabulaire.")
        return
    for sim, i in find_closest(idx, number):
        st.write(f"{idx2word[i]} — {sim:.4f}")

# ─── 7. Interface Streamlit ───────────────────────────────────────────────────
mot = st.text_input("Entrez un mot (en anglais) :")
k   = st.slider("Nombre de voisins à afficher :", 1, 50, 10)

if mot:
    # On pré‐traite l’entrée pour être sûr qu’elle matche le vocabulaire
    mot_proc = preprocess_sentence(mot)
    st.subheader(f"Les {k} mots les plus proches de « {mot_proc} »")
    print_closest(mot_proc, k)
