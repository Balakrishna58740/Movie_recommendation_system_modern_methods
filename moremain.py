from gensim.models import Word2Vec

# Example data: movie descriptions
descriptions = [
    "dream within a dream thriller",
    "space exploration family love",
    "virtual reality hacking thriller",
    "astronaut survival space",
    "alien communication space language"
]

# Tokenize descriptions
tokenized = [desc.split() for desc in descriptions]

# Train Word2Vec
model = Word2Vec(sentences=tokenized, vector_size=100, window=5, min_count=1, workers=4)
def get_similarity(word1, word2, model):
    return model.wv.similarity(word1, word2)

# Example similarity
print("Similarity between 'space' and 'alien':", get_similarity("space", "alien", model))
