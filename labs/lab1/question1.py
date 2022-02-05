# Francois RD
# CSC2611 - Lab 1 - Question 1: Synchronic word embedding
# Feb 2022

# Step 1: Download the pre-trained word2vec embeddings
# Done.

# Step 2: Using gensim, extract embeddings of words in Table 1 of RG65 that also
# appeared in the set W from the earlier exercise, i.e., the pairs of words
# should be identical in all analyses.

# Because Step 2 of the earlier exercise says "Update W by adding n words where
# n is the set of words in Table 1 of RG65 that were not included in the top
# 5000 words from the Brown corpus" then, by definition, all words from RG65 do
# appear in W.

# Import words from RG65 Table 1.
with open("../../data/RG65-Table1.csv") as f:
    rg65_table1 = [line.strip().split(',') for line in f.readlines()]
P = [(w1, w2) for w1, w2, _ in rg65_table1]
S = [float(row[2]) for row in rg65_table1]

# Load word2vec using gensim.
from gensim.models import KeyedVectors

word2vec = KeyedVectors.load_word2vec_format(
    "../../data/GoogleNews-vectors-negative300.bin", binary=True)

# Extract the word embeddings for each pair of words.
P_emb = [(word2vec[w1], word2vec[w2]) for w1, w2 in P]

# Step 3: Calculate cosine distance between each pair of word embeddings you
# have extracted, and report the Pearson correlation between word2vec-based and
# human similarities.

# Calculate the cosine similarity for each pair of embeddings.
from sklearn.metrics.pairwise import cosine_similarity as cosim

S_w2v = [cosim([emb1], [emb2])[0][0] for emb1, emb2 in P_emb]

# Calculate the Pearson correlation.
from scipy.stats import pearsonr

print("Corr(S, S_w2v):", pearsonr(S, S_w2v)[0])

# Step 4:

# Import the PPMI, LSA. and word indices from the earlier exercise.
from labs.prelab.main import M1_plus, M2_300, W_indices

# Load the analogies data and split it according to semantic vs syntactic test.
semantic, syntactic = [], []
with open("../../data/analogies.txt") as f:
    current = semantic
    for line in f.readlines():
        if "gram1-adjective" in line.strip():
            current = syntactic
        if not line.strip().startswith(":"):
            current.append(line.strip().lower().split())

# Filter out words that aren't in the LSA model.
def filter_words(analogies):
    analogies_filtered = []
    for w1, w2, w3, w4 in analogies:
        if w1 in W_indices and w2 in W_indices \
                and w3 in W_indices and w4 in W_indices:
            analogies_filtered.append((w1, w2, w3, w4))
    return analogies_filtered

semantic, syntactic = filter_words(semantic), filter_words(syntactic)

# Perform the analogy test for word2vec.
def word2vec_test(analogies):
    successes = 0
    count = 0
    for w1, w2, w3, w4 in analogies:
        if count % 20 == 0:
            print("{} of {} tests performed".format(count, len(analogies)))
        count += 1
        result = word2vec.most_similar(positive=[w1, w4], negative=[w2], topn=1)
        if result[0][0] == w3:
            successes += 1
    return successes / len(analogies)

print("Word2vec semantic analogical test accuracy:", word2vec_test(semantic))
print("Word2vec syntactic analogical test accuracy:", word2vec_test(syntactic))

# Perform the analogy test for the LSA model.
embed = lambda word: M2_300.transform(M1_plus[W_indices[word]])
all_embeddings = {word: embed(word) for word in W_indices}

def lsa_test(analogies):
    successes = 0
    count = 0
    for w1, w2, w3, w4 in analogies:
        if count % 20 == 0:
            print("{} of {} tests performed".format(count, len(analogies)))
        count += 1
        analogy = embed(w1) + embed(w4) - embed(w2)
        max_cosim, closest_word = 0, None
        for word, embedding in all_embeddings.items():
            if word not in [w1, w2, w4]:  # Input words aren't possible answers.
                current_cosim = cosim(analogy, embedding)[0][0]
                if current_cosim > max_cosim:
                    max_cosim = current_cosim
                    closest_word = word
        if closest_word == w3:
            successes += 1
    return successes / len(analogies)

print("LSA semantic analogical test accuracy:", lsa_test(semantic))
print("LSA syntactic analogical test accuracy:", lsa_test(syntactic))
