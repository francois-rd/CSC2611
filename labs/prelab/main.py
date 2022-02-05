# Francois RD
# CSC2611 - Prelab
# Jan 2022

# Step 1:
from nltk.corpus import brown

# Step 2:
import nltk

# Using Brown's definition of 'word' here. This includes punctuation.
words = [w.lower() for w in brown.words()]
# Compute the unigram frequencies.
unigram_freqs = nltk.FreqDist(words)
# W = 5000 most frequent words.
W = [w for w, _ in unigram_freqs.most_common(5000)]
print("Report the 5 most and least common words from W:")
print(W[:5])
print("-----")
print(W[-5:])
# Import words from RG65 Table 1.
with open("../../data/RG65-Table1-words.txt") as f:
    rg65_words = [w.strip() for w in f.readlines()]
# Add to W all words RG65 that are not already in W.
W = list(set(W) | set(rg65_words))

# Step 3:
from scipy.sparse import lil_matrix

# Create a sparse word-context matrix.
M1 = lil_matrix((len(W), len(W)))
# Compute bigrams for each sentence individually. This is different from
# computing the bigrams over all the words, since that would be including
# bigrams from two sentences that are not actually consecutive. That is to
# say, using "brown.words()" means that you get all sentences from all Brown
# Corpus documents collapsed into a single sequence, so you would be finding
# bigrams that join the last sentence of document 1 with the first sentence
# of document 2 and so on.
bigrams = [list(nltk.bigrams(w.lower() for w in s)) for s in brown.sents()]
# Compute bigram frequencies over all sentences.
bigram_freqs = nltk.FreqDist([b for sb in bigrams for b in sb]).most_common()
# Compute word indices.
W_indices = {w: i for i, w in enumerate(W)}
# Fill in the matrix: matrix[word][context] = bigram_freq
for bigram in bigram_freqs:
    try:
        M1[W_indices[bigram[0][1]], W_indices[bigram[0][0]]] = bigram[1]
    except KeyError:
        # This means that one or both bigram words is >5000 in commonness.
        continue

# Step 4:
import math

# Convert the sparse matrix to a more efficient format.
M1_coo = M1.tocoo()
# Since the bigram frequencies are computed differently than the unigram
# frequencies (see comments in Step 3), we can't use unigram counts from Step 1
# in these PPMI calculations. Instead, we recompute unigram frequencies from row
# and column sums of M1.
M1_row_sums, M1_col_sums = M1_coo.sum(axis=1).transpose(), M1_coo.sum(axis=0)
M1_row_sums, M1_col_sums = M1_row_sums.tolist()[0], M1_col_sums.tolist()[0]
# Same argument for recomputing the total sum.
M1_sum = M1_coo.sum()
# Create a sparse PPMI matrix.
M1_plus = lil_matrix((len(W), len(W)))
# Efficiently iterate over the sparse matrix to compute the PPMI.
for r, c, f in zip(M1_coo.row, M1_coo.col, M1_coo.data):
    pmi = math.log2(f * M1_sum) - math.log2(M1_row_sums[r] * M1_col_sums[c])
    M1_plus[r, c] = max(pmi, 0)

# Step 5:
from sklearn.decomposition import TruncatedSVD

# Apply TruncatedSVD to perform LSA on a PPMI matrix. Do this 3 times for each
# of 10, 100, and 300 dimensions. The reason for using TruncatedSVD instead of
# PCA is that PCA cannot operate on sparse matrices:
#   https://stackoverflow.com/questions/33603787/performing-pca-on-large-sparse-matrix-by-using-sklearn
# In addition to this point, the documentation of TruncatedSVD explicitly states
# that TruncatedSVD operating on word-context matrices is a way to perform LSA.
M2_10 = TruncatedSVD(n_components=10)
M2_10.fit(M1_plus)
M2_100 = TruncatedSVD(n_components=100)
M2_100.fit(M1_plus)
M2_300 = TruncatedSVD(n_components=300)
M2_300.fit(M1_plus)

# Step 6:

# By Step 1, all words in RG65 are in W already, or were added to it, so we just
# load the entirety of Table 1 of RG65.
with open("../../data/RG65-Table1.csv") as f:
    rg65_table1 = [line.strip().split(',') for line in f.readlines()]
P = [(w1, w2) for w1, w2, _ in rg65_table1]
S = [float(row[2]) for row in rg65_table1]

# Step 7:
from sklearn.metrics.pairwise import cosine_similarity as cosim

P_indices = [(W_indices[w1], W_indices[w2]) for w1, w2 in P]

def similarity_from_map(map_func):
    return [cosim(map_func(i), map_func(j))[0][0] for i, j in P_indices]

S_M1 = similarity_from_map(lambda i: M1[i])
S_M1_plus = similarity_from_map(lambda i: M1_plus[i])
S_M2_10 = similarity_from_map(lambda i: M2_10.transform(M1_plus[i]))
S_M2_100 = similarity_from_map(lambda i: M2_100.transform(M1_plus[i]))
S_M2_300 = similarity_from_map(lambda i: M2_300.transform(M1_plus[i]))

# Step 8:
from scipy.stats import pearsonr

print("\nReport the Pearson correlation between S and each of the "
      "model-predicted similarities:")
print("S_M1:    ", pearsonr(S, S_M1)[0])
print("S_M1+:   ", pearsonr(S, S_M1_plus)[0])
print("S_M2_10: ", pearsonr(S, S_M2_10)[0])
print("S_M2_100:", pearsonr(S, S_M2_100)[0])
print("S_M2_300:", pearsonr(S, S_M2_300)[0])
