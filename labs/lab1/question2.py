# Francois RD
# CSC2611 - Lab 1 - Question 2: Diachronic word embedding
# Feb 2022

# Step 1: Download the diachronic word2vec embeddings from the course webpage.
import pickle
with open("../../data/embeddings/data.pkl", 'rb') as f:
    embeddings = pickle.load(f)

# Several of the embeddings are the zero vector over multiple decades. This is
# clearly an artifact/error and these words should be removed.
import numpy as np

to_keep = []
for i in range(len(embeddings['w'])):
    if np.all(np.any(embeddings['E'][i], axis=1)):
        to_keep.append(i)
embeddings['w'] = [embeddings['w'][i] for i in to_keep]
embeddings['E'] = [embeddings['E'][i] for i in to_keep]

# Step 2: Propose three different methods for measuring degree of semantic
# change for individual words and report the top 20 most and least changing
# words in table(s) from each measure. Measure the intercorrelations (of
# semantic change in all words, given the embeddings from Step 1) among the
# three methods you have proposed and summarize the Pearson correlations in a
# 3-by-3 table.
from sklearn.metrics.pairwise import cosine_similarity as cosim

# Method 1: Consecutive semantic displacement.
consecutive_displacement = []
for decs_word in embeddings['E']:
    consecutive_displacement.append([
        1 - cosim([dec1_word], [dec2_word])[0][0]
        for dec1_word, dec2_word in zip(decs_word, decs_word[1:])
    ])

# Method 2: Original semantic displacement.
original_displacement = []
for decs_word in embeddings['E']:
    original_displacement.append([
        1 - cosim([decs_word[0]], [dec_word])[0][0]
        for dec_word in decs_word[1:]
    ])

# Method 3: Nearest neighbour semantic displacement.
neighbour_displacement = []
for i in range(len(embeddings['E'])):
    if i % 100 == 0:
        print("{} of {} words processed".format(i, len(embeddings['E'])))
    # First, find the closest other word to the given word in the first decade.
    max_cosim, closest_word = 0, None
    for j in range(len(embeddings['E'])):
        if i != j:
            res = cosim([embeddings['E'][i][0]], [embeddings['E'][j][0]])[0][0]
            if res > max_cosim:
                max_cosim = res
                closest_word = j
    # Next, find the distance between these two words in each decade.
    i_decs_word, j_decs_word = embeddings['E'][i], embeddings['E'][closest_word]
    neighbour_displacement.append([
        1 - cosim([i_word], [j_word])[0][0]
        for i_word, j_word in zip(i_decs_word, j_decs_word)
    ])

# Compute the maximum times series value for each word. The maximum 1-cosim
# value is the most changed time point. The most changed word is the one with
# the most changed time point (i.e. the maximum of maximums). The least changed
# word is the one with the least changed of the most changed time points (i.e.
# the minimum of the maximums).
def most_and_least_changed(time_series_list, topn=20):
    max_list = [max(time_series) for time_series in time_series_list]
    indices = np.argsort(max_list)
    most_changed = [(i, max_list[i]) for i in reversed(indices[-topn:])]
    least_changed = [(i, max_list[i]) for i in reversed(indices[:topn])]
    return most_changed, least_changed, max_list

c_most, c_least, c_all = most_and_least_changed(consecutive_displacement)
o_most, o_least, o_all = most_and_least_changed(original_displacement)
n_most, n_least, n_all = most_and_least_changed(neighbour_displacement)

# Display the 20 most and least changed for each method in a table.
def format_one(i, d):
    return "{:<20}".format("{} ({:.2f})".format(embeddings['w'][i], d))

print("Table 1: Most changed word. Change score in parentheses.")
print("{:<20} {:<20} {:<20}".format("Consecutive", "Original", "Neighbour"))
for c, o, n in zip(c_most, o_most, n_most):
    print(format_one(*c), format_one(*o), format_one(*n))

print("Table 2: Least changed word. Change score in parentheses.")
print("{:<20} {:<20} {:<20}".format("Consecutive", "Original", "Neighbour"))
for c, o, n in zip(c_least, o_least, n_least):
    print(format_one(*c), format_one(*o), format_one(*n))

# Measure and report the Pearson correlation between the three methods.
from scipy.stats import pearsonr

print("Pairwise Pearson correlations:")
print("            Consecutive    Original    Neighbour")
print("Consecutive -              {:.2f}        {:.2f}".format(
    pearsonr(c_all, o_all)[0], pearsonr(c_all, n_all)[0]))
print("Original    -              -           {:.2f}".format(
    pearsonr(o_all, n_all)[0]))
print("Neighbour   -              -           -")

# Step 3: Propose and justify a procedure for evaluating the accuracy of the
# methods you have proposed in Step 2, and then evaluate the three methods
# following this proposed procedure and report Pearson correlations or relevant
# test statistics.

# Import list of known changed words from the reference papers. Prune away the
# ones not included in this small subset of English.
with open("../../data/known-change-words.txt") as f:
    known_words = [line.strip() for line in f.readlines()
                   if line.strip().lower() in embeddings['w']]
known_words_indices = [embeddings['w'].index(word) for word in known_words]
print(known_words)

def evaluate(method):
    median = np.median(method)
    acc = [method[idx] > median for idx in known_words_indices]
    return sum(acc) / len(acc)

print("Consecutive:", evaluate(c_all))
print("Original:", evaluate(o_all))
print("Neighbour:", evaluate(n_all))

# Step 4: Extract the top 3 changing words using the best method from Steps 2
# and 3. Propose and implement a simple way of detecting the point(s) of
# semantic change in each word based on its diachronic embedding time course.
# Visualize the time course and the detected change point(s).

# The best method is: Original.
import matplotlib.pyplot as plt

for top3 in o_most[:3]:
    word_idx = top3[0]
    change_point = None
    for i, time_point in enumerate(original_displacement[word_idx]):
        if time_point > 0.5:
            change_point = i
            break
    if change_point is None:
        change_point = np.argmax(original_displacement[word_idx])
    past_points = original_displacement[word_idx][:change_point]
    future_points = original_displacement[word_idx][1 + change_point:]
    xticks = np.arange(len(embeddings['d'][1:])).tolist()
    del xticks[change_point]
    plt.figure()
    plt.plot(xticks, past_points + future_points, 'bo')
    plt.plot(change_point, original_displacement[word_idx][change_point], 'r*')
    plt.xticks(np.arange(len(embeddings['d'][1:])), embeddings['d'][1:],
               rotation=20)
    plt.xlabel("Decade")
    plt.ylabel("Semantic Change")
    plt.title("Original Semantic Change Time Series for '{}'".format(
        embeddings['w'][word_idx]
    ))
    plt.show()
