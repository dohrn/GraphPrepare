import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse
import numpy as np



dataset_name = "Input_CSV/{}".format(input("Bitte Namen des CSV-Dataset eingeben (z.B. Data.csv):   "))
column = int(input("Welche Spalte soll betrachtet werden? [0 = erste Spalte]:    "))
to_mask = int(input("Soll der Datensatz gekürzt werden? [1/0]:   "))


if to_mask == 1:
	mask_weight = float(input("Welches Verhältnis? [z.B. 0.5]:    "))
else:
	mask_weight = 1

cust_sep = int(input("Hat die CSV einen anderen Serperator als \",\" [0/1]:   "))

if cust_sep == 1:
	sep = str(input("Welcher Seperator?  Tab = \"\\t\" / Semikolon = ;:  "))
else:
	sep = ","
#comment

# try:
# 	data = pd.read_csv(dataset_name, error_bad_lines=False)
# except:
# 	data = pd.read_csv(dataset_name, error_bad_lines=False, sep=";")

data = pd.read_csv(dataset_name, error_bad_lines=False, sep=sep, skip_blank_lines=True)
print("read_csv - check")
print(data.head())
if to_mask == 1:
	mask = np.random.rand(len(data)) <= mask_weight

	data = data[mask]

print(len(data))

subtext = str(data.columns[column])
working_data = data[subtext]

print("Betrachte Spalte {}".format(subtext))
working_data = working_data.dropna()

max_features_working_data = int(4 * 10 ** 9 / (80*len(data))) - 2

if subtext == "Authors":
	author_sep = str(input("Welcher Seperator trennt die Autoren in einem Feld? Tabe = \"\\t\" / Semikolon = ; / Komma = , :"))
	vectorizer = CountVectorizer(stop_words = 'english', max_features = max_features_working_data, tokenizer = lambda x: x.split(author_sep)).fit(working_data)
else:
	vectorizer = CountVectorizer(ngram_range = (1,1), stop_words = 'english', max_features = max_features_working_data).fit(working_data)
X = vectorizer.transform(working_data)

print("CountVectorizer - check")
names = vectorizer.get_feature_names()
array_data = X.toarray()


new_df = pd.DataFrame(data = array_data, columns = names)
print("starting JacSim")
#np_array = scipy.sparse.csr_matrix(new_df.T.to_numpy())
np_array = new_df.T
jac_sim = 1 - pairwise_distances(np_array, metric = "cosine")
jac_sim = pd.DataFrame(jac_sim, index=new_df.columns, columns=new_df.columns)
print("JacSim - check")

#top = int(0.2*len(jac_sim))
#jac_sim = jac_sim.mask(jac_sim.rank(axis=0, method='min', ascending=False) > top, 0)
print("JacSim_Reduce - check")
output_file_name = "{}_{}_GraphInput.csv".format(dataset_name[:-4], subtext)
jac_sim.to_csv(output_file_name)

print("Input für Graphen erfolgreich erzeugt. Liegt ab unter {}".format(output_file_name))
