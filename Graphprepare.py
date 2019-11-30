import pandas as pd
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import pairwise_distances
import scipy.sparse

dataset_name = input("Bitte Namen des CSV-Dataset eingeben (z.B. Data.csv):   ")
column = int(input("Welche Spalte soll betrachtet werden? [0 = erste Spalte]:    "))

# try:
# 	data = pd.read_csv(dataset_name, error_bad_lines=False)
# except:
# 	data = pd.read_csv(dataset_name, error_bad_lines=False, sep=";")
data = pd.read_csv(dataset_name, error_bad_lines=False, sep=";")
print("read_csv - check")
# if int(column) == 1:
# 	working_data = data["Document Title"]
# 	subtext = "Document Title"
# elif int(column) == 2:
# 	working_data = data["Author Keywords"]
# 	subtext = "Author Keywords"
# elif int(column) == 3:
# 	working_data = data["Abstract"]
# 	subtext = "Abstract"
# elif int(column) == 4:
# 	working_data = data["Authors"]
# 	subtext = "Authors"


subtext = str(data.columns[column])
working_data = data[subtext]

print("Betrachte Spalte {}".format(subtext))
working_data = working_data.dropna()

vectorizer = CountVectorizer(ngram_range=(1,1), stop_words = 'english')
X = vectorizer.fit_transform(working_data)
print("CountVectorizer - check")
names = vectorizer.get_feature_names()
array_data = X.toarray()
new_df = pd.DataFrame(data = array_data, columns = names)
print("starting JacSim")
#np_array = scipy.sparse.csr_matrix(new_df.T.to_numpy())
np_array = new_df.T
jac_sim = 1 - pairwise_distances(np_array, metric = "jaccard")
jac_sim = pd.DataFrame(jac_sim, index=new_df.columns, columns=new_df.columns)
print("JacSim - check")

top = int(0.2*len(jac_sim))
jac_sim = jac_sim.mask(jac_sim.rank(axis=0, method='min', ascending=False) > top, 0)
print("JacSim_Reduce - check")
output_file_name = "{}_{}_GraphInput.csv".format(dataset_name[:-4], subtext)
jac_sim.to_csv(output_file_name)

print("Input für Graphen erfolgreich erzeugt. Liegt ab unter {}".format(output_file_name))