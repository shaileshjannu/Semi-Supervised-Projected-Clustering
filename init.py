import time
import numpy as np
import scipy as sp
from pprint import pprint
from utils import preprocess, calc_selection_threshold, select_dim, initialize
import json

t0 = time.time()
data = preprocess('dataset_diabetes/diabetic_data.csv')
N, V = data.shape

# labeled_objects[i][j] = j-th object in i-th cluster.
labeled_objects = [[9, 23, 543], [456, 34, 52], [76, 500, 381, 245]]

# labeled_dimensions[i][j] = j-th dimension relevant to i-th cluster.
labeled_dimensions = [[1, 3, 5], [9, 7, 8], [4, 2, 6]]    

G, selected_dims = initialize(data, 5, labeled_objects, labeled_dimensions)
t_tot = time.time() - t0

res = {'G': G, 'selected_dims': selected_dims, 't_tot': t_tot}
with open('results.json', 'w') as outfile:
    json.dump(res, outfile)
