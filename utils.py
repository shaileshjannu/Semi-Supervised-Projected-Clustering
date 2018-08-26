import numpy as np
import pandas
from sklearn import preprocessing


k = 6    # Number of clusters
m = 0.5  # Threshold coefficient, m varies between (0,1)
building_dim_num = 3


def preprocess(input_path):
    # Read data and store it in a dataframe.
    df = pandas.read_csv(input_path)

    # Convert string data columns to categorical.
    obj_columns = df.select_dtypes(['object']).columns
    
    # Ignore categorical values for now.
    df = df.drop(df[obj_columns], axis=1)
    
    # TODO: Convert categorical data to a reasonable form.
    # df[obj_columns] = df[obj_columns].astype('category').apply(lambda x: x.cat.codes)
    
    # TODO: Think about scaling.
    # data = preprocessing.scale(np.array(df))
    data = np.array(df)
    return data


def calc_selection_threshold(data):
    V = data.shape[1]
    selection_threshold = [m * np.var(data[:, j], ddof=1) for j in range(V)]
    return selection_threshold


def calc_stats_i(data_i):
    V = data_i.shape[1]
    mu_i = []
    mu_tilde_i = []
    sample_var_i = []

    # Loop over dimensions.
    for j in range(V):
        data_ij = data_i[:, j]
        mu_ij = np.mean(data_ij)
        mu_tilde_ij = np.median(data_ij)
        sample_var_ij = np.var(data_ij, ddof=1)

        mu_i.append(mu_ij)
        mu_tilde_i.append(mu_tilde_ij)
        sample_var_i.append(sample_var_ij)
    return mu_i, mu_tilde_i, sample_var_i


def calc_stats(data):
    C, V = data.shape
    
    # Calculate sample median, sample mean, sample variance.
    mu = []
    mu_tilde = []
    sample_var = []

    # Loop over clusters.
    for i in C:
        data_i = data[i]
        mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i)
        mu.append(mu_i)
        mu_tilde.append(mu_tilde_i)
        sample_var.append(sample_var_i)

    return {'mu': mu,
            'mu_tilde': mu_tilde,
            'sample_var': sample_var}


def select_dim(data_i, selection_threshold, mu_hat_i=None):
    """
    Select relevant dimensions for clusters, which must obey:
    s[i][j] ** 2 + (mu[i][j] - mu_tilde[i][j]) ** 2 < s_hat[i][j] ** 2
    """
    V = data_i.shape[1]
    
    # Return a list of 0/1 values to show whether dim_j is selected.
    selected_dims = []

    # Calculate relevant statistics.
    mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i)

    # Use medoids if given.
    if mu_hat_i is not None:
        mu_tilde_i = mu_hat_i

    # Loop over clusters to find relevant dimensions.
    for j in range(V):
        if sample_var_i[j] ** 2 + (mu_i[j] - mu_tilde_i[j]) ** 2 < selection_threshold[j] ** 2:
            selected_dims.append(j)
    return selected_dims


def bin_cell(data, edges):
    """
    Bin samples in data into the cell defined by edges.
    """
    res = []
    for i in range(len(data)):
        sample = data[i]
        
        # Check whether sample lies in the cell.
        in_cell = True
        for j in range(len(sample)):
            if sample[j] < edges[j][0] or sample[j] > edges[j][1]:
                in_cell = False
                break
        
        # Add it to the results if so.
        if in_cell:
            res.append(i)
    return res


def define_edges(centre, edge_lengths):
    """
    Define the cell edges from a centre and the edge lengths.
    """
    edges = [(centre[i] - edge_lengths[i] / 2, centre[i] + edge_lengths[i] / 2) for i in range(len(edge_lengths))]
    return edges


def hill_climb(data, curr_centre, step_lengths):
    """
    Hill-climbing to find the cell with highest density. 
    """
    # Find the central cell count.
    curr_edges = define_edges(curr_centre, step_lengths)
    curr_bin = bin_cell(data, curr_edges)
    
    # Find the denser cell than current centre.
    denser_found = False
    max_centre = curr_centre
    max_bin = curr_bin
    
    # Explore the neighbouring cells.
    for i in range(len(step_lengths)):
        for sign in [-1, 1]:
            
            # Move to the neighbouring centre.
            step_length = step_lengths[i]
            cell_centre = curr_centre.copy()
            cell_centre[i] += sign * step_length

            # Bin the neighbouring cell.
            cell_edges = define_edges(cell_centre, step_lengths)
            cell_bin = bin_cell(data, cell_edges)

            # Find the most dense cell.
            if len(cell_bin) > len(max_bin):
                max_centre = cell_centre
                max_bin = cell_bin
                denser_found = True
    
    # Found the maximum.
    if not denser_found:
        return max_bin
    else:
        return hill_climb(data, max_centre, step_lengths)
    

def get_peak(data, step_lengths):
    """
    Find the absolute density peak of the grid.
    """
    
    # Bin the data into cells.
    vals_range = np.ptp(data, axis=0)
    bin_nums = [vals_range[i] / step_lengths[i] for i in range(len(step_lengths))]
    H, edges = np.histogramdd(data, bin_nums)
    
    # Find absolute density peak.
    max_indices = np.unravel_index(H.argmax(), H.shape)
    max_edges = [(edges[i][max_indices[i]], edges[i][max_indices[i] + 1]) for i in range(len(edges))]
    
    # Find data points in the peak cell.
    return bin_cell(data, max_edges)


def max_min_dist(data, sd_data, G, selected_dims):
    """
    Sort the ungrouped data points according to min-dist to grouped points.
    """
    # Find all ungrouped points.
    ungrouped_points = np.arange(len(data))
    ungrouped_points = np.delete(ungrouped_points, np.concatenate((G.values())))
    up_dist = {}
    
    # Loop over private seed groups.
    for i, G_i in G.iteritems():
        
        # Loop over private seeds.
        for j in G_i:
            
            # Loop over ungrouped points.
            for up in ungrouped_points:
                
                # Calculate the standardized distance.
                sd_dist = 0
                for dim in selected_dims[i]:
                    sd_dist += ((data[up][dim] - data[j][dim]) / sd_data[dim]) ** 2
                sd_dist = sd_dist ** 0.5
                
                # Compare it with current minimum.
                if up not in up_dist or up_dist[up] > sd_dist:
                    up_dist[up] = sd_dist
    return max(up_dist.iterkeys(), key=(lambda key: up_dist[key]))


def private_seeds_for_labeled_objects(data, labeled_objects_i, labeled_dimensions_i=None):
    selection_threshold = calc_selection_threshold(data)
    
    # Extract i-th cluster's labeled objects.
    data_i = data[labeled_objects_i]
    med_i = np.median(data_i, axis=0)

    # Define a candidate set that includes the dimensions selected by SelectDim as well as 
    # those in labeled_dimensions (if any).
    candidate_set = select_dim(data_i, selection_threshold, med_i)
    if labeled_dimensions_i: 
        candidate_set = list(set(np.concatenate((candidate_set, labeled_dimensions_i))))

    # TODO: Ask Vic to fix score function.
    # score_ij = score_function_i(i, candidate_set, med_i)
    score_ij = np.random.rand(len(candidate_set))

    # Each dimension in the set has a probability proportional to 
    # the score function of being selected as a building dimension of a grid.
    score_ij /= sum(score_ij)
    building_dims = np.random.choice(a=candidate_set, size=building_dim_num, replace=False, p=score_ij)

    # Extract the data with building dimensions.
    data_grid = data[:, building_dims]
    sd_grid = np.std(data_grid, axis=0)
    med_i_grid = med_i[building_dims]

    # Apply hill-climbing search to find most dense cell.
    G_i = hill_climb(data_grid, med_i_grid, sd_grid)
    return G_i


def initialize(data, k, labeled_objects, labeled_dimensions):
    selection_threshold = calc_selection_threshold(data)
    sd_data = np.std(data, axis=0)
    
    # i-th sample belongs to seed_group_label[i] seed group.
    seed_group_label = np.zeros(len(data))
    
    # Denote the current distance to the closest median.
    med_dist = np.zeros(len(data))
    
    # Private seeds.
    G = {}
    
    # Selected dimensions.
    selected_dims = {}
    
    # Loop over clusters with both labeled objects and labeled dimensions.
    for i in range(k):
        if ((i < len(labeled_objects) and len(labeled_objects[i]) > 0) 
            and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
            G[i] = private_seeds_for_labeled_objects(data, labeled_objects[i], labeled_dimensions[i])
            
            # Find the relevant dimensions for this cluster.
            selected_dims[i] = select_dim(data[G[i]], selection_threshold)
    
    # Loop over clusters with only labeled objects.
    for i in range(k):
        if ((i < len(labeled_objects) and len(labeled_objects[i]) > 0) 
            and not ((i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0))):
            G[i] = private_seeds_for_labeled_objects(data, labeled_objects[i])
            
            # Find the relevant dimensions for this cluster.
            selected_dims[i] = select_dim(data[G[i]], selection_threshold)
        
    # Loop over clusters with only labeled dimensions.
    for i in range(k):
        if ((not (i < len(labeled_objects) and len(labeled_objects[i]) > 0)) 
            and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
            candidate_set = labeled_dimensions[i]
            building_dims = np.random.choice(a=candidate_set, size=building_dim_num, replace=False)

            # Extract the data with building dimensions.
            data_grid = data[:, building_dims]
            sd_grid = np.std(data_grid, axis=0)

            # Find absolute peak.
            G[i] = get_peak(data_grid, sd_grid)
            
            # Find the relevant dimensions for this cluster.
            selected_dims[i] = select_dim(data[G[i]], selection_threshold)
    
    # Loop over clusters with no input.
    for i in range(k):
        if not ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
            and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
            
            # Pick the seed whose minimum distance to all the seeds already picked 
            # by other seed groups is maximum, as the median.
            med_i = data[max_min_dist(data, sd_data, G, selected_dims)]
            G[i] = hill_climb(data, med_i, sd_data)
            
            # Find the relevant dimensions for this cluster.
            selected_dims[i] = select_dim(data[G[i]], selection_threshold)
    return G, selected_dims


def column(matrix, i):
    return [row[i] for row in matrix]


def score_function_ij(i, data_j, medoids=None):
    ni = len(data_j)
    miu = np.mean(data_j)
    sample_var = np.var(data_j, ddof=1)
    selection_threshold = m * sample_var
    
    if medoids:
        miu_tilda = data_j[medoids[i]]  # Assuming the medoids are in list format
    else:
        # Assuming correct input to the function where the lengths of two features are the same
        miu_tilda = np.median(data_j)

    phi_ij = (ni - 1) * (1 - (sample_var + (miu - miu_tilda)**2)/selection_threshold)
    return phi_ij


def score_function_i(i, selected_dimension, medoids=None):
    
    phi_ij = []
    
    for j in selected_dimension:
        phi_ij.append(score_function_ij(i, column(data,j), medoids))
    
    # TODO: phi_ij can be re-used maybe?
    # phi_i = sum(phi_ij)
    return phi_ij


def score_function_all(data):
    
    phi_i = []
    n = len(data)
    d = len(data[0])
    
    for k in clusters:
        phi_i.append(score_function_i_updated(k,data[k]))
        
    phi = sum(phi_i)/(n*d)
    return phi