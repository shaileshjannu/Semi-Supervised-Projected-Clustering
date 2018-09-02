import numpy as np


def calc_stats_i(data_i):
    V = data_i.shape[-1]
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


def calc_stats(data, clusters):

    # Calculate sample median, sample mean, sample variance.
    mu = []
    mu_tilde = []
    sample_var = []

    # Loop over clusters.
    for i in clusters:
        data_i = data[i]
        mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i)
        mu.append(mu_i)
        mu_tilde.append(mu_tilde_i)
        sample_var.append(sample_var_i)

    return {'mu': mu,
            'mu_tilde': mu_tilde,
            'sample_var': sample_var}


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


def hill_climb(data, curr_centre, edge_lengths, step_lengths):
    """
    Hill-climbing to find the cell with highest density. 
    """
    # Find the central cell count.
    curr_edges = define_edges(curr_centre, edge_lengths)
    curr_bin = bin_cell(data, curr_edges)
    
    # Find the denser cell than current centre.
    denser_found = False
    max_centre = curr_centre
    max_bin = curr_bin
    
    # Explore the neighbouring cells.
    for i in range(len(step_lengths)):
        for sign in range(-5, 6, 1):
            
            # Move to the neighbouring centre.
            step_length = step_lengths[i]
            cell_centre = curr_centre.copy()
            cell_centre[i] += sign * step_length

            # Bin the neighbouring cell.
            cell_edges = define_edges(cell_centre, edge_lengths)
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
        return hill_climb(data, max_centre, edge_lengths, step_lengths)
    

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
    if G is None or len(G) == 0:
        return np.random.randint(0, len(data))

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
    return max(up_dist.keys(), key=(lambda key: up_dist[key]))
