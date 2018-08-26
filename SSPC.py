import numpy as np
from utils import calc_stats_i, hill_climb, get_peak, max_min_dist


class SSPC(object):
    """
    Semi-Supervised Projected Clustering algorithm.
    """

    def __init__(self, k=6, m=0.5, building_dim_num=3):
        """
        :param k: number of clusters
        :param m: threshold coefficient, m varies between (0,1)
        :param building_dim_num: number of grid-building dimensions
        """
        self.k = k
        self.m = m
        self.building_dim_num = building_dim_num
        self.data = None
        self.labeled_objects = None
        self.labeled_dimensions = None

    def fit(self, data, labeled_objects=None, labeled_dimensions=None):
        self.data = data
        self.labeled_objects = labeled_objects
        self.labeled_dimensions = labeled_dimensions

    @property
    def selection_threshold(self):
        data = self.data
        V = data.shape[1]
        m = self.m
        selection_threshold = [m * np.var(data[:, j], ddof=1) for j in range(V)]
        return selection_threshold

    @staticmethod
    def select_dim(data_i, selection_threshold, mu_hat_i=None):
        """
        Select relevant dimensions for clusters, which must obey:
        s[i][j] ** 2 + (mu[i][j] - mu_tilde[i][j]) ** 2 < s_hat[i][j] ** 2
        """
        V = data_i.shape[1]

        # Return a list of selected dimensions.
        selected_dims = []

        # Calculate relevant statistics.
        mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i)

        # Use medoids if given.
        if mu_hat_i is not None:
            mu_tilde_i = mu_hat_i

        # Loop to find relevant dimensions.
        for j in range(V):
            if sample_var_i[j] ** 2 + (mu_i[j] - mu_tilde_i[j]) ** 2 < selection_threshold[j] ** 2:
                selected_dims.append(j)
        return selected_dims

    def private_seeds_for_labeled_objects(self, labeled_objects_i, labeled_dimensions_i=None):
        data = self.data
        selection_threshold = self.selection_threshold
        building_dim_num = self.building_dim_num

        # Extract i-th cluster's labeled objects.
        data_i = data[labeled_objects_i]
        med_i = np.median(data_i, axis=0)

        # Define a candidate set that includes the dimensions selected by SelectDim as well as 
        # those in labeled_dimensions (if any).
        candidate_set = SSPC.select_dim(data_i, selection_threshold, med_i)
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

    def initialize(self):
        data = self.data
        labeled_objects = self.labeled_objects
        labeled_dimensions = self.labeled_dimensions
        selection_threshold = self.selection_threshold
        sd_data = np.std(data, axis=0)
        k = self.k
        building_dim_num = self.building_dim_num

        # Private seeds.
        G = {}

        # Selected dimensions.
        selected_dims = {}

        # Loop over clusters with both labeled objects and labeled dimensions.
        for i in range(k):
            if ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                G[i] = self.private_seeds_for_labeled_objects(labeled_objects[i], labeled_dimensions[i])

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[G[i]], selection_threshold)

        # Loop over clusters with only labeled objects.
        for i in range(k):
            if ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and not (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                G[i] = self.private_seeds_for_labeled_objects(labeled_objects[i])

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[G[i]], selection_threshold)

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
                selected_dims[i] = SSPC.select_dim(data[G[i]], selection_threshold)

        # Loop over clusters with no input.
        for i in range(k):
            if not ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                # Pick the seed whose minimum distance to all the seeds already picked
                # by other seed groups is maximum, as the median.
                med_i = data[max_min_dist(data, sd_data, G, selected_dims)]
                G[i] = hill_climb(data, med_i, sd_data)

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[G[i]], selection_threshold)
        return G, selected_dims

    @staticmethod
    def replace_cluster_rep(reps, phi_i, G):
        """
        Find the worst performing cluster and replace its representative.
        :param reps: current representatives used for each cluster
        :param phi_i: objective function score for cluster i
        :param G: private seed groups
        :return: worst performing cluster and the new representative for it
        """
        # worst-performing cluster
        wp_cluster = phi_i.index(min(phi_i))

        # Find private seeds which are not used as other cluster reps.
        available_reps = []
        for rep in G[wp_cluster]:
            if rep not in reps:
                available_reps.append(rep)

        # Randomly pick a seed as the new rep.
        new_rep = np.random.choice(available_reps)
        return {'worst_performing_cluster': wp_cluster,
                'new_rep': new_rep}
