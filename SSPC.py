import numpy as np
import random
import pandas as pd
from utils import calc_stats_i, hill_climb, get_peak, max_min_dist


class SSPC(object):
    """
    Semi-Supervised Projected Clustering algorithm.
    """

    def __init__(self, k, m=0.5, building_dim_num=3):
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
        self.seed_groups = None
        self.selected_dims = None
        self.medoid_bank = None
        self.reps = None
        self.clusters = None

    def fit(self, data, labeled_objects=None, labeled_dimensions=None):
        self.data = data
        self.labeled_objects = labeled_objects
        self.labeled_dimensions = labeled_dimensions
        self.initialize()
        self.draw_medoids()

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

        # Make sure the candidate set is big enough for grid building.
        if len(candidate_set) < building_dim_num:
            candidate_set = np.concatenate((
                candidate_set,
                np.random.choice(
                    a=np.delete(range(data.shape[1]), candidate_set),
                    size=building_dim_num - len(candidate_set),
                    replace=False
                )))

        # Each dimension in the set has a probability proportional to
        # the score function of being selected as a building dimension of a grid.
        score_ij = self.score_function_i(data_i, candidate_set, med_i)

        # Standardize for probability distribution.
        if min(score_ij) < 0:
            score_ij -= 2 * min(score_ij)
        score_ij /= sum(score_ij)
        building_dims = np.random.choice(a=candidate_set, size=building_dim_num, replace=False, p=score_ij)

        # Extract the data with building dimensions.
        data_grid = data[:, building_dims]
        sd_grid = np.std(data_grid, axis=0)
        med_i_grid = med_i[building_dims]

        # Apply hill-climbing search to find most dense cell.
        seed_group_i = hill_climb(data_grid, med_i_grid, sd_grid)
        return seed_group_i

    def initialize(self):
        data = self.data
        labeled_objects = self.labeled_objects
        labeled_dimensions = self.labeled_dimensions
        selection_threshold = self.selection_threshold
        sd_data = np.std(data, axis=0)
        k = self.k
        building_dim_num = self.building_dim_num

        # Private seeds.
        seed_groups = {}

        # Selected dimensions.
        selected_dims = {}

        # Loop over clusters with both labeled objects and labeled dimensions.
        for i in range(k):
            if ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                seed_groups[i] = self.private_seeds_for_labeled_objects(labeled_objects[i], labeled_dimensions[i])

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)

        # Loop over clusters with only labeled objects.
        for i in range(k):
            if ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and not (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                seed_groups[i] = self.private_seeds_for_labeled_objects(labeled_objects[i])

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)

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
                seed_groups[i] = get_peak(data_grid, sd_grid)

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)

        # Loop over clusters with no input.
        for i in range(k):
            if not ((i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                # Pick the seed whose minimum distance to all the seeds already picked
                # by other seed groups is maximum, as the median.
                med_i = data[max_min_dist(data, sd_data, seed_groups, selected_dims)]
                seed_groups[i] = hill_climb(data, med_i, sd_data)

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)
        self.seed_groups = seed_groups
        self.selected_dims = selected_dims

    def draw_medoids(self):
        """
        Randomly initialize a list of medoids for each seed_groups without repetition.
        """
        seed_groups = self.seed_groups
        medoids = {}
        for k in range(len(seed_groups)):
            number_to_choose = len(seed_groups[k])
            medoids.update({k: random.sample(seed_groups[k], number_to_choose)})
        self.medoid_bank = medoids

    def replace_cluster_rep(self, phi_i):
        """
        Find the worst performing cluster and replace its representative.
        :param phi_i: objective function score for cluster i
        :return: worst performing cluster and the new representative for it
        """
        # current representatives used for each cluster
        reps = self.reps
        medoids = self.medoid_bank

        # worst-performing cluster
        wp_cluster = phi_i.index(min(phi_i))

        # Find potential medoid which is not used as another cluster's rep.
        for i in range(len(medoids[wp_cluster])):
            new_medoid = medoids[wp_cluster][i]
            if new_medoid not in reps:
                reps[wp_cluster] = new_medoid
                medoids[wp_cluster].pop(i)
                return {'worst_performing_cluster': wp_cluster,
                        'new_rep': new_medoid}

    def score_function_ij(self, data_ij, medoid_used_j=None):
        """
        The score function of the cluster i in j-th dimension: input - data in i-th group of specific dimension.
        Reminds that the miu used in first iteration is the medoid; while in later iterations it became the median.
        m should be within range(0,1)
        """
        m = self.m
        ni = len(data_ij)
        miu = np.mean(data_ij)
        sample_var = np.var(data_ij, ddof=1)
        selection_threshold = m * sample_var

        # Check whether medoids are given.
        if medoid_used_j:
            miu_tilde = medoid_used_j
        else:
            miu_tilde = np.median(data_ij)

        # Calculate the score for i-th cluster, j-th dimension.
        if np.isclose(selection_threshold, 0):
            phi_ij = ni - 1
        else:
            phi_ij = (ni - 1) * (1 - (sample_var + (miu - miu_tilde) ** 2) / selection_threshold)
        return phi_ij

    def score_function_i(self, data_i, selected_dims, medoid_used=None):
        """
        Calculate the score component of each cluster: sum of score component phi_ij over set of selected dimensions.
        :param data_i: cluster i
        :param selected_dims: selected dimensions for this cluster
        :param medoid_used: medoid of the labelled objects clusters.
        """
        phi_ij = []

        # Calculate the score of each dimension in data_k
        for j in selected_dims:

            # Check whether medoid is given.
            if medoid_used is None:
                medoid_used_j = None
            else:
                medoid_used_j = medoid_used[j]
            phi_ij.append(self.score_function_ij(data_i[:, j], medoid_used_j))

        # Sum over selected dimensions
        # phi_i = sum(phi_ij)
        return phi_ij

    def score_function_all(self, clusters, selected_dims, medoids_used=None):
        data = self.data
        phi_i = []
        for i in range(len(clusters)):
            cluster = clusters[i]

            # Check whether medoids are given.
            if medoids_used is None:
                medoid_used = None
            else:
                medoid_used = medoids_used[i]
            phi_i.append(sum(self.score_function_i(data[cluster], selected_dims, medoid_used)))

        phi = sum(phi_i) / (data.shape[0] * data.shape[1])
        return phi

    def assign_max(self):
        """
        Assign every object in the dataset to the cluster
        (or outlier list) that gives the greatest improvement to the objective score.
        """
        data = self.data
        k = self.k
        selected_dimensions = self.selected_dims

        # the medoids list selected for each cluster
        medoids = self.medoid_bank

        # a list of assigned clusters for each individual data
        cluster_assigned = []

        # a list of relevant phi_i scores for assigned clusters
        phi_i_scores = np.zeros(k)
        clusters = {}

        # Initialize clusters assigned.
        for i in range(k):
            clusters.update({i: []})
            medoids_used = medoids[i].pop()
            clusters[i].append(medoids_used)

        # list for currently used medoids
        lst = [item[0] for item in clusters.values()]
        self.reps = lst

        # For each new data input, calculate the new score_i for every cluster and assign the maximum
        for n_i in range(len(data)):
            new_dat_point = data[n_i]
            phi_i = []

            for i in range(k):
                cluster = clusters[i]

                # current available data in i-th cluster
                data_i = data[cluster]

                # Calculate the score_ij.
                phi_ij = []
                for j in selected_dimensions[i]:

                    # j-th dimension of data in i-th cluster
                    data_ij = data_i[j]
                    data_ij.append(new_dat_point[j])
                    data_ij = pd.DataFrame(data_ij)
                    phi_ij.append(self.score_function_ij(data_ij))
                phi_i.append(sum(phi_ij))

            # Handle medoids separately.
            if n_i in lst:
                cluster_assigned.append(lst.index(n_i))
            else:
                cluster_assigned.append(np.argmax(phi_i))
                clusters[cluster_assigned[n_i]].append(n_i)
            phi_i_scores[cluster_assigned[n_i]] = max(phi_i)
        return cluster_assigned, phi_i_scores
