import numpy as np
import random
from utils import calc_stats_i, hill_climb, get_peak, max_min_dist


class SSPC(object):
    """
    Semi-Supervised Projected Clustering algorithm.
    """

    def __init__(self, k, m=0.5, building_dim_num=3, max_drop_len=3, grid_size=1.0, climb_step_size=0.1):
        """
        :param k: number of clusters
        :param m: threshold coefficient, m varies between (0,1)
        :param building_dim_num: number of grid-building dimensions
        :param max_drop_len: maximal tolerable length of continuously decreasing sequence
        :param grid_size: ratio of the hill climb grid size to std
        :param climb_step_size: ratio of the hill climb step size to std
        """
        self.k = k
        self.m = m
        self.building_dim_num = building_dim_num
        self.max_drop_len = max_drop_len
        self.grid_size = grid_size
        self.climb_step_size = climb_step_size
        self.data = None
        self.labeled_objects = None
        self.labeled_dimensions = None
        self.seed_groups = None
        self.selected_dims = None
        self.medoid_bank = None
        self.medoids_used = None
        self.reps = None
        self.clusters = None

    def fit_and_predict(self, data, labeled_objects=None, labeled_dimensions=None):
        self.data = np.asarray(data)
        data = self.data
        self.labeled_objects = labeled_objects
        self.labeled_dimensions = labeled_dimensions
        print("selection threshold: ", self.selection_threshold)
        self._initialize()
        self._draw_medoids()
        print("reps: ", self.reps)
        print("medoids: ", self.medoids_used)
        print("seed groups: ", self.medoid_bank)
        print("selected dimension:", self.selected_dims)

        # best score so far
        best_phi = None

        # length of consecutive decrease
        drop_length = 0
        while drop_length < self.max_drop_len:

            # Run cluster assignment.
            clusters = self._assign_max()

            # Find selected dimensions for each cluster.
            selected_dims = []
            for i in clusters:
                selected_dims.append(self.select_dim(data[i], self.selection_threshold))

            # Calculate current score.
            phi_i = self._score_function_all(clusters, selected_dims, self.reps)
            phi = sum(phi_i) / (data.shape[0] * data.shape[1])

            # Update best score and best clustering.
            if best_phi is None or phi > best_phi:
                best_phi = phi
                self.clusters = clusters
                self.selected_dims = selected_dims
                drop_length = 0
                print("clusters: ", clusters)
                print("selected dimensions: ", selected_dims)
                print("phi i: ", phi_i)
            else:
                drop_length += 1

            # Replace rep (medoid) of the worst cluster.
            self._replace_cluster_rep(list(phi_i))
        return {'clusters': self.clusters,
                'selected dimensions': self.selected_dims}

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
        V = data_i.shape[-1]

        # Return a list of selected dimensions.
        selected_dims = []

        # Calculate relevant statistics.
        mu_i, mu_tilde_i, sample_var_i = calc_stats_i(data_i)

        # Use medoids if given.
        if mu_hat_i is not None:
            mu_tilde_i = mu_hat_i

        # Loop to find relevant dimensions.
        for j in range(V):
            if sample_var_i[j] + (mu_i[j] - mu_tilde_i[j]) ** 2 < selection_threshold[j]:
                selected_dims.append(j)
        return selected_dims

    def _private_seeds_for_labeled_objects(self, labeled_objects_i, labeled_dimensions_i=None):
        data = self.data
        selection_threshold = self.selection_threshold
        building_dim_num = self.building_dim_num
        grid_size = self.grid_size
        climb_step_size = self.climb_step_size

        # Extract i-th cluster's labeled objects.
        data_i = data[labeled_objects_i]
        med_i = np.median(data_i, axis=0)

        # Define a candidate set that includes the dimensions selected by SelectDim as well as 
        # those in labeled_dimensions (if any).
        candidate_set = SSPC.select_dim(data_i, selection_threshold, med_i)
        if labeled_dimensions_i:
            candidate_set = list(set(candidate_set + labeled_dimensions_i))

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
        score_ij = self._score_function_i(data_i, candidate_set, med_i)

        # Standardize for probability distribution.
        if max(score_ij) == min(score_ij):
            score_ij = np.array([1.0] * len(score_ij))
        else:
            score_ij = np.array([(score_ij[i] - min(score_ij)) / (max(score_ij) - min(score_ij))
                                 for i in range(len(score_ij))])
        score_ij /= sum(score_ij)
        building_dims = np.random.choice(a=candidate_set, size=building_dim_num, replace=False, p=score_ij)

        # Extract the data with building dimensions.
        data_grid = data[:, building_dims]
        sd_grid = np.std(data_grid, axis=0)
        med_i_grid = med_i[building_dims]

        # Apply hill-climbing search to find most dense cell.
        seed_group_i = hill_climb(data_grid, med_i_grid, sd_grid * grid_size, sd_grid * climb_step_size)
        return seed_group_i

    def _initialize(self):
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
            if ((labeled_objects and i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (labeled_dimensions and i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
                seed_groups[i] = self._private_seeds_for_labeled_objects(labeled_objects[i], labeled_dimensions[i])

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)

        # Loop over clusters with only labeled objects.
        for i in range(k):
            if ((labeled_objects and i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (labeled_dimensions is None or
                         not (i < len(labeled_dimensions) and
                              len(labeled_dimensions[i]) > 0))):
                seed_groups[i] = self._private_seeds_for_labeled_objects(labeled_objects[i])

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)

        # Loop over clusters with only labeled dimensions.
        for i in range(k):
            if ((labeled_objects is None or not (i < len(labeled_objects) and len(labeled_objects[i]) > 0))
                    and (labeled_dimensions and i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0)):
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
            if (labeled_objects is None or
                    not (i < len(labeled_objects) and len(labeled_objects[i]) > 0)
                    and (labeled_dimensions is None or
                         not (i < len(labeled_dimensions) and len(labeled_dimensions[i]) > 0))):
                # Pick the seed whose minimum distance to all the seeds already picked
                # by other seed groups is maximum, as the median.
                med_i = data[max_min_dist(data, sd_data, seed_groups, selected_dims)]
                seed_groups[i] = hill_climb(data, med_i, sd_data, sd_data)

                # Find the relevant dimensions for this cluster.
                selected_dims[i] = SSPC.select_dim(data[seed_groups[i]], selection_threshold)
        self.seed_groups = seed_groups
        self.selected_dims = selected_dims

    def _draw_medoids(self):
        """
        Randomly _initialize a list of medoids for each seed_groups without repetition.
        """
        reps = []
        seed_groups = self.seed_groups
        medoids = {}
        medoids_used = []
        for k in range(len(seed_groups)):
            number_to_choose = len(seed_groups[k])
            medoids.update({k: random.sample(seed_groups[k], number_to_choose)})
            reps.append(self.data[medoids[k][0]])
            medoids_used.append(medoids[k][0])
        self.medoid_bank = medoids
        self.reps = reps
        self.medoids_used = medoids_used

    def _replace_cluster_rep(self, phi_i):
        """
        Find the worst performing cluster and replace its representative.
        :param phi_i: objective function score for cluster i
        :return: worst performing cluster and the new representative for it
        """
        # current representatives used for each cluster
        reps = self.reps
        medoids = self.medoid_bank
        medoids_used = self.medoids_used
        clusters = self.clusters
        data = self.data

        # worst-performing cluster
        wp_cluster = phi_i.index(min(phi_i))

        # cluster with no points
        empty_clusters = []
        for i in range(len(clusters)):
            if len(clusters[i]) == 0:
                empty_clusters.append(i)

        # clusters to replace with medoids
        med_clusters = list(set([wp_cluster] + empty_clusters))

        # Find potential medoid which is not used as another cluster's rep.
        for med_cluster in med_clusters:
            for i in range(len(medoids[med_cluster])):
                new_medoid = medoids[med_cluster][i]
                if new_medoid not in medoids_used:
                    reps[med_cluster] = data[new_medoid]

                    # Replace the front of the medoid bank by the new medoid.
                    medoids[med_cluster].pop(i)
                    if len(medoids[med_cluster]) == 0:
                        medoids[med_cluster] = [new_medoid]
                    else:
                        medoids[med_cluster][0] = new_medoid
                    self.medoids_used = [new_medoid]
                    break

        # Replace all other cluster reps by current medians.
        for i in range(len(reps)):
            if i != wp_cluster:
                reps[i] = [np.median(data[clusters[i]][:, j]) for j in range(data.shape[1])]
        self.reps = reps

    @staticmethod
    def score_function_ij(data_ij, medoid_used_j, selection_threshold_ij):
        """
        The score function of the cluster i in j-th dimension: input - data in i-th group of specific dimension.
        Reminds that the miu used in first iteration is the medoid; while in later iterations it became the median.
        m should be within range(0,1)
        """
        ni = len(data_ij)
        miu = np.mean(data_ij)
        sample_var = np.var(data_ij, ddof=1)

        # Check whether medoids are given.
        miu_tilde = medoid_used_j

        # Calculate the score for i-th cluster, j-th dimension.
        if ni == 1 or np.isclose(selection_threshold_ij, 0):
            phi_ij = ni - 1
        else:
            phi_ij = (ni - 1) * (1 - (sample_var + (miu - miu_tilde) ** 2) / selection_threshold_ij)
        return phi_ij

    def _score_function_i(self, data_i, selected_dims, medoid_used):
        """
        Calculate the score component of each cluster: sum of score component phi_ij over set of selected dimensions.
        :param data_i: cluster i
        :param selected_dims: selected dimensions for this cluster
        :param medoid_used: medoid of the labelled objects clusters.
        """
        phi_ij = []
        selection_threshold = self.selection_threshold

        # Calculate the score of each dimension in data_k
        for j in selected_dims:
            medoid_used_j = medoid_used[j]
            phi_ij.append(self.score_function_ij(data_i[:, j], medoid_used_j, selection_threshold[j]))
        return phi_ij

    def _score_function_all(self, clusters, selected_dims, medoids_used):
        data = self.data
        phi_i = []
        for i in range(len(clusters)):
            cluster = clusters[i]
            medoid_used = medoids_used[i]
            selected_dim = selected_dims[i]
            phi_i.append(sum(self._score_function_i(data[cluster], selected_dim, medoid_used)))
        return phi_i

    def _assign_max(self):
        """
        Assign every object in the dataset to the cluster
        (or outlier list) that gives the greatest improvement to the objective score.
        """
        data = self.data
        k = self.k
        selected_dimensions = self.selected_dims

        # a list of assigned clusters for each individual data
        cluster_assigned = []

        # a list of relevant phi_i scores for assigned clusters
        clusters = []

        # list for currently used reps
        reps = self.reps

        # Initialize clusters assigned.
        for i in range(k):
            clusters.append([])

        # For each new data input, calculate the new score_i for every cluster and assign the maximum
        for n_i in range(len(data)):
            new_dat_point = data[n_i]
            phi_i = []

            for i in range(k):
                cluster = clusters[i]
                rep = reps[i]

                # current available data in i-th cluster
                data_i = data[cluster].copy()
                data_i = np.append(data_i, [new_dat_point], axis=0)

                # Calculate the score_ij.
                phi_ij = self._score_function_i(data_i, selected_dimensions[i], rep)
                phi_i.append(sum(phi_ij))

            # Add the point to a cluster which maximizes phi_i.
            cluster_assigned.append(phi_i.index(max(phi_i)))
            clusters[cluster_assigned[n_i]].append(n_i)
        return clusters
