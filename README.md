# Semi-Supervised-Projected-Clustering
Semi-Supervised Projected Clustering according to 'On Discovery of Extremely Low-Dimensional Clusters using Semi-Supervised Projected Clustering' by Kevin Y. Yip et. al.

Dataset used: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

# Overview
SSPC algorithm aims to apply projected clustering with semi-supervised heuristics onto high-dimensional clustering data, to improve on current unsupervised clustering algorithms.

We will briefly discuss the algorithm as suggested in the paper and compare it with K-Means.

## Parital labels
The term semi-supervised refers to the case where we have partial labels for our datasets. SSPC use the partial labels to create 'seed groups' for each cluster, and draw medoids from these seed groups to act as the centroid as in K-Means.

## Projected clustering
For high-dimensional data clustering, it is often the case where only a subset of the features (dimensions) are relevant for a specific cluster, so we can create a mechanism (select_dim) to find those relevant features for a cluster, according to some metric (details in the paper).

Similarly, we can incorporate this idea into the objective function so that when looking for the 'optimal' clustering, we only consider the dimensions which are most relevant to a cluster.
