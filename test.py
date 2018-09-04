import unittest
import numpy as np
from SSPC import SSPC


class TestWithMockData(unittest.TestCase):
    """
    Mock up data to test individual functions in SSPC.
    """
    def setUp(self):
        """
        Set up mock data.
        """
        self.sspc = SSPC(k=3)
        self.data = [[11,  1.5,  1.0,  900],    # cluster 0
                     [10,  -1.5, 1.2,  902],    # cluster 0
                     [-10, -1.1, 2.3,  -1002],  # cluster 1
                     [-9,  -0.9, 0.1,  -987],   # cluster 1
                     [9,   2.3,  0.99, 899],    # cluster 0
                     [0,   0,    0,    789],    # cluster 2
                     [1,   0.2,  -0.1, 456],    # cluster 2
                     [-9,  -1.0, -2.3, -999],   # cluster 1
                     [0,   -0.1, 0.1,  -345]]   # cluster 2
        self.labeled_objects = [[0], [2], [5]]
        self.labeled_dimensions = [[0], [0], [0]]
        self.clusters = [[0, 1, 4], [2, 3, 7], [5, 6, 8]]
        self.selected_dims = [[0, 2, 3], [0, 1, 3], [0, 1, 2]]

    def test_initialize(self):
        """
        Make sure the seed groups are reasonable.
        """
        sspc = self.sspc
        sspc.data = np.asarray(self.data)
        sspc.labeled_objects = self.labeled_objects
        sspc.labeled_dimensions = self.labeled_dimensions
        sspc.initialize()
        for seed_group_i in range(len(sspc.seed_groups)):
            seed_group = sspc.seed_groups[seed_group_i]
            labeled_object = sspc.labeled_objects[seed_group_i]
            for lo in labeled_object:
                self.assertTrue(lo in seed_group)

    def test_fit_and_predict(self):
        res = self.sspc.fit_and_predict(self.data,
                                        labeled_objects=self.labeled_objects,
                                        labeled_dimensions=self.labeled_dimensions)
        clusters, selected_dims = res['clusters'], res['selected dimensions']
        print(clusters)
        print(selected_dims)
        # Test clustering and selected dimensions.
        for i in range(len(clusters)):
            self.assertEqual(clusters[i], self.clusters[i])
            self.assertEqual(selected_dims[i], self.selected_dims[i])


if __name__ == '__main__':
    unittest.main()
