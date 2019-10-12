import unittest
import numpy as np

import rechelper.movielens
import rechelper.dataset

class MovieLensTestSuite(unittest.TestCase):
    """Note: Requires data to be set up correctly"""

    def test_load_ml100k(self):
      ratings_all, movies_all = rechelper.movielens.load_movielens("ml-100k")
      self.assertEqual(ratings_all.shape, (100000,4))
      self.assertEqual(movies_all.shape, (1682,5))

      ds = rechelper.dataset.create(ratings_all)
      self.assertEqual(ds.ratings.shape, (100000,3))
      self.assertEqual(ds.unique_users, 943)
      self.assertEqual(ds.unique_items, 1682)

      self.assertEqual(ds.user_idx_map[0], 1)
      self.assertEqual(ds.user_idx_map[1], 2)
      self.assertEqual(ds.user_idx_map[942], 943)

      self.assertEqual(ds.item_idx_map[0], 1)
      self.assertEqual(ds.item_idx_map[1], 2)
      self.assertEqual(ds.item_idx_map[1681], 1682)      

    def test_load_ml_latest_small(self):
      ratings_all, movies_all = rechelper.movielens.load_movielens("ml-latest-small")
      self.assertEqual(ratings_all.shape, (100836,4))
      self.assertEqual(movies_all.shape, (9742,5))

      ds = rechelper.dataset.create(ratings_all)
      self.assertEqual(ds.ratings.shape, (100836,3))
      self.assertEqual(ds.unique_users, 610)
      self.assertEqual(ds.unique_items, 9724)

      self.assertEqual(ds.user_idx_map[0], 1)
      self.assertEqual(ds.user_idx_map[1], 2)
      self.assertEqual(ds.user_idx_map[609], 610)

      self.assertEqual(ds.item_idx_map[0], 1)
      self.assertEqual(ds.item_idx_map[1], 2)
      self.assertEqual(ds.item_idx_map[9723], 193609)

if __name__ == '__main__':
    unittest.main()