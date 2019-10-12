import unittest

import rechelper.movielens

class MovieLensTestSuite(unittest.TestCase):
    """Note: Requires data to be set up correctly"""

    def test_load_ml100k(self):
      ratings,movies = rechelper.movielens.load_movielens("ml-100k")
      self.assertEqual(ratings.shape, (100000, 4))
      self.assertEqual(movies.shape, (1682, 5))

if __name__ == '__main__':
    unittest.main()