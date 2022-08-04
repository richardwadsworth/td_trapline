from math import ceil
from utils import map_index_to_coord
import unittest

class Test_Map(unittest.TestCase):

    def mutate(self, x, y):
        return [x, y]

    def test(self):

        self.assertEqual(map_index_to_coord(4, 0), self.mutate(0,0))
        self.assertEqual(map_index_to_coord(4, 4), self.mutate(0,1))
        self.assertEqual(map_index_to_coord(4, 5), self.mutate(1,1))
        self.assertEqual(map_index_to_coord(4, 15), self.mutate(3,3))
        self.assertEqual(map_index_to_coord(8, 63), self.mutate(7,7))


unittest.main('utils', argv=[''], verbosity=2, exit=False)
