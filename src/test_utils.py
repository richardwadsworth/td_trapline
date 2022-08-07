from utils import map_index_to_coord, map_coord_to_index
import unittest

class Test_Map(unittest.TestCase):

    def mutate(self, x, y):
        return [x, y]

    def test_map_index_to_coord(self):

        self.assertEqual(map_index_to_coord(4, 0), self.mutate(0,0))
        self.assertEqual(map_index_to_coord(4, 4), self.mutate(0,1))
        self.assertEqual(map_index_to_coord(4, 3), self.mutate(3,0))
        self.assertEqual(map_index_to_coord(4, 7), self.mutate(3,1))
        self.assertEqual(map_index_to_coord(4, 5), self.mutate(1,1))
        self.assertEqual(map_index_to_coord(4, 15), self.mutate(3,3))
        self.assertEqual(map_index_to_coord(8, 63), self.mutate(7,7))


    def test_map_coord_to_index(self):

        self.assertEqual(map_coord_to_index(4, 0, 0), 0),
        self.assertEqual(map_coord_to_index(4, 0, 1), 4),
        self.assertEqual(map_coord_to_index(4, 3, 0), 3),
        self.assertEqual(map_coord_to_index(4, 3, 1), 7), 
        self.assertEqual(map_coord_to_index(4, 1, 1), 5),
        self.assertEqual(map_coord_to_index(4, 3, 3), 15),
        self.assertEqual(map_coord_to_index(8, 7, 7), 63)

unittest.main('test_utils', argv=[''], verbosity=2, exit=False)
