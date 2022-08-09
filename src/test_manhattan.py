import unittest
from manhattan import get_manhattan_distance

class Test_Map(unittest.TestCase):

    def test_manhattan_distance_different(self):

        size =4
        route_1_index = [0,1,5,6]
        route_2_index = [0,4,8,9,5,6]

        distance = get_manhattan_distance(size, route_1_index, route_2_index)
        
        self.assertEqual(distance, 8)

    def test_manhattan_distance_same(self):

        size =4
        route_1_index = [0,4,8,9,5,6]
        route_2_index = [0,4,8,9,5,6]

        distance = get_manhattan_distance(size, route_1_index, route_2_index)
        
        self.assertEqual(distance, 0)


unittest.main('test_manhattan', argv=[''], verbosity=2, exit=False)