import unittest
from manhattan import get_manhattan_distance

class Test_Map(unittest.TestCase):

    def test_manhattan_distance_different_length(self):

        size =4
        route_1_index = [0,1,5,6]
        route_2_index = [0,4,8,9,5,6]

        distance = get_manhattan_distance(size, route_1_index, route_2_index)
        
        self.assertEqual(distance, 8)

    def test_manhattan_distance_same_length(self):

        size =4
        route_1_index = [0,4,8,9,5,6]
        route_2_index = [0,4,8,9,5,6]

        distance = get_manhattan_distance(size, route_1_index, route_2_index)
        
        self.assertEqual(distance, 0)

    def test_manhattan_distance_different_route_same_length(self):

        size =4
        route_1_index = [0,1,5,6,10,11,15]
        route_2_index = [0,4,5,6,10,11,15]

        distance = get_manhattan_distance(size, route_1_index, route_2_index)
        
        self.assertEqual(distance, 2)

unittest.main('test_manhattan', argv=[''], verbosity=2, exit=False)