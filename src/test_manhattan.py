import unittest
from manhattan import get_manhattan_similarity, get_manhattan_distance

class Test_Map(unittest.TestCase):

    def test_manhattan_similarity_different_length(self):

        size =4
        route_1_index = [0,1,5,6]
        route_2_index = [0,4,8,9,5,6]

        similarity = get_manhattan_similarity(size, route_1_index, route_2_index)
        
        self.assertEqual(similarity, 8)

    def test_manhattan_similarity_same_length(self):

        size =4
        route_1_index = [0,4,8,9,5,6]
        route_2_index = [0,4,8,9,5,6]

        similarity = get_manhattan_similarity(size, route_1_index, route_2_index)
        
        self.assertEqual(similarity, 0)

    def test_manhattan_similarity_different_route_same_length(self):

        size =4
        route_1_index = [0,1,5,6,10,11,15]
        route_2_index = [0,4,5,6,10,11,15]

        similarity = get_manhattan_similarity(size, route_1_index, route_2_index)
        
        self.assertEqual(similarity, 2)

class Test_Distance(unittest.TestCase):

    def test_get_manhattan_distance(self):

            size =4
            route = [0,1,5,6,10,11,15]
            distance = get_manhattan_distance(size, route)
            self.assertEqual(distance, 6)

            route = [0,6,13]
            distance = get_manhattan_distance(size, route)
            self.assertEqual(distance, 6)

            route = [10,4,7,12]
            distance = get_manhattan_distance(size, route)
            self.assertEqual(distance, 11)
            


unittest.main('test_manhattan', argv=[''], verbosity=2, exit=False)