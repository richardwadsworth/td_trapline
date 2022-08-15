import unittest
from utils import map_coord_to_index
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

            
    def test_get_manhattan_distance_opposites(self):
            # opposite routes

            size =4
        
            route1 = [1,7]
            route2 = [7,1]
            distance1 = get_manhattan_distance(size, route1)
            distance2 = get_manhattan_distance(size, route2)
            self.assertEqual(distance1, distance2)

            route1 = [0,3,15]
            route2 = [15,3,0]
            distance1 = get_manhattan_distance(size, route1)
            distance2 = get_manhattan_distance(size, route2)
            self.assertEqual(distance1, distance2)

            route1 = [2,5,9,14,11,7]
            route2 = [2,7,11,14,9,5]
            distance1 = get_manhattan_distance(size, route1)
            distance2 = get_manhattan_distance(size, route2)
            self.assertEqual(distance1, distance2)


            size =17
            route1 = [76, 111,  213, 109]
            route2 = [76, 109,  213, 111]
            distance1 = get_manhattan_distance(size, route1)
            distance2 = get_manhattan_distance(size, route2)
            self.assertEqual(distance1, distance2)

            size =17
            route1 = [76, 111, 145, 179, 213, 246, 211, 177, 143, 109]
            route2 = [76, 109, 143, 177, 211, 246, 213, 179, 145, 111] 
            distance1 = get_manhattan_distance(size, route1)
            distance2 = get_manhattan_distance(size, route2)
            self.assertEqual(distance1, distance2)

            size =20
            route1=[x[0] for x in [
                    (map_coord_to_index(size, 9, 4),1.0), 
                    (map_coord_to_index(size, 10, 6),1.0),
                    (map_coord_to_index(size, 10, 9),1.0),
                    (map_coord_to_index(size, 10, 12),1.0),
                    (map_coord_to_index(size, 10, 15),1.0),
                    (map_coord_to_index(size, 9, 18),1.0),
                    (map_coord_to_index(size, 8, 15),1.0),
                    (map_coord_to_index(size, 8, 12),1.0),
                    (map_coord_to_index(size, 8, 9),1.0),
                    (map_coord_to_index(size, 8, 6),1.0)
                    ]]
            distance1 = get_manhattan_distance(size, route1)
            1==1


unittest.main('test_manhattan', argv=[''], verbosity=2, exit=False)