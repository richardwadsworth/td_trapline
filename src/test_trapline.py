import unittest
import numpy as np
from trapline import cluster_common_route_segments, is_stable_trapline_1, is_stable_trapline_2
#from trapline import is_stable_trapline, is_stable_trapline_2,cluster_common_route_segments
from utils import get_sliding_window_sequence

class Test_Map(unittest.TestCase):

    def test_is_stable_trapline(self):
        
        
        routes = []
        routes.append([0,1,5,6,10,15]) # route 1
        routes.append([0,4,5,6,10,15]) # route 2, slight variation
        routes.append([0,1,5,6,10,15]) # route 3

        sequence = get_sliding_window_sequence(2,len(routes),1)

        is_trapline = is_stable_trapline_1(16, sequence, routes, 3) # on threshold
        self.assertEqual(is_trapline, True)

        is_trapline = is_stable_trapline_1(16, sequence, routes, 2.9) # below threshold
        self.assertEqual(is_trapline, False)
        
    def test_is_stable_trapline_2(self):

        routes = []
        routes.append([0,1,5,6,10,15]) # route 1
        routes.append([0,4,5,6,10,15]) # route 2, slight variation
        routes.append([0,1,5,6,10,15]) # route 3

        sequence = get_sliding_window_sequence(2,len(routes),1)

        is_trapline = is_stable_trapline_2(16, sequence, routes, 3) # on threshold
        self.assertEqual(is_trapline, True)

        is_trapline = is_stable_trapline_1(16, sequence, routes, 2.9) # below threshold
        self.assertEqual(is_trapline, False)
    

class Test_Clustering(unittest.TestCase):

    def test_cluster_common_route_segments_same(self):

        
        route_1 = [6,2,3,7,11,15] # route 1
        route_2 = route_1 # route 2, same
    
        expected_route1_segments = [[6,2,3,7,11,15]]
        expected_route2_segments = expected_route1_segments

        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

        # swap the routes
        route2_segments, route1_segments  = cluster_common_route_segments(route_2, route_1)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)


    def test_cluster_common_route_segments_similar_1(self):

        
        route_1 = [6,2,3,7,11,15] # route 1
        route_2 =[6,10,11,15] # route 2, slight variation
    
        expected_route1_segments = [[6],[6,2,3,7,11],[11,15]]
        expected_route2_segments = [[6],[6,10,11],[11,15]]

        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

        # swap the routes
        route2_segments, route1_segments  = cluster_common_route_segments(route_2, route_1)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

    def test_cluster_common_route_segments_similar_2(self):

        route_1 = [0,1,5,6,2,3,7,11,15] # route 1
        route_2 =[0,4,8,9,5,6,10,9,5,6,7,11,15] # route 2, slight variation
    
        expected_route1_segments = [[0],[0,1,5],[5,6],[6,2,3,7],[7,11,15]]
        expected_route2_segments = [[0],[0,4,8,9,5],[5,6],[6,10,9,5,6,7],[7,11,15]]

        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

        # swap the routes
        route2_segments, route1_segments  = cluster_common_route_segments(route_2, route_1)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)


    def test_cluster_common_route_segments_different(self):

        
        route_1 = [6,2,3,7,11,15] # route 1
        route_2 =[6,5,4,8] # route 2, very different
    
        expected_route1_segments = [[6],[6,2,3,7,11,15]]
        expected_route2_segments = [[6],[6,5,4,8]]

        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

        # swap the routes
        route2_segments, route1_segments  = cluster_common_route_segments(route_2, route_1)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

    def test_cluster_common_route_segments_repeated_vist(self):

        
        route_1 = [6,2,3,7,11,15] # route 1
        route_2 =[6,10,11,7,11,15] # route 2, slight variation
    
        expected_route1_segments = [[6],[6,2,3,7],[7,11,15]]
        expected_route2_segments = [[6],[6,10,11,7],[7,11,15]]

        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

        # swap the routes

        # slight different segments
        expected_route1_segments = [[6],[6,2,3,7,11],[11,15]]
        expected_route2_segments = [[6],[6,10,11],[11,7,11,15]]

        route2_segments, route1_segments  = cluster_common_route_segments(route_2, route_1)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)
        
    
    def test_cluster_common_route_segments_long_routes(self):

        
        route_1 = [34, 100, 116, 117, 133, 149, 150, 166, 167, ]
        route_2 = [34, 100, 116, 132, 133, 149, 149, 150, 118, 117, 133, 149, 150]

        expected_route1_segments = [[34],[34,100,116],[116,117],[117, 133, 149, 150, 166, 167]]
        expected_route2_segments = [[34],[34,100,116],[116, 132, 133, 149, 149, 150, 118, 117],[117, 133, 149, 150]]

        route1_segments, route2_segments = cluster_common_route_segments(route_1, route_2)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)

        # swap the routes

        # slight different segments
        expected_route1_segments = [[34],[34,100,116],[116, 117, 133], [133, 149], [149, 150], [150, 166, 167]]
        expected_route2_segments = [[34],[34,100,116],[116, 132, 133],[133, 149],[149, 149, 150],[150, 118, 117, 133, 149, 150]]

        route2_segments, route1_segments  = cluster_common_route_segments(route_2, route_1)
        self.assertEqual(expected_route1_segments, route1_segments)
        self.assertEqual(expected_route2_segments, route2_segments)
        
unittest.main('test_trapline', argv=[''], verbosity=2, exit=False)