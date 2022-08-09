import unittest
import numpy as np
from trapline import is_stable_trapline
from utils import get_sliding_window_sequence

class Test_Map(unittest.TestCase):

    def test_is_stable_trapline(self):
        
        
        routes = []
        routes.append([0,1,5,6,10,15]) # route 1
        routes.append([0,4,5,6,10,15]) # route 2, slight variation
        routes.append([0,1,5,6,10,15]) # route 3

        sequence = get_sliding_window_sequence(2,len(routes),1)

        is_trapline = is_stable_trapline(16, sequence, routes, 3) # on threshold

        self.assertEqual(is_trapline, True)

        is_trapline = is_stable_trapline(16, sequence, routes, 2.9) # below threshold
        self.assertEqual(is_trapline, False)
        

unittest.main('test_trapline', argv=[''], verbosity=2, exit=False)