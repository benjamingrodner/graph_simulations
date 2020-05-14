import time
import numpy as np


# time kernel functions
def normal():
    t0 = time.time()
    a = np.zeros([10]*2)
    b = np.ones([5]*2)
    a[0:5,0:5] = b
    t1 = time.time()
    print('t1-t0',t1-t0)

def place():
    t0 = time.time()
    a = np.zeros([10]*2)
    b = np.ones([5]*2)
    np.place
    t1 = time.time()
    print('t1-t0',t1-t0)


normal()
