import os
import sys
import operator


multiadd = lambda a,b: map(operator.add, a,b)

def ptw_add(v1,v2):
    return list(map(operator.add, v1, v2))

def ptw_sub(v1,v2):
    return list(map(operator.sub, v1, v2))

def ptw_mul(v1,v2):
    return list(map(operator.mul, v1, v2))

def scalar_mul(v1,c):
    return [x*c for x in v1]


def dot(v1,v2):
    if len(v1)!=len(v2):
        raise ValueError("length of two vectors does not match.")
    return sum(x*y for x,y in zip(v1,v2))




def import_models():
    sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/models/')



