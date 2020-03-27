# Helper functions

def mapFromTo(x,a,b,c,d):
    """
    Maps x from range a...b to c...d
    :param x: Input
    :param a: from lower
    :param b: from upper
    :param c: to lower
    :param d: to upper
    :return: mapped
    """
    y = (x-a)/(b-a)*(d-c)+c
    return y