import taichi as ti


@ti.func
def mid(x,y):
    """
        Return the mid point between x and y
    """
    return (x[0]+y[0])/2, (x[1]+y[1])/2


@ti.func
def length2(x, y):
    """
        Return the sqaure of the length of line segment xy
    """
    return (x[0]-y[0])**2 + (x[1]-y[1])**2


@ti.func
def length(x, y):
    """
        Return the length of line segment xy
    """
    return ti.sqrt(length2(x,y))

@ti.func
def perpend(p):
    """
        Return the perpendicular vector
        (There are two implementations)
    """
    p[0], p[1] = -p[1], p[0]
    #p[0], p[1] = p[1], -p[0]
    return p

@ti.func
def cross(x,y):
    """
        Return the 2-D cross product of x and y
    """
    product = x[0]*y[1] - x[1]*y[0]
    return product

@ti.func
def crossv(x,y):
    """
        Return the 2-D cross product vector of x and y
    """
    v = perpend(x - y)
    return v