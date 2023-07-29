import numpy as np


def polygon(n, center, radius, offset=0):
    """Compute the vertices of a polygon.

    Args:
        n (int): Number of vertices.
        center (numpy.ndarray, (2,)): Center of polygon.
        radius (float): Distance from center to vertices.
        offset (float, optional): Angle of polygon's top vertex in radians (default 0).

    Returns:
        list(numpy.ndarray, (2,)): List of vertices
    """
    points = []
    for i in range(n):
        th = i / n * 2 * np.pi + offset
        v = np.array([np.sin(th), -np.cos(th)])
        points.append(radius * v + center)
    return points


def star(n, center, radius_outer, radius_inner):
    """Compute the vertices of a star.

    Args:
        n (int): Number of star points.
        center (numpy.ndarray, (2,)): Center of star.
        radius_outer (float): Distance from center to outer vertices.
        radius_inner (float): Distance from center to inner vertices.

    Returns:
        list(numpy.ndarray, (2,)): List of vertices
    """
    points = []
    radii = [radius_outer, radius_inner]
    for i in range(2 * n):
        th = i / n * np.pi
        v = np.array([np.sin(th), -np.cos(th)])
        r = radius_outer if i % 2 == 0 else radius_inner
        points.append(r * v + center)
    return points
