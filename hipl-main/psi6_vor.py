from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import numpy as np
from image_helpers import sort_clockwise
from scipy.optimize import linprog
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from image_helpers import get_dist


def get_theta_psi(psi, tilt_dir=None, mode="psi6"):
    # Accepts: psi6 or psi4
    # Returns: theta IN DEGREES
    # mode == 'psi6' or 'psi4'
    # if tilt_dir is given (angle in deg), arrows will be centered on that angle
    theta = np.rad2deg(np.angle(psi))
    if mode.lower() == "psi6":
        if tilt_dir is not None:
            theta = (theta % 60 - 30) + tilt_dir + 90
        else:
            theta = theta % 60
    elif mode.lower() == "psi4":
        if tilt_dir is not None:
            theta = (theta % 90 - 45) + tilt_dir + 90
        else:
            theta = theta % 90
    else:
        print("Mode must be either 'psi4' or 'psi6")
        return
    return theta


def get_vertices(pos_ind, pos, vor, cutoff=100):
    """Part of the Voronoi implementation of psin.
    Given the voronoi object, the pos point you care about, and its index in the original pos_list
    this gives you the local edges that surround the point.
    sometimes vertices get put thousands of pixels off the screen, so this doesnt count those edges
    cutoff in pixels
    """
    ind = vor.point_region[pos_ind]
    retlist = []
    for i in vor.regions[ind]:
        p = vor.vertices[i]
        if abs(p[0] - pos[0]) < cutoff and abs(p[1] - pos[1]) < cutoff:
            retlist.append(p)
    return np.array(retlist)


def get_edges(verts):
    """Part of the Voronoi implementation of psi4.
    for a list of vertices, this gives (vertex, vertex) pairs of each segment of the
    convex polygon.
    Relies on the fact that voronoi gives you these points in a nice order going around
    the polygon (which it does)."""
    retlist = []
    # toss out cases on the edge w
    if np.shape(verts)[0] > 2:
        for i in range(np.shape(verts)[0]):
            ends = (verts[i - 1], verts[i])
            retlist.append(ends)
    return np.array(retlist)


def psi6_Vor(Im):
    """
    psi6 and psi4 are weighted by the edge length of the voronoi cell
    also calculates the conventional psi (psi6c and psi4c) which are not.
    """
    pos_list = np.array([skyrm.pos[::-1] for skyrm in Im.skyrm_list])
    retlist = []
    vor = Voronoi(pos_list)
    i = 0
    for skyrm in Im.skyrm_list:
        points = skyrm.pos
        psi6 = 0
        psi4 = 0
        psi6c = 0
        psi4c = 0
        A = get_vertices(i, pos_list[i], vor, cutoff=100)
        i += 1

        # needs to be bordering more than two cells to be considered legit
        if np.shape(A)[0] > 2:
            hull = ConvexHull(A)
            skyrm.vor_area = hull.volume
            edges = get_edges(A)
            nedges = len(edges)
            totL = 0

            # need to calculate total border length first
            for edge in edges:
                length = get_dist(edge[0], edge[1])
                totL += length

            # this loops over each edge and calculates its component of the psi4 total
            for edge in edges:
                length = get_dist(edge[0], edge[1])

                # the slope between the two points is the inverse of the slope of the edge
                dx = edge[0, 0] - edge[1, 0]
                dy = edge[0, 1] - edge[1, 1]
                theta = np.arctan2(dy, dx)

                # component of psi4 for this point due to this individual edge
                psi6 += length / totL * np.exp(6j * theta)
                psi4 += length / totL * np.exp(4j * theta)
                psi6c += np.exp(6j * theta) / nedges
                psi4c += np.exp(4j * theta) / nedges

            # assign dot to have psi4vor value
            skyrm.psi6 = psi6
            skyrm.psi4 = psi4
            skyrm.psi6c = psi6c
            skyrm.psi4c = psi4c
        else:
            skyrm.vor_area = None
            skyrm.psi6 = None
            skyrm.psi4 = None
            skyrm.psi6c = None
            skyrm.psi4c = None
    return


def voronoi_neighbors(Im):
    pos_list = np.array(
        [skyrm.pos[::-1] for skyrm in Im.skyrm_list]
    )  # expects x,y points
    vor = Voronoi(pos_list)

    bad = False
    i = 0
    num = len(Im.skyrm_list)
    print(f"{0}/{num} .. ", end="")
    for skyrm in Im.skyrm_list:
        if (i + 1) % 100 == 0:
            print(f"\rVoronoi neighbors: {i+1}/{num}", end="")
        nlist = []
        neighb_vert = []
        A = get_vertices(i, pos_list[i], vor, cutoff=100)
        j = 0
        # gets the neighbors' index and puts in skyrm.neighbors_vor
        for nskyrm in Im.skyrm_list:
            B = get_vertices(j, pos_list[j], vor, cutoff=100)
            for vertex in A:
                if vertex in B and i != j:
                    neighb_vert.append([nskyrm.index, vertex[1], vertex[0]])
                    if nskyrm.index not in nlist:
                        nlist.append(nskyrm.index)
            j += 1

        skyrm.neighbors = nlist  # just neighbors indices
        skyrm.neighbors_vert = np.array(
            neighb_vert
        )  # neighbors indicies with all shared vertices

        # print('skyrm: ', skyrm)
        # print()
        # print("skyrm neighbors_vert:", skyrm.neighbors_vert)
        # print()
        # print("skyrm neighbors_vert shape:", np.shape(skyrm.neighbors_vert))
        # print()
        # print("skyrm neighbors_vert[:,1:]:", np.array(skyrm.neighbors_vert)[:,1])
        # print()
        # print("skyrm neighbors_vert[:,1:]:", skyrm.neighbors_vert[:,1])
        # print("-----")

        if len(skyrm.neighbors_vert) > 0:
            skyrm.vor_cell = sort_clockwise(
                skyrm.neighbors_vert[:, 1:], skyrm.pos
            )  # just verticies
        else:
            skyrm.edge = True
            i += 1
            bad = True
            continue
        vor_circ = get_vor_circle(
            skyrm.vor_cell
        )  # position and rad of largest inscribed circle
        if vor_circ[0] is not None:  # will return Nones if doesnt have enough points
            skyrm.vor_circ = vor_circ  # which happens if edges are too long
        else:
            skyrm.edge = True
        i += 1
    print(f"\rVoronoi neighbors: {num}/{num}")

    if bad:
        print("1 or more skyrmions has bad edge values.\nSkipping averages.")
    else:
        Im.get_skyrm_averages()  # calculate average values across image
    return


def get_vor_circle(points, show=False):
    # points should be in clockwise ordering, [[y1,x1], [y2,x2], ...]
    # returns [center_y, center_x, radius]

    points = np.array(points)

    hslist = gen_hslist(points)
    try:
        norm_vector = np.reshape(
            np.linalg.norm(hslist[:, :-1], axis=1), (hslist.shape[0], 1)
        )
    except Exception as e:
        # print('\n',e)
        # print('hslist: ', hslist.shape, hslist)
        # print('points: ', points)
        return [None, None, None]
    c = np.zeros((hslist.shape[1],))
    c[-1] = -1

    A = np.hstack((hslist[:, :-1], norm_vector))
    b = -hslist[:, -1:]

    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    center = res.x[:-1]  # center (x,y)
    radius = res.x[-1]  # radius

    if show:
        show_max_circle(points[:, ::-1], [center[1], center[0], radius])

    return [center[1], center[0], radius]


def show_max_circle(points, vor_circ, pos=None):
    # points [[x1,y1], [x2,y2], ...]
    # vor_circ [y,x,rad]
    # pos [y, x]
    center = vor_circ[:2]
    center = center[::-1]
    rad = vor_circ[2]
    fig, ax = plt.subplots()
    x = np.pad(points[:, 0], 1, mode="wrap")[1:]
    y = np.pad(points[:, 1], 1, mode="wrap")[1:]
    plt.plot(x, y, "ro-", c="g", alpha=1)
    circle = Circle(center, radius=rad, alpha=0.5)  # center is x,y
    center = Circle(center, radius=0.1, color="k", label="max circle center")
    ax.add_patch(circle)
    ax.add_patch(center)
    if pos is not None:
        pos = Circle(pos[::-1], radius=0.1, color="r", label="skyrm pos")
        ax.add_patch(pos)
        plt.legend()
    ax.set_aspect(1)
    plt.show()
    return


def gen_hslist(points):
    """
    generate half space intersection of vertex points
    """
    hslist = []
    for i in range(len(points)):
        y1, x1 = points[i - 1]
        y2, x2 = points[i]
        if x2 != x1:
            m = (y2 - y1) / (x2 - x1)
            halfspace = np.array([m, -1, y1 - m * x1])
            if x2 > x1:
                hslist.append(-1 * halfspace)
            else:
                hslist.append(halfspace)
        else:  # m undefined
            if y2 > y1:
                hslist.append([-1, 0, x1])
            elif y2 < y1:
                hslist.append([1, 0, -1 * x1])
    return np.array(hslist)
