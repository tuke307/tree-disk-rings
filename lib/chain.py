import numpy as np
import matplotlib.pyplot as plt
from typing import List

from lib.canny_devernay_edge_detector import write_curves_to_image
from lib.filter_edges import write_filter_curves_to_image


def euclidean_distance(pix1, pix2):
    return np.sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2)


def get_node_from_list_by_angle(dot_list, angle):
    try:
        dot = next(dot for dot in dot_list if (dot.angle == angle))
    except StopIteration as e:
        dot = None
    return dot


def get_chain_from_list_by_id(chain_list, chain_id):
    try:
        chain_in_list = next(chain for chain in chain_list if (chain.id == chain_id))

    except StopIteration:
        chain_in_list = None
    return chain_in_list

########################################################################################################################
#Class Node
########################################################################################################################
class Node:
    def __init__(self, x, y, chain_id, radial_distance, angle):
        self.x = x
        self.y = y
        self.chain_id = chain_id
        self.radial_distance = radial_distance
        self.angle = angle

    def __repr__(self):
        return (f'({self.x},{self.y}) ang:{self.angle} radio:{self.radial_distance:0.2f} cad.id {self.chain_id}\n')

    def __str__(self):
        return (f'({self.x},{self.y}) ang:{self.angle} radio:{self.radial_distance:0.2f} id {self.chain_id}')

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.angle == other.angle


def euclidean_distance_between_nodes(d1: Node, d2: Node):
    v1 = np.array([d1.x, d1.y], dtype=float)
    v2 = np.array([d2.x, d2.y], dtype=float)
    return euclidean_distance(v1, v2)

def copy_node(node: Node):
    return Node(**{'y': node.y, 'x': node.x, 'angle': node.angle , 'radial_distance':
    node.radial_distance, 'chain_id': node.chain_id})

########################################################################################################################
#Class Chain
########################################################################################################################
class TypeChains:
    center = 0
    normal = 1
    border = 2


class ClockDirection:
    clockwise = 0
    anti_clockwise = 1



class Chain:
    def __init__(self, chain_id: int, Nr: int, center, M: int, N:int, type: TypeChains = TypeChains.normal, A_up=None, A_down=None, B_up=None,
                 B_down=None):
        self.nodes_list = []
        self.id = chain_id
        self.label_id = chain_id
        self.size = 0
        self.Nr = Nr
        self.A_up = A_up
        self.A_down = A_down
        self.B_up = B_up
        self.B_down = B_down
        self.type = type
        self.extA = None
        self.extB = None
        self.center = center
        self.M = M
        self.N = N

    def __eq__(self, other):
        if other is None:
            return False
        return self.id == other.id and self.size == other.size

    def is_full(self, regions_count=16):
        if len(self.nodes_list) >= (regions_count - 1) * self.Nr / regions_count:
            return True
        else:
            return False

    def sort_dots(self, direction=ClockDirection.clockwise):
        return self.clockwise_sorted_dots if direction == ClockDirection.clockwise else self.clockwise_sorted_dots[::-1]

    def _sort_dots(self, direction = ClockDirection.clockwise):
        clock_wise_sorted_dots = []
        step = 360 / self.Nr
        angle_k = self.extB.angle if direction == ClockDirection.clockwise else self.extA.angle
        while len(clock_wise_sorted_dots) < self.size:
            dot = self.get_node_by_angle(angle_k)
            assert dot is not None
            assert dot.chain_id == self.id
            clock_wise_sorted_dots.append(dot)
            angle_k = (angle_k - step) % 360 if direction == ClockDirection.clockwise else (angle_k + step) % 360

        return clock_wise_sorted_dots

    def __repr__(self):
        return (f'(id_l:{self.label_id},id:{self.id}, size {self.size}')

    def __find_borders(self):
        diff = np.zeros(self.size)
        extA_init = self.extA if self.extA is not None else None
        extB_init = self.extB if self.extB is not None else None
        self.nodes_list.sort(key=lambda x: x.angle, reverse=False)
        diff[0] = (self.nodes_list[0].angle + 360 - self.nodes_list[-1].angle) % 360

        for i in range(1, self.size):
            diff[i] = (self.nodes_list[i].angle - self.nodes_list[i - 1].angle)

        border1 = diff.argmax()
        if border1 == 0:
            border2 = diff.shape[0] - 1
        else:
            border2 = border1 - 1

        self.extAind = border1
        self.extBind = border2

        change_border = True if (extA_init is None or extB_init is None) or \
                                (extA_init != self.nodes_list[border1] or extB_init != self.nodes_list[
                                    border2]) else False
        self.extA = self.nodes_list[border1]
        self.extB = self.nodes_list[border2]

        return change_border

    def add_nodes_list(self, nodes_list):
        self.nodes_list += nodes_list
        change_border = self.update()
        return change_border

    def update(self):
        self.size = len(self.nodes_list)
        if self.size > 1:
            change_border = self.__find_borders()
            self.clockwise_sorted_dots = self._sort_dots()
        else:
            raise

        return change_border

    def get_nodes_coordinates(self):
        x = [dot.x for dot in self.nodes_list]
        y = [dot.y for dot in self.nodes_list]
        x_rot = np.roll(x, -self.extAind)
        y_rot = np.roll(y, -self.extAind)
        return x_rot, y_rot

    def get_dot_angle_values(self):
        return [dot.angle for dot in self.nodes_list]

    def get_node_by_angle(self, angle):
        return get_node_from_list_by_angle(self.nodes_list, angle)

    def change_id(self, index):
        for dot in self.nodes_list:
            dot.chain_id = index
        self.id = index
        return 0

    def to_array(self):
        x1, y1 = self.get_nodes_coordinates()
        nodes = np.vstack((x1, y1)).T

        c1a = np.array([self.extA.x, self.extA.y], dtype=float)
        c1b = np.array([self.extB.x, self.extB.y], dtype=float)
        return nodes.astype(float), c1a, c1b

    def check_if_nodes_are_missing(self):
        angle_domain = [self.extA.angle]
        step = 360/self.Nr
        while angle_domain[-1] != self.extB.angle:
            angle_domain.append((angle_domain[-1] + step) % 360)

        return not self.size == len([angle for angle in angle_domain if self.get_node_by_angle(angle) is not None])

def copy_chain(chain : Chain):
    aux_chain = Chain(chain.id, chain.Nr, chain.center, chain.M, chain.N, type=chain.type)
    aux_chain_node_list = [ copy_node(node)
                        for node in chain.nodes_list]
    aux_chain.add_nodes_list(aux_chain_node_list)

    return aux_chain
class EndPoints:
    A = 0
    B = 1

class ChainLocation:
    inwards = 0
    outwards = 1
def angular_distance_between_chains_endpoints(cad_1, cad_2, border):
    """
    Compute angular distance between chains endpoints. If border == A then compute distance between ch1.extA and
    cad2.extB. In the other case, compute distance between ch1.extB and cad2.extA
    @param cad_1: chain source
    @param cad_2: chain dst
    @param border: cad_1 endpoint
    @return: angular distance between enpoints in degrees
    """

    cte_degrees_in_a_circle = 360
    ext_cad_2 = cad_2.extB if border == EndPoints.A else cad_2.extA
    ext_cad_1 = cad_1.extA if border == EndPoints.A else cad_1.extB
    if border == EndPoints.B:
        if ext_cad_2.angle > ext_cad_1.angle:
            angular_distance = ext_cad_2.angle - ext_cad_1.angle
        else:
            angular_distance = ext_cad_2.angle + (cte_degrees_in_a_circle - ext_cad_1.angle)

    else:

        if ext_cad_2.angle > ext_cad_1.angle:
            angular_distance = ext_cad_1.angle + (cte_degrees_in_a_circle - ext_cad_2.angle)

        else:
            angular_distance = ext_cad_1.angle - ext_cad_2.angle

    return angular_distance


def distance_to_endpoint(ext, matriz):
    distances = np.sqrt(np.sum((matriz - ext) ** 2, axis=1))
    return np.min(distances)


def minimum_euclidean_distance_between_chains_endpoints(c1: Chain, c2: Chain):
    """
    Compute minimum distance between chain endpoints.
    @param c1:
    @param c2:
    @return:
    """
    nodes1, c1a, c1b = c1.to_array()
    nodes2, c2a, c2b = c2.to_array()
    c2a_min = distance_to_endpoint(nodes1, c2a)
    c2b_min = distance_to_endpoint(nodes1, c2b)
    c1a_min = distance_to_endpoint(nodes2, c1a)
    c1b_min = distance_to_endpoint(nodes2, c1b)
    return np.min([c2a_min, c2b_min, c1a_min, c1b_min])

def get_chains_within_angle(angle: int, chain_list: List[Chain]):
    chains_list = []
    for chain in chain_list:
        A = chain.extA.angle
        B = chain.extB.angle
        if ((A <= B and A <= angle <= B) or
                (A > B and (A <= angle or angle <= B))):
            chains_list.append(chain)

    return chains_list

def get_closest_chain_border_to_angle(chain: Chain, angle: int):
    B = chain.extB.angle
    A = chain.extA.angle
    if B < A:
        dist_to_b = 360 - angle + B if angle > B else B - angle
        dist_to_a = angle - A if angle > B else 360 - A + angle

    else:
        dist_to_a = A - angle
        dist_to_b = angle - B
    #assert dist_to_a > 0 and dist_to_b > 0
    dot = chain.extB if dist_to_b < dist_to_a else chain.extA
    return dot


def get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(chains_list: List[Chain], angle: int):
    node_list_over_ray = []
    for chain in chains_list:
        #dot = get_closest_chain_dot_to_angle(chain, angle)
        try:
            node =  [node for node in chain.nodes_list if node.angle == angle][0]
        except IndexError:
            node = get_closest_chain_border_to_angle(chain, angle)
            pass

        if node not in node_list_over_ray:
            node_list_over_ray.append(node)
    #write_log(MODULE_NAME,label,f"{lista_puntos_perfil}")
    if len(node_list_over_ray)>0:
        node_list_over_ray= sorted(node_list_over_ray, key=lambda x: x.radial_distance, reverse=False)
    return node_list_over_ray


def get_nodes_from_chain_list(chain_list: List[Chain]):
    inner_nodes = []
    for chain in chain_list:
        inner_nodes += chain.nodes_list
    return inner_nodes

def get_nodes_angles_from_list_nodes(node_list: List[Node]):
    return [node.angle for node in node_list]
###########################################
##Display
###########################################

def visualize_chains_over_image(chain_list=[], img=None, filename=None, devernay=None, filter=None):
    if devernay is not None:
        img = write_curves_to_image(devernay, img)
    elif filter is not None:
        img = write_filter_curves_to_image(filter, img)

    figsize = (10, 10)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    for chain in chain_list:
        x,y = chain.get_nodes_coordinates()
        if chain.type == TypeChains.normal:
            if chain.is_full():
                x = x.tolist() + [x[0]]
                y = y.tolist() + [y[0]]
                plt.plot(x, y, 'b', linewidth=1)
            else:
                plt.plot(x, y, 'r', linewidth=1)
        elif chain.type == TypeChains.border:
            plt.plot(x, y, 'k', linewidth=1)

        else:
            plt.scatter(int(x[0]), int(y[0]), c='k')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def visualize_selected_ch_and_chains_over_image_(selected_ch=[], chain_list=[], img=None, filename=None, devernay=None, filter=None):
    if devernay is not None:
        img = write_curves_to_image(devernay, img)
    elif filter is not None:
        img = write_filter_curves_to_image(filter, img)

    figsize = (10, 10)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    for chain in chain_list:
        x, y = chain.get_nodes_coordinates()
        plt.plot(x, y, 'w', linewidth=3)


    #draw selected chain
    for ch in selected_ch:
        x, y = ch.get_nodes_coordinates()
        plt.plot(x, y,  linewidth=3)
        plt.annotate(str(ch.label_id), (x[0], y[0]), c='b')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig(filename)
    plt.close()