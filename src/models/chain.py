from enum import Enum
from typing import List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.utils.drawing import Drawing
from src.models.node import Node
from src.models.curve import Curve


class TypeChains(Enum):
    center = 0
    normal = 1
    border = 2


class ClockDirection(Enum):
    clockwise = 0
    anti_clockwise = 1


class EndPoints(Enum):
    A = 0
    B = 1


class ChainLocation(Enum):
    inwards = 0
    outwards = 1


class Chain:
    def __init__(
        self,
        chain_id: int,
        nr: int,
        center: Tuple[float, float],
        img_height: int,
        img_width: int,
        type: TypeChains = TypeChains.normal,
        A_outward: Optional[Node] = None,
        A_inward: Optional[Node] = None,
        B_outward: Optional[Node] = None,
        B_inward: Optional[Node] = None,
    ) -> None:
        """
        Represents a chain of nodes.

        Args:
            chain_id (int): Identifier for the chain.
            nr (int): Number of rays.
            center (Tuple[float, float]): Center coordinates (x, y).
            img_height (int): Image height.
            img_width (int): Image width.
            type (TypeChains): Type of the chain.
            A_outward (Optional[Node]): Outward node at endpoint A.
            A_inward (Optional[Node]): Inward node at endpoint A.
            B_outward (Optional[Node]): Outward node at endpoint B.
            B_inward (Optional[Node]): Inward node at endpoint B.

        Returns:
            None
        """
        self.l_nodes: List[Node] = []
        self.id = chain_id
        self.size = 0
        self.nr = nr
        self.extA: Optional[Node] = None
        self.extB: Optional[Node] = None
        self.type = type
        self.A_outward = A_outward
        self.A_inward = A_inward
        self.B_outward = B_outward
        self.B_inward = B_inward

        # Less important attributes
        self.center = center  # center of the disk (pith pixel location)
        self.img_height = img_height  # image height
        self.img_width = img_width  # image width
        self.label_id = chain_id  # debug purpose

    def __eq__(self, other: Any) -> bool:
        """
        Compares two chains for equality.

        Args:
            other (Any): Object to compare to.

        Returns:
            bool: True if the chains are equal, False otherwise.
        """
        return (
            isinstance(other, Chain) and self.id == other.id and self.size == other.size
        )

    def is_closed(self, threshold: float = 0.95) -> bool:
        """
        Checks if the chain is closed based on the threshold.

        Args:
            threshold (float): Threshold ratio for being considered closed.

        Returns:
            bool: True if the chain is closed, False otherwise.
        """
        return len(self.l_nodes) >= threshold * self.nr

    def sort_dots(
        self, direction: ClockDirection = ClockDirection.clockwise
    ) -> List[Node]:
        """
        Returns the nodes sorted in the specified direction.

        Args:
            direction (ClockDirection): Direction to sort the nodes.

        Returns:
            List[Node]: Sorted list of nodes.
        """
        return (
            self.clockwise_sorted_dots
            if direction == ClockDirection.clockwise
            else self.clockwise_sorted_dots[::-1]
        )

    def _sort_dots(
        self, direction: ClockDirection = ClockDirection.clockwise
    ) -> List[Node]:
        """
        Sorts the nodes in the chain in the specified direction.

        Args:
            direction (ClockDirection): Direction to sort the nodes.

        Returns:
            List[Node]: Sorted list of nodes.
        """
        clock_wise_sorted_dots = []
        step = 360 / self.nr
        angle_k = (
            self.extB.angle
            if direction == ClockDirection.clockwise
            else self.extA.angle
        )
        while len(clock_wise_sorted_dots) < self.size:
            dot = self.get_node_by_angle(angle_k)
            assert dot is not None
            assert dot.chain_id == self.id
            clock_wise_sorted_dots.append(dot)
            angle_k = (
                (angle_k - step) % 360
                if direction == ClockDirection.clockwise
                else (angle_k + step) % 360
            )
        return clock_wise_sorted_dots

    def __repr__(self) -> str:
        """
        Returns a string representation of the chain.

        Args:
            None

        Returns:
            str: String representation of the chain.
        """
        return f"(id_l:{self.label_id},id:{self.id}, size {self.size}"

    def __find_endpoints(self) -> bool:
        """
        Finds the endpoints of the chain.

        Args:
            None

        Returns:
            bool: True if endpoints changed, False otherwise.
        """
        diff = np.zeros(self.size)
        extA_init = self.extA
        extB_init = self.extB
        self.l_nodes.sort(key=lambda x: x.angle, reverse=False)
        diff[0] = (self.l_nodes[0].angle + 360 - self.l_nodes[-1].angle) % 360

        for i in range(1, self.size):
            diff[i] = self.l_nodes[i].angle - self.l_nodes[i - 1].angle

        border1 = diff.argmax()
        if border1 == 0:
            border2 = diff.shape[0] - 1
        else:
            border2 = border1 - 1

        self.extAind = border1
        self.extBind = border2

        change_border = (
            True
            if (extA_init is None or extB_init is None)
            or (
                extA_init != self.l_nodes[border1] or extB_init != self.l_nodes[border2]
            )
            else False
        )
        self.extA = self.l_nodes[border1]
        self.extB = self.l_nodes[border2]

        return change_border

    def add_nodes_list(self, l_nodes: List[Node]) -> bool:
        """
        Adds a list of nodes to the chain.

        Args:
            l_nodes (List[Node]): List of nodes to add.

        Returns:
            bool: True if endpoints changed after adding nodes, False otherwise.
        """
        self.l_nodes += l_nodes
        change_border = self.update()
        return change_border

    def update(self) -> bool:
        """
        Updates the chain after adding nodes.

        Args:
            None

        Returns:
            bool: True if endpoints changed, False otherwise.
        """
        self.size = len(self.l_nodes)
        if self.size > 1:
            change_border = self.__find_endpoints()
            self.clockwise_sorted_dots = self._sort_dots()
        else:
            raise Exception("Chain must have at least two nodes.")

        return change_border

    def get_nodes_coordinates(self) -> Tuple[np.array, np.array]:
        """
        Gets the coordinates of the nodes in the chain.

        Args:
            None

        Returns:
            Tuple[np.array, np.array]: Arrays of x and y coordinates.
        """
        x = [dot.x for dot in self.l_nodes]
        y = [dot.y for dot in self.l_nodes]
        x_rot = np.roll(x, -self.extAind)
        y_rot = np.roll(y, -self.extAind)

        return np.array(x_rot), np.array(y_rot)

    def get_dot_angle_values(self) -> List[float]:
        """
        Gets the angle values of the nodes in the chain.

        Args:
            None

        Returns:
            List[float]: List of angle values.
        """
        return [dot.angle for dot in self.l_nodes]

    def get_node_by_angle(self, angle: float) -> Optional[Node]:
        """
        Retrieves a node by its angle.

        Args:
            angle (float): Angle to search for.

        Returns:
            Optional[Node]: The node with the specified angle, or None if not found.
        """
        return get_node_from_list_by_angle(self.l_nodes, angle)

    def change_id(self, index: int) -> int:
        """
        Changes the ID of the chain and its nodes.

        Args:
            index (int): New ID for the chain.

        Returns:
            int: 0 upon completion.
        """
        for dot in self.l_nodes:
            dot.chain_id = index
        self.id = index
        return 0

    def to_array(
        self,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Returns nodes coordinates in a numpy array, along with endpoint coordinates.

        Args:
            None

        Returns:
            Tuple[np.array, np.array, np.array]:
                nodes (np.ndarray): Coordinates of nodes.
                c1a (np.array): Coordinates of extA.
                c1b (np.array): Coordinates of extB.
        """
        x1, y1 = self.get_nodes_coordinates()
        nodes = np.vstack((x1, y1)).T
        c1a = np.array([self.extA.x, self.extA.y], dtype=float)
        c1b = np.array([self.extB.x, self.extB.y], dtype=float)

        return nodes.astype(float), c1a, c1b


def euclidean_distance(pix1: np.array, pix2: np.array) -> float:
    """
    Calculates the Euclidean distance between two pixels.

    Args:
        pix1 (np.array): Coordinates of the first pixel.
        pix2 (np.array): Coordinates of the second pixel.

    Returns:
        float: Euclidean distance between pix1 and pix2.
    """
    return np.sqrt((pix1[0] - pix2[0]) ** 2 + (pix1[1] - pix2[1]) ** 2)


def get_node_from_list_by_angle(dot_list: List[Node], angle: float) -> Optional[Node]:
    """
    Retrieves a node from a list by its angle.

    Args:
        dot_list (List[Node]): List of nodes to search.
        angle (float): Angle to match.

    Returns:
        Optional[Node]: The node with the specified angle, or None if not found.
    """
    try:
        dot = next(dot for dot in dot_list if (dot.angle == angle))
    except StopIteration:
        dot = None
    return dot


def get_chain_from_list_by_id(
    chain_list: List["Chain"], chain_id: int
) -> Optional["Chain"]:
    """
    Retrieves a chain from a list by its ID.

    Args:
        chain_list (List[Chain]): List of chains to search.
        chain_id (int): ID of the chain to find.

    Returns:
        Optional[Chain]: The chain with the specified ID, or None if not found.
    """
    try:
        chain_in_list = next(chain for chain in chain_list if (chain.id == chain_id))
    except StopIteration:
        chain_in_list = None
    return chain_in_list


def euclidean_distance_between_nodes(d1: Node, d2: Node) -> float:
    """
    Computes Euclidean distance between two nodes.

    Args:
        d1 (Node): First node.
        d2 (Node): Second node.

    Returns:
        float: Euclidean distance between the two nodes.
    """
    v1 = np.array([d1.x, d1.y], dtype=float)
    v2 = np.array([d2.x, d2.y], dtype=float)

    return euclidean_distance(v1, v2)


def copy_node(node: Node) -> Node:
    """
    Creates a copy of a node.

    Args:
        node (Node): Node to copy.

    Returns:
        Node: A new node with the same attributes as the original.
    """
    return Node(
        x=node.x,
        y=node.y,
        chain_id=node.chain_id,
        radial_distance=node.radial_distance,
        angle=node.angle,
    )


def copy_chain(chain: Chain) -> Chain:
    """
    Creates a copy of a chain.

    Args:
        chain (Chain): Chain to copy.

    Returns:
        Chain: A new chain with copied nodes.
    """
    aux_chain = Chain(
        chain.id,
        chain.nr,
        chain.center,
        chain.img_height,
        chain.img_width,
        type=chain.type,
    )
    aux_chain_node_list = [copy_node(node) for node in chain.l_nodes]
    aux_chain.add_nodes_list(aux_chain_node_list)
    return aux_chain


def angular_distance_between_endpoints(endpoint_j: Node, endpoint_k: Node) -> float:
    """
    Compute angular distance between endpoints.

    Args:
        endpoint_j (Node): Endpoint node j of chain ch_j.
        endpoint_k (Node): Endpoint node k of chain ch_k.

    Returns:
        float: Angular distance in degrees.
    """
    cte_degrees_in_a_circle = 360
    angular_distance = (
        endpoint_j.angle - endpoint_k.angle + cte_degrees_in_a_circle
    ) % cte_degrees_in_a_circle
    return angular_distance


def angular_distance_between_chains(
    ch_j: Chain, ch_k: Chain, endpoint_j_type: EndPoints
) -> float:
    """
    Compute angular distance between chains' endpoints.

    If endpoint_j == A then compute distance between chain.extA and ch_k.extB.
    Otherwise, compute distance between chain.extB and ch_k.extA.

    Args:
        ch_j (Chain): Chain j.
        ch_k (Chain): Chain k.
        endpoint_j_type (EndPoints): Endpoint type of chain j (A or B).

    Returns:
        float: Angular distance between endpoints in degrees.
    """
    endpoint_k = ch_k.extB if endpoint_j_type == EndPoints.A else ch_k.extA
    endpoint_j = ch_j.extA if endpoint_j_type == EndPoints.A else ch_j.extB

    angular_distance = (
        angular_distance_between_endpoints(endpoint_k, endpoint_j)
        if endpoint_j_type == EndPoints.B
        else angular_distance_between_endpoints(endpoint_j, endpoint_k)
    )

    return angular_distance


def minimum_euclidean_distance_between_vector_and_matrix(
    vector: np.array, matrix: np.array
) -> float:
    """
    Compute minimum Euclidean distance between a vector and each row in a matrix.

    Args:
        vector (np.array): Vector.
        matrix (np.array): Matrix.

    Returns:
        float: Minimum distance.
    """
    distances = np.sqrt(np.sum((matrix - vector) ** 2, axis=1))

    return np.min(distances)


def minimum_euclidean_distance_between_chains_endpoints(
    ch_j: Chain, ch_k: Chain
) -> float:
    """
    Compute minimum Euclidean distance between ch_j and ch_k endpoints.

    Args:
        ch_j (Chain): Chain j.
        ch_k (Chain): Chain k.

    Returns:
        float: Minimum Euclidean distance between endpoints.
    """
    nodes1, c1a, c1b = ch_j.to_array()
    nodes2, c2a, c2b = ch_k.to_array()
    c2a_min = minimum_euclidean_distance_between_vector_and_matrix(c2a, nodes1)
    c2b_min = minimum_euclidean_distance_between_vector_and_matrix(c2b, nodes1)
    c1a_min = minimum_euclidean_distance_between_vector_and_matrix(c1a, nodes2)
    c1b_min = minimum_euclidean_distance_between_vector_and_matrix(c1b, nodes2)
    return np.min([c2a_min, c2b_min, c1a_min, c1b_min])


def get_chains_within_angle(angle: float, chain_list: List[Chain]) -> List[Chain]:
    """
    Get chains that cover a specific angle.

    Args:
        angle (float): Angle in degrees.
        chain_list (List[Chain]): List of chains.

    Returns:
        List[Chain]: List of chains that cover the angle.
    """
    chains_list = []
    for chain in chain_list:
        A = chain.extA.angle
        B = chain.extB.angle
        if (A <= B and A <= angle <= B) or (A > B and (A <= angle or angle <= B)):
            chains_list.append(chain)
    return chains_list


def get_closest_chain_border_to_angle(chain: Chain, angle: float) -> Node:
    """
    Get the closest endpoint of a chain to a given angle.

    Args:
        chain (Chain): Chain to search.
        angle (float): Angle in degrees.

    Returns:
        Node: Closest endpoint node.
    """
    B = chain.extB.angle
    A = chain.extA.angle
    if B < A:
        dist_to_b = 360 - angle + B if angle > B else B - angle
        dist_to_a = angle - A if angle > B else 360 - A + angle
    else:
        dist_to_a = A - angle
        dist_to_b = angle - B
    # assert dist_to_a > 0 and dist_to_b > 0
    dot = chain.extB if dist_to_b < dist_to_a else chain.extA
    return dot


def get_closest_dots_to_angle_on_radial_direction_sorted_by_ascending_distance_to_center(
    chains_list: List[Chain], angle: float
) -> List[Node]:
    """
    Get nodes of all chains that are over the ray defined by angle and sort them by ascending distance to center.

    Args:
        chains_list (List[Chain]): List of chains.
        angle (float): Ray angle direction.

    Returns:
        List[Node]: Nodes sorted by ascending distance to center over ray direction angle.
    """
    node_list_over_ray = []
    for chain in chains_list:
        try:
            node = next(node for node in chain.l_nodes if node.angle == angle)
        except StopIteration:
            node = get_closest_chain_border_to_angle(chain, angle)

        if node not in node_list_over_ray:
            node_list_over_ray.append(node)

    if node_list_over_ray:
        node_list_over_ray.sort(key=lambda x: x.radial_distance)

    return node_list_over_ray


def get_nodes_from_chain_list(chain_list: List[Chain]) -> List[Node]:
    """
    Retrieves all nodes from a list of chains.

    Args:
        chain_list (List[Chain]): List of chains.

    Returns:
        List[Node]: List of all nodes in the chains.
    """
    inner_nodes = []
    for chain in chain_list:
        inner_nodes.extend(chain.l_nodes)
    return inner_nodes


def get_nodes_angles_from_list_nodes(node_list: List[Node]) -> List[float]:
    """
    Retrieves angles from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[float]: List of node angles.
    """
    return [node.angle for node in node_list]


def visualize_chains_over_image(
    chain_list: List[Chain] = [],
    img: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
    devernay: Optional[np.ndarray] = None,
    filter: Optional[np.ndarray] = None,
) -> None:
    """
    Visualizes chains over an image.

    Args:
        chain_list (List[Chain]): List of chains to visualize.
        img (Optional[np.ndarray]): Image to display chains over.
        filename (Optional[str]): Filename to save the image.
        devernay (Optional[np.ndarray]): Devernay curves.
        filter (Optional[np.ndarray]): Filtered curves.

    Returns:
        None
    """
    if devernay is not None:
        img = Drawing.write_curves_to_image(devernay, img)
    elif filter is not None:
        img = write_filter_curves_to_image(filter, img)

    figsize = (10, 10)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")

    for chain in chain_list:
        x, y = chain.get_nodes_coordinates()
        if chain.type == TypeChains.normal:
            if chain.is_closed():
                x = x.tolist() + [x[0]]
                y = y.tolist() + [y[0]]
                plt.plot(x, y, "b", linewidth=1)
            else:
                plt.plot(x, y, "r", linewidth=1)
        elif chain.type == TypeChains.border:
            plt.plot(x, y, "k", linewidth=1)
        else:
            plt.scatter(int(x[0]), int(y[0]), c="k")

    plt.tight_layout()
    plt.axis("off")

    if filename:
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()


def visualize_selected_ch_and_chains_over_image_(
    selected_ch: List[Chain] = [],
    chain_list: List[Chain] = [],
    img: Optional[np.ndarray] = None,
    filename: Optional[str] = None,
    devernay: Optional[np.ndarray] = None,
    filter: Optional[np.ndarray] = None,
) -> None:
    """
    Visualizes selected chains and chain list over an image.

    Args:
        selected_ch (List[Chain]): List of selected chains to highlight.
        chain_list (List[Chain]): List of chains to visualize.
        img (Optional[np.ndarray]): Image to display chains over.
        filename (Optional[str]): Filename to save the image.
        devernay (Optional[np.ndarray]): Devernay curves.
        filter (Optional[np.ndarray]): Filtered curves.

    Returns:
        None
    """
    if devernay is not None:
        img = Drawing.write_curves_to_image(devernay, img)
    elif filter is not None:
        img = write_filter_curves_to_image(filter, img)

    figsize = (10, 10)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap="gray")
    for chain in chain_list:
        x, y = chain.get_nodes_coordinates()
        plt.plot(x, y, "w", linewidth=3)

    # draw selected chains
    for ch in selected_ch:
        x, y = ch.get_nodes_coordinates()
        plt.plot(x, y, linewidth=3)
        plt.annotate(str(ch.label_id), (x[0], y[0]), c="b")

    plt.tight_layout()
    plt.axis("off")
    if filename:
        plt.savefig(filename)
    plt.close()


def write_filter_curves_to_image(curves: List[Curve], img: np.ndarray) -> np.ndarray:
    """
    Draws filtered curves onto an image.

    Args:
        curves (List[Curve]): List of filtered curves.
        img (np.ndarray): Image to draw on.

    Returns:
        np.ndarray: Image with curves drawn.
    """
    img = np.full_like(img, 255)

    for c in curves:
        img = c.draw(img, thickness=3)

    return img
