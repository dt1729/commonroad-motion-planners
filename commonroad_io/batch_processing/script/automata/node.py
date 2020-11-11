from commonroad.scenario.trajectory import State
from typing import List
from automata.MotionPrimitive import MotionPrimitive


class Node:
    def __init__(self, path: List[List[State]], primitives: List[MotionPrimitive], tree_depth: int):
        self.path: List[List[State]] = path
        self.primitives = primitives
        self.tree_depth = tree_depth

    def get_successors(self):
        """
        Returns all possible successor primitives of the current primitive.
        """
        return self.primitives[-1].Successors


class PrioNode(Node):
    def __init__(self, path: List[List[State]], primitives: List[MotionPrimitive], tree_depth: int,
                 current_cost: float):
        super(PrioNode, self).__init__(path, primitives, tree_depth)
        self.current_cost = current_cost

