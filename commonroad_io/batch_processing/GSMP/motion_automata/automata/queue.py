from abc import ABC, abstractmethod
import heapq
import numpy as np


class Queue(ABC):
    def __init__(self):
        self.elements = []

    def empty(self):
        """
        Test whether the queue is empty: Returns true if the queue is empty.
        """
        return len(self.elements) == 0

    def insert(self, item):
        """
        Insert the item into the queue.
        """
        self.elements.append(item)

    @abstractmethod
    def pop(self):
        """
            Pop elements from the queue.
        """
        pass


class FIFOQueue(Queue):
    def __init__(self):
        super(FIFOQueue, self).__init__()

    def pop(self):
        """
        Chooses the shallowest node in the queue.
        """
        if self.empty():
            return None
        return self.elements.pop(0)


class LIFOQueue(Queue):
    def __init__(self):
        super(LIFOQueue, self).__init__()

    def pop(self):
        """
        Chooses the deepest node in the queue.
        """
        if self.empty():
            return None
        return self.elements.pop()


class PriorityQueue:
    def __init__(self):
        self.elements = []
        self.count = 0

    def empty(self):
        """
        Test whether the queue is empty. Returns true if the queue is empty.
        """
        return len(self.elements) == 0

    def insert(self, item, priority):
        """
        Put an item into the queue and count the number of elements in the queue. The number is saved in self.count.

        :param priority: the priority used to sort the queue. It's often the value of some cost function.
        """
        heapq.heappush(self.elements, (priority*10000, self.count, item))
        self.count += 1

    def pop(self):
        """
        Pop the smallest item off the heap (Priority queue) if the queue is not empty.
        """
        if self.empty():
            return None
        return heapq.heappop(self.elements)[2]
