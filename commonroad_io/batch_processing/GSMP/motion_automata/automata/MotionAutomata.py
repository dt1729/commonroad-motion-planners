import copy
import numpy as np
import xml.etree.ElementTree as et
from automata.MotionPrimitiveParser import MotionPrimitiveParser
from automata.MotionPrimitive import MotionPrimitive
from commonroad.geometry.shape import Rectangle


class MotionAutomata:
    """
    Class to handle motion primitives for motion planning
    """
    def __init__(self, veh_type_id):
        self.numPrimitives = 0
        self.Primitives = []
        self.veh_type_id = veh_type_id

        self.egoShape = None
        if veh_type_id == 1:
            self.egoShape = Rectangle(length=4.298, width=1.674)
        elif veh_type_id == 2:
            self.egoShape = Rectangle(length=4.508, width=1.610)
        elif veh_type_id == 3:
            self.egoShape = Rectangle(length=4.569, width=1.844)

    def readFromXML(self, filename: str) -> None:
        """
        Reads all MotionPrimitives from the given file and stores them in the primitives array.

        :param filename: the name of the file to be read from
        """
        # parse XML file
        xmlTree = et.parse(filename).getroot()

        # add trajectories
        self.numPrimitives = self.numPrimitives + len(xmlTree.findall('Trajectory'))
        for t in xmlTree.findall('Trajectory'):
            self.Primitives.append(MotionPrimitiveParser.createFromNode(t))

        self.setVehicleTypeIdPrimitives()
        return

    def sort_primitives(self):
        self.Primitives.sort(key=lambda x: x.finalState.y, reverse=False)

    def createConnectivityListPrimitive(self, primitive: MotionPrimitive) -> None:
        """
        Creates the successor list for a single primitive and stores them in a successor list of the given primitive.

        :param primitive: the primitive to be connected
        """
        for self_primitive in self.Primitives:
            if primitive.checkConnectivityToNext(self_primitive):
                primitive.Successors.append(self_primitive)

    def createConnectivityLists(self) -> None:
        """
        Creates a connectivity list for every primitive (let every primitive has its corresponding successor list).
        """
        for self_primitive in self.Primitives:
            self.createConnectivityListPrimitive(self_primitive)
        return

    def createMirroredPrimitives(self) -> None:
        """
        Creates the mirrored motion primitives since the file to load primitives by default only contains left curves. This function computes the right curves.
        """
        oldPrimitives = self.Primitives
        self.numPrimitives = 2 * self.numPrimitives
        self.Primitives = np.empty(self.numPrimitives, dtype=MotionPrimitive)

        for i in range(len(oldPrimitives)):
            self.Primitives[i] = oldPrimitives[i]
            # create mirrored primitives for the old primitives
            newPrimitive = copy.deepcopy(self.Primitives[i])
            newPrimitive.mirror()
            # add the new mirrored primitive to self primitive list
            self.Primitives[i + len(oldPrimitives)] = newPrimitive
        return

    def getClosestStartVelocity(self, initial_velocity):
        """
        get the velocity among start states that is closest to the given initial velocity
        """
        min_error = float('inf')
        min_idx = 0

        for i in range(len(self.Primitives)):
            primitive = self.Primitives[i]

            error_velocity = abs(initial_velocity - primitive.startState.velocity)
            if error_velocity < min_error:
                min_error = error_velocity
                min_idx = i

        return self.Primitives[min_idx].startState.velocity

    def setVehicleTypeIdPrimitives(self):
        """
        assign vehicle type id to all primitives
        """
        for primitive in self.Primitives:
            primitive.veh_type_id = self.veh_type_id

