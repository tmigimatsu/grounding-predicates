import os
import random
import typing

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import numpy as np
import pygame
import symbolic

from . import actions, objects, propositions
from .constants import COLOR_BG, DIM, DIM_GRID, RES_GRID
from .state import State


class World:
    """World class.

    Args:
        pddl (symbolic.Pddl): Pddl object.
        config (dict): World config.
        state (world.State, optional): Initial state. If None, uses pddl initial state.
        log (text file, optional): Print debug statements to this file (default None).
        validate (bool, optional): Validate axioms (default False).
    """

    def __init__(self, pddl, config, state=None, log=None, validate=False):
        self._pddl = pddl
        self._logfile = log

        # Add objects
        self._objects = {}
        self._children = set()
        for obj in self._pddl.objects:
            self._objects[obj.name] = objects.create(self, obj, config[obj.name])

            if obj.type.is_subtype("location"):
                self.add_child(self._objects[obj.name])

        # Add all states without applying
        self._state = State(self)
        pending_props = []
        if state is None:
            for str_prop in self._pddl.initial_state:
                prop = self._state.add(str_prop, apply=False)
                pending_props.append(prop)
        else:
            # Fix derived predicates
            state = set(propositions.alias(str_prop) for str_prop in state)
            state = self.pddl.derived_state(state)
            state = set(propositions.alias(str_prop) for str_prop in state)
            for str_prop in state:
                prop = self._state.add(str_prop, apply=False)
                pending_props.append(prop)

        if validate:
            if not self.pddl.is_valid_state(self.state.stringify()):
                raise propositions.PropositionValueError(
                    f"Cannot apply {str_prop} due to axiom violation."
                )

        # Match all keys to lockables.
        keys = set()
        lockables = set()
        for prop in pending_props:
            for arg in prop.args:
                if isinstance(arg, objects.Key):
                    keys.add(str(arg))
                elif isinstance(arg, objects.Lockable):
                    lockables.add(str(arg))
        for key in keys:
            lockable = key.split("_")[0]
            if not lockable in lockables:
                continue
            self._state.add(f"matches({key}, {lockable})")

        # Apply all added states in order of priority (highest first)
        pending_props.sort(reverse=True)
        for prop in pending_props:
            prop.apply()

        self._canvas = pygame.display.set_mode(DIM, depth=24)

    @property
    def pddl(self):
        """Pddl object.

        :type: symbolic.Pddl
        """
        return self._pddl

    @property
    def objects(self):
        """World objects list.

        :type: dict(str, objects.Object)
        """
        return self._objects

    @property
    def locations(self):
        """World locations list.

        :type: list(objects.Object)
        """
        return [obj for obj in self._children if obj.is_subtype("location")]

    @property
    def occupancy(self):
        """Occupancy grid.

        :type: numpy.ndarray, (``DIM_GRID[0]+1``, ``DIM_GRID[1]+1``), bool
        """
        return self._occupancy

    @property
    def state(self):
        """World state.

        :type: world.State
        """
        return self._state

    @property
    def logfile(self):
        """Log file.

        Gets written to with the method 'logfile.write(str)'."""
        return self._logfile

    @logfile.setter
    def logfile(self, logfile):
        self._logfile = logfile

    def log(self, message):
        """Print debug message to log file.

        Args:
            message (str): Debug message.
        """
        if self.logfile is None:
            return
        self.logfile.write(message)

    def add_child(self, obj):
        self._children.add(obj)

    def list_valid_actions(self):
        """List valid actions from current state.

        Returns:
            list(str): List of valid actions.
        """
        return self.pddl.list_valid_actions(self.state.stringify())

    def is_goal_satisfied(self):
        """Determine whether goal is satisfied at current state.

        Returns:
            bool: Whether goal is satisfied.
        """
        return self.pddl.is_goal_satisfied(self.state.stringify())

    def execute(self, str_action):
        """Execute action.

        Args:
            str_action (str): Action string.
        """
        str_state = self.state.stringify()
        str_state_next = self.pddl.next_state(str_state, str_action)
        self.state.update(str_state_next)

    def place_objects(self):
        """Place objects without a location."""
        # Find all used objects
        locations = set(self.locations)
        used_objects = set(arg for prop in self.state for arg in prop.args) - locations
        used_objects.add(self.objects["agent"])
        try:
            used_objects.remove(self.objects["hand"])
        except:
            pass

        # Find all container roots
        in_props = self.state.get_propositions(propositions.In)
        children = set(prop.obj for prop in in_props) - locations
        roots = used_objects - children

        # Place doors
        doors = [obj for obj in used_objects if isinstance(obj, objects.Door)]
        for door in doors:
            loc_candidates = set(self.locations)
            locs = []
            if door in children:
                # Add existing locations
                in_door_props = [prop for prop in in_props if prop.obj is door]
                for prop in in_door_props:
                    locs.append(prop.container)

                # TODO: Handle case where door is assigned to more than 2 locations
                assert len(locs) <= 2

            # Generate random locations
            loc_candidates = set(self.locations) - set(locs)
            locs += random.sample(loc_candidates, 2 - len(locs))
            loc_a, loc_b = locs

            # Add propositions
            self.state.add(f"connects({door}, {loc_a}, {loc_b})")
            self.state.add(f"connects({door}, {loc_b}, {loc_a})")
            self.state.add(f"in({door}, {loc_a})")
            self.state.add(f"in({door}, {loc_b})")
        roots -= set(doors)

        # Place all roots
        for root in roots:
            loc = random.choice(self.locations)
            self.state.add(f"in({root}, {loc})")

    def compute_occupancy(self, exclude=[], padding=0):
        """Compute occupancy grid with (i,j) indexing.

        Args:
            exclude (list(objects.Object), optional): Exclude objects from occupancy grid.
            padding (int): Padding around the objects (excludes rooms).
        """
        self._occupancy = np.zeros(DIM_GRID + 1, dtype=bool)
        for obj in self.objects.values():
            if obj in exclude:
                continue
            self._occupancy |= obj.compute_occupancy(padding=padding)

    def is_occupied(self, pos):
        """Find whether grid position is occupied.

        Args:
            pos (tuple(int, int)): Grid position.

        Returns:
            bool: Whether grid position is occupied.
        """
        if pos[0] < 0 or pos[0] >= self._occupancy.shape[1]:
            return False
        if pos[1] < 1 or pos[1] >= self._occupancy.shape[0]:
            return False
        return self._occupancy[pos[1], pos[0]]

    def nearest_unoccupied(self, pos):
        """Find nearest unoccupied position to given position.

        Args:
            pos (tuple(int, int)): Starting position.

        Returns:
            tuple(int, int): Nearest unoccupied position.
        """
        directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
        random.shuffle(directions)
        directions.insert(0, (0, 0))
        for direction in directions:
            # Convert (x,y) coordinates to (i,j)
            pos_test = (pos[1] + direction[0], pos[0] + direction[1])
            if not self.occupancy[pos_test]:
                # Convert (i,j) coordinates to (x,y)
                return (pos_test[1], pos_test[0])

    def nearest_in_room(self, pos_start, loc):
        """Find nearest position in room to given position.

        Args:
            pos (tuple(int, int)): Starting position.
            loc (objects.Location): Room location.

        Returns:
            tuple(int, int): Nearest unoccupied position.
        """
        interior = loc.compute_interior()
        dist = 1
        while dist < DIM_GRID[0]:
            directions = [(0, dist), (dist, 0), (-dist, 0), (0, -dist)]
            random.shuffle(directions)
            for direction in directions:
                pos = (pos_start[0] + direction[0], pos_start[1] + direction[1])
                if (
                    pos[0] < 0
                    or pos[1] < 0
                    or pos[0] > DIM_GRID[0]
                    or pos[1] > DIM_GRID[1]
                ):
                    continue

                if interior[pos[1], pos[0]]:
                    return pos
            dist += 1

    def a_star(self, pos_start, pos_goal):
        """Find second to last position in shortest path between the start and
        goal position.

        Args:
            pos_start (tuple(int, int)): Start position.
            pos_goal (tuple(int, int)): Goal position.

        Returns:
            tuple(int, int): Last position before the goal.
        """
        import queue

        # Convert (x,y) coordinates to (i,j)
        start = (pos_start[1], pos_start[0])
        goal = (pos_goal[1], pos_goal[0])

        def h(node):
            return np.linalg.norm(np.array(goal) - start)

        def neighbors(node):
            node = np.array(node)
            nodes = []
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            random.shuffle(directions)
            for direction in directions:
                neighbor = tuple(node + direction)
                if not self.occupancy[neighbor] or neighbor == goal:
                    nodes.append(neighbor)
            return nodes

        frontier = queue.PriorityQueue()
        frontier.put((h(start), start))
        parent = {}
        cost = np.full(DIM_GRID + 1, float("inf"))
        cost[start] = 0.0

        while not frontier.empty():
            score, node = frontier.get()
            if node == goal:
                # Convert (i,j) coordinates to (x,y)
                return (parent[node][1], parent[node][0])

            for neighbor in neighbors(node):
                cost_to_neighbor = cost[node] + 1
                if cost_to_neighbor >= cost[neighbor]:
                    continue

                cost[neighbor] = cost_to_neighbor
                parent[neighbor] = node
                frontier.put((cost_to_neighbor + h(neighbor), neighbor))

        return None

    def render(self) -> np.ndarray:
        """Renders the world environment.

        Returns:
            (img [220, 220, 3] (H/W/rgb) uint8 array,
             bounding boxes <obj name> -> [4,] (x1/y1/x2/y2) float32 array).
        """
        self._canvas.fill(COLOR_BG)
        for child in self._children:
            child.render(self._canvas)
        img = pygame.surfarray.array3d(self._canvas)
        return np.swapaxes(img, 0, 1)

    def get_bounding_boxes(self) -> np.ndarray:
        """Extracts bounding boxes of objects.

        Returns:
            [O, 4] (idx_object, x1/y1/x2/y2)
        """
        obj_indices = {}
        for idx_obj, obj in enumerate(self.pddl.objects):
            obj_indices[obj.name] = idx_obj

        O = len(self.pddl.objects)
        boxes = np.full((O, 4), np.nan, dtype=np.float32)
        for obj_name, obj in self.objects.items():
            idx_obj = obj_indices[obj_name]
            if obj.pos is None:
                continue
            boxes[idx_obj][:2] = obj.xy_min
            boxes[idx_obj][2:] = obj.xy_max
            if isinstance(obj, objects.Location):
                boxes[idx_obj][2:] += RES_GRID

        return boxes
