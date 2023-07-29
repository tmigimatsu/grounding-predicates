"""
Gridworld Objects
=================

.. rubric:: Base classes

* :class:`Object`
* :class:`Movable`
* :class:`Container`
* :class:`Openable`
* :class:`Lockable`

.. rubric:: Object classes

* :class:`Lock`
* :class:`Door`
* :class:`Location`
* :class:`Trophy`
* :class:`Chest`
* :class:`Key`
* :class:`Agent`
* :class:`Hand`
"""
import functools
import random
import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

try:
    import pygame
except:
    pass
import numpy as np
import symbolic

from .constants import COLOR_BG, DIM_GRID, RES_GRID
from . import propositions


class ObjectPlacementError(Exception):
    pass


@functools.total_ordering
class Object:
    """Object base class.

    Args:
        world (symbolic.World): World object.
        obj (symbolic.Object): Symbolic object.
        pos (tuple(int, int), optional): Top-left position of object in the grid world.
        size (tuple(int, int), optional): Size of object in the grid world.
    """

    def __init__(self, world, obj, pos=None, size=(1, 1)):
        self._world = world
        self._obj = obj
        if pos is None:
            self._pos = None
        else:
            self._pos = np.array(pos)
        self._size = np.array(size)

    def __repr__(self):
        """Object name."""
        return str(self.obj)

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, other):
        """Compare object name."""
        return repr(self) == repr(other)

    def __lt__(self, other):
        """Compare object name."""
        return repr(self) < repr(other)

    @property
    def world(self):
        """World object.

        :type: symbolic.World
        """
        return self._world

    @property
    def obj(self):
        """Symbolic object.

        :type: symbolic.Object
        """
        return self._obj

    @property
    def pos(self):
        """Grid position.

        :type: numpy.ndarray, (2,)
        """
        return self._pos

    @pos.setter
    def pos(self, pos):
        self.world.log(f"  . Setting position of {self} to {pos}")
        if pos is None:
            self._pos = None
        else:
            self._pos = np.array(pos)

    @property
    def size(self):
        """Grid size.

        :type: numpy.ndarray, (2,)
        """
        return self._size

    @property
    def xy_min(self):
        """Pixel position of top-left corner.

        :type: numpy.ndarray, (2,)
        """
        return self.pos * RES_GRID

    @property
    def xy_max(self):
        """Pixel position of bottom-right corner.

        :type: numpy.ndarray, (2,)
        """
        return (self.pos + self.size) * RES_GRID

    @property
    def xy_center(self):
        """Pixel position of center.

        :type: numpy.ndarray, (2,)
        """
        return 0.5 * (self.xy_min + self.xy_max)

    @property
    def dim(self):
        """Pixel size of object.

        :type: numpy.ndarray, (2,)
        """
        return self.xy_max - self.xy_min + RES_GRID

    def is_subtype(self, type):
        """Check if object is instance of given type.

        Args:
            type (str): Type string.
        Returns:
            (bool): Whether object is instance of given type.
        """
        return self.obj.type.is_subtype(type)

    def compute_occupancy(self, padding=0):
        """Compute occupancy grid with (i,j) indexing.

        Args:
            padding (int): Padding around the object by grid distance.
        Returns:
            numpy.ndarray, (``DIM_GRID+1``, ``DIM_GRID+1``): Occupancy grid.

        Example:
            >>> from env.gridworld import objects
            >>> world = None    # Properly initialize world.World
            >>> obj = 'trophy'  # Properly initialize symbolic.Object
            >>> obj_config = {
            ...     'object': 'Trophy',
            ...     'args': {
            ...         'pos': [8, 8],
            ...         'color': [255, 200, 0],
            ...     },
            ... }
            >>> trophy = objects.create(world, obj, obj_config)
            >>> trophy.compute_occupancy(padding=2)
            array([[False, False, False, False, False, False, False, False, False,
                    False, False],
                   [False, False, False, False, False, False, False, False, False,
                    False, False],
                   [False, False, False, False, False, False, False, False, False,
                    False, False],
                   [False, False, False, False, False, False, False, False, False,
                    False, False],
                   [False, False, False, False, False, False, False, False, False,
                    False, False],
                   [False, False, False, False, False, False, False, False, False,
                    False, False],
                   [False, False, False, False, False, False, False, False,  True,
                    False, False],
                   [False, False, False, False, False, False, False,  True,  True,
                     True, False],
                   [False, False, False, False, False, False,  True,  True,  True,
                     True,  True],
                   [False, False, False, False, False, False, False,  True,  True,
                     True, False],
                   [False, False, False, False, False, False, False, False,  True,
                    False, False]])
            """
        occupancy = np.zeros(DIM_GRID + 1, dtype=bool)
        if self.pos is None:
            return occupancy

        # Create diamond shape
        for i in range(padding + 1):
            p = np.array((i, padding - i))
            left, top = self.pos - p
            right, bottom = self.pos + self.size + p
            occupancy[top:bottom, left:right] = True
        return occupancy

    def render(self, screen):
        """Virtual render method.

        Args:
            screen (pygame.Surface): Pygame screen.
        """
        pass


class Container(Object):
    """Container base class.

    Args:
        world (symbolic.World): World object.
        obj (symbolic.Object): Symbolic object.
        pos (tuple(int, int), optional): Top-left position of object in the grid world.
        size (tuple(int, int), optional): Size of object in the grid world.
    """

    def __init__(self, world, obj, pos=None, size=(1, 1)):
        super().__init__(world, obj, pos, size)
        self._children = []

    @property
    def children(self):
        """Children objects of container.

        :type: list(objects.Object)
        """
        return self._children

    @Object.pos.setter
    def pos(self, pos):
        """Set the position of all children."""
        Object.pos.fset(self, pos)
        for child in self.children:
            child.pos = pos

    def add_child(self, obj):
        """Add object to container and set its position to the container's position.

        Args:
            obj (objects.Object): Child object.
        """
        # Return if obj is already in its container
        if obj in self.children:
            return

        # Add obj to container's children
        self.children.append(obj)
        obj.pos = self.pos

    def remove_child(self, obj):
        """Remove the object from the container.

        Args:
            obj (objects.Object): Child object.
        """
        self.children.remove(obj)


class Movable(Object):
    """Movable base class.

    Args:
        world (symbolic.World): World object.
        obj (symbolic.Object): Symbolic object.
        pos (tuple(int, int)): Top-left position of object in grid-world.
    """

    def __init__(self, world, obj, pos):
        super().__init__(world, obj, pos)


class Openable(Container):
    """Openable base class.

    Args:
        world (symbolic.World): World object.
        obj (symbolic.Object): Symbolic object.
        pos (tuple(int, int)): Top-left position of object in grid-world.

    Attributes:
        is_closed (bool): Whether object is closed.
    """

    def __init__(self, world, obj, pos):
        super().__init__(world, obj, pos)
        self.is_closed = False

    def render(self, screen):
        """Render children if object is open.

        Args:
            screen (pygame.Surface): Pygame screen.
        """
        # if self.is_closed:
        #     # Render children only if object is not closed
        #     return

        for child in self.children:
            child.render(screen)


class Lock(Object):
    """Lock object for :class:`Lockable` types.

    Args:
        parent (symdb.Lockable): Parent object.
        color (tuple(int, int, int)): RGB color of object.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
    """

    def __init__(self, parent, color):
        self._parent = parent
        self.color = color

    @property
    def obj(self):
        """Name of parent + '_lock'.

        :type: str
        """
        return f"{self._parent.obj}_lock"

    @property
    def pos(self):
        """Grid position of parent + (0.1, 0.1).

        :type: numpy.ndarray, (2,)
        """
        return self._parent.pos + np.array([0.1, 0.1])

    @property
    def size(self):
        """Grid size = (0.8, 0.8).

        :type: numpy.ndarray, (2,)
        """
        return np.array([0.8, 0.8])

    def render(self, screen):
        """Render X mark.

        Args:
            screen (pygame.Surface): Pygame screen.
        """
        pygame.draw.line(
            screen, self.color, self.xy_min, self.xy_max, 4,
        )
        pygame.draw.line(
            screen,
            self.color,
            (self.xy_min[0], self.xy_max[1]),
            (self.xy_max[0], self.xy_min[1]),
            4,
        )


class Lockable(Openable):
    """Lockable base class.

    Args:
        world (symbolic.World): World object.
        obj (symbolic.Object): Symbolic object.
        pos (tuple(int, int)): Top-left position of object in grid-world.
        lock (dict): Lock config.

    Attributes:
        is_locked (bool): Whether object is locked.
    """

    def __init__(self, world, obj, pos, lock):
        super().__init__(world, obj, pos)
        self.is_locked = False
        self._lock = OBJECTS[lock["object"]](self, **lock["args"])

    def render(self, screen):
        """Render lock if locked.

        Args:
            screen (pygame.Surface): Pygame screen.
        """
        super().render(screen)
        if self.is_locked:
            self._lock.render(screen)


class Door(Lockable):
    """Door object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        lock (dict): Lock config.
        color (tuple(int, int, int)): RGB color of object.
        pos (tuple(int, int), optional): Top-left position of object in grid-world.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
        size (np.ndarray, (2,)): (1, 1)
    """

    def __init__(self, world, obj, lock, color, pos=None):
        super().__init__(world, obj, pos, lock)
        self.color = color
        self._locations = []

    @property
    def locations(self):
        """Locations that the door connects.

        :type: tuple(objects.Location, objects.Location)
        """
        return self._locations

    def register_connection(self, prop):
        """Register connection between this door and its locations and randomly
        place door along the wall between the rooms.

        Args:
            prop (propositions.Connects): Connection proposition.
        """

        # Check if connection has already been registered
        new_locations = sorted(prop.locations)
        if new_locations == self._locations:
            return
        self._locations = new_locations
        loc_a, loc_b = self._locations

        # Compute wall between rooms
        wall = loc_a.compute_occupancy() & loc_b.compute_occupancy()
        wall_i, wall_j = np.nonzero(wall)

        # Compute occupancy without door and rooms
        self.world.compute_occupancy(exclude=[self, loc_a, loc_b], padding=2)

        # Put the door randomly along the wall
        indices = list(range(wall_i.size))
        random.shuffle(indices)
        is_placed = False
        for idx in indices:
            pos = (wall_j[idx], wall_i[idx])
            if (
                self.world.is_occupied(pos)
                or self.world.nearest_in_room(pos, loc_a) is None
                or self.world.nearest_in_room(pos, loc_b) is None
            ):
                # Position is occupied or doesn't lead to a valid room opening
                continue

            self.pos = pos
            is_placed = True
            break

        if not is_placed:
            raise ObjectPlacementError(f"Unable to place door for {prop}.")

    def render(self, screen):
        """Render square if closed, or empty square if open.

        Args:
            screen (pygame.Surface): Pygame screen.
        """
        pygame.draw.rect(screen, self.color, (*self.xy_min, RES_GRID, RES_GRID))
        if not self.is_closed:
            pygame.draw.rect(
                screen,
                COLOR_BG,
                (*(self.xy_min + 0.1 * RES_GRID), 0.8 * RES_GRID, 0.8 * RES_GRID,),
            )
        super().render(screen)


class Location(Container):
    """Location object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        pos (tuple(int, int)): Top-left position of object in grid-world.
        size (tuple(int, int), optional): Size of object in the grid world.
        color (tuple(int, int, int)): RGB color of object.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
    """

    def __init__(self, world, obj, pos, size, color):
        super().__init__(world, obj, pos, size)
        self.color = color
        self._connections = []

    def register_connection(self, prop):
        """Register connection between this location and another location.

        Add the connecting door to this location's list of children.

        Args:
            prop (propositions.Connects): Connection proposition.
        """
        assert self in prop.locations

        # Return if connection is registered already
        for connection in self._connections:
            if prop.door == connection.door:
                return

        self._connections.append(prop)
        if prop.door not in self.children:
            self.children.append(prop.door)

    @property
    def doors(self):
        """Get the doors connected to this location."""
        return [prop.door for prop in self._connections]

    def get_doors_to(self, loc):
        """Get the doors to the given location.

        Returns:
            list(objects.Door): List of doors.
        """
        return [prop.door for prop in self._connections if loc in prop.locations]

    def compute_interior(self):
        """Compute interior grid with (i,j) indexing.

        Returns:
            numpy.ndarray, (``DIM_GRID+1``, ``DIM_GRID+1``): Interior grid.
        """
        interior = np.zeros(DIM_GRID + 1, dtype=bool)
        left, top = self.pos + 1
        right, bottom = self.pos + self.size
        interior[top:bottom, left:right] = True
        return interior

    def compute_occupancy(self, padding=0):
        """Compute occupancy grid with (i,j) indexing.

        Args:
            padding (int): Padding around the object (does nothing for locations).
        Returns:
            numpy.ndarray, (``DIM_GRID+1``, ``DIM_GRID+1``): Occupancy grid.
        """
        occupancy = np.zeros(DIM_GRID + 1, dtype=bool)
        left, top = self.pos
        right, bottom = self.pos + self.size
        occupancy[top, left:right] = True
        occupancy[bottom, left:right] = True
        occupancy[top:bottom, left] = True
        occupancy[top:bottom, right] = True
        return occupancy

    def add_child(self, obj):
        """Add object to container and set its position to the container's position.

        Args:
            obj (objects.Object): Child object.
        """
        # Return if obj is already in its container
        if obj in self.children:
            return

        # Add obj to container's children
        self.children.append(obj)

        # Move agent to door position or place randomly place in this location.
        if isinstance(obj, Agent):
            obj.location = self
            return

        # Place object randomly in this location
        if obj.pos is None:
            self.world.compute_occupancy(exclude=[obj], padding=2)
            interior = self.compute_interior()
            interior_i, interior_j = np.nonzero(interior)
            indices = list(range(interior_i.size))
            random.shuffle(indices)
            is_placed = False
            for idx in indices:
                pos = (interior_j[idx], interior_i[idx])
                if not self.world.is_occupied(pos):
                    self.world.log(f"  . Random set obj position in {self}")
                    obj.pos = pos
                    is_placed = True
                    break
            if not is_placed:
                raise ObjectPlacementError(f"Unable to place {obj} in {self}.")

    def render(self, screen):
        """Render location rectangle and all its children.

        Args:
            screen (pygame.Surface): Pygame screen.
        """
        pygame.draw.rect(screen, self.color, (*self.xy_min, RES_GRID, self.dim[1]))
        pygame.draw.rect(screen, self.color, (*self.xy_min, self.dim[0], RES_GRID))
        pygame.draw.rect(
            screen, self.color, (self.xy_max[0], self.xy_min[1], RES_GRID, self.dim[1])
        )
        pygame.draw.rect(
            screen, self.color, (self.xy_min[0], self.xy_max[1], self.dim[0], RES_GRID)
        )

        for child in self.children:
            child.render(screen)


class Trophy(Movable):
    """Trophy object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        color (tuple(int, int, int)): RGB color of object.
        pos (tuple(int, int), optional): Top-left position of object in grid-world.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
        size (np.ndarray, (2,)): (1, 1)
    """

    def __init__(self, world, obj, color, pos=None):
        super().__init__(world, obj, pos)
        self.color = color

    def render(self, screen):
        from .utils import star

        pygame.draw.polygon(
            screen, self.color, star(5, self.xy_center, RES_GRID // 2, RES_GRID // 5),
        )
        super().render(screen)


class Chest(Lockable):
    """Chest object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        lock (dict): Lock config.
        color (tuple(int, int, int)): RGB color of object.
        pos (tuple(int, int), optional): Top-left position of object in grid-world.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
        size (np.ndarray, (2,)): (1, 1)
    """

    def __init__(self, world, obj, lock, color, pos=None):
        super().__init__(world, obj, pos, lock)
        self.color = color

    def render(self, screen):
        pygame.draw.rect(screen, self.color, (*self.xy_min, RES_GRID, RES_GRID))
        if not self.is_closed:
            inner = (*(self.xy_min + 0.1 * RES_GRID), 0.8 * RES_GRID, 0.8 * RES_GRID)
            pygame.draw.rect(screen, COLOR_BG, inner)
        super().render(screen)


class Key(Movable):
    """Key object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        color (tuple(int, int, int), optional): RGB color of object.
        pos (tuple(int, int), optional): Top-left position of object in grid-world.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
        size (np.ndarray, (2,)): (1, 1)
    """

    def __init__(self, world, obj, color=None, pos=None):
        super().__init__(world, obj, pos)
        self.color = color
        self.offset = 0.0
        lockable = obj.name.split("_")[0]
        # self.offset = np.pi if obj.name == "chest_key" else 0.0

    def render(self, screen):
        from .utils import polygon

        if self.color is None:
            raise ValueError(
                f"{self} has no color. Color is set by the lockable object."
            )
        pygame.draw.polygon(
            screen, self.color, polygon(3, self.xy_center, RES_GRID // 2, self.offset),
        )
        super().render(screen)


class Agent(Object):
    """Agent object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        color (tuple(int, int, int)): RGB color of object.
        pos (tuple(int, int), optional): Top-left position of object in grid-world.

    Attributes:
        color (tuple(int, int, int)): RGB color of object.
        hand (Hand): Agent's hand (linked in Hand's constructor).
    """

    def __init__(self, world, obj, color, pos=None):
        super().__init__(world, obj, pos)
        self.color = color
        self.hand = None
        self._location = None

    @Object.pos.setter
    def pos(self, pos):
        Object.pos.fset(self, pos)
        if self.hand is not None:
            for child in self.hand.children:
                child.pos = pos

    @property
    def location(self):
        """Agent's location.

        :type: objects.Location
        """
        return self._location

    @location.setter
    def location(self, location):
        def is_free(pos, interior):
            """Determine whether position is free and does not have any
            reachable objects."""

            # Return false if spot is occupied
            if self.world.is_occupied(pos):
                return False

            # Return false if any neighboring spots are occupied
            pos = np.array(pos)
            directions = [(0, 1), (1, 0), (-1, 0), (0, -1)]
            for direction in directions:
                nbor = pos + direction
                if interior[nbor[1], nbor[0]] and self.world.is_occupied(nbor):
                    return False

            return True

        loc_prev = self.location
        self._location = location

        # Move the agent to the room at the position nearest to the door
        if loc_prev is not None:
            # TODO: Handle case with multiple doors
            pos_door = loc_prev.get_doors_to(location)[0].pos

            self.world.log(f"  . Set agent position in {self}")
            self.pos = self.world.nearest_in_room(pos_door, location)
            return

        # Put the agent randomly in the location
        if self.pos is None:
            self.world.compute_occupancy(exclude=[self], padding=2)
            interior = location.compute_interior()
            room_i, room_j = np.nonzero(interior)
            for i in range(len(room_i)):
                idx = np.random.randint(room_i.size)
                pos = (room_j[idx], room_i[idx])
                if not self.world.is_occupied(pos):
                    self.world.log(f"  . Random set obj position in {self}")
                    self.pos = pos
                    return

            # Could not find any free spots
            raise propositions.PropositionValueError(
                f"No free space for agent in {location}: \n{self.world.occupancy}."
            )

    def goto(self, obj):
        """Go to object.

        Args:
            obj (objects.Object):
        """
        if obj.is_subtype("location"):
            raise ArgumentTypeError("Agent.goto({obj}): cannot goto a location.")
        if (obj.pos == self.pos).all():
            raise propositions.PropositionValueError(
                f"Agent.goto({obj}): obj and agent have the same position {self.pos}"
            )

        # Move the agent to the object
        self.world.compute_occupancy(exclude=[self])
        pos_target = self.world.a_star(self.pos, obj.pos)
        if pos_target is None:
            raise ValueError(f"No valid path to target '{obj}'.")
        self.world.log(f"  . Set agent position in agent.goto()")
        self.pos = pos_target

    def render(self, screen):
        pygame.draw.circle(
            screen, self.color, self.xy_center.astype(int), RES_GRID // 2
        )
        if self.hand is not None:
            for child in self.hand.children:
                child.render(screen)
        super().render(screen)


class Hand(Container):
    """Hand object.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        agent (Agent): Agent to which hand is attached.

    Attributes:
        agent (Agent): Agent to which hand is attached.
    """

    def __init__(self, world, obj, agent):
        super().__init__(world, obj)
        if agent in world.objects:
            self.agent = world.objects[agent]
            self.agent.hand = self
        else:
            # If agent hasn't been created yet, store agent's name so we can
            # retrieve it later.
            self.agent = None
            self._name_agent = agent

    def _try_update_agent(self):
        if self.agent is None:
            # Attach agent if not yet attached.
            self.agent = self.world.objects[self._name_agent]
            self.agent.hand = self

    @property
    def pos(self):
        self._try_update_agent()
        return self.agent.pos

    @pos.setter
    def pos(self, pos):
        self.world.log(f"  . Setting position of {self} to {pos}")
        self._try_update_agent()
        self.agent.pos = pos

    @property
    def size(self):
        self._try_update_agent()
        return self.agent.size

    def remove_child(self, obj):
        """Remove the object from the hand and place it in the room.

        Args:
            obj (objects.Object): Child object.
        """
        super().remove_child(obj)
        self.world.compute_occupancy(exclude=[obj])
        self.world.log(f"  . Set obj position in Hand.remove_child()")
        obj.pos = self.world.nearest_unoccupied(self.pos)


OBJECTS = {
    "Agent": Agent,
    "Chest": Chest,
    "Door": Door,
    "Hand": Hand,
    "Key": Key,
    "Location": Location,
    "Lock": Lock,
    "Trophy": Trophy,
}
"""Map from object type strings to object types."""


def create(world, obj, obj_config):
    """Create object from config.

    Args:
        world (world.World): World environment.
        obj (symbolic.Object): Symbolic object.
        obj_config (dict): Object config.

    Example:
        >>> from env.gridworld import objects
        >>> world = None    # Dummy world.World
        >>> obj = 'trophy'  # Dummy symbolic.Object
        >>> obj_config = {
        ...     'object': 'Trophy',
        ...     'args': {
        ...         'pos': [8, 8],
        ...         'color': [255, 200, 0],
        ...     },
        ... }
        >>> objects.create(world, obj, obj_config)
        trophy
    """

    Object = OBJECTS[obj_config["object"]]
    if "args" not in obj_config or obj_config["args"] is None:
        return Object(world, obj)
    return Object(world, obj, **obj_config["args"])
