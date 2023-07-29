import functools
import typing

import numpy as np

import symbolic
from . import objects


class ArgumentTypeError(Exception):
    pass


class PropositionValueError(Exception):
    pass


@functools.total_ordering
class Proposition:
    """Proposition base class.

    Args:
        name (str): Predicate name.
        *args (objects.Object): Proposition arguments.
    """

    is_static = False

    def __init__(self, name, *args):
        self.name = name
        self.args = args
        self._world = self.args[0].world

    def __eq__(self, other):
        """Compare priority, predicate name, and arguments."""
        return (
            self.priority == other.priority
            and self.name == other.name
            and self.args == other.args
        )

    def __lt__(self, other):
        """Compare priority, predicate name, and arguments."""
        a = (self.priority, self.name, self.args)
        b = (other.priority, other.name, other.args)
        return a < b

    def __repr__(self):
        return "{}({})".format(self.name, ", ".join([str(arg) for arg in self.args]))

    def __hash__(self):
        return hash(repr(self))

    @property
    def priority(self):
        """Priority of proposition, where 0 is lowest.

        Higher priority propositions get applied first. Base proposition has
        priority 0.

        :type: int
        """
        return 0

    @property
    def world(self):
        """World object.

        :type: symbolic.World
        """
        return self._world

    def apply(self):
        """Virtual apply method."""
        self.args[0].world.log(f"+ Applying {self}")

    def negate(self):
        """Virtual negate method."""
        self.args[0].world.log(f"- Negating {self}")


class Matches(Proposition):
    """Proposition that indicates a match between a lockable object and its key.

    Args:
        a (objects.Lockable or objects.Key): First argument.
        b (objects.Lockable or objects.Key): Second argument.
    """

    name_predicate = "matches"
    is_static = True

    def __init__(self, a, b):
        super().__init__(Matches.name_predicate, a, b)
        if isinstance(a, objects.Key) and isinstance(b, objects.Lockable):
            self._key = a
            self._lockable = b
            if not str(self.lockable) in str(self.key):
                raise PropositionValueError(f"{self} not applicable due to mismatching arguments.")
        # elif isinstance(a, objects.Lockable) and isinstance(b, objects.Key):
        #     self._key = b
        #     self._lockable = a
        else:
            raise ArgumentTypeError(f"{self} requires one Key and one Lockable type.")

    @property
    def key(self):
        """Key object.

        :type: objects.Key
        """
        return self._key

    @property
    def lockable(self):
        """Lockable object.

        :type: objects.Lockable
        """
        return self._lockable

    @property
    def priority(self):
        """Priority 4.

        :type: int
        """
        return 4

    def apply(self):
        """Change the color of the key to match the locker."""
        super().apply()

        # Check to make sure proposition doesn't conflict with another
        # matches_props = self.world.state.get_propositions(Matches)
        # matches_props = [prop for prop in matches_props if prop.key == self.key]
        # for prop in matches_props:
        #     if prop.lockable != self.lockable:
        #         raise PropositionValueError(f"{self} conflicts with {prop}.")
        # r, g, b = self.lockable.color
        # self.key.color = (int(0.8 * r), int(0.8 * g), int(0.8 * b))

        # self.world.state.add(f"matches({self.args[1]}, {self.args[0]})", apply=False)


class Connects(Proposition):
    """Proposition that indicates door *a* connects locations *b* and *c*.

    Updates each location's :py:attr:`env.gridworld.objects.Location.doors` map
    with the other location and corresponding door.

    Args:
        door (objects.Door): Door.
        loc_a (objects.Location): First location.
        loc_b (objects.Location): Second location.
    """

    name_predicate = "connects"
    is_static = True

    def __init__(self, door, loc_a, loc_b):
        super().__init__(Connects.name_predicate, door, loc_a, loc_b)
        if not isinstance(door, objects.Door):
            raise ArgumentTypeError(f"{door} in {self} must be a Door type.")
        if not loc_a.is_subtype("location"):
            raise ArgumentTypeError(f"{loc_a} in {self} must be a Location type.")
        if not loc_b.is_subtype("location"):
            raise ArgumentTypeError(f"{loc_b} in {self} must be a Location type.")
        if loc_a == loc_b:
            raise ArgumentTypeError(f"Locations must be different in {self}.")

        self._door = door
        self._locations = (loc_a, loc_b)

    @property
    def door(self):
        """Door object.

        :type: objects.Door
        """
        return self._door

    @property
    def locations(self):
        """Locations *a* and *b*.

        :type: tuple(objects.Location, objects.Location)
        """
        return self._locations

    @property
    def priority(self):
        """Priority 4.

        :type: int
        """
        return 4

    def apply(self):
        """Register the door and locations with each other and randomly place
        the door along the wall between the rooms."""
        super().apply()
        self.door.register_connection(self)
        loc_a, loc_b = self.locations
        loc_a.register_connection(self)
        loc_b.register_connection(self)


class Closed(Proposition):
    """Proposition that indicates object *a* is closed.

    Args:
        obj (objects.Openable): Openable.
    """

    name_predicate = "closed"
    is_static = False

    def __init__(self, obj):
        super().__init__(Closed.name_predicate, obj)
        if not isinstance(obj, objects.Openable):
            raise ArgumentTypeError(f"{obj} in {self} must be an Openable type.")

        self._obj = obj

    @property
    def obj(self):
        """Closed object.

        :type: objects.Openable
        """
        return self._obj

    @property
    def priority(self):
        """Priority 1.

        :type: int
        """
        return 1

    def apply(self):
        """Sets object's :py:attr:`env.gridworld.objects.Openable.is_closed`
        attribute to ``True``."""
        super().apply()
        self.obj.is_closed = True

    def negate(self):
        """Sets object's :py:attr:`env.gridworld.objects.Openable.is_closed`
        attribute to ``False``."""
        super().negate()
        self.obj.is_closed = False


class Locked(Proposition):
    """Proposition that indicates object *a* is locked.

    Args:
        obj (objects.Lockable): Lockable.
    """

    name_predicate = "locked"
    is_static = False

    def __init__(self, obj):
        super().__init__(Locked.name_predicate, obj)
        if not isinstance(obj, objects.Lockable):
            raise ArgumentTypeError(f"{obj} in locked({obj}) must be a Lockable type.")

        self._obj = obj

    @property
    def obj(self):
        """Locked object.

        :type: objects.Lockable
        """
        return self._obj

    def apply(self):
        """Sets object's :py:attr:`env.gridworld.objects.Lockable.is_locked`
        attribute to ``True``."""
        super().apply()

        # Check to make sure locked object is closed
        world = self.obj.world
        is_closed = False
        for prop in world.state.get_propositions(Closed):
            if prop.obj == self.obj:
                is_closed = True
                break
        if not is_closed:
            raise PropositionValueError(
                f"{self} can only be applied if closed({self.obj})."
            )

        self.obj.is_locked = True

    def negate(self):
        """Sets object's :py:attr:`env.gridworld.objects.Lockable.is_locked`
        attribute to ``False``."""
        super().negate()
        self.obj.is_locked = False


class In(Proposition):
    """Proposition that indicates object is in a container.

    Args:
        obj (objects.Object): Contained object.
        container (objects.Container): Container object.
    """

    name_predicate = "in"
    is_static = False

    def __init__(self, obj, container):
        super().__init__(In.name_predicate, obj, container)
        if isinstance(obj, (objects.Location, objects.Hand)):
            raise ArgumentTypeError(f"Cannot apply {self}.")
        if not obj.is_subtype("movable") and not container.is_subtype("location"):
            raise ArgumentTypeError(f"Cannot apply {self}.")
        if isinstance(container, objects.Door):
            raise ArgumentTypeError(f"Cannot apply {self}.")
        if not isinstance(container, objects.Container):
            raise ArgumentTypeError(f"{container} in {self} must be a Container type.")
        if obj == container:
            raise ArgumentTypeError(f"{self} must contain two different arguments.")

        self._obj = obj
        self._container = container

    @property
    def priority(self):
        """Priority 3.

        :type: int
        """
        return 3

    @property
    def obj(self):
        """Contained object.

        :type: objects.Object
        """
        return self._obj

    @property
    def container(self):
        """Container object.

        :type: objects.Object
        """
        return self._container

    def apply(self):
        """Adds object to the container's
        :py:attr:`env.gridworld.objects.Object.children` list.

        Sets the object's position to a random position inside the container, or
        if the container is a :py:attr:`objects.Location`, then move the agent
        into the room at the position next to the door.
        """
        super().apply()

        # Recursive function to find location of object chain, or default if
        # location doesn't exist.
        def find_location(obj, default=None):
            # Object is location
            if obj.is_subtype("location"):
                return obj

            # Find parent
            in_props = self.world.state.get_propositions(In)
            in_props = [prop for prop in in_props if prop.obj == obj]
            assert len(in_props) < 2 or isinstance(obj, objects.Door)

            if not in_props:
                # Nothing was found, so return this object (highest ancestor in tree)
                return default

            prop = in_props[0]
            return find_location(prop.container)

        def make_container_reachable(obj):
            if obj.is_subtype("location") or isinstance(
                obj, (objects.Agent, objects.Hand)
            ):
                return

            self.world.log(f"  + Adding reachable({obj})")
            self.world.state.add(f"reachable({obj})", apply=False)

            # Find parent
            in_props = self.world.state.get_propositions(In)
            in_props = [prop for prop in in_props if prop.obj == obj]
            assert len(in_props) < 2 or isinstance(obj, objects.Door)

            # Nothing was found, so return
            if not in_props:
                return

            # Make container reachable
            prop = in_props[0]
            make_container_reachable(prop.container)

        # Check to make sure proposition doesn't conflict with another
        if not isinstance(self.obj, objects.Door):
            in_props = self.world.state.get_propositions(In)
            in_props = [prop for prop in in_props if prop.obj == self.obj]
            for prop in in_props:
                if prop.container != self.container:
                    raise PropositionValueError(f"{self} conflicts with {prop}.")

            if not isinstance(self.obj, objects.Agent) and not isinstance(
                self.container, objects.Hand
            ):
                reachable_props = self.world.state.get_propositions(Reachable)
                reachable_props = [
                    prop for prop in reachable_props if prop.obj == self.obj
                ]
                if reachable_props:
                    agent = self.world.objects["agent"]
                    loc_agent = agent.location
                    if loc_agent is not None:
                        loc_obj = find_location(self.obj, default=self.container)
                        if loc_obj != loc_agent:
                            raise PropositionValueError(
                                f"{self} conflicts with {reachable_props[0]}."
                            )

        # Fix reachability
        if self.world.state.contains(f"reachable({self.obj})"):
            # If object is reachable, make its containers reachable too
            make_container_reachable(self.container)
        elif isinstance(self.container, objects.Hand):
            self.world.state.add(f"reachable({self.obj})", apply=False)

        self.container.add_child(self.obj)

    def negate(self):
        """Removes object from the container's
        :py:attr:`env.gridworld.objects.Object.children` list.

        If the container is a :py:attr:`objects.Hand`, then move the object to
        the nearest unoccupied space.
        """
        super().negate()
        self.container.remove_child(self.obj)


class Reachable(Proposition):
    """Proposition that indicates the object is reachable.

    Args:
        obj (objects.Object): Reachable object.
    """

    name_predicate = "reachable"
    is_static = False

    def __init__(self, obj):
        super().__init__(Reachable.name_predicate, obj)
        if isinstance(obj, (objects.Location, objects.Hand, objects.Agent)):
            raise ArgumentTypeError(f"Cannot apply {obj} to {self}.")
        self._obj = obj
        self._agent = obj.world.objects["agent"]

    @property
    def obj(self):
        """Reachable object.

        :type: objects.Object
        """
        return self._obj

    @property
    def priority(self):
        """Priority 2.

        :type: int
        """
        return 2

    def apply(self):
        """Moves agent to nearest position near object using A-star."""
        import random

        super().apply()

        # Recursive function to find location of object chain, or the highest
        # ancestor if the location doesn't exist.
        def find_location(obj):
            # Object is location
            if obj.is_subtype("location"):
                return obj

            # Find parent
            in_props = self.world.state.get_propositions(In)
            in_props = [prop for prop in in_props if prop.obj == obj]
            assert len(in_props) < 2 or isinstance(obj, objects.Door)

            if not in_props:
                # Nothing was found, so return this object (highest ancestor in tree)
                return obj

            prop = in_props[0]
            return find_location(prop.container)

        def find_container(obj):
            # Object is location
            if obj.is_subtype("location"):
                return None

            # Find parent
            in_props = self.world.state.get_propositions(In)
            in_props = [prop for prop in in_props if prop.obj == obj]
            assert len(in_props) < 2 or isinstance(obj, objects.Door)

            # Return obj if its container is a location or it has no container
            if not in_props or in_props[0].container.is_subtype("location"):
                return obj

            # Return parent's container
            return find_container(in_props[0].container)

        # Check to make sure there are no other reachable objects
        # TODO: Handle multiple reachables
        reachable_props = self.world.state.get_propositions(Reachable)
        reachable_props = [prop for prop in reachable_props if prop != self]
        for prop in reachable_props:
            container_prop = find_container(prop.obj)
            container = find_container(self.obj)

            # If one object is in hand, then both can be reachable
            if self.world.objects["hand"] in (container_prop, container):
                continue

            # If containers are equal, then both can be reachable
            if container == container_prop and self.world.state.contains(
                f"reachable({container})"
            ):
                continue

            # Find the location of the newly reachable object
            if self.obj.pos is not None:
                # If the newly reachable object is already reachable, then both can be reachable.
                if np.abs(self._agent.pos - self.obj.pos).sum() <= 1:
                    continue

            self.world.log(f"  . CONTAINER: {container} {container_prop}")
            raise PropositionValueError(f"{self} conflicts with {prop}")

        # If object is in hand, ignore
        if self.world.state.contains(f"in({self.obj}, {self.world.objects['hand']})"):
            return

        # If agent has a location, go to the object
        if self._agent.location is not None:
            loc = self._agent.location
            if self.obj.pos is None:
                self.world.state.add(f"in({self.obj}, {loc})")
            else:
                self._agent.goto(self.obj)
            return

        # Find the location of the reachable object
        loc = find_location(self.obj)

        # If no location was found, assign highest ancestor to one
        ancestor = self.obj
        if not loc.is_subtype("location"):
            ancestor = loc
            loc = random.choice(self.world.locations)
            self.world.state.add(f"in({ancestor}, {loc})")
            # loc.children.append(ancestor)

        # Add agent to location
        self.world.state.add(f"in({self._agent}, {loc})")

        if self.obj.pos is not None:
            self._agent.goto(self.obj)
            return

        # Object doesn't have a position, so move the object
        self.world.log("  . Move object")
        self.world.compute_occupancy(exclude=[self._agent])
        self.world.log(f"  . Set obj position in {self}")
        ancestor.pos = self.world.nearest_unoccupied(self._agent.pos)

    # TODO: Enforce not reachable


PROPOSITIONS: typing.Dict[str, typing.Type[Proposition]] = {
    "matches": Matches,  # 4
    "connects": Connects,  # 4
    "closed": Closed,  # 1
    "locked": Locked,  # 0
    "in": In,  # 3
    "reachable": Reachable,  # 2
    "reachable_in": Reachable,  # 2
}
"""Map from predicate names to Proposition types."""


def parse_head(str_prop):
    import re

    matches = re.match("([^\(]*)\([^\)]*", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse proposition from '{str_prop}'.")
    name_pred = matches.group(1)
    return name_pred


def parse_args(str_prop):
    import re

    matches = re.match("[^\(]*\(([^\)]*)", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse objects from '{str_prop}'.")
    str_args = matches.group(1).replace(" ", "").split(",")
    return str_args


def alias(str_prop):
    """Returns proposition with its normalized predicate name.

    Args:
        str_prop (str): Proposition string.
    Returns:
        (str): Normalized proposition string.

    Example:
        >>> from env.gridworld import propositions
        >>> propositions.alias('reachable_in(trophy)')
        'reachable(trophy)'
    """
    import re

    matches = re.match("([^\(]*)\(([^\)]*)", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse proposition from '{str_prop}'.")
    name_pred = matches.group(1)
    try:
        Pred = PROPOSITIONS[name_pred]
    except:
        raise ValueError(f"Unsupported predicate name in '{str_prop}'.")
    name_pred = Pred.name_predicate
    str_args = matches.group(2)

    return f"{name_pred}({str_args})"


def is_static(str_prop: str) -> bool:
    """Returns whether the proposition is static (unmutable)."""
    str_pred = parse_head(str_prop)
    return PROPOSITIONS[str_pred].is_static


def is_valid(pddl: symbolic.Pddl, str_prop: str) -> bool:
    """Returns whether the proposition is valid."""

    def create_world(pddl, world_config="../env/gridworld/config.yaml"):
        import os

        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        import yaml

        import pygame

        pygame.init()

        from env.gridworld.world import World

        # Create world for checking proposition validity
        with open(world_config) as f:
            config = yaml.full_load(f)

        return World(pddl, config)

    world = create_world(pddl)

    from gpred import dnf_utils

    args = dnf_utils.parse_args(str_prop)

    # Make sure there are no duplicate args
    if len(set(args)) != len(args):
        return False

    try:
        prop = create(world, str_prop)
    except:
        return False
    return True


def create(world, str_prop):
    """Create proposition from string.

    Args:
        world (world.World): World environment.
        str_prop (str): Proposition string.
    Returns:
        (propositions.Proposition): Proposition object.

    Example:
        >>> from env.gridworld import objects, propositions
        >>> class World:  # Dummy World class
        ...     def __init__(self):
        ...         self.objects = { 'box': objects.Openable(self, 'box', (0, 0)) }
        >>> world = World()
        >>> propositions.create(world, 'closed(box)')
        closed(box)
    """

    import re

    matches = re.match("([^\(]*)\(([^\)]*)", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse proposition from '{str_prop}'.")
    name_pred = matches.group(1)
    try:
        Pred = PROPOSITIONS[name_pred]
    except:
        raise ValueError(f"Unsupported predicate name in '{str_prop}'.")
    str_args = matches.group(2).replace(" ", "").split(",")
    args = [world.objects[arg] for arg in str_args]
    return Pred(*args)
