from . import propositions


class State:
    """State class.

    Args:
        world (world.World): World environment.
    """

    def __init__(self, world):
        self._propositions = set()
        self._world = world

    @property
    def propositions(self):
        """Set of propositions.

        :type: set(propositions.Proposition)
        """
        return self._propositions

    def __repr__(self):
        return repr(self._propositions)

    def __iter__(self):
        return iter(self._propositions)

    def get_propositions(self, prop_type):
        """Get propositions of a certain type."""
        return [prop for prop in self._propositions if isinstance(prop, prop_type)]

    def add(self, str_prop, apply=True, validate=False):
        """Apply the proposition and add it to the state.

        Args:
            str_prop (str): Proposition string.
            apply (bool, optional): Whether to apply the proposition (default True).
            validate (bool, optional): Whether to validate proposition against axioms (default False).
        Returns:
            (propositions.Proposition): Proposition object.
        """
        prop = propositions.create(self._world, str_prop)

        # Validate proposition by checking pddl axioms
        if validate:
            test_state = self.stringify()
            test_state.add(str_prop)
            test_state = self._world.pddl.derived_state(test_state)
            if not self._world.pddl.is_valid_state(test_state):
                raise propositions.PropositionValueError(
                    f"Cannot apply {str_prop} due to axiom violation."
                )

        # Add proposition
        if prop not in self._propositions:
            self._propositions.add(prop)

            # Apply proposition
            if apply:
                prop.apply()

        return prop

    def remove(self, str_prop, apply=True):
        """Negate the proposition and remove it from the state.

        Args:
            str_prop (str): Proposition string.
            apply (bool, optional): Whether to apply the proposition's negation (default True).
        Returns:
            (propositions.Proposition): Proposition object.
        """
        prop = propositions.create(self._world, str_prop)

        # Remove proposition
        self._propositions.remove(prop)

        # Apply negation
        if apply:
            prop.negate()

        return prop

    def stringify(self):
        """Stringify propositions.

        Returns:
            set(str): Set of proposition strings.
        """
        return set(map(str, self._propositions))

    def contains(self, str_prop):
        """Finds whether the proposition exists in the current state.

        Args:
            str_prop (str): Proposition string.
        Returns:
            (bool): Whether the proposition is true.
        """
        for prop in self._propositions:
            if str_prop == str(prop):
                return True
        return False

    def update(self, str_state):
        """Update state by doing a diff with the given new state.

        Args:
            str_state (str): New state string.
        """
        str_state_prev = self.stringify()
        str_props_to_add = str_state.difference(str_state_prev)
        str_props_to_remove = str_state_prev.difference(str_state)

        # Add and remove props without applying
        props_to_add = [
            self.add(str_prop, apply=False) for str_prop in str_props_to_add
        ]
        props_to_remove = [
            self.remove(str_prop, apply=False) for str_prop in str_props_to_remove
        ]

        # Sort application order by highest priority first, then remove before add
        props = [(prop.priority, 0, prop) for prop in props_to_add]
        props += [(prop.priority, 1, prop) for prop in props_to_remove]
        props.sort(reverse=True)

        # Apply changes
        for (_, remove, prop) in props:
            if remove:
                prop.negate()
            else:
                prop.apply()
