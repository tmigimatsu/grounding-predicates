import functools
import typing

import symbolic

def alias(str_prop: str) -> str:
    """Returns proposition with its normalized predicate name.

    Args:
        str_prop: Proposition string.
    Returns:
        Normalized proposition string.

    Example:
        >>> from env.gridworld import propositions
        >>> propositions.alias('reachable_in(trophy)')
        'reachable(trophy)'
    """
    return str_prop

def is_static(str_prop: str) -> bool:
    """Returns whether the proposition is static (unmutable)."""
    return str_prop.startswith("is-")

def is_valid(pddl: symbolic.Pddl, str_prop: str) -> bool:
    """Returns whether the proposition is valid."""
    from gpred import dnf_utils
    args = dnf_utils.parse_args(str_prop)
    # Make sure there are no duplicate args
    return len(set(args)) == len(args)
