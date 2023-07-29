import functools
import itertools
import pathlib
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import h5py
import numpy as np
import pandas as pd  # type: ignore
import symbolic  # type: ignore
import tqdm


class EmptyDisjunctiveFormula:
    def __init__(self):
        self.conjunctions = set()


def parse_head(str_prop):
    import re

    matches = re.match(r"([^\(]*)\([^\)]*", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse proposition from '{str_prop}'.")
    name_pred = matches.group(1)
    return name_pred


def parse_args(str_prop):
    import re

    matches = re.match(r"[^\(]*\(([^\)]*)", str_prop)
    if matches is None:
        raise ValueError(f"Unable to parse objects from '{str_prop}'.")
    str_args = matches.group(1).replace(" ", "").split(",")
    return str_args


def stringify_prop(head: str, args: List[str]) -> str:
    return f"{head}({', '.join(args)})"


def get_action(pddl: symbolic.Pddl, action_name: str) -> Optional[symbolic.Action]:
    """Gets PDDL action symbol."""
    try:
        idx_action = [action.name for action in pddl.actions].index(action_name)
    except ValueError:
        return None
    return pddl.actions[idx_action]


def get_object(pddl: symbolic.Pddl, object_name: str) -> Optional[symbolic.Object]:
    """Gets PDDL object symbol."""
    try:
        idx_object = [object.name for object in pddl.objects].index(object_name)
    except ValueError:
        return None
    return pddl.objects[idx_object]


@functools.lru_cache(maxsize=1024 * 1024)
def get_normalized_conditions(
    pddl: symbolic.Pddl,
    action: str,
    apply_axioms: bool = False,
) -> Tuple[symbolic.DisjunctiveFormula, symbolic.DisjunctiveFormula]:
    """Gets dnfs from cache.

    Args:
        pddl: Pddl instance.
        action: Action call.
        apply_axioms: Whether to apply axioms.
    Returns:
        Pre/post conditions.
    """
    args = parse_args(action)
    if args[0] == "void":
        return EmptyDisjunctiveFormula(), EmptyDisjunctiveFormula()
    return symbolic.DisjunctiveFormula.normalize_conditions(pddl, action, apply_axioms)


@functools.lru_cache(maxsize=1024)
def get_static_props(pddl: symbolic.Pddl) -> np.ndarray:
    """Computes which propositions cannot be changed by actions.

    Args:
        pddl: Pddl instance.
    Returns:
        [N] bool array with 1 if the proposition at that index is static.
    """
    if pddl.name == "twentybn":
        from env.twentybn import propositions
    elif pddl.name == "table":
        from env.table import propositions  # type: ignore
    elif pddl.name == "gridworld":
        from env.gridworld import propositions  # type: ignore
    else:
        raise NotImplementedError(f"Unknown env {pddl.name}.")

    N = len(pddl.state_index)
    idx_static = np.zeros((N,), dtype=bool)

    # Iterate over all propositions
    for i in range(N):
        # Check whether predicate is static
        str_prop = pddl.state_index.get_proposition(i)
        idx_static[i] = propositions.is_static(str_prop)

    return idx_static


@functools.lru_cache(maxsize=1024)
def get_valid_props(pddl: symbolic.Pddl) -> np.ndarray:
    """Computes which propositions are valid.

    Args:
        pddl: Pddl instance.
    Returns:
        [N] bool array with 1 if the proposition at that index is valid.
    """
    if pddl.name == "twentybn":
        from env.twentybn import propositions
    elif pddl.name == "table":
        from env.table import propositions  # type: ignore
    elif pddl.name == "gridworld":
        from env.gridworld import propositions  # type: ignore
    else:
        raise NotImplementedError(f"Unknown env {pddl.name}.")

    N = len(pddl.state_index)
    idx_valid = np.zeros((N,), dtype=bool)

    # Iterate over all propositions
    for i in range(N):
        # Check whether proposition is valid
        str_prop = pddl.state_index.get_proposition(i)
        idx_valid[i] = propositions.is_valid(pddl, str_prop)
    return idx_valid


@functools.lru_cache(maxsize=1024)
def get_consistent_props(pddl: symbolic.Pddl) -> np.ndarray:
    """Gets propositions that can be transferred from pre-conditions to post-conditions.

    Args:
        pddl: Pddl instance.
    Returns:
        [N] bool array with 1 if the proposition at that index is consistent.
    """
    N = len(pddl.state_index)
    idx_consistent = np.ones((N,), dtype=bool)

    if pddl.name == "twentybn":
        idx_visible = [
            idx_prop
            for idx_prop in range(N)
            if parse_head(pddl.state_index.get_proposition(idx_prop)) == "visible"
        ]
        idx_consistent[idx_visible] = 0

    return idx_consistent


@functools.lru_cache(maxsize=1024)
def get_predicate_props(pddl: symbolic.Pddl, pred: str) -> np.ndarray:
    """Gets propositions corresponding to the given predicate.

    Args:
        pddl: Pddl instance.
        pred: Predicate name.
    Returns:
        [N] bool array with 1 if the proposition at that index has the given predicate.
    """
    N = len(pddl.state_index)
    idx_props = np.zeros((N,), dtype=bool)
    for idx_prop in range(N):
        prop = pddl.state_index.get_proposition(idx_prop)
        if parse_head(prop) == pred:
            idx_props[idx_prop] = True
    return idx_props


def compute_max_num_conjunctions(pddl: symbolic.Pddl, actions: Iterable[str]) -> int:
    """Compute maximum number of conjunctions in pre/post-condition DNFs.

    Args:
        pddl: Pddl instance.
        actions: Iterable list of actions.
    Returns:
        Maximum number of conjunctions.
    """
    max_num_conj = 0
    for action in set(actions):
        dnf_pre, dnf_post = get_normalized_conditions(pddl, action)
        num_conj = max(len(dnf_pre.conjunctions), len(dnf_post.conjunctions))
        max_num_conj = max(max_num_conj, num_conj)
    print(
        f"Maximum number of conjunctions M = {max_num_conj}. Record this for future use."
    )
    return max_num_conj


@functools.lru_cache(maxsize=1024 * 1024)
def get_dnf(
    pddl: symbolic.Pddl,
    action: str,
    M: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get DNF proposition indices from cache.

    Args:
        pddl: Pddl instance.
        action: Action call.
        M: Max number of conjunctions in pre/post-conditions DNFs, computed by
            compute_max_num_conjunctions().
    Returns:
        (dnf, mask), where dnf is a [2 x 2 x N x M] index array with pre- and
        post- conditions along axis 0 and positive and negative propositions
        along axis 1. mask is a [2 x M] boolean array indicating the used slots
        along the M-index (since DNFs have a variable number of conjunctions).
    """

    def idx_conditions(
        conjunctions: List[symbolic.PartialState], N: int, M: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the proposition indices for the given pre- or post-conditions.

        Args:
            conjunctions: Pre- or post-condition conjunctions.
            N: Number of propositions.
            M: Max number of conjunctions.
        Returns:
            Two [2 x N x M] arrays, one with proposition indices for the partial
            state, the other with a mask over the used conjunction slots.
        """

        def filter_props(state):
            """Replace derived predicates with normal ones."""
            if pddl.name == "twentybn":
                from env.twentybn import propositions
            elif pddl.name == "table":
                from env.table import propositions
            elif pddl.name == "gridworld":
                from env.gridworld import propositions
            else:
                raise NotImplementedError(f"Unknown env {pddl.name}.")
            return set(propositions.alias(str_prop) for str_prop in state)

        idx = np.zeros((2, N, M), dtype=bool)
        for j, conj in enumerate(conjunctions):
            s_pos = filter_props(conj.pos)
            s_neg = filter_props(conj.neg)
            idx[0, :, j] = pddl.state_index.get_indexed_state(s_pos)
            idx[1, :, j] = pddl.state_index.get_indexed_state(s_neg)

        # Create mask over conjunctions
        mask = np.zeros((M,), dtype=bool)
        mask[: len(conjunctions)] = True

        return idx, mask

    N = len(pddl.state_index)

    try:
        dnf_pre, dnf_post = get_normalized_conditions(pddl, action)
    except RuntimeError:
        raise RuntimeError(f"Could not normalize conditions for {action}.")

    if M is None:
        M = max(len(dnf_pre.conjunctions), len(dnf_post.conjunctions))

    idx_pre, mask_pre = idx_conditions(dnf_pre.conjunctions, N, M)
    idx_post, mask_post = idx_conditions(dnf_post.conjunctions, N, M)

    dnf = np.stack((idx_pre, idx_post), axis=0)  # [2 x 2 x N x M] bool array
    mask = np.vstack((mask_pre, mask_post))  # [2 x M] bool array
    return dnf, mask


def convert_dnf_to_partial_state(
    pddl: symbolic.Pddl, dnf: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Converts the dnf into a partial state consistent with the axioms.

    Args:
        pddl: Pddl instance.
        dnf: [2 x 2 x N x M] bool array output by get_dnf().
        mask: [2 x M] bool array output by get_dnf().
    Returns:
        [2 x 2 x N] bool array representing (pre/post), (pos/neg) partial state.
    """
    # Expand pre/post masks
    # [M]
    c_mask_pre = mask[0, :]
    c_mask_post = mask[1, :]

    # Expand pre/post dnfs
    # [2, N, M]
    c_pre = dnf[0, :, :, :]
    c_post = dnf[1, :, :, :]

    # Compute propositions common to all pre dnfs
    # [2, N]
    if c_mask_pre.sum() == 0:
        s_pre = np.zeros(c_pre.shape[:2], dtype=c_pre.dtype)
    else:
        s_pre = c_pre.sum(axis=-1) == c_mask_pre.sum()

    # Expand pos/neg pre propositions
    # [N]
    s_pre_pos = s_pre[0, :]
    s_pre_neg = s_pre[1, :]

    # Apply axioms to pre state
    pos = pddl.state_index.get_state(s_pre_pos)
    neg = pddl.state_index.get_state(s_pre_neg)
    pos, neg = pddl.consistent_state(pos, neg)
    s_pre_pos[:] = pddl.state_index.get_indexed_state(pos)
    s_pre_neg[:] = pddl.state_index.get_indexed_state(neg)

    # Expand pos/neg post dnfs
    # [N, M]
    c_post_pos = c_post[0, :, :]
    c_post_neg = c_post[1, :, :]

    # Mask out post dnfs that conflict with the pre state (static pre neg
    # and post pos or static pre pos and post neg).
    # [M]
    idx_static = get_static_props(pddl)
    mask_post_pos_violations = (c_post_pos & (s_pre_neg & idx_static)[:, None]).any(
        axis=0
    )
    mask_post_neg_violations = (c_post_neg & (s_pre_pos & idx_static)[:, None]).any(
        axis=0
    )
    mask_post_violations = mask_post_pos_violations | mask_post_neg_violations

    # [2, N, M]
    c_post &= ~mask_post_violations[None, None, :]

    # [M]
    c_mask_post &= ~mask_post_violations

    # Compute propositions common to all post dnfs
    # [2, N]
    if c_mask_post.sum() == 0:
        s_post = np.zeros(c_post.shape[:2], dtype=c_post.dtype)
    else:
        s_post = c_post.sum(axis=-1) == c_mask_post.sum()

    # Carry over static pre propositions to post state
    # [N]
    s_pre_pos = s_pre[0, :]
    s_pre_neg = s_pre[1, :]
    s_post_pos = s_post[0, :]
    s_post_neg = s_post[1, :]
    s_post_pos[:] |= s_pre_pos & idx_static
    s_post_neg[:] |= s_pre_neg & idx_static

    # Apply axioms to post state
    # [N]
    pos = pddl.state_index.get_state(s_post_pos)
    neg = pddl.state_index.get_state(s_post_neg)
    pos, neg = pddl.consistent_state(pos, neg)
    s_post_pos[:] = pddl.state_index.get_indexed_state(pos)
    s_post_neg[:] = pddl.state_index.get_indexed_state(neg)

    # Apply additional static post propositions to pre state
    # [N]
    s_pre_pos[:] |= s_post_pos & idx_static
    s_pre_neg[:] |= s_post_neg & idx_static

    # Carry over unchanged pre propositions to post state
    # [N]
    idx_consistent = get_consistent_props(pddl)
    s_post_pos[:] |= idx_consistent & s_pre_pos & ~s_post_neg & ~c_post_neg.any(axis=1)
    s_post_neg[:] |= idx_consistent & s_pre_neg & ~s_post_pos & ~c_post_pos.any(axis=1)

    # Reassemble state
    # [2, 2, N]
    s = np.stack((s_pre, s_post), axis=0)

    # Filter out invalid propositions.
    idx_valid = get_valid_props(pddl)
    s &= idx_valid[None, None, :]

    return s


# def compute_correlation(
#     pddl: symbolic.Pddl, actions: Iterable[str], proposition: str
# ) -> np.ndarray:
#     M = compute_max_num_conjunctions(pddl, actions)
#     idx_static = get_static_props(pddl)
#     for action in actions:
#         dnf, mask = get_dnf(pddl, action, M)
#         s_partial = convert_to_partial_state(pddl, dnf, mask, idx_static)
#     dnf_pre, dnf_post = get_normalized_conditions(pddl, action)


def get_used_objects(
    pddl: symbolic.Pddl, s_pre: np.ndarray, s_post: np.ndarray
) -> Set[str]:
    """Finds the set objects that appear in either the pre- or post-state.

    Args:
        pddl: Pddl instance.
        s_pre: Precondition state.
        s_post: Postcondition state.
    Returns:
        Set of objects that appear in the state.
    """
    props = pddl.state_index.get_state(s_pre | s_post)
    objects = []
    for prop in props:
        objects += parse_args(prop)
    return set(objects)


def get_negative_props(
    pddl: symbolic.Pddl, idx_valid: np.ndarray, used_objects: Set[str]
) -> np.ndarray:
    """Finds the set of propositions among the valid ones that must be
    negative due to non-existent objects.

    Args:
        pddl: Pddl instance.
        idx_valid: Valid propositions output by get_valid_props().
        used_objects: Used objects output by get_used_objects().
    Returns:
        Set of negative propositions.
    """
    valid_props = pddl.state_index.get_state(idx_valid)
    neg_props = []
    for prop in valid_props:
        args = parse_args(prop)
        if set(args) - used_objects:
            neg_props.append(prop)
    return pddl.state_index.get_indexed_state(set(neg_props))


def get_used_props(pddl: symbolic.Pddl, boxes: np.ndarray) -> np.ndarray:
    """Finds the set of used props based on objects that exist in the scene.

    Boxes can either be -inf or nan for nonexistent objects.

    Args:
        pddl: Pddl instance.
        boxes: [O, 4] (num_objects, x1/y1/x2/y2) float32.
    """

    def is_not_numeric(x):
        return np.isnan(x) or np.isinf(x)

    O = len(pddl.objects)  # noqa: E741
    idx_objects = {pddl.objects[idx_obj].name: idx_obj for idx_obj in range(O)}

    idx_used = get_valid_props(pddl).copy()
    for idx_prop in idx_used.nonzero()[0]:
        prop = pddl.state_index.get_proposition(idx_prop)
        args = parse_args(prop)
        for arg in args:
            idx_obj = idx_objects[arg]
            if is_not_numeric(boxes[idx_obj, 0]):
                idx_used[idx_prop] = False
                break

    return idx_used


def get_actions(
    path: Union[str, pathlib.Path] = "../data/gridworld",
    dataset: str = "dataset.hdf5",
) -> List[str]:
    """Gets the set of all actions in the dataset.

    Args:
        path: Path of dataset.
        dataset: Filename of dataset.
    Returns:
        List of all actions.
    """
    with h5py.File(pathlib.Path(path) / dataset, "r") as f:
        actions = list(set(tqdm.tqdm(f["actions"])))
    return actions


@functools.lru_cache(maxsize=1024 * 1024)
def get_partial_state(pddl: symbolic.Pddl, action: str) -> np.ndarray:
    """Loads the action pre/post-condition DNFs as partial states.

    Args:
        pddl: Pddl instance.
        action: Action call.
    Returns:
        [2 x 2 x N] bool array representing (pre/post), (pos/neg) partial state.
    """
    # [2 x 2 x N x M] array (pre/post, pos/neg, N props, M conjs)
    dnf, mask = get_dnf(pddl, action)

    # [2 x 2 x N]
    s_partial = convert_dnf_to_partial_state(pddl, dnf, mask)

    return s_partial


def get_partial_states(pddl: symbolic.Pddl, actions: List[str]) -> np.ndarray:
    """Loads the action DNFs as partial states.

    Args:
        pddl: Pddl instance.
        actions: List of actions obtained from get_actions().
    Returns:
        [num_actions, 2, 2, N] array where axis 1 is (pre/post) and axis 2 is
        (pos/neg) propositions.
    """
    N = len(pddl.state_index)
    A = len(actions)

    # [A, 2, 2, N] (actions, pre/post, pos/neg, props)
    s_partial = np.zeros((A, 2, 2, N), dtype=bool)
    for idx_action, action in enumerate(actions):
        # [2, 2, N] (pre/post, pos/neg, props)
        s_partial[idx_action] = get_partial_state(pddl, action)

    return s_partial


@functools.lru_cache(maxsize=1024 * 1024)
def get_partial_predicate_state(pddl: symbolic.Pddl, action: str) -> np.ndarray:
    """Loads the action pre/post-condition DNFs as predicate partial states.

    Args:
        pddl: Pddl instance.
        action: Action call.
    Returns:
        [2, 16, 2, P] (pre/post, arg_combos, pos/neg, num_props).
    """
    # [2, 2, N] (pre/post, pos/neg, N)
    s_partial = get_partial_state(pddl, action)

    # [2, 2, N] -> [2, 2, 16, P] (pre/post, pos/neg, arg_combos, num_props)
    sp_partial = proposition_to_predicate_indices(pddl, s_partial)

    # [2, 2, 16, P] -> [2, 16, 2, P] (pre/post, arg_combos, pos/neg, num_props)
    sp_partial = np.swapaxes(sp_partial, 2, 1)

    return sp_partial


@functools.lru_cache(maxsize=10)
def max_num_parameters(pddl: symbolic.Pddl) -> int:
    """Computes the maximum number of parameters among all the predicates.

    Args:
        pddl: Pddl instance.
    Returns:
        Maximum number of parameters.
    """
    return max(len(pred.parameters) for pred in pddl.predicates)


@functools.lru_cache(maxsize=10)
def get_arg_combo_mapping(
    pddl: symbolic.Pddl,
    indices: bool = False,
) -> Tuple[List[Tuple[Union[str, int], ...]], Dict[Tuple[Union[str, int], ...], int]]:
    """Assigns indices to all the argument combinations.

    Argument combinations are represented as tuples of strings.

    Args:
        pddl: Pddl instance.
        indices: Whether to map object indices or objects.
    Returns:
        (list of arg combos, dict from arg combo to indices).
    """
    # Collect all objects.
    if indices:
        args: List[Union[str, int]] = [i for i, obj in enumerate(pddl.objects)]
    else:
        args = [str(obj) for obj in pddl.objects]

    # Compute the max number of parameters in a predicate.
    max_num_params = max_num_parameters(pddl)

    # Build a map from idx_arg_combo to arg_combo.
    arg_combos = []
    for num_params in range(max_num_params):
        arg_combos += list(itertools.permutations(args, num_params + 1))

    # Build a map from arg_combo to idx_arg_combo.
    idx_arg_combos = {}
    for idx_arg_combo, arg_combo in enumerate(arg_combos):
        idx_arg_combos[arg_combo] = idx_arg_combo

    return arg_combos, idx_arg_combos


@functools.lru_cache(maxsize=10)
def num_arg_combos(pddl: symbolic.Pddl) -> int:
    """Computes the number of argument combinations.

    Args:
        pddl: Pddl isnstance.
    Returns:
        Total number of argument combinations.
    """
    return len(get_arg_combo_mapping(pddl)[0])


def arg_combo_index(pddl: symbolic.Pddl, args: Tuple[str, ...]) -> Optional[int]:
    """Gets the index for the current argument combination.

    Args:
        args: Tuple of arguments as strings.
    Returns:
        Arg combo index, or None if the arguments are the same.
    """
    _, idx_arg_combos = get_arg_combo_mapping(pddl)

    assert type(args) is tuple

    try:
        return idx_arg_combos[args]
    except KeyError:
        return None


def arg_combo(
    pddl: symbolic.Pddl, idx_args: int, indices: bool = False
) -> Tuple[Union[str, int], ...]:
    """Gets the argument list for the given argument combo index.

    Args:
        idx_args: Argument combo index.
        indices: Whether to map object indices or objects.
    Returns:
        List of arguments.
    """
    arg_combos, _ = get_arg_combo_mapping(pddl, indices=indices)

    return arg_combos[idx_args]


@functools.lru_cache(maxsize=10)
def param_to_arg_combo_indices(pddl: symbolic.Pddl) -> np.ndarray:
    """Creates an index map from parameter indices to argument combination
    indices that can be used to convert Numpy arrays with advanced indexing.

    Parameter index values equal to num_params represent empty slots in the
    argument combination.

    Args:
        pddl: Pddl instance.
    Returns:
        [num_arg_combos, max_num_params] array of integer indices into a
        [num_params + 1] array.
    """
    NUM_ARG_COMBOS = num_arg_combos(pddl)
    MAX_NUM_PARAMS = max_num_parameters(pddl)

    params = [str(obj) for obj in pddl.objects]

    idx_param_to_arg_combo = np.zeros((NUM_ARG_COMBOS, MAX_NUM_PARAMS), dtype=np.uint32)
    arg_combos, _ = get_arg_combo_mapping(pddl)
    for idx_arg_combo, args in enumerate(arg_combos):
        idx_params = [params.index(arg) for arg in args]  # type: ignore
        idx_params += [len(params)] * (MAX_NUM_PARAMS - len(idx_params))

        idx_param_to_arg_combo[idx_arg_combo, :] = np.array(idx_params, dtype=np.uint32)

    return idx_param_to_arg_combo


def idx_prop_to_idx_pred(
    pddl: symbolic.Pddl, idx_prop: int
) -> Tuple[int, Optional[int]]:
    """Converts the proposition index into a (predicate index, arg combo index) pair.

    Arg combo index may be None if the arguments are repeated.

    Args:
        pddl: Pddl instance.
        idx_prop: Proposition index.
    Returns:
        (predicate index, arg combo index) pair
    """
    preds = [pred.name for pred in pddl.predicates]
    prop = pddl.state_index.get_proposition(idx_prop)
    pred = parse_head(prop)
    idx_pred = preds.index(pred)

    args = tuple(parse_args(prop))
    idx_arg_combo = arg_combo_index(pddl, args)

    return idx_pred, idx_arg_combo


@functools.lru_cache(maxsize=10)
def proposition_to_predicate_index_map(pddl: symbolic.Pddl) -> np.ndarray:
    """Creates an index map from proposition indices to predicate indices that
    can be used to convert Numpy arrays with advanced indexing.

    Predicate elements that do not correspond to a proposition will be assigned
    to index N (proposition N+1).

    Args:
        pddl: Pddl instance.
    Returns:
        [num_arg_combo, num_preds] array of integer indices into a [N] array.
    """
    P = len(pddl.predicates)
    N = len(pddl.state_index)
    NUM_ARG_COMBOS = num_arg_combos(pddl)

    idx_prop_to_pred = np.full((NUM_ARG_COMBOS, P), N, dtype=np.uint32)
    for idx_prop in range(N):
        idx_pred, idx_arg_combo = idx_prop_to_idx_pred(pddl, idx_prop)
        if idx_arg_combo is None:
            continue

        idx_prop_to_pred[idx_arg_combo, idx_pred] = idx_prop

    return idx_prop_to_pred


@functools.lru_cache(maxsize=10)
def predicate_to_proposition_index_map(pddl: symbolic.Pddl) -> np.ndarray:
    """Creates an index map from predicate indices to proposition indices that
    can be used to convert Numpy arrays with advanced indexing.

    Proposition elements that do not correspond to a predicate will be assigned
    to index num_arg_combo * num_pred.

    Args:
        pddl: Pddl instance.
    Returns:
        [N] array of integer indices into a [num_arg_combo * num_preds] array.
    """
    P = len(pddl.predicates)
    N = len(pddl.state_index)
    NUM_ARG_COMBOS = num_arg_combos(pddl)

    idx_pred_to_prop = np.full((N,), P * NUM_ARG_COMBOS, dtype=np.uint32)
    for idx_prop in range(N):
        idx_pred, idx_arg_combo = idx_prop_to_idx_pred(pddl, idx_prop)
        if idx_arg_combo is None:
            continue

        idx_pred_to_prop[idx_prop] = idx_arg_combo * P + idx_pred

    return idx_pred_to_prop


def proposition_to_predicate_indices(
    pddl: symbolic.Pddl,
    idx_props: np.ndarray,
    default_value=0,
) -> np.ndarray:
    """Converts proposition state to predicate state.

    Args:
        pddl: Pddl instance.
        idx_props: [..., N] array.
        default_value: Default value to assign to predicates that do not
            correspond to any propositions.
    Returns:
        [..., 16, P] array.
    """
    dim = idx_props.shape[:-1]

    # [..., N] -> [-1, N]
    N = idx_props.shape[-1]
    idx_props = idx_props.reshape((-1, N))

    # [-1, 1]
    default = np.full((idx_props.shape[0], 1), default_value, dtype=idx_props.dtype)

    # [-1, N], [-1, 1] -> [-1, N + 1]
    idx_props = np.concatenate((idx_props, default), axis=1)

    # [16, P]
    idx_prop_to_pred = proposition_to_predicate_index_map(pddl)

    # [-1, N + 1] -> [-1, 16, P]
    idx_preds = idx_props[:, idx_prop_to_pred]

    # [-1, 16, P] -> [..., 16, P]
    idx_preds = idx_preds.reshape((*dim, *idx_preds.shape[-2:]))

    return idx_preds


def predicate_to_proposition_indices(
    pddl: symbolic.Pddl,
    idx_preds: np.ndarray,
    idx_used: Optional[np.ndarray] = None,
    idx_arg_combo: Optional[int] = None,
    default_value=0,
) -> np.ndarray:
    """Converts predicate state to proposition state.

    Args:
        pddl: Pddl instance.
        idx_preds: [..., MC, P], [..., U, P], or [..., P] array.
        idx_used: [MC] index array with U entries (must be provided if idx_preds
            is [..., U, P]).
        idx_arg_combo: Arg combo index (must be provided if idx_preds is [..., P]).
        default_value: Default value to assign to propositions that do not
            correspond to any predicates.
    Returns:
        [..., N] array.
    """
    # [N]
    idx_pred_to_prop = predicate_to_proposition_index_map(pddl)

    if idx_arg_combo is not None:
        dim = idx_preds.shape[:-1]
        P = idx_preds.shape[-1]
        U = 1

        # Shift predicate combo range.
        # [N]
        idx_pred_to_prop = idx_pred_to_prop - idx_arg_combo * P

        idx_pred_to_prop[idx_pred_to_prop > P] = P

    elif idx_used is not None:
        dim = idx_preds.shape[:-2]
        U, P = idx_preds.shape[-2:]
        MC = idx_used.shape[0]

        # Shift predicate combo range.
        # [10, 16, 48, 11, 17, 48, 18, 32, 33]
        # idx_pred_to_prop

        # [MC + 2]
        # [0, 16, 32, 48, 64]
        bins = np.arange(0, (MC + 2) * P, P, dtype=np.uint32)

        # [N]
        # [0,  1,  3,  0,  1,  3,  1,  2,  2 ]
        idx_mc_to_n = np.digitize(idx_pred_to_prop, bins) - 1

        # [MC + 1]
        # [1, 0, 1] + [0]
        idx_used = np.append(idx_used, False)

        # [0, 1, 1] + [2]
        idx_offset = (~idx_used).cumsum()  # type: ignore

        # [0, 0, 1] + [0]
        idx_offset[~idx_used] = 0  # type: ignore

        # [N]
        # [ 0,  0,  0,  0,  0,  0,  0,  1,  1 ]
        idx_offset_n = idx_offset[idx_mc_to_n]

        # [10, 11,  -,  -,  -,  -,  -,  16, 17]
        idx_pred_to_prop = idx_pred_to_prop - idx_offset_n * P

        # [ 1,  1,  0,  0,  0,  0,  0,  1,  1 ]
        idx_used_n = idx_used[idx_mc_to_n]  # type: ignore

        # [10, 32, 32, 32, 11, 32, 32,  16, 17]
        idx_pred_to_prop[~idx_used_n] = U * P
    else:
        dim = idx_preds.shape[:-2]
        U, P = idx_preds.shape[-2:]

    dim_flat = np.prod(dim) if dim else 1

    # [-1, MC * P + 1]
    idx_preds_aug = np.empty((dim_flat, U * P + 1), dtype=idx_preds.dtype)

    # [..., MC, P] -> [-1, MC * P] (arg_combos * preds)
    idx_preds_aug[:, :-1] = idx_preds.reshape((-1, U * P))

    # [-1, 1]
    idx_preds_aug[:, -1] = default_value

    # [-1, MC * P + 1] -> [-1, N] (num_props)
    idx_props = idx_preds_aug[:, idx_pred_to_prop]

    # [-1, N] -> [..., N]
    idx_props = idx_props.reshape((*dim, idx_props.shape[-1]))

    return idx_props


def count_prop_occurrences(
    pddl: symbolic.Pddl, s_partial: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Counts co-occurrences of propositions among all the DNFs.

    Args:
        pddl: Pddl instance.
        s_partial: Partial states from DNFs obtained from get_partial_states().
    Returns:
        (num_xy [N, N, 2, 2], num_xy_total [N, N]) tuple.
    """
    N = len(pddl.state_index)

    num_xy = np.zeros((N, N, 2, 2), dtype=int)
    for i in range(s_partial.shape[0]):
        # [N, 2]
        ss = s_partial[i, :, :].T

        # [N, N, 2, 2]
        num_xy += ss[:, None, :, None] & ss[None, :, None, :]

    # [N, N] (x idx, y idx)
    num_xy_total = num_xy.sum(axis=-1).sum(axis=-1)
    return num_xy, num_xy_total


def count_prop_occurrences_with_object_detection(
    pddl: symbolic.Pddl,
    actions: List[str],
    idx_valid: np.ndarray,
    s_partial: np.ndarray,
    path: Union[str, pathlib.Path] = "../data/gridworld",
    dataset: str = "dataset.hdf5",
) -> Tuple[np.ndarray, np.ndarray]:
    """Counts co-occurrences of propositions among all the DNFs, augmented with
    negative propositions inferred from non-existant objects.

    Args:
        pddl: Pddl instance.
        actions: List of actions obtained from get_actions().
        idx_valid: [N] boolean array of valid propositions obtained from get_valid_props().
        s_partial: Partial states from DNFs obtained from get_partial_states().
        path: Path of dataset.
        dataset: Filename of dataset.
    """
    N = len(pddl.state_index)

    # [A, N]
    count_unused = np.zeros((len(actions), N), dtype=int)
    count_actions = np.zeros((len(actions),), dtype=int)
    with h5py.File(pathlib.Path(path) / dataset, "r") as f:
        D = len(f["actions"])
        for idx in tqdm.tqdm(range(D)):
            action = f["actions"][idx]
            idx_action = actions.index(action)

            s_pre = f["s_pre"][idx]
            s_post = f["s_post"][idx]
            used_objects = get_used_objects(pddl, s_pre, s_post)
            s_neg = get_negative_props(pddl, idx_valid, used_objects)

            # Filter out negative props already covered by dnf.
            s_dnf_pre = s_partial[2 * idx_action, 1, :]
            s_dnf_post = s_partial[2 * idx_action + 1, 1, :]
            s_neg &= ~(s_dnf_pre | s_dnf_post)

            count_unused[idx_action, :] += s_neg
            count_actions[idx_action] += 1

    # [A, N]
    p_unused = count_unused / count_actions[:, None]

    # [2*A, 2, N]
    s_partial_aug = (s_partial & idx_valid[None, None, :]).astype(float)
    s_partial_aug[0::2, 1, :] += p_unused
    s_partial_aug[1::2, 1, :] += p_unused

    num_xy_aug = np.zeros((N, N, 2, 2))
    num_xy_aug_total = np.zeros((N, N))
    for i in range(s_partial.shape[0]):
        # [N, 2]
        ss = s_partial_aug[i, :, :].T

        # [N, N, 2, 2]
        ssss = ss[:, None, :, None] * ss[None, :, None, :]
        num_xy_aug += ss[:, None, :, None] * ss[None, :, None, :]

        # [N, N]
        num_xy_aug_total += (ssss > 0).sum(axis=-1).sum(axis=-1)

    return num_xy_aug, num_xy_aug_total


def compute_mutual_information(
    num_xy: np.ndarray, num_xy_total: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the joint probability and mutual information from the
    proposition counts.

    Args:
        num_xy: [N, N, 2, 2] matrix of pos/neg proposition co-occurrences.
        num_xy_total: [N, N] matrix of total proposition co-occurrences.
    Returns:
        (joint_probability [N, N, 2, 2], mutual_information [N, N]) tuple.
    """

    def safe_divide(numer, denom):
        return np.divide(numer, denom, out=np.zeros(numer.shape), where=denom != 0)

    def safe_log2(arg):
        return np.log2(arg, out=np.zeros(arg.shape), where=arg != 0)

    # [N, N] (x idx, y idx)
    num_xy_total = num_xy.sum(axis=-1).sum(axis=-1)
    # [N, N, 2, 2] (x idx, y idx, x pos/neg, y pos/neg)
    p_xy = safe_divide(num_xy, num_xy_total[:, :, None, None])
    # [N, N, 2] (x idx, y idx, x pos/neg)
    p_x = safe_divide(num_xy.sum(axis=3), num_xy_total[:, :, None])
    # [N, N, 2] (x idx, y idx, y pos/neg)
    p_y = safe_divide(num_xy.sum(axis=2), num_xy_total[:, :, None])

    # [N, N, 2, 2] (x idx, y idx, x pos/neg, y pos/neg)
    p_x_p_y = p_x[:, :, :, None] * p_y[:, :, None, :]  # [N, N, 2, 1] x [N, N, 1, 2]
    log_arg = safe_divide(p_xy, p_x_p_y)

    # [N, N] Mutual information
    I_xy = (p_xy * safe_log2(log_arg)).sum(axis=-1).sum(axis=-1)

    # [N] Entropy
    # H_x = I_xy.diagonal()
    # H_x = -(np.diagonal(p_x) * safe_log2(np.diagonal(p_x))).sum(axis=0)

    return p_xy, I_xy
    # print("p_xy:\n", p_xy[idx_prop_other, idx_prop])
    # print("p_x:\n", p_x[idx_prop_other, idx_prop])
    # print("p_y:\n", p_y[idx_prop_other, idx_prop])


def diff_state(states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Finds the unique and common propositions among M states.

    Args:
        states: List of M state indices as a [M, N] array.
    Returns:
        (unique propositions [M, N], common propositions [N]) pair.
    """
    common = np.logical_and.reduce(states != 0.0)
    idx_common = common.nonzero()
    unique = states.copy()
    unique[:, idx_common] = 0
    return unique, common


def print_pre_post_state(
    pddl: symbolic.Pddl, s: np.ndarray, name: Optional[str] = None
):
    """Prints the state or pre/post-conditions.

    Args:
        pddl: Pddl instance.
        s: State as a length [N] array or pre/post-conditions as a [2, N] array.
        name: Name to print before state.
    """

    def print_props(s: np.ndarray, name: Optional[str]):
        """Prints propositions.

        Args:
            s: State index.
            name: Name of state index.
        """
        tab = ""
        if name is not None:
            print(f"{name}:")
            tab = "  "
        for idx in s.nonzero()[0]:
            prop = pddl.state_index.get_proposition(idx)
            print(f"{tab}+ {prop}")
        print("")

    if name is not None:
        print(f"{name}\n=====")
    if len(s.shape) == 1 or s.shape[0] == 1:
        if s.shape[0] == 1:
            s = s[0]
        print_props(s, None)
        return

    unique, common = diff_state(s)
    print_props(unique[0], "pre")
    print_props(unique[1], "post")
    print_props(common, "common")


def print_partial_state(
    pddl: symbolic.Pddl,
    s_pos: np.ndarray,
    s_neg: np.ndarray,
    name: Optional[str] = None,
):
    """Prints the partial state.

    Args:
        pddl: Pddl instance.
        s_pos: Positive (pre/post) state indices as a [2, N] array.
        s_neg: Negative (pre/post) state indices as a [2, N] array.
        name: Name to print before state.
    """
    print("DEPRECATED: Use dnf_utils.print_state().")

    def print_props(pos: np.ndarray, neg: np.ndarray, name: Optional[str]):
        """Prints propositions.

        Args:
            pos: Positive state index.
            neg: Negative state index.
            name: Name of state index.
        """
        if name is not None:
            print(f"{name}:")
        for idx in np.logical_or(pos > 0, neg > 0).nonzero()[0]:
            char = "+" if pos[idx] > 0 else "-"
            prop = pddl.state_index.get_proposition(idx)
            print(f"  {char} {prop}")
        print("")

    if len(s_pos.shape) == 1 or s_pos.shape[0] == 1:
        if s_pos.shape[0] == 1:
            s_pos = s_pos[0]
            s_neg = s_neg[0]
        print_props(s_pos, s_neg, name)
        return

    unique_pos, common_pos = diff_state(s_pos)
    unique_neg, common_neg = diff_state(s_neg)
    if name is not None:
        print(f"{name}\n=====")
    print_props(unique_pos[0], unique_neg[0], "pre")
    print_props(unique_pos[1], unique_neg[1], "post")
    print_props(common_pos, common_neg, "common")


def print_dnf(
    pddl: symbolic.Pddl,
    dnf: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    common_only: bool = False,
    name: Optional[str] = None,
):
    """Prints raw dnfs.

    Args:
        pddl: Pddl instance.
        dnf: [2, 2, N, M] (pre/post, pos/neg, N propositions, M conjunctions) array
            or (pos, neg) tuple of [2, N, M] arrays.
        common_only: If true, only prints common propositions across conjunctions.
        name: Name to print before state.
    """

    def print_conjunctions(pos_neg: np.ndarray, name: str, common_only: bool):
        """Prints conjunctions.

        Args:
            pos_neg: [2, N, M] (pos/neg) pair of M state indices.
            name: Name of dnf set.
            common_only: Print only common propositions.
        """

        def print_props(pos: np.ndarray, neg: np.ndarray, name: str):
            """Print propositions.

            Args:
                pos: Positive state indices as a [N] array.
                neg: Negative state indices as a [N] array.
                name: Name of state indices.
            """
            print(f"{name}:")
            for idx in np.logical_or(pos > 0, neg > 0).nonzero()[0]:
                char = "+" if pos[idx] > 0 else "-"
                prop = pddl.state_index.get_proposition(idx)
                print(f"  {char} {prop}")
            print("")

        M = pos_neg.shape[-1]
        M = pos_neg.reshape((-1, M)).sum(axis=0).nonzero()[0][-1] + 1
        unique_pos, common_pos = diff_state(pos_neg[0, :, :M].T)
        unique_neg, common_neg = diff_state(pos_neg[1, :, :M].T)

        if not common_only:
            for m in range(M):
                print_props(unique_pos[m], unique_neg[m], f"{name} ({m})")
        print_props(common_pos, common_neg, f"{name} (common)")

    if name is not None:
        print(f"{name}\n=====")
    print_conjunctions(dnf[0], "pre", common_only)
    print_conjunctions(dnf[1], "post", common_only)


def convert_to_partial_state(
    pddl: symbolic.Pddl,
    s: np.ndarray,
    idx_valid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Converts the state to a partial state.

    Args:
        pddl: Pddl instance.
        s: [2, N] (pos/neg, num_props) or [N] (num_props).
        idx_valid: Valid propositions to print.
    Returns:
        [2, N] partial state.
    """
    N = s.shape[-1]

    # Full state.
    if len(s.shape) == 1:
        # [N]
        if idx_valid is None:
            idx_valid = np.ones((N,), dtype=bool)

        # [2, N] (pos/neg, num_props)
        s_partial = np.zeros((2, N), dtype=s.dtype)

        if s.dtype == bool:
            s_partial[0] = s & idx_valid
            s_partial[1] = ~s & idx_valid
        else:
            s_partial[0, :] = s
            s_partial[1, :] = 1 - s
            s_partial[s_partial <= 0.5] = 0
            s_partial[:, ~idx_valid] = 0

        # [2, N] (pos/neg, num_props)
        return s_partial

    # Partial state.
    assert s.shape[0] == 2

    # [2, N] (pos/neg, num_props)
    return s


def print_state(
    pddl: symbolic.Pddl,
    s: np.ndarray,
    idx_valid: Optional[np.ndarray] = None,
    name: Optional[str] = None,
):
    """Prints (partial and not partial) states.

    Args:
        pddl: Pddl instance.
        s: [2, N] (pos/neg, num_pred) or [N] (num_pred).
        idx_valid: Valid propositions to print.
        name: Name to print before state.
    """
    s_partial = convert_to_partial_state(pddl, s, idx_valid=idx_valid)

    if name is None:
        name = "State" if len(s.shape) == 1 else "Partial state"
    print(f"\n{name}\n=====")

    for idx_prop in s_partial.sum(axis=0).nonzero()[0]:
        is_pos = s_partial[0, idx_prop] > 0

        char = "+" if is_pos else "-"

        if s_partial.dtype == bool:
            val = s_partial[0, idx_prop] if is_pos else ~s_partial[1, idx_prop]
        else:
            val = s_partial[0, idx_prop] if is_pos else 1 - s_partial[1, idx_prop]

        prop = pddl.state_index.get_proposition(idx_prop)

        print(f"  {char} {prop}: {val}")

    print("")


def print_predicate_state(
    pddl: symbolic.Pddl,
    sp: np.ndarray,
    idx_arg_combo: int,
    idx_valid: Optional[np.ndarray] = None,
    name: Optional[str] = None,
):
    """Prints predicate (partial and not partial) states.

    Args:
        pddl: Pddl instance.
        sp: [2, P] (pos/neg, num_pred) or [P] (num_pred).
        idx_arg_combo: Arg combo index.
        idx_valid: Valid propositions to print.
        name: Name to print before state.
    """
    s = predicate_to_proposition_indices(pddl, sp, idx_arg_combo=idx_arg_combo)
    print_state(pddl, s, idx_valid, name)


def compute_pddl_statistics(
    pddl: symbolic.Pddl, actions: Optional[List[str]] = None
) -> pd.DataFrame:
    """Computes proposition statistics for all the actions in the pddl.

    Args:
        pddl: Pddl instance.
        actions: List of action calls.
    Returns:
        Long-form dataframe with columns:
         - Proposition: "pred(a, b)".
         - Predicate: "pred".
         - Action: "action(a, b)".
         - Condition: "pre" or "post".
         - Label: True or False.
    """
    if actions is None:
        actions = pddl.actions

    stats: Dict[str, List] = {
        "Proposition": [],
        "Predicate": [],
        "Action": [],
        "Condition": [],
        "Label": [],
    }
    for action in actions:
        str_action = str(action)

        # [2, 2, N] (pre/post, pos/neg, num_props)
        s_partial = get_partial_state(pddl, str_action)

        for idx_pre_post in range(s_partial.shape[0]):
            idx_used = s_partial[idx_pre_post].sum(axis=0).nonzero()[0]
            for idx_prop in idx_used:
                val_prop = s_partial[idx_pre_post, 0, idx_prop]
                prop = pddl.state_index.get_proposition(idx_prop)
                stats["Proposition"].append(prop)
                stats["Predicate"].append(parse_head(prop))
                stats["Action"].append(str_action)
                stats["Condition"].append("pre" if idx_pre_post == 0 else "post")
                stats["Label"].append(val_prop)

    return pd.DataFrame(stats)


def compute_predicate_class_weights(
    pddl: symbolic.Pddl,
    actions: Optional[List[str]] = None,
    action_instances: Optional[List[str]] = None,
) -> np.ndarray:
    r"""Computes predicate class weights for weighted cross entropy.

    Given a vector :math:`c \in \mathbb{R}^N` of all the class counts:

    .. math::
       p = \frac{c}{\sum{c}_i}
       w = \frac{1}{N * p}

    This is constructed such that weights for class in a uniform distribution would be 1.

    Args:
        pddl: Pddl instance.
        actions: List of action calls.
        action_instances: Action instances in dataset.
    Returns:
        [2, P] (pos/neg, num_preds) float32 array of cross entropy weights.
    """
    # Compute pddl statistics.
    stats = compute_pddl_statistics(pddl, actions=actions)
    if action_instances is not None:
        action_instances = pd.DataFrame(action_instances, columns=["Action"])
        stats = stats.merge(action_instances, left_on="Action", right_on="Action")

    preds = [pred.name for pred in pddl.predicates]

    # [P, 2] (num_preds, pos/neg)
    pred_counts = np.zeros((len(preds), 2), dtype=int)

    # | Proposition | Predicate | Action | Condition | Label |
    stats_pos = stats[stats.Label == True]  # noqa: E712
    stats_neg = stats[stats.Label == False]  # noqa: E712

    # | Predicate | Count |
    num_pos = stats_pos.groupby("Predicate").count().Label
    num_neg = stats_neg.groupby("Predicate").count().Label

    for idx_pred, pred in enumerate(preds):
        if pred in num_pos:
            pred_counts[idx_pred, 0] = num_pos[pred]
        if pred in num_neg:
            pred_counts[idx_pred, 1] = num_neg[pred]

    # w_inv = pred_counts.sum() / (pred_counts.size * pred_counts)
    # w_inv[w_inv == float("inf")] = np.nan

    # [P, 2]
    # Set minimum count to 1 to avoid infinite weights.
    pred_counts = pred_counts.astype(float)
    pred_counts[pred_counts == 0] = np.nan

    # Theoretical size of classes.
    # ~1000
    N = np.exp(np.log(pred_counts[~np.isnan(pred_counts)]).mean())
    beta = (N - 1) / N

    # Effective number of samples
    # max 1000
    E_n = (1 - np.power(beta, pred_counts)) / (1 - beta)

    pred_weights = 1 / E_n

    # E_n[np.isnan(E_n)] = 0
    # Zero out weights with no samples.
    pred_weights[np.isnan(pred_weights)] = 0

    # Scale weights so sum equals number of classes.
    pred_weights *= pred_weights.size / pred_weights.sum()
    # pred_weights *= E_n.sum() / E_n.size
    # pred_weights /= pred_weights.mean()
    # pred_weights *= pred_counts.size / pred_weights.sum()
    # BETA = 0.9993
    # a = pred_counts.flatten().copy()
    # a.sort()
    # print(a)
    # pred_weights = (1 - BETA) / (1 - np.power(BETA, np.maximum(143, pred_counts)))
    # pred_weights *= pred_counts.size / pred_weights.sum()

    # [P, 2] -> [2, P] (pos/neg, num_preds)
    pred_weights = pred_weights.T.astype(np.float32)
    # w_inv = w_inv.T.astype(np.float32)

    return pred_weights


def bbox_masks(
    width: int,
    height: int,
    boxes: np.ndarray,
    labels: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Creates bounding box image masks from a list of bounding boxes.

    If idx_used is specified, the mask will be filled with the corresponding
    object indices.

    Args:
        width: Image width.
        height: Image height.
        boxes: [..., num_boxes, 4] array of boxes (x1/y1/x2/y2).
        labels: Optional [..., num_boxes] float32 array of labels.
        mask: Optional [..., num_boxes, H, W] float32 array to populate.
    Returns:
        [..., num_boxes, H, W] float32 array with ones inside the box and zeros everywhere else.
    """
    # Select all dimensions except last (x1/y1/x2/y2).
    dim = boxes.shape[:-1]

    # Preallocate mask image.
    if mask is None:
        mask = np.zeros((*dim, height, width), dtype=np.float32)

    # Iterate over all boxes.
    for sub in np.ndindex(*dim):
        # [4] (x1/y1/x2/y2)
        box = boxes[sub]
        if box[0] == -float("inf"):
            continue

        # Round to nearest integer.
        box = (box + 0.5).astype(int)

        # Get label.
        label = 1 if labels is None else labels[sub]

        # Append (y1:y2, x1:x2) to indexing slice.
        sub += (slice(box[1], box[3]), slice(box[0], box[2]))

        mask[sub] = label

    return mask


def bbox_arg_combos(pddl: symbolic.Pddl, boxes: np.ndarray) -> np.ndarray:
    """Expands bounding boxes into arg combos.

    Args:
        pddl: Pddl instance.
        boxes: [O, 4] (hand/a/b/c, x1/y1/x2/y2) boxes with -1 or nan for null boxes.
    Returns:
        [MC, M + 1, 4] (arg_combos, roi/a/b, x1/y1/x2/y2) boxes with -inf for null boxes.
    """
    # [4, 4] (hand/a/b/c, x1/y1/x2/y2)
    boxes = np.array(boxes)
    boxes[boxes < 0] = np.nan

    # [5, 4] (hand/a/b/c/nan, x1/y1/x2/y2)
    # NaN channel gets selected by parameter to arg combo map.
    O = len(pddl.objects)  # noqa: E741
    param_boxes = np.concatenate(
        (boxes, np.full((O + 1 - boxes.shape[1], 4), np.nan, dtype=np.float32)), axis=0
    )

    # [16, 2] (num_arg_combos, num_args)
    idx_param_to_arg_combo = param_to_arg_combo_indices(pddl)

    # [MC, M + 1, 4] (arg_combos, roi/a/b, x1/y1/x2/y2)
    MC, M = idx_param_to_arg_combo.shape
    boxes = np.empty((MC, M + 1, 4), np.float32)

    # [5, 4] -> [MC, M, 4] (arg_combos, num_args, x1/y1/x2/y2)
    boxes[:, 1:, :] = param_boxes[idx_param_to_arg_combo, :]

    # [MC, M, 2] -> [MC, 2] (arg_combos, x/y)
    boxes[:, 0, :2] = np.nanmin(boxes[:, 1:, :2], axis=1)  # xy1
    boxes[:, 0, 2:] = np.nanmax(boxes[:, 1:, 2:], axis=1)  # xy2

    # Convert nans to -inf.
    boxes[np.isnan(boxes)] = -float("inf")

    return boxes
