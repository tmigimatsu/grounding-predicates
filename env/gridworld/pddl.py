#!/usr/bin/env python3

# TODO: Turn into test

import symbolic

if __name__ == "__main__":
    pddl = symbolic.Pddl("domain.pddl", "problem.pddl")
    planner = symbolic.Planner(pddl)
    print(planner.root, "\n")
    bfs = symbolic.BreadthFirstSearch(planner.root, max_depth=14)

    # for plan in bfs:
    #     for node in plan[1:]:
    #         print("{}: {}".format(node.depth, node.action))

    # pre, post = symbolic.NormalizeConditions(pddl, "goto(chest)")
    # print(pre)
    # print(post)
    # print(len(post.conjunctions))

    # exit()

    s = pddl.initial_state
    print(s)

    action_skeleton = [
        "goto(door_key)",
        "pick(door_key, room_a)",
        "goto(door)",
        "unlock(door, door_key)",
        "place(door_key, room_a)",
        "open(door)",
        "goto(chest_key)",
        "pick(chest_key, room_a)",
        "enter(room_b, door)",
        "goto(chest)",
        "unlock(chest, chest_key)",
        "place(chest_key, room_b)",
        "open(chest)",
        "pick(trophy, chest)",
    ]
    print(len(action_skeleton))

    for a in action_skeleton:
        # print(s)
        print(pddl.list_valid_actions(s))
        print(s)
        print(a)

        assert pddl.is_valid_action(s, a)
        s = pddl.next_state(s, a)

    # assert pddl.is_goal_satisfied(s)
    assert pddl.is_valid_plan(action_skeleton)
