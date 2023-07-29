# class Action:
#     def __init__(self, name, args):
#         self.name = name
#         self.args = args

#     def __repr__(self):
#         return "{}({})".format(self.name, ", ".join([str(arg) for arg in self.args]))


# def goto(world, obj):
#     agent = world.objects["agent"]


# class Goto(Action):
#     def __init__(self, world, obj):
#         super().__init__("goto", obj)


# def apply(world, str_action):
#     # Apply action
#     str_state = world.state.stringify()
#     str_state_next = world.pddl.next_state(str_state, str_action)
#     world.state.update(str_state_next)
