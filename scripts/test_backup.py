import unified_planning
from unified_planning.shortcuts import *

import multi_agent_rekeps.planning


Location = UserType('Location')

robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
robot_has_been_at = unified_planning.model.Fluent('robot_has_been_at', BoolType(), l=Location)
connected = unified_planning.model.Fluent('connected', BoolType(), l_from=Location, l_to=Location)

problem = unified_planning.model.Problem('robot')
problem.add_fluent(robot_at, default_initial_value=False)
problem.add_fluent(robot_has_been_at, default_initial_value=False)
problem.add_fluent(connected, default_initial_value=False)

NLOC = 5
locations = [unified_planning.model.Object('l%s' % i, Location) for i in range(NLOC)]
problem.add_objects(locations)

move = unified_planning.model.InstantaneousAction('move', l_from=Location, l_to=Location)
l_from = move.parameter('l_from')
l_to = move.parameter('l_to')
move.add_precondition(connected(l_from, l_to))
move.add_precondition(robot_at(l_from))
move.add_effect(robot_at(l_from), False)
move.add_effect(robot_at(l_to), True)
move.add_effect(robot_has_been_at(l_to), True)

problem.add_action(move)

problem.set_initial_value(robot_at(locations[0]), True)
problem.set_initial_value(robot_has_been_at(locations[0]), True)

problem.set_initial_value(connected(locations[0], locations[1]), True)
problem.set_initial_value(connected(locations[1], locations[2]), True)
problem.set_initial_value(connected(locations[1], locations[3]), True)
problem.set_initial_value(connected(locations[2], locations[3]), True)
problem.set_initial_value(connected(locations[3], locations[2]), True)

problem.set_initial_value(connected(locations[2], locations[4]), True)
problem.set_initial_value(connected(locations[3], locations[4]), True)

problem.add_goal(robot_at(locations[-1]))
# problem.add_goal(robot_has_been_at(locations[2]))
# problem.add_goal(robot_has_been_at(locations[3]))
print(problem)

# %% [markdown]
# ### Solving Planning Problems
#
# The most direct way to solve a planning problem is to select an available planning engine by name and use it to solve the problem. In the following we use `pyperplan` to solve the problem and print the plan.

# %%
with OneshotPlanner(name='pop') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("Planner returned: %s" % result.plan)
    else:
        print("No plan found.")

# %% [markdown]
# The unified_planning can also automatically select, among the available planners installed on the system, one that is expressive enough for the problem at hand.

# %%
# with OneshotPlanner(problem_kind=problem.kind) as planner:
#     result = planner.solve(problem)
#     print("%s returned: %s" % (planner.name, result.plan))

# %% [markdown]
# In this example, Pyperplan was selected. The `problem.kind` property, returns an object that describes the characteristics of the problem.

# # %%
# print(problem.kind.features)

# %% [markdown]
# #### Beyond plan generation

# %% [markdown]
# `OneshotPlanner` is not the only operation mode we can invoke from the unified_planning, it is just one way to interact with a planning engine. Another useful functionality is `PlanValidation` that checks if a plan is valid for a problem.

# # %%
# plan = result.plan
# with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
#     if validator.validate(problem, plan):
#         print('The plan is valid')
#     else:
#         print('The plan is invalid')

# %% [markdown]
# It is also possible to use the `Compiler` operation mode with `compilation_kind=CompilationKind.GROUNDING` to create an equivalent formulation of a problem that does not use parameters for the actions.
#
# For an in-depth tutorial about the `Compiler` operation mode check the [Notebook on Compilers](https://colab.research.google.com/github/aiplan4eu/unified-planning/blob/master/notebooks/Compilers_example.ipynb).

# # %%
# with Compiler(problem_kind=problem.kind, compilation_kind=CompilationKind.GROUNDING) as grounder:
#     grounding_result = grounder.compile(problem, CompilationKind.GROUNDING)
#     ground_problem = grounding_result.problem
#     print(ground_problem)

#     # The grounding_result can be used to "lift" a ground plan back to the level of the original problem
#     with OneshotPlanner(problem_kind=ground_problem.kind) as planner:
#         ground_plan = planner.solve(ground_problem).plan
#         print('Ground plan: %s' % ground_plan)
#         # Replace the action instances of the grounded plan with their correspoding lifted version
#         lifted_plan = ground_plan.replace_action_instances(grounding_result.map_back_action_instance)
#         print('Lifted plan: %s' % lifted_plan)
#         # Test the problem and plan validity
#         with PlanValidator(problem_kind=problem.kind, plan_kind=ground_plan.kind) as validator:
#             ground_validation = validator.validate(ground_problem, ground_plan)
#             lift_validation = validator.validate(problem, lifted_plan)
#             Valid = up.engines.ValidationResultStatus.VALID
#             assert ground_validation.status == Valid
#             assert lift_validation.status == Valid

# %% [markdown]
# #### Parallel planning

# %% [markdown]
# We can invoke different instances of a planner in parallel or different planners and return the first plan that is generated effortlessly.

# # %%
# with OneshotPlanner(names=['tamer', 'tamer', 'pyperplan'],
#                     params=[{'heuristic': 'hadd'}, {'heuristic': 'hmax'}, {}]) as planner:
#     plan = planner.solve(problem).plan
#     print("%s returned: %s" % (planner.name, plan))

# # %%
# from unified_planning.plot import plot_sequential_plan

# # %% [markdown]
# # Ignore the code below, it's used to make this notebook also runnable in the Countinuous Intergation.

# # %%
# # Redefine the plot package methods imported above to print the plot to a temp file
# # if the exception "could not locate runnable browser" is raised. This usually happens
# # in the Continuous Integration.

# from inspect import getmembers, isfunction
# from unified_planning import plot
# from functools import partial
# import os, uuid, tempfile as tf

# # Define the function that will be executed instead
# def _function(original_function, *args, **kwargs):
#     try:
#         original_function(*args, **kwargs)
#     except Exception as e:
#         if "could not locate runnable browser" in str(e):
#             original_function(*args, **kwargs,
#                 filename=f"{os.path.join(tf.gettempdir(), str(uuid.uuid1()))}.png"
#             )
#         else:
#             raise e

# # Iterate over all the functions of the plot package
# for function_name, function in getmembers(plot, isfunction):
#     # Override the original function with the new one
#     globals()[function_name] = partial(_function, function)

# # %%
# if plan is not None:
#     plot_sequential_plan(plan, figsize=(16, 4), node_size=4000, font_size=10)
