import unified_planning
from unified_planning.shortcuts import *

import ucpop


Location = UserType('Location')

robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
connected = unified_planning.model.Fluent('connected', BoolType(), l_from=Location, l_to=Location)

problem = unified_planning.model.Problem('robot')
problem.add_fluent(robot_at, default_initial_value=False)
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

problem.add_action(move)

problem.set_initial_value(robot_at(locations[0]), True)

problem.set_initial_value(connected(locations[0], locations[1]), True)
problem.set_initial_value(connected(locations[1], locations[2]), True)
problem.set_initial_value(connected(locations[1], locations[3]), True)
problem.set_initial_value(connected(locations[2], locations[3]), True)
problem.set_initial_value(connected(locations[3], locations[2]), True)

problem.set_initial_value(connected(locations[2], locations[4]), True)
problem.set_initial_value(connected(locations[3], locations[4]), True)

problem.add_goal(robot_at(locations[-1]))
print(problem)

### Solving

# %%
with OneshotPlanner(name='pop') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("Planner returned: %s" % result.plan)
    else:
        print("No plan found.")
