import logging
from unified_planning.shortcuts import *
from unified_planning.model import Problem, Fluent, InstantaneousAction, Object

import ucpop

ucpop.search.logger.setLevel(logging.INFO)
ucpop.pop.logger.setLevel(logging.INFO)
ucpop.pop2.logger.setLevel(logging.INFO)
ucpop.classes.logger.setLevel(logging.INFO)

# set up logging to log file
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', # Format of log messages
    filename='test.log',
    level=logging.INFO
)


# Define a single type for both blocks and locations
Location = UserType('Location')
Robot = UserType('Robot')

# Define fluents
holding = Fluent('holding', BoolType(), r=Robot)
on = Fluent('on', BoolType(), b=Location, l=Location)  # block b is on location l
clear = Fluent('clear', BoolType(), l=Location)        # location is clear

# Create objects
robot1 = Object('robot1', Robot)
robot2 = Object('robot2', Robot)
A = Object('A', Location)
B = Object('B', Location)
table = Object('table', Location)

# Define actions
pickup = InstantaneousAction('PickUp', r=Robot, b=Location, l=Location)
r, b, l = pickup.parameters
pickup.add_precondition(on(b, l))
pickup.add_precondition(clear(b))  # block b must be clear to be picked up
pickup.add_precondition(clear(l))  # location must be clear (optional, or remove)
pickup.add_precondition(Not(holding(r)))
pickup.add_effect(on(b, l), False)
pickup.add_effect(clear(l), True)
pickup.add_effect(clear(b), False)
pickup.add_effect(holding(r), True)

putdown = InstantaneousAction('PutDown', r=Robot, b=Location, l=Location)
r, b, l = putdown.parameters
putdown.add_precondition(holding(r))
putdown.add_precondition(clear(l))
putdown.add_precondition(Not(Equals(b, l)))  # can't put a block down on itself
putdown.add_effect(on(b, l), True)
putdown.add_effect(clear(l), False, condition=Not(Equals(l, table)))
putdown.add_effect(clear(b), True)
putdown.add_effect(holding(r), False)

# Create the problem
problem = Problem('BlockStackingRobots')
problem.add_fluent(holding, default_initial_value=False)
problem.add_fluent(on, default_initial_value=False)
problem.add_fluent(clear, default_initial_value=False)

problem.add_action(pickup)
problem.add_action(putdown)

problem.add_object(robot1)
problem.add_object(robot2)
problem.add_object(A)
problem.add_object(B)
problem.add_object(table)

# Initial state
problem.set_initial_value(on(A, table), True)
problem.set_initial_value(on(B, table), True)
problem.set_initial_value(clear(A), True)
problem.set_initial_value(clear(B), True)
problem.set_initial_value(clear(table), True)
problem.set_initial_value(holding(robot1), False)
problem.set_initial_value(holding(robot2), False)

# Goal: B on A
problem.add_goal(on(B, A))

# Optional: print the problem
print(problem)

### Solving

# %%
with OneshotPlanner(name='pop2') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("Planner returned: %s" % result.plan)
    else:
        print("No plan found.")
