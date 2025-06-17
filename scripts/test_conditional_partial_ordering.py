import logging
import unified_planning
from unified_planning.shortcuts import *

import ucpop


ucpop.search.logger.setLevel(logging.INFO)
ucpop.pop.logger.setLevel(logging.INFO)
ucpop.pop2.logger.setLevel(logging.INFO)
ucpop.classes.logger.setLevel(logging.INFO)

# set up logging to log file
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', # Format of log messages
    # filename='test.log',
    level=logging.INFO
)


# Define the types
Robot = UserType('Robot')

# Define fluents
done_a = Fluent('done_a')
done_b = Fluent('done_b')
sticky = Fluent('sticky', BoolType(), robot=Robot)

# Define the actions
do_a = InstantaneousAction('DoA', r=Robot)
r = do_a.parameter('r')
do_a.add_precondition(Not(done_a))
do_a.add_effect(done_a, True)
do_a.add_effect(sticky(r), True)

do_b = InstantaneousAction('DoB', r=Robot)
r = do_b.parameter('r')
do_b.add_precondition(Not(done_b))
do_b.add_precondition(Not(sticky(r)))
do_b.add_effect(done_b, True)

# Define objects
robot1 = Object('robot1', Robot)
robot2 = Object('robot2', Robot)

# Create the problem
problem = Problem('StickyRobots')
problem.add_fluent(done_a)
problem.add_fluent(done_b)
problem.add_fluent(sticky)
problem.add_action(do_a)
problem.add_action(do_b)
problem.add_object(robot1)
problem.add_object(robot2)

# Set initial values
problem.set_initial_value(done_a, False)
problem.set_initial_value(done_b, False)
problem.set_initial_value(sticky(robot1), False)
problem.set_initial_value(sticky(robot2), False)

# Set goals
problem.add_goal(done_a)
problem.add_goal(done_b)

# Optional: print the problem
print(problem)

### Solving

# %%
with OneshotPlanner(name='pcop') as planner:
    result = planner.solve(problem)
    if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
        print("Planner returned: %s" % result.plan)
    else:
        print("No plan found.")
