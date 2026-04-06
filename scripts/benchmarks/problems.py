import logging
import unified_planning
from unified_planning.shortcuts import *

import ucpop

# Keep solver logs quiet during benchmarking.
ucpop.search.logger.setLevel(logging.ERROR)
ucpop.pop.logger.setLevel(logging.ERROR)
ucpop.pop2.logger.setLevel(logging.ERROR)
ucpop.classes.logger.setLevel(logging.ERROR)

def make_robot_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')
    Hat = UserType('Hat')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    robot_has_been_at = unified_planning.model.Fluent('robot_has_been_at', BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )

    problem = unified_planning.model.Problem('robot')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(robot_has_been_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)

    nloc = 5
    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(nloc)
    ]
    problem.add_objects(locations)

    hat1 = unified_planning.model.Object('hat1', Hat)
    hat2 = unified_planning.model.Object('hat2', Hat)
    problem.add_objects([hat1, hat2])

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location, hat=Hat
    )
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

    problem.add_goal(robot_at(locations[2]))
    return problem

PROBLEM_BUILDERS = {
    'robot_reach_l2': make_robot_problem,
}


