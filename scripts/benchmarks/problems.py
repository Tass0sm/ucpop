import logging
import unified_planning
from unified_planning.shortcuts import *

import ucpop

# Keep solver logs quiet during benchmarking.
ucpop.search.logger.setLevel(logging.ERROR)
ucpop.pop.logger.setLevel(logging.ERROR)
ucpop.pop2.logger.setLevel(logging.ERROR)
ucpop.classes.logger.setLevel(logging.ERROR)

get_environment().credits_stream = None

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

def make_bowl_delivery_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')
    Bowl = UserType('Bowl')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )
    bowl_at = unified_planning.model.Fluent(
        'bowl_at', BoolType(), b=Bowl, l=Location
    )
    holding = unified_planning.model.Fluent('holding', BoolType(), b=Bowl)
    handempty = unified_planning.model.Fluent('handempty', BoolType())

    problem = unified_planning.model.Problem('bowl_delivery')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(bowl_at, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(handempty, default_initial_value=False)

    nloc = 5
    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(nloc)
    ]
    problem.add_objects(locations)

    bowl1 = unified_planning.model.Object('bowl1', Bowl)
    problem.add_object(bowl1)

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location
    )
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(connected(l_from, l_to))
    move.add_precondition(robot_at(l_from))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)

    pick = unified_planning.model.InstantaneousAction(
        'pick', b=Bowl, l=Location
    )
    b = pick.parameter('b')
    l = pick.parameter('l')
    pick.add_precondition(robot_at(l))
    pick.add_precondition(bowl_at(b, l))
    pick.add_precondition(handempty())
    pick.add_effect(bowl_at(b, l), False)
    pick.add_effect(holding(b), True)
    pick.add_effect(handempty(), False)

    place = unified_planning.model.InstantaneousAction(
        'place', b=Bowl, l=Location
    )
    b = place.parameter('b')
    l = place.parameter('l')
    place.add_precondition(robot_at(l))
    place.add_precondition(holding(b))
    place.add_effect(bowl_at(b, l), True)
    place.add_effect(holding(b), False)
    place.add_effect(handempty(), True)

    problem.add_action(move)
    problem.add_action(pick)
    problem.add_action(place)

    # Initial state
    problem.set_initial_value(robot_at(locations[0]), True)
    problem.set_initial_value(bowl_at(bowl1, locations[3]), True)
    problem.set_initial_value(handempty(), True)

    # Same small directed graph shape as the first benchmark.
    problem.set_initial_value(connected(locations[0], locations[1]), True)
    problem.set_initial_value(connected(locations[1], locations[2]), True)
    problem.set_initial_value(connected(locations[1], locations[3]), True)
    problem.set_initial_value(connected(locations[2], locations[3]), True)
    problem.set_initial_value(connected(locations[3], locations[2]), True)
    problem.set_initial_value(connected(locations[2], locations[4]), True)
    problem.set_initial_value(connected(locations[3], locations[4]), True)

    # Deliver the bowl to l4.
    problem.add_goal(bowl_at(bowl1, locations[4]))
    return problem


def make_fill_and_deliver_bowl_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')
    Bowl = UserType('Bowl')
    WaterSource = UserType('WaterSource')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )
    bowl_at = unified_planning.model.Fluent(
        'bowl_at', BoolType(), b=Bowl, l=Location
    )
    source_at = unified_planning.model.Fluent(
        'source_at', BoolType(), s=WaterSource, l=Location
    )
    holding = unified_planning.model.Fluent('holding', BoolType(), b=Bowl)
    handempty = unified_planning.model.Fluent('handempty', BoolType())
    bowl_filled = unified_planning.model.Fluent('bowl_filled', BoolType(), b=Bowl)
    water_delivered = unified_planning.model.Fluent(
        'water_delivered', BoolType(), l=Location
    )

    problem = unified_planning.model.Problem('fill_and_deliver_bowl')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(bowl_at, default_initial_value=False)
    problem.add_fluent(source_at, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(handempty, default_initial_value=False)
    problem.add_fluent(bowl_filled, default_initial_value=False)
    problem.add_fluent(water_delivered, default_initial_value=False)

    nloc = 5
    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(nloc)
    ]
    problem.add_objects(locations)

    bowl1 = unified_planning.model.Object('bowl1', Bowl)
    faucet1 = unified_planning.model.Object('faucet1', WaterSource)
    problem.add_objects([bowl1, faucet1])

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location
    )
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(connected(l_from, l_to))
    move.add_precondition(robot_at(l_from))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)

    pick = unified_planning.model.InstantaneousAction(
        'pick', b=Bowl, l=Location
    )
    b = pick.parameter('b')
    l = pick.parameter('l')
    pick.add_precondition(robot_at(l))
    pick.add_precondition(bowl_at(b, l))
    pick.add_precondition(handempty())
    pick.add_effect(bowl_at(b, l), False)
    pick.add_effect(holding(b), True)
    pick.add_effect(handempty(), False)

    fill = unified_planning.model.InstantaneousAction(
        'fill', b=Bowl, s=WaterSource, l=Location
    )
    b = fill.parameter('b')
    s = fill.parameter('s')
    l = fill.parameter('l')
    fill.add_precondition(robot_at(l))
    fill.add_precondition(holding(b))
    fill.add_precondition(source_at(s, l))
    fill.add_effect(bowl_filled(b), True)

    pour = unified_planning.model.InstantaneousAction(
        'pour', b=Bowl, l=Location
    )
    b = pour.parameter('b')
    l = pour.parameter('l')
    pour.add_precondition(robot_at(l))
    pour.add_precondition(holding(b))
    pour.add_precondition(bowl_filled(b))
    pour.add_effect(bowl_filled(b), False)
    pour.add_effect(water_delivered(l), True)

    problem.add_action(move)
    problem.add_action(pick)
    problem.add_action(fill)
    problem.add_action(pour)

    # Initial state
    problem.set_initial_value(robot_at(locations[0]), True)
    problem.set_initial_value(bowl_at(bowl1, locations[3]), True)
    problem.set_initial_value(source_at(faucet1, locations[2]), True)
    problem.set_initial_value(handempty(), True)

    # Same directed graph.
    problem.set_initial_value(connected(locations[0], locations[1]), True)
    problem.set_initial_value(connected(locations[1], locations[2]), True)
    problem.set_initial_value(connected(locations[1], locations[3]), True)
    problem.set_initial_value(connected(locations[2], locations[3]), True)
    problem.set_initial_value(connected(locations[3], locations[2]), True)
    problem.set_initial_value(connected(locations[2], locations[4]), True)
    problem.set_initial_value(connected(locations[3], locations[4]), True)

    # Goal: deliver the water to l4 by pouring there.
    problem.add_goal(water_delivered(locations[4]))
    return problem

def make_robot_reach_2_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )

    problem = unified_planning.model.Problem('robot_reach_2')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)

    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(3)
    ]
    problem.add_objects(locations)

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location
    )
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

    problem.add_goal(robot_at(locations[2]))
    return problem


def make_robot_visit_2_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    robot_has_been_at = unified_planning.model.Fluent(
        'robot_has_been_at', BoolType(), l=Location
    )
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )

    problem = unified_planning.model.Problem('robot_visit_2')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(robot_has_been_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)

    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(3)
    ]
    problem.add_objects(locations)

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location
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

    problem.add_goal(robot_has_been_at(locations[1]))
    problem.add_goal(robot_at(locations[2]))
    return problem


def make_bowl_delivery_small_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')
    Bowl = UserType('Bowl')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )
    bowl_at = unified_planning.model.Fluent(
        'bowl_at', BoolType(), b=Bowl, l=Location
    )
    holding = unified_planning.model.Fluent('holding', BoolType(), b=Bowl)
    handempty = unified_planning.model.Fluent('handempty', BoolType())

    problem = unified_planning.model.Problem('bowl_delivery_small')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(bowl_at, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(handempty, default_initial_value=False)

    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(3)
    ]
    problem.add_objects(locations)

    bowl1 = unified_planning.model.Object('bowl1', Bowl)
    problem.add_object(bowl1)

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location
    )
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(connected(l_from, l_to))
    move.add_precondition(robot_at(l_from))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)

    pick = unified_planning.model.InstantaneousAction(
        'pick', b=Bowl, l=Location
    )
    b = pick.parameter('b')
    l = pick.parameter('l')
    pick.add_precondition(robot_at(l))
    pick.add_precondition(bowl_at(b, l))
    pick.add_precondition(handempty())
    pick.add_effect(bowl_at(b, l), False)
    pick.add_effect(holding(b), True)
    pick.add_effect(handempty(), False)

    place = unified_planning.model.InstantaneousAction(
        'place', b=Bowl, l=Location
    )
    b = place.parameter('b')
    l = place.parameter('l')
    place.add_precondition(robot_at(l))
    place.add_precondition(holding(b))
    place.add_effect(bowl_at(b, l), True)
    place.add_effect(holding(b), False)
    place.add_effect(handempty(), True)

    problem.add_action(move)
    problem.add_action(pick)
    problem.add_action(place)

    problem.set_initial_value(robot_at(locations[0]), True)
    problem.set_initial_value(bowl_at(bowl1, locations[1]), True)
    problem.set_initial_value(handempty(), True)

    problem.set_initial_value(connected(locations[0], locations[1]), True)
    problem.set_initial_value(connected(locations[1], locations[2]), True)

    problem.add_goal(bowl_at(bowl1, locations[2]))
    return problem


def make_fill_and_deliver_bowl_small_problem() -> unified_planning.model.Problem:
    Location = UserType('Location')
    Bowl = UserType('Bowl')
    WaterSource = UserType('WaterSource')

    robot_at = unified_planning.model.Fluent('robot_at', BoolType(), l=Location)
    connected = unified_planning.model.Fluent(
        'connected', BoolType(), l_from=Location, l_to=Location
    )
    bowl_at = unified_planning.model.Fluent(
        'bowl_at', BoolType(), b=Bowl, l=Location
    )
    source_at = unified_planning.model.Fluent(
        'source_at', BoolType(), s=WaterSource, l=Location
    )
    holding = unified_planning.model.Fluent('holding', BoolType(), b=Bowl)
    handempty = unified_planning.model.Fluent('handempty', BoolType())
    bowl_filled = unified_planning.model.Fluent('bowl_filled', BoolType(), b=Bowl)
    water_delivered = unified_planning.model.Fluent(
        'water_delivered', BoolType(), l=Location
    )

    problem = unified_planning.model.Problem('fill_and_deliver_bowl_small')
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    problem.add_fluent(bowl_at, default_initial_value=False)
    problem.add_fluent(source_at, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(handempty, default_initial_value=False)
    problem.add_fluent(bowl_filled, default_initial_value=False)
    problem.add_fluent(water_delivered, default_initial_value=False)

    locations = [
        unified_planning.model.Object(f'l{i}', Location)
        for i in range(4)
    ]
    problem.add_objects(locations)

    bowl1 = unified_planning.model.Object('bowl1', Bowl)
    faucet1 = unified_planning.model.Object('faucet1', WaterSource)
    problem.add_objects([bowl1, faucet1])

    move = unified_planning.model.InstantaneousAction(
        'move', l_from=Location, l_to=Location
    )
    l_from = move.parameter('l_from')
    l_to = move.parameter('l_to')
    move.add_precondition(connected(l_from, l_to))
    move.add_precondition(robot_at(l_from))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)

    pick = unified_planning.model.InstantaneousAction(
        'pick', b=Bowl, l=Location
    )
    b = pick.parameter('b')
    l = pick.parameter('l')
    pick.add_precondition(robot_at(l))
    pick.add_precondition(bowl_at(b, l))
    pick.add_precondition(handempty())
    pick.add_effect(bowl_at(b, l), False)
    pick.add_effect(holding(b), True)
    pick.add_effect(handempty(), False)

    fill = unified_planning.model.InstantaneousAction(
        'fill', b=Bowl, s=WaterSource, l=Location
    )
    b = fill.parameter('b')
    s = fill.parameter('s')
    l = fill.parameter('l')
    fill.add_precondition(robot_at(l))
    fill.add_precondition(holding(b))
    fill.add_precondition(source_at(s, l))
    fill.add_effect(bowl_filled(b), True)

    pour = unified_planning.model.InstantaneousAction(
        'pour', b=Bowl, l=Location
    )
    b = pour.parameter('b')
    l = pour.parameter('l')
    pour.add_precondition(robot_at(l))
    pour.add_precondition(holding(b))
    pour.add_precondition(bowl_filled(b))
    pour.add_effect(bowl_filled(b), False)
    pour.add_effect(water_delivered(l), True)

    problem.add_action(move)
    problem.add_action(pick)
    problem.add_action(fill)
    problem.add_action(pour)

    problem.set_initial_value(robot_at(locations[0]), True)
    problem.set_initial_value(bowl_at(bowl1, locations[1]), True)
    problem.set_initial_value(source_at(faucet1, locations[2]), True)
    problem.set_initial_value(handempty(), True)

    problem.set_initial_value(connected(locations[0], locations[1]), True)
    problem.set_initial_value(connected(locations[1], locations[2]), True)
    problem.set_initial_value(connected(locations[2], locations[3]), True)

    problem.add_goal(water_delivered(locations[3]))
    return problem


PROBLEM_BUILDERS = {
    'robot_reach_2': make_robot_reach_2_problem,
    'robot_visit_2': make_robot_visit_2_problem,
    'bowl_delivery_small': make_bowl_delivery_small_problem,
    'fill_and_deliver_bowl_small': make_fill_and_deliver_bowl_small_problem,
}


