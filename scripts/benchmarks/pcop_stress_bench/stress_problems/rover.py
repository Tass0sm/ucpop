import unified_planning
from unified_planning.shortcuts import *


def _add_rover_core(problem, Rover, Data, Location):
    at = unified_planning.model.Fluent('at', BoolType(), r=Rover, l=Location)
    edge = unified_planning.model.Fluent('edge', BoolType(), r=Rover, l_from=Location, l_to=Location)
    sample_target = unified_planning.model.Fluent('sample_target', BoolType(), d=Data, l=Location)
    image_target = unified_planning.model.Fluent('image_target', BoolType(), d=Data, l=Location)
    carrying = unified_planning.model.Fluent('carrying', BoolType(), r=Rover, d=Data)
    uploaded = unified_planning.model.Fluent('uploaded', BoolType(), d=Data)
    uplink = unified_planning.model.Fluent('uplink', BoolType(), l=Location)

    for fl in [at, edge, sample_target, image_target, carrying, uploaded, uplink]:
        problem.add_fluent(fl, default_initial_value=False)

    navigate = unified_planning.model.InstantaneousAction('navigate', r=Rover, l_from=Location, l_to=Location)
    r = navigate.parameter('r')
    l_from = navigate.parameter('l_from')
    l_to = navigate.parameter('l_to')
    navigate.add_precondition(at(r, l_from))
    navigate.add_precondition(edge(r, l_from, l_to))
    navigate.add_effect(at(r, l_from), False)
    navigate.add_effect(at(r, l_to), True)

    sample = unified_planning.model.InstantaneousAction('sample', r=Rover, d=Data, l=Location)
    r = sample.parameter('r')
    d = sample.parameter('d')
    l = sample.parameter('l')
    sample.add_precondition(at(r, l))
    sample.add_precondition(sample_target(d, l))
    sample.add_effect(carrying(r, d), True)

    observe = unified_planning.model.InstantaneousAction('observe', r=Rover, d=Data, l=Location)
    r = observe.parameter('r')
    d = observe.parameter('d')
    l = observe.parameter('l')
    observe.add_precondition(at(r, l))
    observe.add_precondition(image_target(d, l))
    observe.add_effect(carrying(r, d), True)

    relay = unified_planning.model.InstantaneousAction('relay', r_from=Rover, r_to=Rover, d=Data, l=Location)
    r_from = relay.parameter('r_from')
    r_to = relay.parameter('r_to')
    d = relay.parameter('d')
    l = relay.parameter('l')
    relay.add_precondition(at(r_from, l))
    relay.add_precondition(at(r_to, l))
    relay.add_precondition(carrying(r_from, d))
    relay.add_effect(carrying(r_from, d), False)
    relay.add_effect(carrying(r_to, d), True)

    uplink_action = unified_planning.model.InstantaneousAction('uplink_data', r=Rover, d=Data, l=Location)
    r = uplink_action.parameter('r')
    d = uplink_action.parameter('d')
    l = uplink_action.parameter('l')
    uplink_action.add_precondition(at(r, l))
    uplink_action.add_precondition(uplink(l))
    uplink_action.add_precondition(carrying(r, d))
    uplink_action.add_effect(carrying(r, d), False)
    uplink_action.add_effect(uploaded(d), True)

    problem.add_action(navigate)
    problem.add_action(sample)
    problem.add_action(observe)
    problem.add_action(relay)
    problem.add_action(uplink_action)

    return {
        'at': at,
        'edge': edge,
        'sample_target': sample_target,
        'image_target': image_target,
        'carrying': carrying,
        'uploaded': uploaded,
        'uplink': uplink,
    }


def _set_bidirectional_edges(problem, edge, rover, path):
    for l_from, l_to in zip(path, path[1:]):
        problem.set_initial_value(edge(rover, l_from, l_to), True)
        problem.set_initial_value(edge(rover, l_to, l_from), True)


def make_rover_relay_small() -> unified_planning.model.Problem:
    """Two rovers gather data in disjoint regions and relay at a center node."""
    Rover = UserType('Rover')
    Data = UserType('Data')
    Location = UserType('Location')

    problem = unified_planning.model.Problem('rover_relay_small')
    fl = _add_rover_core(problem, Rover, Data, Location)

    rover1 = unified_planning.model.Object('rover1', Rover)
    rover2 = unified_planning.model.Object('rover2', Rover)
    left = unified_planning.model.Object('left', Location)
    center = unified_planning.model.Object('center', Location)
    right = unified_planning.model.Object('right', Location)
    base = unified_planning.model.Object('base', Location)
    rock1 = unified_planning.model.Object('rock1', Data)
    obj1 = unified_planning.model.Object('obj1', Data)
    problem.add_objects([rover1, rover2, left, center, right, base, rock1, obj1])

    problem.set_initial_value(fl['at'](rover1, left), True)
    problem.set_initial_value(fl['at'](rover2, base), True)
    problem.set_initial_value(fl['sample_target'](rock1, left), True)
    problem.set_initial_value(fl['image_target'](obj1, right), True)
    problem.set_initial_value(fl['uplink'](base), True)

    _set_bidirectional_edges(problem, fl['edge'], rover1, [left, center])
    _set_bidirectional_edges(problem, fl['edge'], rover2, [base, center, right])

    problem.add_goal(fl['uploaded'](rock1))
    problem.add_goal(fl['uploaded'](obj1))
    return problem


def make_rover_relay_medium() -> unified_planning.model.Problem:
    """Three rovers must coordinate two relays across sparsely connected regions."""
    Rover = UserType('Rover')
    Data = UserType('Data')
    Location = UserType('Location')

    problem = unified_planning.model.Problem('rover_relay_medium')
    fl = _add_rover_core(problem, Rover, Data, Location)

    rover1 = unified_planning.model.Object('rover1', Rover)
    rover2 = unified_planning.model.Object('rover2', Rover)
    rover3 = unified_planning.model.Object('rover3', Rover)
    north = unified_planning.model.Object('north', Location)
    left = unified_planning.model.Object('left', Location)
    hub = unified_planning.model.Object('hub', Location)
    right = unified_planning.model.Object('right', Location)
    base = unified_planning.model.Object('base', Location)
    rock1 = unified_planning.model.Object('rock1', Data)
    rock2 = unified_planning.model.Object('rock2', Data)
    obj1 = unified_planning.model.Object('obj1', Data)
    problem.add_objects([rover1, rover2, rover3, north, left, hub, right, base, rock1, rock2, obj1])

    problem.set_initial_value(fl['at'](rover1, left), True)
    problem.set_initial_value(fl['at'](rover2, base), True)
    problem.set_initial_value(fl['at'](rover3, north), True)
    problem.set_initial_value(fl['sample_target'](rock1, left), True)
    problem.set_initial_value(fl['sample_target'](rock2, north), True)
    problem.set_initial_value(fl['image_target'](obj1, right), True)
    problem.set_initial_value(fl['uplink'](base), True)

    _set_bidirectional_edges(problem, fl['edge'], rover1, [left, hub])
    _set_bidirectional_edges(problem, fl['edge'], rover2, [base, hub, right])
    _set_bidirectional_edges(problem, fl['edge'], rover3, [north, hub])

    problem.add_goal(fl['uploaded'](rock1))
    problem.add_goal(fl['uploaded'](rock2))
    problem.add_goal(fl['uploaded'](obj1))
    return problem


def make_rover_relay_hard() -> unified_planning.model.Problem:
    """Three rovers must gather four data objects across two disconnected outer regions."""
    Rover = UserType('Rover')
    Data = UserType('Data')
    Location = UserType('Location')

    problem = unified_planning.model.Problem('rover_relay_hard')
    fl = _add_rover_core(problem, Rover, Data, Location)

    rover1 = unified_planning.model.Object('rover1', Rover)
    rover2 = unified_planning.model.Object('rover2', Rover)
    rover3 = unified_planning.model.Object('rover3', Rover)
    north = unified_planning.model.Object('north', Location)
    left = unified_planning.model.Object('left', Location)
    hub = unified_planning.model.Object('hub', Location)
    right = unified_planning.model.Object('right', Location)
    south = unified_planning.model.Object('south', Location)
    base = unified_planning.model.Object('base', Location)
    data_objs = [unified_planning.model.Object(name, Data) for name in ['rock1', 'rock2', 'obj1', 'obj2']]
    rock1, rock2, obj1, obj2 = data_objs
    problem.add_objects([rover1, rover2, rover3, north, left, hub, right, south, base] + data_objs)

    problem.set_initial_value(fl['at'](rover1, left), True)
    problem.set_initial_value(fl['at'](rover2, base), True)
    problem.set_initial_value(fl['at'](rover3, north), True)
    problem.set_initial_value(fl['sample_target'](rock1, left), True)
    problem.set_initial_value(fl['sample_target'](rock2, north), True)
    problem.set_initial_value(fl['image_target'](obj1, right), True)
    problem.set_initial_value(fl['image_target'](obj2, south), True)
    problem.set_initial_value(fl['uplink'](base), True)

    _set_bidirectional_edges(problem, fl['edge'], rover1, [left, hub])
    _set_bidirectional_edges(problem, fl['edge'], rover2, [base, hub, right, south])
    _set_bidirectional_edges(problem, fl['edge'], rover3, [north, hub])

    for d in data_objs:
        problem.add_goal(fl['uploaded'](d))
    return problem
