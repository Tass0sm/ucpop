import unified_planning
from unified_planning.shortcuts import *


def _add_pick_place_actions(problem, Agent, Block, Cell, *, use_access=True):
    at = unified_planning.model.Fluent('at', BoolType(), b=Block, c=Cell)
    clear = unified_planning.model.Fluent('clear', BoolType(), c=Cell)
    holding = unified_planning.model.Fluent('holding', BoolType(), a=Agent, b=Block)
    handempty = unified_planning.model.Fluent('handempty', BoolType(), a=Agent)
    reachable = unified_planning.model.Fluent('reachable', BoolType(), a=Agent, c=Cell)
    access_for = unified_planning.model.Fluent('access_for', BoolType(), c=Cell, acc=Cell)

    problem.add_fluent(at, default_initial_value=False)
    problem.add_fluent(clear, default_initial_value=False)
    problem.add_fluent(holding, default_initial_value=False)
    problem.add_fluent(handempty, default_initial_value=False)
    problem.add_fluent(reachable, default_initial_value=False)
    if use_access:
        problem.add_fluent(access_for, default_initial_value=False)

    if use_access:
        pick = unified_planning.model.InstantaneousAction(
            'pick', a=Agent, b=Block, c=Cell, acc=Cell
        )
        a = pick.parameter('a')
        b = pick.parameter('b')
        c = pick.parameter('c')
        acc = pick.parameter('acc')
        pick.add_precondition(reachable(a, c))
        pick.add_precondition(at(b, c))
        pick.add_precondition(handempty(a))
        pick.add_precondition(access_for(c, acc))
        pick.add_precondition(clear(acc))
        pick.add_effect(at(b, c), False)
        pick.add_effect(clear(c), True)
        pick.add_effect(holding(a, b), True)
        pick.add_effect(handempty(a), False)

        place = unified_planning.model.InstantaneousAction(
            'place', a=Agent, b=Block, c=Cell, acc=Cell
        )
        a = place.parameter('a')
        b = place.parameter('b')
        c = place.parameter('c')
        acc = place.parameter('acc')
        place.add_precondition(reachable(a, c))
        place.add_precondition(holding(a, b))
        place.add_precondition(clear(c))
        place.add_precondition(access_for(c, acc))
        place.add_precondition(clear(acc))
        place.add_effect(at(b, c), True)
        place.add_effect(clear(c), False)
        place.add_effect(holding(a, b), False)
        place.add_effect(handempty(a), True)
    else:
        pick = unified_planning.model.InstantaneousAction('pick', a=Agent, b=Block, c=Cell)
        a = pick.parameter('a')
        b = pick.parameter('b')
        c = pick.parameter('c')
        pick.add_precondition(reachable(a, c))
        pick.add_precondition(at(b, c))
        pick.add_precondition(handempty(a))
        pick.add_effect(at(b, c), False)
        pick.add_effect(clear(c), True)
        pick.add_effect(holding(a, b), True)
        pick.add_effect(handempty(a), False)

        place = unified_planning.model.InstantaneousAction('place', a=Agent, b=Block, c=Cell)
        a = place.parameter('a')
        b = place.parameter('b')
        c = place.parameter('c')
        place.add_precondition(reachable(a, c))
        place.add_precondition(holding(a, b))
        place.add_precondition(clear(c))
        place.add_effect(at(b, c), True)
        place.add_effect(clear(c), False)
        place.add_effect(holding(a, b), False)
        place.add_effect(handempty(a), True)

    problem.add_action(pick)
    problem.add_action(place)
    return {
        'at': at,
        'clear': clear,
        'holding': holding,
        'handempty': handempty,
        'reachable': reachable,
        'access_for': access_for if use_access else None,
    }


def make_blocks_clearance_small() -> unified_planning.model.Problem:
    """Two agents clear two blockers to free a target block and its goal cell."""
    Agent = UserType('Agent')
    Block = UserType('Block')
    Cell = UserType('Cell')

    problem = unified_planning.model.Problem('blocks_clearance_small')
    fluents = _add_pick_place_actions(problem, Agent, Block, Cell, use_access=True)
    at = fluents['at']
    clear = fluents['clear']
    handempty = fluents['handempty']
    reachable = fluents['reachable']
    access_for = fluents['access_for']

    arm1 = unified_planning.model.Object('arm1', Agent)
    arm2 = unified_planning.model.Object('arm2', Agent)
    target = unified_planning.model.Object('target', Block)
    obs1 = unified_planning.model.Object('obs1', Block)
    obs2 = unified_planning.model.Object('obs2', Block)
    c0 = unified_planning.model.Object('c0', Cell)
    c1 = unified_planning.model.Object('c1', Cell)
    c2 = unified_planning.model.Object('c2', Cell)
    c3 = unified_planning.model.Object('c3', Cell)
    c4 = unified_planning.model.Object('c4', Cell)
    c5 = unified_planning.model.Object('c5', Cell)
    problem.add_objects([arm1, arm2, target, obs1, obs2, c0, c1, c2, c3, c4, c5])

    for a in [arm1, arm2]:
        for c in [c0, c1, c2, c3, c4, c5]:
            problem.set_initial_value(reachable(a, c), True)
        problem.set_initial_value(handempty(a), True)

    # Access structure: target start requires c2 clear; goal cell requires c3 clear.
    problem.set_initial_value(access_for(c0, c5), True)
    problem.set_initial_value(access_for(c1, c2), True)
    problem.set_initial_value(access_for(c2, c5), True)
    problem.set_initial_value(access_for(c3, c0), True)
    problem.set_initial_value(access_for(c4, c3), True)
    problem.set_initial_value(access_for(c5, c0), True)

    problem.set_initial_value(at(target, c1), True)
    problem.set_initial_value(at(obs1, c2), True)
    problem.set_initial_value(at(obs2, c3), True)

    for empty in [c0, c4, c5]:
        problem.set_initial_value(clear(empty), True)

    problem.add_goal(at(target, c4))
    return problem


def make_blocks_clearance_medium() -> unified_planning.model.Problem:
    """Three blockers create two independent clearance subgoals plus one parking bottleneck."""
    Agent = UserType('Agent')
    Block = UserType('Block')
    Cell = UserType('Cell')

    problem = unified_planning.model.Problem('blocks_clearance_medium')
    fluents = _add_pick_place_actions(problem, Agent, Block, Cell, use_access=True)
    at = fluents['at']
    clear = fluents['clear']
    handempty = fluents['handempty']
    reachable = fluents['reachable']
    access_for = fluents['access_for']

    agents = [unified_planning.model.Object(f'arm{i}', Agent) for i in [1, 2]]
    blocks = [
        unified_planning.model.Object('target', Block),
        unified_planning.model.Object('obs1', Block),
        unified_planning.model.Object('obs2', Block),
        unified_planning.model.Object('obs3', Block),
    ]
    cells = [unified_planning.model.Object(f'c{i}', Cell) for i in range(8)]
    problem.add_objects(agents + blocks + cells)
    target, obs1, obs2, obs3 = blocks
    c0, c1, c2, c3, c4, c5, c6, c7 = cells

    for a in agents:
        for c in cells:
            problem.set_initial_value(reachable(a, c), True)
        problem.set_initial_value(handempty(a), True)

    access_pairs = {
        c0: c7,
        c1: c2,
        c2: c7,
        c3: c4,
        c4: c7,
        c5: c6,
        c6: c0,
        c7: c0,
    }
    for cell, acc in access_pairs.items():
        problem.set_initial_value(access_for(cell, acc), True)

    placements = {
        target: c1,
        obs1: c2,
        obs2: c4,
        obs3: c6,
    }
    for b, c in placements.items():
        problem.set_initial_value(at(b, c), True)

    for empty in [c0, c3, c5, c7]:
        problem.set_initial_value(clear(empty), True)

    problem.add_goal(at(target, c5))
    return problem


def make_blocks_handover_small() -> unified_planning.model.Problem:
    """Target must cross a shared transfer cell because no agent can reach both zones."""
    Agent = UserType('Agent')
    Block = UserType('Block')
    Cell = UserType('Cell')

    problem = unified_planning.model.Problem('blocks_handover_small')
    fluents = _add_pick_place_actions(problem, Agent, Block, Cell, use_access=False)
    at = fluents['at']
    clear = fluents['clear']
    handempty = fluents['handempty']
    reachable = fluents['reachable']

    arm_left = unified_planning.model.Object('arm_left', Agent)
    arm_right = unified_planning.model.Object('arm_right', Agent)
    target = unified_planning.model.Object('target', Block)
    blocker = unified_planning.model.Object('blocker', Block)
    left_a = unified_planning.model.Object('left_a', Cell)
    left_b = unified_planning.model.Object('left_b', Cell)
    transfer = unified_planning.model.Object('transfer', Cell)
    right_a = unified_planning.model.Object('right_a', Cell)
    right_goal = unified_planning.model.Object('right_goal', Cell)
    problem.add_objects([
        arm_left, arm_right, target, blocker,
        left_a, left_b, transfer, right_a, right_goal,
    ])

    # Left arm owns left cells + transfer, right arm owns right cells + transfer.
    for c in [left_a, left_b, transfer]:
        problem.set_initial_value(reachable(arm_left, c), True)
    for c in [transfer, right_a, right_goal]:
        problem.set_initial_value(reachable(arm_right, c), True)
    problem.set_initial_value(handempty(arm_left), True)
    problem.set_initial_value(handempty(arm_right), True)

    problem.set_initial_value(at(target, left_a), True)
    problem.set_initial_value(at(blocker, right_goal), True)
    problem.set_initial_value(clear(left_b), True)
    problem.set_initial_value(clear(transfer), True)
    problem.set_initial_value(clear(right_a), True)

    problem.add_goal(at(target, right_goal))
    return problem


def make_blocks_handover_medium() -> unified_planning.model.Problem:
    """Two targets must cross zones through a single transfer bottleneck."""
    Agent = UserType('Agent')
    Block = UserType('Block')
    Cell = UserType('Cell')

    problem = unified_planning.model.Problem('blocks_handover_medium')
    fluents = _add_pick_place_actions(problem, Agent, Block, Cell, use_access=False)
    at = fluents['at']
    clear = fluents['clear']
    handempty = fluents['handempty']
    reachable = fluents['reachable']

    arm_left = unified_planning.model.Object('arm_left', Agent)
    arm_right = unified_planning.model.Object('arm_right', Agent)
    target1 = unified_planning.model.Object('target1', Block)
    target2 = unified_planning.model.Object('target2', Block)
    blocker = unified_planning.model.Object('blocker', Block)
    cells = [unified_planning.model.Object(name, Cell) for name in [
        'left_a', 'left_b', 'left_parking', 'transfer', 'right_a', 'right_b', 'right_goal1', 'right_goal2'
    ]]
    left_a, left_b, left_parking, transfer, right_a, right_b, right_goal1, right_goal2 = cells
    problem.add_objects([arm_left, arm_right, target1, target2, blocker] + cells)

    for c in [left_a, left_b, left_parking, transfer]:
        problem.set_initial_value(reachable(arm_left, c), True)
    for c in [transfer, right_a, right_b, right_goal1, right_goal2]:
        problem.set_initial_value(reachable(arm_right, c), True)
    problem.set_initial_value(handempty(arm_left), True)
    problem.set_initial_value(handempty(arm_right), True)

    problem.set_initial_value(at(target1, left_a), True)
    problem.set_initial_value(at(target2, left_b), True)
    problem.set_initial_value(at(blocker, right_goal1), True)

    for empty in [left_parking, transfer, right_a, right_b, right_goal2]:
        problem.set_initial_value(clear(empty), True)

    problem.add_goal(at(target1, right_goal1))
    problem.add_goal(at(target2, right_goal2))
    return problem
