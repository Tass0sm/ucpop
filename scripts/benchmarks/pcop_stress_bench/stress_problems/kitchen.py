import unified_planning
from unified_planning.shortcuts import *


def _add_kitchen_core(problem, Agent, Item, Slot):
    at = unified_planning.model.Fluent('at', BoolType(), i=Item, s=Slot)
    clear = unified_planning.model.Fluent('clear', BoolType(), s=Slot)
    holding = unified_planning.model.Fluent('holding', BoolType(), a=Agent, i=Item)
    handempty = unified_planning.model.Fluent('handempty', BoolType(), a=Agent)
    reachable = unified_planning.model.Fluent('reachable', BoolType(), a=Agent, s=Slot)
    dirty = unified_planning.model.Fluent('dirty', BoolType(), i=Item)
    clean = unified_planning.model.Fluent('clean', BoolType(), i=Item)
    raw = unified_planning.model.Fluent('raw', BoolType(), i=Item)
    cooked = unified_planning.model.Fluent('cooked', BoolType(), i=Item)
    sink_slot = unified_planning.model.Fluent('sink_slot', BoolType(), s=Slot)
    stove_slot = unified_planning.model.Fluent('stove_slot', BoolType(), s=Slot)
    pass_slot = unified_planning.model.Fluent('pass_slot', BoolType(), s=Slot)
    serve_slot = unified_planning.model.Fluent('serve_slot', BoolType(), s=Slot)

    for fl in [at, clear, holding, handempty, reachable, dirty, clean, raw, cooked,
               sink_slot, stove_slot, pass_slot, serve_slot]:
        problem.add_fluent(fl, default_initial_value=False)

    pick = unified_planning.model.InstantaneousAction('pick', a=Agent, i=Item, s=Slot)
    a = pick.parameter('a')
    i = pick.parameter('i')
    s = pick.parameter('s')
    pick.add_precondition(reachable(a, s))
    pick.add_precondition(at(i, s))
    pick.add_precondition(handempty(a))
    pick.add_effect(at(i, s), False)
    pick.add_effect(clear(s), True)
    pick.add_effect(holding(a, i), True)
    pick.add_effect(handempty(a), False)

    place = unified_planning.model.InstantaneousAction('place', a=Agent, i=Item, s=Slot)
    a = place.parameter('a')
    i = place.parameter('i')
    s = place.parameter('s')
    place.add_precondition(reachable(a, s))
    place.add_precondition(holding(a, i))
    place.add_precondition(clear(s))
    place.add_effect(at(i, s), True)
    place.add_effect(clear(s), False)
    place.add_effect(holding(a, i), False)
    place.add_effect(handempty(a), True)

    clean_action = unified_planning.model.InstantaneousAction('clean_item', a=Agent, i=Item, s=Slot)
    a = clean_action.parameter('a')
    i = clean_action.parameter('i')
    s = clean_action.parameter('s')
    clean_action.add_precondition(reachable(a, s))
    clean_action.add_precondition(at(i, s))
    clean_action.add_precondition(sink_slot(s))
    clean_action.add_precondition(dirty(i))
    clean_action.add_effect(dirty(i), False)
    clean_action.add_effect(clean(i), True)

    cook_action = unified_planning.model.InstantaneousAction('cook_item', a=Agent, i=Item, s=Slot)
    a = cook_action.parameter('a')
    i = cook_action.parameter('i')
    s = cook_action.parameter('s')
    cook_action.add_precondition(reachable(a, s))
    cook_action.add_precondition(at(i, s))
    cook_action.add_precondition(stove_slot(s))
    cook_action.add_precondition(raw(i))
    cook_action.add_precondition(clean(i))
    cook_action.add_effect(raw(i), False)
    cook_action.add_effect(cooked(i), True)

    problem.add_action(pick)
    problem.add_action(place)
    problem.add_action(clean_action)
    problem.add_action(cook_action)

    return {
        'at': at,
        'clear': clear,
        'holding': holding,
        'handempty': handempty,
        'reachable': reachable,
        'dirty': dirty,
        'clean': clean,
        'raw': raw,
        'cooked': cooked,
        'sink_slot': sink_slot,
        'stove_slot': stove_slot,
        'pass_slot': pass_slot,
        'serve_slot': serve_slot,
    }


def _add_basic_reachability(problem, reachable, chef_left, chef_right, *, left_slots, right_slots, shared_slots):
    for s in left_slots + shared_slots:
        problem.set_initial_value(reachable(chef_left, s), True)
    for s in right_slots + shared_slots:
        problem.set_initial_value(reachable(chef_right, s), True)


def make_kitchen_pipeline_small() -> unified_planning.model.Problem:
    """Two chefs must clean, hand off through a pass slot, and cook two items."""
    Agent = UserType('Agent')
    Item = UserType('Item')
    Slot = UserType('Slot')

    problem = unified_planning.model.Problem('kitchen_pipeline_small')
    fl = _add_kitchen_core(problem, Agent, Item, Slot)

    chef_left = unified_planning.model.Object('chef_left', Agent)
    chef_right = unified_planning.model.Object('chef_right', Agent)
    item1 = unified_planning.model.Object('item1', Item)
    item2 = unified_planning.model.Object('item2', Item)
    left_a = unified_planning.model.Object('left_a', Slot)
    left_b = unified_planning.model.Object('left_b', Slot)
    sink = unified_planning.model.Object('sink', Slot)
    pas = unified_planning.model.Object('pass', Slot)
    stove = unified_planning.model.Object('stove', Slot)
    serve1 = unified_planning.model.Object('serve1', Slot)
    serve2 = unified_planning.model.Object('serve2', Slot)
    problem.add_objects([chef_left, chef_right, item1, item2, left_a, left_b, sink, pas, stove, serve1, serve2])

    _add_basic_reachability(problem, fl['reachable'], chef_left, chef_right,
                            left_slots=[left_a, left_b, sink], right_slots=[stove, serve1, serve2], shared_slots=[pas])
    for a in [chef_left, chef_right]:
        problem.set_initial_value(fl['handempty'](a), True)

    problem.set_initial_value(fl['sink_slot'](sink), True)
    problem.set_initial_value(fl['stove_slot'](stove), True)
    problem.set_initial_value(fl['pass_slot'](pas), True)
    problem.set_initial_value(fl['serve_slot'](serve1), True)
    problem.set_initial_value(fl['serve_slot'](serve2), True)

    problem.set_initial_value(fl['at'](item1, left_a), True)
    problem.set_initial_value(fl['at'](item2, left_b), True)
    for s in [sink, pas, stove, serve1, serve2]:
        problem.set_initial_value(fl['clear'](s), True)

    for i in [item1, item2]:
        problem.set_initial_value(fl['dirty'](i), True)
        problem.set_initial_value(fl['raw'](i), True)

    problem.add_goal(fl['cooked'](item1))
    problem.add_goal(fl['cooked'](item2))
    problem.add_goal(fl['at'](item1, serve1))
    problem.add_goal(fl['at'](item2, serve2))
    return problem


def make_kitchen_pipeline_medium() -> unified_planning.model.Problem:
    """Three items share one pass slot and one stove, creating long chained subplans."""
    Agent = UserType('Agent')
    Item = UserType('Item')
    Slot = UserType('Slot')

    problem = unified_planning.model.Problem('kitchen_pipeline_medium')
    fl = _add_kitchen_core(problem, Agent, Item, Slot)

    chef_left = unified_planning.model.Object('chef_left', Agent)
    chef_right = unified_planning.model.Object('chef_right', Agent)
    items = [unified_planning.model.Object(f'item{i}', Item) for i in [1, 2, 3]]
    left_slots = [unified_planning.model.Object(f'left_{name}', Slot) for name in ['a', 'b', 'c']]
    sink = unified_planning.model.Object('sink', Slot)
    pas = unified_planning.model.Object('pass', Slot)
    stove = unified_planning.model.Object('stove', Slot)
    serves = [unified_planning.model.Object(f'serve{i}', Slot) for i in [1, 2, 3]]
    problem.add_objects([chef_left, chef_right, sink, pas, stove] + items + left_slots + serves)

    _add_basic_reachability(problem, fl['reachable'], chef_left, chef_right,
                            left_slots=left_slots + [sink], right_slots=[stove] + serves, shared_slots=[pas])
    for a in [chef_left, chef_right]:
        problem.set_initial_value(fl['handempty'](a), True)

    problem.set_initial_value(fl['sink_slot'](sink), True)
    problem.set_initial_value(fl['stove_slot'](stove), True)
    problem.set_initial_value(fl['pass_slot'](pas), True)
    for s in serves:
        problem.set_initial_value(fl['serve_slot'](s), True)

    for item, slot in zip(items, left_slots):
        problem.set_initial_value(fl['at'](item, slot), True)
        problem.set_initial_value(fl['dirty'](item), True)
        problem.set_initial_value(fl['raw'](item), True)
    for s in [sink, pas, stove] + serves:
        problem.set_initial_value(fl['clear'](s), True)

    for item, serve in zip(items, serves):
        problem.add_goal(fl['cooked'](item))
        problem.add_goal(fl['at'](item, serve))
    return problem


def make_kitchen_buffer_trap_small() -> unified_planning.model.Problem:
    """One sink, one stove, and one pass slot create an order-sensitive bottleneck."""
    Agent = UserType('Agent')
    Item = UserType('Item')
    Slot = UserType('Slot')

    problem = unified_planning.model.Problem('kitchen_buffer_trap_small')
    fl = _add_kitchen_core(problem, Agent, Item, Slot)

    chef_left = unified_planning.model.Object('chef_left', Agent)
    chef_right = unified_planning.model.Object('chef_right', Agent)
    item1 = unified_planning.model.Object('item1', Item)
    item2 = unified_planning.model.Object('item2', Item)
    clutter = unified_planning.model.Object('clutter', Item)
    left_a = unified_planning.model.Object('left_a', Slot)
    left_b = unified_planning.model.Object('left_b', Slot)
    sink = unified_planning.model.Object('sink', Slot)
    pas = unified_planning.model.Object('pass', Slot)
    stove = unified_planning.model.Object('stove', Slot)
    right_hold = unified_planning.model.Object('right_hold', Slot)
    serve1 = unified_planning.model.Object('serve1', Slot)
    serve2 = unified_planning.model.Object('serve2', Slot)
    problem.add_objects([
        chef_left, chef_right, item1, item2, clutter,
        left_a, left_b, sink, pas, stove, right_hold, serve1, serve2,
    ])

    _add_basic_reachability(problem, fl['reachable'], chef_left, chef_right,
                            left_slots=[left_a, left_b, sink], right_slots=[stove, right_hold, serve1, serve2], shared_slots=[pas])
    for a in [chef_left, chef_right]:
        problem.set_initial_value(fl['handempty'](a), True)

    problem.set_initial_value(fl['sink_slot'](sink), True)
    problem.set_initial_value(fl['stove_slot'](stove), True)
    problem.set_initial_value(fl['pass_slot'](pas), True)
    problem.set_initial_value(fl['serve_slot'](serve1), True)
    problem.set_initial_value(fl['serve_slot'](serve2), True)

    problem.set_initial_value(fl['at'](item1, left_a), True)
    problem.set_initial_value(fl['at'](item2, left_b), True)
    problem.set_initial_value(fl['at'](clutter, right_hold), True)

    for s in [sink, pas, stove, serve1, serve2]:
        problem.set_initial_value(fl['clear'](s), True)

    for i in [item1, item2]:
        problem.set_initial_value(fl['dirty'](i), True)
        problem.set_initial_value(fl['raw'](i), True)

    problem.add_goal(fl['cooked'](item1))
    problem.add_goal(fl['cooked'](item2))
    problem.add_goal(fl['at'](item1, serve1))
    problem.add_goal(fl['at'](item2, serve2))
    # clutter must remain on right_hold, so the planner cannot just use it as arbitrary free parking.
    problem.add_goal(fl['at'](clutter, right_hold))
    return problem


def make_kitchen_buffer_trap_medium() -> unified_planning.model.Problem:
    """Three items share single-capacity sink, stove, and pass slots."""
    Agent = UserType('Agent')
    Item = UserType('Item')
    Slot = UserType('Slot')

    problem = unified_planning.model.Problem('kitchen_buffer_trap_medium')
    fl = _add_kitchen_core(problem, Agent, Item, Slot)

    chef_left = unified_planning.model.Object('chef_left', Agent)
    chef_right = unified_planning.model.Object('chef_right', Agent)
    items = [unified_planning.model.Object(f'item{i}', Item) for i in [1, 2, 3]]
    left_slots = [unified_planning.model.Object(f'left_{name}', Slot) for name in ['a', 'b', 'c']]
    sink = unified_planning.model.Object('sink', Slot)
    pas = unified_planning.model.Object('pass', Slot)
    stove = unified_planning.model.Object('stove', Slot)
    buffer = unified_planning.model.Object('buffer', Slot)
    serves = [unified_planning.model.Object(f'serve{i}', Slot) for i in [1, 2, 3]]
    problem.add_objects([chef_left, chef_right, sink, pas, stove, buffer] + items + left_slots + serves)

    _add_basic_reachability(problem, fl['reachable'], chef_left, chef_right,
                            left_slots=left_slots + [sink], right_slots=[stove, buffer] + serves, shared_slots=[pas])
    for a in [chef_left, chef_right]:
        problem.set_initial_value(fl['handempty'](a), True)

    problem.set_initial_value(fl['sink_slot'](sink), True)
    problem.set_initial_value(fl['stove_slot'](stove), True)
    problem.set_initial_value(fl['pass_slot'](pas), True)
    for s in serves:
        problem.set_initial_value(fl['serve_slot'](s), True)

    for item, slot in zip(items, left_slots):
        problem.set_initial_value(fl['at'](item, slot), True)
        problem.set_initial_value(fl['dirty'](item), True)
        problem.set_initial_value(fl['raw'](item), True)

    for s in [sink, pas, stove, buffer] + serves:
        problem.set_initial_value(fl['clear'](s), True)

    for item, serve in zip(items, serves):
        problem.add_goal(fl['cooked'](item))
        problem.add_goal(fl['at'](item, serve))
    return problem
