"""Exercise POP3, UCPOP and PCOP on a few small problems and check each returned
plan is valid under every grounding allowed by its binding constraints and
every linearization of its partial order in that grounding."""

import itertools
import logging

import networkx as nx
import unified_planning
from unified_planning.shortcuts import *
from unified_planning.plans import SequentialPlan, ActionInstance

import ucpop
from ucpop.pop3 import POP3
from ucpop.ucpop import UCPOP
from ucpop.pcop import PCOP
from ucpop.variable import Var

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.ERROR)


def sticky_robots():
    Robot = UserType('Robot')
    done_a = Fluent('done_a')
    done_b = Fluent('done_b')
    sticky = Fluent('sticky', BoolType(), robot=Robot)

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

    problem = Problem('StickyRobots')
    for f in [done_a, done_b, sticky]:
        problem.add_fluent(f, default_initial_value=False)
    problem.add_actions([do_a, do_b])
    problem.add_objects([Object('robot1', Robot), Object('robot2', Robot)])
    problem.add_goal(done_a)
    problem.add_goal(done_b)
    return problem


def robot_moves(name='robot', goal=None):
    Location = UserType('Location')
    robot_at = Fluent('robot_at', BoolType(), l=Location)
    connected = Fluent('connected', BoolType(), l_from=Location, l_to=Location)

    move = InstantaneousAction('move', l_from=Location, l_to=Location)
    l_from, l_to = move.parameter('l_from'), move.parameter('l_to')
    move.add_precondition(connected(l_from, l_to))
    move.add_precondition(robot_at(l_from))
    move.add_effect(robot_at(l_from), False)
    move.add_effect(robot_at(l_to), True)

    problem = Problem(name)
    problem.add_fluent(robot_at, default_initial_value=False)
    problem.add_fluent(connected, default_initial_value=False)
    ls = [Object(f'l{i}', Location) for i in range(4)]
    problem.add_objects(ls)
    problem.add_action(move)
    problem.set_initial_value(robot_at(ls[0]), True)
    for a, b in [(0, 1), (1, 2), (1, 3), (2, 3)]:
        problem.set_initial_value(connected(ls[a], ls[b]), True)
    problem.add_goal(goal(robot_at, ls) if goal else robot_at(ls[3]))
    return problem


def robot_either():
    """A disjunctive goal: either location will do."""
    return robot_moves('robot_either', lambda robot_at, ls: Or(robot_at(ls[3]), robot_at(ls[2])))


def lamp():
    """One flip lights the lamp for both reading and the goal, so its
    conditional effect supports two links but its antecedent is posted once
    (step 4 of Figure 18). The disjunct (lit) of the last goal already has a
    link when it is chosen, so it is dropped (step 2(d)); failing there as the
    paper says would lose the only plan, since (broken) is unachievable."""
    powered, lit, done, broken = (Fluent(n) for n in ['powered', 'lit', 'done', 'broken'])

    turn_on = InstantaneousAction('turn_on')
    turn_on.add_effect(powered, True)

    flip = InstantaneousAction('flip')
    flip.add_effect(lit, True, condition=powered)

    read = InstantaneousAction('read')
    read.add_precondition(lit)
    read.add_effect(done, True)

    problem = Problem('lamp')
    for f in [powered, lit, done, broken]:
        problem.add_fluent(f, default_initial_value=False)
    problem.add_actions([turn_on, flip, read])
    problem.add_goal(done)
    problem.add_goal(lit)
    problem.add_goal(Or(lit, broken))
    return problem


def briefcase(name='briefcase', goals=None, quantified=False):
    """Pednault's briefcase domain, the classic UCPOP example: moving the
    briefcase moves whatever is in it (a conditional effect). Getting the
    dictionary to the office while leaving the paycheck at home requires
    confronting the threat by taking the paycheck out first. If QUANTIFIED,
    moving uses a universally quantified effect as in the paper."""
    Place = UserType('Place')
    Thing = UserType('Thing')
    b_at = Fluent('b_at', BoolType(), p=Place)
    t_at = Fluent('t_at', BoolType(), t=Thing, p=Place)
    t_in = Fluent('t_in', BoolType(), t=Thing)

    mov_b = InstantaneousAction('mov_b', m=Place, l=Place)
    m, l = mov_b.parameter('m'), mov_b.parameter('l')
    mov_b.add_precondition(b_at(m))
    mov_b.add_precondition(Not(Equals(m, l)))
    mov_b.add_effect(b_at(l), True)
    mov_b.add_effect(b_at(m), False)

    take_out = InstantaneousAction('take_out', x=Thing)
    take_out.add_precondition(t_in(take_out.parameter('x')))
    take_out.add_effect(t_in(take_out.parameter('x')), False)

    put_in = InstantaneousAction('put_in', x=Thing, l=Place)
    x, pl = put_in.parameter('x'), put_in.parameter('l')
    put_in.add_precondition(t_at(x, pl))
    put_in.add_precondition(b_at(pl))
    put_in.add_effect(t_in(x), True)

    problem = Problem(name)
    for f in [b_at, t_at, t_in]:
        problem.add_fluent(f, default_initial_value=False)
    home, office = Object('home', Place), Object('office', Place)
    dictionary, paycheck = Object('dictionary', Thing), Object('paycheck', Thing)
    problem.add_objects([home, office, dictionary, paycheck])

    if quantified:
        z = Variable('z', Thing)
        mov_b.add_effect(t_at(z, l), True, condition=t_in(z), forall=[z])
        mov_b.add_effect(t_at(z, m), False, condition=t_in(z), forall=[z])
    else:
        # conditional effects of moving the briefcase, one per thing, as an
        # expansion of the lisp domain's (forall (?z) (when ...))
        for thing in [dictionary, paycheck]:
            mov_b.add_effect(t_at(thing, l), True, condition=t_in(thing))
            mov_b.add_effect(t_at(thing, m), False, condition=t_in(thing))
    problem.add_actions([mov_b, take_out, put_in])

    problem.set_initial_value(b_at(home), True)
    problem.set_initial_value(t_at(dictionary, home), True)
    problem.set_initial_value(t_at(paycheck, home), True)
    problem.set_initial_value(t_in(paycheck), True)
    if goals:
        for g in goals(t_at, Thing, home, office):
            problem.add_goal(g)
    else:
        problem.add_goal(t_at(dictionary, office))
        problem.add_goal(t_at(paycheck, home))
    return problem


def briefcase_all():
    """A universally quantified goal: everything at the office."""
    t = Variable('t', UserType('Thing'))
    return briefcase('briefcase_all', lambda t_at, Thing, home, office: [Forall(t_at(t, office), t)])


def briefcase_some():
    """An existentially quantified goal: something at the office."""
    t = Variable('t', UserType('Thing'))
    return briefcase('briefcase_some', lambda t_at, Thing, home, office: [Exists(t_at(t, office), t)])


def briefcase_together():
    """Nested quantifiers as in section 7.2: everything together somewhere
    other than home, i.e. a Skolem constant constrained by an inequality
    and by a universal goal."""
    Place, Thing = UserType('Place'), UserType('Thing')
    p, t = Variable('p', Place), Variable('t', Thing)
    return briefcase('briefcase_together',
                     lambda t_at, Thing, home, office: [Exists(And(Not(Equals(p, home)), Forall(t_at(t, p), t)), p)])


def briefcase_forall():
    """The briefcase with a universally quantified effect."""
    return briefcase('briefcase_forall', quantified=True)


def briefcase_forall_all():
    """A universally quantified goal achieved by a universally quantified effect."""
    t = Variable('t', UserType('Thing'))
    return briefcase('briefcase_forall_all', lambda t_at, Thing, home, office: [Forall(t_at(t, office), t)], quantified=True)


def chmod():
    """The paper's chmod * example: protecting every file necessarily
    clobbers (writable a), even though the unifier binds the effect's
    universal variable, so writing a must come first."""
    File = UserType('File')
    writable = Fluent('writable', BoolType(), f=File)
    written = Fluent('written', BoolType(), f=File)
    protected = Fluent('protected')

    write = InstantaneousAction('write', f=File)
    write.add_precondition(writable(write.parameter('f')))
    write.add_effect(written(write.parameter('f')), True)

    chmod_all = InstantaneousAction('chmod_all')
    g = Variable('g', File)
    chmod_all.add_effect(writable(g), False, forall=[g])
    chmod_all.add_effect(protected, True)

    problem = Problem('chmod')
    for fl in [writable, written, protected]:
        problem.add_fluent(fl, default_initial_value=False)
    a, b = Object('a', File), Object('b', File)
    problem.add_objects([a, b])
    problem.add_actions([write, chmod_all])
    problem.set_initial_value(writable(a), True)
    problem.set_initial_value(writable(b), True)
    problem.add_goal(written(a))
    problem.add_goal(protected)
    return problem


def blocks(name='blocks', clear_free=False):
    """Unstack A from B from C onto the Table. With the move of Figure 15,
    moving onto the Table only deletes (clear ?y) when (/= ?y Table), so the
    first move's threat to (clear Table) is confronted with an equality goal.
    If CLEAR_FREE, there is no clear predicate: as suggested in section 7,
    move requires that nothing be on ?b, nor on ?y unless it is the Table."""
    Thing = UserType('Thing')
    on = Fluent('on', BoolType(), b=Thing, x=Thing)
    clear = Fluent('clear', BoolType(), x=Thing)
    a, b, c, table = (Object(n, Thing) for n in ['A', 'B', 'C', 'Table'])

    move = InstantaneousAction('move', b=Thing, x=Thing, y=Thing)
    mb, mx, my = move.parameters
    move.add_precondition(on(mb, mx))
    move.add_precondition(Not(Equals(mb, mx)))
    move.add_precondition(Not(Equals(mb, my)))
    move.add_precondition(Not(Equals(mx, my)))
    move.add_effect(on(mb, my), True)
    move.add_effect(on(mb, mx), False)
    if clear_free:
        z = Variable('z', Thing)
        move.add_precondition(Forall(Not(on(z, mb)), z))
        move.add_precondition(Forall(Or(Equals(my, table), Not(on(z, my))), z))
    else:
        move.add_precondition(clear(mb))
        move.add_precondition(clear(my))
        move.add_effect(clear(mx), True)
        move.add_effect(clear(my), False, condition=Not(Equals(my, table)))

    problem = Problem(name)
    problem.add_fluent(on, default_initial_value=False)
    if not clear_free:
        problem.add_fluent(clear, default_initial_value=False)
    problem.add_objects([a, b, c, table])
    problem.add_action(move)
    for x, y in [(a, b), (b, c), (c, table)]:
        problem.set_initial_value(on(x, y), True)
    if not clear_free:
        problem.set_initial_value(clear(a), True)
        problem.set_initial_value(clear(table), True)
    problem.add_goal(on(a, table))
    problem.add_goal(on(b, table))
    return problem


def blocks_clear_free():
    """Blocks world with universally quantified (in)equality preconditions."""
    return blocks('blocks_clear_free', clear_free=True)


def battery(name='battery', n_robots=2, charged_initially=True):
    """Using a robot drains its battery, and robot1 must end up charged.
    Leaving the robot used unbound, the threat to (charged robot1) holds only
    if it is robot1: PCOP can separate it from robot1, or order its use before
    the charging under that codesignation. With one robot, separation leaves
    no grounding. If not CHARGED_INITIALLY, robot1 must be charged by a step."""
    Robot = UserType('Robot')
    used = Fluent('used')
    charged = Fluent('charged', BoolType(), r=Robot)

    use = InstantaneousAction('use', r=Robot)
    use.add_effect(used, True)
    use.add_effect(charged(use.parameter('r')), False)

    charge = InstantaneousAction('charge', r=Robot)
    charge.add_effect(charged(charge.parameter('r')), True)

    problem = Problem(name)
    problem.add_fluent(used, default_initial_value=False)
    problem.add_fluent(charged, default_initial_value=False)
    problem.add_actions([use, charge])
    robots = [Object(f'robot{i + 1}', Robot) for i in range(n_robots)]
    problem.add_objects(robots)
    if charged_initially:
        problem.set_initial_value(charged(robots[0]), True)
    problem.add_goal(used)
    problem.add_goal(charged(robots[0]))
    return problem


def battery_one():
    return battery('battery_one', n_robots=1)


def battery_uncharged():
    return battery('battery_uncharged', charged_initially=False)


def groundings(plan, problem):
    """Every assignment of objects to the plan's unbound variables that
    respects its noncodesignation constraints."""
    bindings = plan.bindings
    variables = sorted(bindings.unbound, key=lambda v: (v.num, v.name))

    for values in itertools.product(*[problem.objects(v.type) for v in variables]):
        assignment = dict(zip(variables, values))

        def value(x):
            rx = bindings.find(x)
            return assignment.get(rx, bindings.get_grounding(rx))

        if all(value(v) != value(other)
               for v in variables
               for other in bindings.nodes[v].noncodesignation):
            yield value


def check(problem, plan):
    """Validate every grounding x linearization; returns (#checked, failures)."""
    steps = {s.id: s for s in plan.steps if s.id not in (0, -1)}
    graph = nx.DiGraph()
    graph.add_nodes_from(steps)
    for u, vs in plan.adj_list.items():
        for v in vs:
            if u in steps and v in steps:
                graph.add_edge(u, v)

    checked, failures = 0, []
    with PlanValidator(problem_kind=problem.kind) as validator:
        for value in groundings(plan, problem):
            # PCOP's orderings that hold in this grounding
            grounded = graph.copy()
            for u, v, condition in getattr(plan, 'conditional_edges', ()):
                if u in steps and v in steps and all((value(x) == value(y)) == eq for x, y, eq in condition):
                    grounded.add_edge(u, v)

            for order in nx.all_topological_sorts(grounded):
                actions = [ActionInstance(steps[i].action,
                                          tuple(value(Var(p.name, p.type, i)) for p in steps[i].action.parameters))
                           for i in order]
                checked += 1
                if not validator.validate(problem, SequentialPlan(actions)):
                    failures.append(actions)
    return checked, failures


if __name__ == '__main__':
    pop3_problems = [sticky_robots, robot_moves, robot_either, lamp, briefcase, blocks]
    ucpop_problems = pop3_problems + [briefcase_all, briefcase_some, briefcase_together, briefcase_forall, briefcase_forall_all, chmod, blocks_clear_free, battery, battery_one, battery_uncharged]

    for planner_name, planner_class, problems in [('pop3', POP3, pop3_problems), ('ucpop', UCPOP, ucpop_problems), ('pcop', PCOP, ucpop_problems)]:
        for make_problem in problems:
            problem = make_problem()
            plan, path = planner_class(problem).execute()
            label = f"{planner_name} {problem.name}"
            if plan is None:
                print(f"{label}: no plan found")
                continue

            with OneshotPlanner(name=planner_name) as planner:
                print(f"{label}: {planner.solve(problem).plan}")

            checked, failures = check(problem, plan)
            print(f"  search path length {len(path)}; {checked} grounded linearizations checked, {len(failures)} invalid")
            if checked == 0:
                print("  INVALID: no grounded linearization")
            for f in failures[:3]:
                print("  INVALID:", f)
