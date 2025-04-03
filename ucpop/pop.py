"""Implementation of pop based on
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""

import random
from typing import List, Dict, Tuple
from unified_planning.model import FNode, Problem, Action, Effect

from ucpop.classes import PlanStep, Link, Plan
from ucpop.utils import initial_values_to_conjuncts


def create_null_plan_and_agenda(problem: Problem):
    start_step = PlanStep(id=0, effects=initial_values_to_conjuncts(problem.initial_values))
    end_step = PlanStep(id=-1, preconditions=frozenset(problem.goals))

    plan = Plan(steps=[start_step, end_step],
                ordering=[(0, -1)],
                links=[])

    agenda = set([(q, end_step) for q in end_step.preconditions])
    
    return plan, agenda


def select_goal(agenda: List[Tuple[FNode, PlanStep]]):
    return next(iter(agenda))


def select_actions(q: FNode, a_need: PlanStep, plan: Plan, actions: List[Action]):

    def could_work(action):
        return q in [e.fluent for e in action.effects if e.value.is_true()]
    
    def new_step_from_action(action):
        new_step = plan.new_step(action)
        new_constraints = set([(0, new_step.id), (new_step.id, a_need.id), (new_step.id, -1)])
        return new_step, new_constraints

    possible_new_steps = map(new_step_from_action, filter(could_work, actions))

    def step_could_work(step):
        # TODO: must check that it is possibly prior to 
        return q in [e for e in step.effects]

    possible_reusable_steps = [(s, [(s.id, a_need.id)]) for s in filter(step_could_work, plan.steps)]
    possible_steps = list(possible_new_steps) + possible_reusable_steps

    if possible_steps:
        step, new_constraints = random.choice(possible_steps)
        return step, new_constraints
    else:
        return None, []


def threatens(step, link):
    return ~link.condition in step.effects


def can_demote(a_t, link, plan):
    return plan.can_constrain(a_t.id, link.step_p.id)


def can_promote(a_t, link, plan):
    return plan.can_constrain(link.step_c.id, a_t.id)


def protect(link, a_t, plan):
    demotion_c = can_demote(a_t, link, plan)
    promotion_c = can_promote(a_t, link, plan)
    if demotion_c and promotion_c:
        return random.choice([demotion_c, promotion_c])
    elif demotion_c:
        return demotion_c
    elif promotion_c:
        return promotion_c
    else:
        return None


def protect_causal_links(plan):
    for link in plan.links:
        for a_t in [step for step in plan.steps if threatens(step, link)]:
            protecting_edge = protect(link, a_t, plan)
            if not protecting_edge:
                return False
            plan.add_edge(*protecting_edge)

    return True


def pop(problem: Problem):
    # create the null plan
    plan, agenda = create_null_plan_and_agenda(problem)

    step = 1
    while len(agenda) > 0:
        step += 1

        # step 2
        q, a_need = select_goal(agenda)

        # step 3
        a_add, new_constraints = select_actions(q, a_need, plan, problem.actions)
        if not a_add:
            return None, {}

        plan.add_step(a_add)
        for u, v in new_constraints:
            plan.add_edge(u, v)
        plan.add_link(a_add, q, a_need)

        # step 4
        agenda.remove((q, a_need))
        for pc in a_add.preconditions:
            agenda |= set([(pc, a_add)])

        # step 5
        successfully_protected = protect_causal_links(plan)
        if not successfully_protected:
            return None, {}

        if step > 200:
            print("Couldn't find a solution")
            return None, {}

    return plan, {}
