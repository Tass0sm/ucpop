"""Implementation of pop based on
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""


from unified_planning.model import Problem


def create_null_plan():
    return None


def get_agenda(problem: Problem):
    return []




def protect_causal_links(plan):
    for causal_link in plan.links:
        protecting_edge = protect(plan, causal_link)
        # if not protecting_edge:
        #     return None, {}
        plan.add_edge(*protecting_edge)


def pop(problem: Problem):
    # create the null plan
    plan = create_null_plan(problem)
    agenda = get_agenda(problem)

    step = 1
    while len(agenda) > 0:
        step += 1

        # step 2
        q, a_need = select_goal(agenda)

        # step 3
        a_add = select_actions(plan, q)
        # if not a_add:
        #     return None, {}
        plan.add_link(a_add, q, a_need)
        plan.add_edge(a_add, a_need)

        # step 4
        agenda.remove((q, a_need))

        # step 5
        protect_causal_links(plan)

        if step > 200:
            print("Couldn't find a solution")
            return None, None

    return plan, {}
