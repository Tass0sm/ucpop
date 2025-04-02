"""Implementation of ucpop based on
https://homes.cs.washington.edu/~weld/papers/pi.pdf

"""


from unified_planning.model import Problem


def create_null_plan():
    pass
    

def get_agenda(problem: Problem):
    pass


def ucpop(problem: Problem):
    """Execute the algorithm"""

    # create the null plan
    plan = create_null_plan()
    agenda = get_agenda(problem)

    step = 1
    while len(agenda) > 0:
        step += 1

        # step 2 in POP
        try:
            Q, A_need = select_goal(agenda)
        except IndexError:
            print('Probably Wrong')
            break

        # step 3 in POP
        act0 = possible_actions[0]
        # remove <G, act1> from Agenda
        self.agenda.remove((G, act1))

        # For actions with variable number of arguments, use least commitment principle
        # act0_temp, bindings = self.find_action_for_precondition(G)
        # act0 = self.generate_action_object(act0_temp, bindings)

        # Actions = Actions U {act0}
        self.actions.add(act0)

        # Constraints = add_const(start < act0, Constraints)
        self.constraints = self.add_const((self.start, act0), self.constraints)

        # for each CL E CausalLinks do
        #   Constraints = protect(CL, act0, Constraints)
        for causal_link in self.causal_links:
            self.constraints = self.protect(causal_link, act0, self.constraints)

        # Agenda = Agenda U {<P, act0>: P is a precondition of act0}
        for precondition in act0.precond:
            self.agenda.add((precondition, act0))

        # Constraints = add_const(act0 < act1, Constraints)
        self.constraints = self.add_const((act0, act1), self.constraints)

        # CausalLinks U {<act0, G, act1>}
        if (act0, G, act1) not in self.causal_links:
            self.causal_links.append((act0, G, act1))

        # for each A E Actions do
        #   Constraints = protect(<act0, G, act1>, A, Constraints)
        for action in self.actions:
            self.constraints = self.protect((act0, G, act1), action, self.constraints)

        if step > 200:
            print("Couldn't find a solution")
            return None, None

    if display:
        self.display_plan()
    else:
        return self.constraints, self.causal_links
