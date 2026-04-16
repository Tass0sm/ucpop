"""Engine for UCPOP in unified-planning
"""

import random
from typing import Callable, IO, Optional
from functools import reduce

import unified_planning as up
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import ActionInstance

from ucpop.ucpop import POP as UCPOP
from ucpop.constraints import make_constraint_action_instance



class UCPOPEngineImpl(up.engines.Engine,
                      up.engines.mixins.OneshotPlannerMixin):

    def __init__(self, **options):
        # Read known user-options and store them for using in the `solve` method
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        # self.max_tries = options.get('max_tries', None)
        # self.restart_probability = options.get('restart_probability', 0.00001)

    @property
    def name(self) -> str:
        return "UCPOP"

    @staticmethod
    def supported_kind():
        # For this demo we limit ourselves to numeric planning.
        # Other kinds of problems can be modeled in the UP library,
        # see unified_planning.model.problem_kind.
        supported_kind = up.model.ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_typing('HIERARCHICAL_TYPING')
        supported_kind.set_numbers('CONTINUOUS_NUMBERS')
        supported_kind.set_numbers('DISCRETE_NUMBERS')
        supported_kind.set_fluents_type('NUMERIC_FLUENTS')
        supported_kind.set_numbers('BOUNDED_TYPES')
        supported_kind.set_fluents_type('OBJECT_FLUENTS')
        supported_kind.set_conditions_kind('NEGATIVE_CONDITIONS')
        supported_kind.set_conditions_kind('DISJUNCTIVE_CONDITIONS')
        supported_kind.set_conditions_kind('EQUALITIES')
        supported_kind.set_conditions_kind('EXISTENTIAL_CONDITIONS')
        supported_kind.set_conditions_kind('UNIVERSAL_CONDITIONS')
        supported_kind.set_effects_kind('CONDITIONAL_EFFECTS')
        supported_kind.set_effects_kind('INCREASE_EFFECTS')
        supported_kind.set_effects_kind('DECREASE_EFFECTS')
        supported_kind.set_effects_kind('FLUENTS_IN_NUMERIC_ASSIGNMENTS')

        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= UCPOPEngineImpl.supported_kind()

    def _action_adjacency_list_from_plan(self, plan):
        id_to_instance_map = {}
        graph = {}

        for step in plan.steps:
            if step.id in [0, -1]:
                continue
            action_instance = make_constraint_action_instance(step.action)
            id_to_instance_map[step.id] = action_instance
            graph[action_instance] = []

        for u, vs in plan.adj_list.items():
            for v in vs:
                if u in [0, -1] or v in [0, -1] or u == v:
                    continue
                u_inst = id_to_instance_map[u]
                v_inst = id_to_instance_map[v]
                graph[u_inst].append(v_inst)

        return graph

    def _solve(
        self,
        problem: 'up.model.Problem',
        callback=None,
        timeout=None,
        output_stream=None
    ) -> 'up.engines.PlanGenerationResult':
        env = problem.environment

        # Ground first, same as POP
        with env.factory.Compiler(
            problem_kind=problem.kind,
            compilation_kind=up.engines.CompilationKind.GROUNDING
        ) as grounder:
            grounding_result = grounder.compile(
                problem,
                up.engines.CompilationKind.GROUNDING
            )

        grounded_problem = grounding_result.problem

        # Run the Python UCPOP planner directly
        result = UCPOP(grounded_problem).execute()

        # Depending on how execute() is patched, it may return:
        #   plan
        # or:
        #   plan, search_tree
        if isinstance(result, tuple):
            plan, _ = result
        else:
            plan = result

        if plan:
            status = PlanGenerationResultStatus.SOLVED_SATISFICING
            action_adjacency_list = self._action_adjacency_list_from_plan(plan)
            return up.engines.PlanGenerationResult(
                status,
                up.plans.PartialOrderPlan(action_adjacency_list),
                self.name,
                metrics={}
            )
        else:
            status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            return up.engines.PlanGenerationResult(status, None, self.name)

    def destroy(self):
        pass


env = up.environment.get_environment()
env.factory.add_engine('ucpop', __name__, 'UCPOPEngineImpl')
