"""Engine for POP2 in unified-planning
"""

import random
from collections import defaultdict
from typing import Callable, IO, Optional
from functools import reduce

import unified_planning as up
from unified_planning import engines
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.model import Parameter
from unified_planning.plans import ActionInstance

from ucpop.variable import Var
from ucpop.pop2 import POP2



class POP2EngineImpl(up.engines.Engine,
                     up.engines.mixins.OneshotPlannerMixin):

    def __init__(self, **options):
        # Read known user-options and store them for using in the `solve` method
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

    @property
    def name(self) -> str:
        return "POP2"

    @staticmethod
    def supported_kind():
        # For this demo we limit ourselves to numeric planning.
        # Other kinds of problems can be modeled in the UP library,
        # see unified_planning.model.problem_kind.
        supported_kind = up.model.ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_time("DISCRETE_TIME")
        supported_kind.set_typing('FLAT_TYPING')
        supported_kind.set_fluents_type('OBJECT_FLUENTS')
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITIES")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= POP2EngineImpl.supported_kind()

    def _action_adjacency_list_from_plan(self, plan):
        id_to_instance_map = {}
        graph = {}

        for step in plan.steps:
            if step.id in [0, -1]:
                continue

            def get_grounding(p: Parameter):
                v = Var(p.name, p.type, step.id)
                result = plan.bindings.get_grounding_or_variable(v)
                assert not isinstance(result, Var), f"{v} isn't grounded"
                return result

            params = tuple(map(get_grounding, step.action.parameters))
            action_instance = ActionInstance(step.action, params)
            id_to_instance_map[step.id] = action_instance
            graph[action_instance] = []

        # Build the directed graph
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
            callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
            timeout: Optional[float] = None,
            output_stream: Optional[IO[str]] = None
    ) -> 'up.engines.PlanGenerationResult':

        plan = POP2(problem).execute()

        if plan:
            status = PlanGenerationResultStatus.SOLVED_SATISFICING
            action_adjacency_list = self._action_adjacency_list_from_plan(plan)
            return up.engines.PlanGenerationResult(
                status, up.plans.PartialOrderPlan(action_adjacency_list),
                self.name, metrics={}
            )
        else:
            status = PlanGenerationResultStatus.UNSOLVABLE_PROVEN
            return up.engines.PlanGenerationResult(status, None, self.name)

    def destroy(self):
        pass


env = up.environment.get_environment()
env.factory.add_engine('pop2', __name__, 'POP2EngineImpl')
