"""Engine for POP2 in unified-planning
"""

import random
from collections import defaultdict
from typing import Callable, IO, Optional
from functools import reduce

import unified_planning as up
from unified_planning import engines
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.plans import ActionInstance

from ucpop.pop2 import POP2


class POP2EngineImpl(up.engines.Engine,
                     up.engines.mixins.OneshotPlannerMixin):

    def __init__(self, **options):
        # Read known user-options and store them for using in the `solve` method
        up.engines.Engine.__init__(self)
        up.engines.mixins.OneshotPlannerMixin.__init__(self)

        # self.max_tries = options.get('max_tries', None)
        # self.restart_probability = options.get('restart_probability', 0.00001)

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

        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= POP2EngineImpl.supported_kind()

    def _solve(
            self,
            problem: 'up.model.Problem',
            callback: Optional[Callable[['up.engines.PlanGenerationResult'], None]] = None,
            timeout: Optional[float] = None,
            output_stream: Optional[IO[str]] = None
    ) -> 'up.engines.PlanGenerationResult':

        breakpoint()

        plan = POP2(problem).execute()

        if False:
            status = PlanGenerationResultStatus.SOLVED_SATISFICING
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
