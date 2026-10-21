"""Engine for POP3 in unified-planning
"""

import unified_planning as up

from ucpop.pop3 import POP3
from ucpop.up.pop2_engine import POP2EngineImpl


class POP3EngineImpl(POP2EngineImpl):

    @property
    def name(self) -> str:
        return "POP3"

    @staticmethod
    def supported_kind():
        supported_kind = POP2EngineImpl.supported_kind()
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= POP3EngineImpl.supported_kind()

    def _planner(self, problem):
        return POP3(problem)


env = up.environment.get_environment()
env.factory.add_engine('pop3', __name__, 'POP3EngineImpl')
