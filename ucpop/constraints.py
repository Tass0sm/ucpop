class ConstraintGenerator:
    """ Base class for symbolic-to-GoC constraint gen

    Subclass this for each action type that should add constraints to a GoC.
    `add(goc)` should mutate the provided GoC in place.
    """ 

    def add(self, goc):
        raise NotImplementedError


class DummyConstraintGenerator(ConstraintGenerator):
    """Minimal example generator used for testing the action/constraint pipeline.

    Attach to a constraint-enabled action to verify that
    returned action instances can add a constraint to a GoC.
    """

    def __init__(self, action_instance):
        self._action_instance = action_instance

    def add(self, goc):
        goc.add_point_to_point(0, 0, [0, 0, 0])


class ConstraintEnabledInstantaneousAction(InstantaneousAction):
    """InstantaneousAction with an attached constraint-generator type.

    Use instead of `InstantaneousAction` to produce GoC constraints.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._constraint_generator_type = None
 
    def add_constraint_generator(self, generator_type):
        """ Call this in problem construction so engine wrappers know how to wrap
        action instances returned in plans.
        """
        self._constraint_generator_type = generator_type

    @property
    def constraint_generator_type(self):
        return self._constraint_generator_type

    def has_constraint_generator(self):
        return self._constraint_generator_type is not None


class ConstraintActionInstance(ActionInstance):
    """ActionInstance carrying a concrete ConstraintGenerator.

    Engine wrappers should return this instead of a plain ActionInstance when
    the action is constraint-enabled.
    """

    def __init__(self, action, params=tuple(), *, constraint_generator=None, agent=None):
         super().__init__(action, params, agent=agent)
         self._constraint_generator = constraint_generator

    def add_constraint(self, goc):
        """Add this action instance's constraint(s) to the given GoC."""
        if self._constraint_generator is None:
            raise ValueError("No constraint generator attached to this action instance.")
        self._constraint_generator.add(goc)

    @property
    def constraint_generator(self):
        return self._constraint_generator

class ConstraintPartialActionInstance(PartialActionInstance):
    """PartialActionInstance carrying a concrete ConstraintGenerator.

    Used by PCOP when returned actions may still be partially bound but should
    still expose the same constraint-attachment interface.
    """
    def __init__(self, action, params=tuple(), *, constraint_generator=None, agent=None, motion_paths=None):
        super().__init__(action, params, agent=agent, motion_paths=motion_paths)
        self._constraint_generator = constraint_generator

    def add_constraint(self, goc):
        if self._constraint_generator is None:
            raise ValueError("No constraint generator attached to this action instance.")
        self._constraint_generator.add(goc)

    @property
    def constraint_generator(self):
        return self._constraint_generator


# MARK: Helpers ------------------

def make_constraint_action_instance(action, params=tuple(), agent=None):
    """Create the right action-instance wrapper for a planned action.

    Returns a ConstraintActionInstance when `action` is constraint-enabled.
    Otherwise returns a normal ActionInstance. Engine wrappers should use this
    instead of constructing ActionInstance directly.
    """
    if isinstance(action, ConstraintEnabledInstantaneousAction) and action.has_constraint_generator():
        ai = ConstraintActionInstance(action, params, agent=agent)
        generator = action.constraint_generator_type(ai)
        ai._constraint_generator = generator
        return ai
    return ActionInstance(action, params, agent=agent)

def make_constraint_partial_action_instance(action, params=tuple(), agent=None):
    """Create the right partial action-instance wrapper for PCOP.

    Returns a ConstraintPartialActionInstance when `action` is constraint-enabled;
    otherwise returns a normal PartialActionInstance.
    """
    if isinstance(action, ConstraintEnabledInstantaneousAction) and action.has_constraint_generator():
        ai = ConstraintPartialActionInstance(action, params, agent=agent)
        generator = action.constraint_generator_type(ai)
        ai._constraint_generator = generator
        return ai
    return PartialActionInstance(action, params, agent=agent)
