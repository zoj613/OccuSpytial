from types import SimpleNamespace


class BaseStorage(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]


class State(BaseStorage):
    """Store parameter variables so they can be accessed as attributes."""

    def __iter__(self):
        yield from self.__dict__


class FixedState(BaseStorage):
    """Store parameter variables so they can be accessed as attributes.

    Values of variables assigned to an instance of this class cannot be changed
    Thus this class should be used for values that remain constant during
    sampling.
    """

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise KeyError('cannot change attributes already set')
        super().__setattr__(name, value)
