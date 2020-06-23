from types import SimpleNamespace


class BaseStorage(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]


class State(BaseStorage):
    def __iter__(self):
        yield from self.__dict__


class FixedState(BaseStorage):
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise KeyError('cannot change attributes already set')
        super().__setattr__(name, value)
