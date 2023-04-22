"""See `TestDict` for an example."""
from typing import MutableMapping, Type

import pytest

from blackboard_pagi.testing.collections import unordered_equal


class MutableMappingTests:
    mutable_mapping: Type

    @classmethod
    def create_mutable_mapping(cls) -> MutableMapping:
        return cls.mutable_mapping()

    @staticmethod
    def get_key(i):
        return i

    @staticmethod
    def get_value(i):
        return str(i)

    def get_key_value(self, i):
        return self.get_key(i), self.get_value(i)

    def test_get_missing_fails(self):
        instance = self.create_mutable_mapping()
        with pytest.raises(KeyError):
            # noinspection PyStatementEffect
            instance[self.get_key(1)]

    def test_del_missing_fails(self):
        instance = self.create_mutable_mapping()
        with pytest.raises(KeyError):
            del instance[self.get_key(1)]

    def test_integration(self):
        instance = self.create_mutable_mapping()

        assert len(instance) == 0
        assert list(iter(instance)) == []

        key, value = self.get_key_value(1)
        instance[key] = value

        assert instance[key] == value
        assert unordered_equal(list(iter(instance)), [key])
        assert len(instance) == 1

        key2, value2 = self.get_key_value(2)
        instance[key2] = value2

        assert instance[key2] == value2
        assert unordered_equal(list(iter(instance)), [key, key2])
        assert len(instance) == 2

        value3 = self.get_value(3)
        instance[key] = value3
        assert instance[key] == value3

        del instance[key]
        assert key not in instance
        assert unordered_equal(list(iter(instance)), [key2])
        assert len(instance) == 1
