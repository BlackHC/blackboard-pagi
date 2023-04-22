"""See `TestSet` for an example."""
from typing import MutableSet, Type

from blackboard_pagi.testing.collections import unordered_equal


class MutableSetTests:
    mutable_set: Type

    @classmethod
    def create_mutable_set(cls) -> MutableSet:
        return cls.mutable_set()

    @staticmethod
    def get_element(i):
        return i

    def test_add_coverage(self):
        instance = self.create_mutable_set()
        element1 = self.get_element(1)

        instance.add(element1)
        # And add a second time for good measure
        instance.add(element1)

    def test_discard_missing_element_passes(self):
        instance = self.create_mutable_set()
        element1 = self.get_element(1)

        instance.discard(element1)

    def test_discard_passes(self):
        instance = self.create_mutable_set()
        element1 = self.get_element(1)

        instance.add(element1)
        instance.discard(element1)

    def test_contains_len(self):
        instance = self.create_mutable_set()
        element1 = self.get_element(1)

        assert len(instance) == 0
        assert element1 not in instance
        instance.add(element1)
        assert element1 in instance
        assert len(instance) == 1

        element2 = self.get_element(2)
        assert element2 not in instance
        instance.add(element2)
        assert element2 in instance
        assert len(instance) == 2

        assert element1 in instance
        instance.discard(element1)
        assert element1 not in instance
        assert len(instance) == 1

        assert element2 in instance
        instance.discard(element2)
        assert element1 not in instance
        assert element2 not in instance
        assert len(instance) == 0

    def test_iter(self):
        instance = self.create_mutable_set()
        element1 = self.get_element(1)
        element2 = self.get_element(2)

        assert list(iter(instance)) == []
        instance.add(element1)
        assert unordered_equal(iter(instance), [element1])

        instance.add(element2)
        assert unordered_equal(iter(instance), [element1, element2])
