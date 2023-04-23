from dataclasses import dataclass

import pydantic

from blackboard_pagi.utils.object_converter import ObjectConverter


def test_object_converter():
    # create an ObjectConverter
    converter = ObjectConverter()

    # add a converter for a dataclass
    @dataclass
    class Test:
        a: int
        b: str

    # Check that a vanilla dataclass is converted automatically
    assert converter(Test(1, '2')) == {'a': 1, 'b': '2'}

    def convert_test(converter: ObjectConverter, test: Test):
        return {'a': test.a + 1, 'b': test.b}

    converter.register_converter(convert_test, Test)

    # Check that the converter is used
    assert converter(Test(1, '2')) == {'a': 1 + 1, 'b': '2'}

    # add a converter for a Pydantic model
    @converter.add_converter()
    class TestModel(pydantic.BaseModel):
        a: int
        b: str

    # convert a Pydantic model
    assert converter(TestModel(a=1, b='2')) == {'a': 1, 'b': '2'}

    class TestModel2(pydantic.BaseModel):
        a: int
        b: str

    # does not convert a Pydantic model by default
    assert converter(TestModel2(a=1, b='2')) == repr(TestModel2(a=1, b='2'))

    # convert a dict
    assert converter({'a': 1, 'b': '2'}) == {'a': 1, 'b': '2'}

    # convert an object that is not a dict, dataclass, or Pydantic model
    assert converter(TestModel2) == repr(TestModel2)

    # convert a list
    assert converter([1, 2, 3]) == [1, 2, 3]

    # convert a set
    assert converter({1, 2, 3}) == {1, 2, 3}

    # convert a tuple
    assert converter((1, 2, 3)) == (1, 2, 3)

    # convert a string
    assert converter('test') == 'test'

    # convert an int
    assert converter(1) == 1

    # convert a nested object
    assert converter({'a': Test(1, '2')}) == {'a': {'a': 2, 'b': '2'}}
