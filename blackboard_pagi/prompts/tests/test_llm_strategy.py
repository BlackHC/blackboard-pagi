from dataclasses import dataclass

from langchain import OpenAI

from blackboard_pagi.prompts.llm_strategy import can_wrap_member_in_llm, check_not_implemented, llm_strategy
from blackboard_pagi.testing.fake_llm import FakeLLM


@dataclass
class NotImplementedDataclass:
    @staticmethod
    def static_method(a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    @classmethod
    def class_method(cls, a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    @property
    def property_getter(self) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    def bound_method(self, a: int, b: int) -> int:
        """Add two numbers."""
        raise NotImplementedError()

    def bound_method_raises_class(self, a: int, b: int = 1) -> int:
        """Add two numbers."""
        raise NotImplementedError

    def bound_method_implemented(self, a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @staticmethod
    def static_method_implemented(a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @classmethod
    def class_method_implemented(cls, a: int, b: int = 1) -> int:
        """Add two numbers."""
        return a + b

    @property
    def property_getter_implemented(self) -> int:
        """Return 1."""
        return 1


def test_can_wrap_member_in_llm():
    """Test that can_wrap_member_in_llm works as expected."""
    # This automatically also checks check_not_implemented.
    assert can_wrap_member_in_llm(NotImplementedDataclass.static_method)
    assert can_wrap_member_in_llm(NotImplementedDataclass.bound_method)
    assert can_wrap_member_in_llm(NotImplementedDataclass.class_method)
    assert can_wrap_member_in_llm(NotImplementedDataclass.property_getter)
    assert can_wrap_member_in_llm(NotImplementedDataclass.bound_method_raises_class)

    assert not can_wrap_member_in_llm(NotImplementedDataclass.bound_method_implemented)
    assert not can_wrap_member_in_llm(NotImplementedDataclass.static_method_implemented)
    assert not can_wrap_member_in_llm(NotImplementedDataclass.class_method_implemented)
    assert not can_wrap_member_in_llm(NotImplementedDataclass.property_getter_implemented)


def not_implemented_function():
    raise NotImplementedError


def test_check_not_implemented_functon():
    assert check_not_implemented(not_implemented_function)
    assert not check_not_implemented(lambda: 1)


def test_llm_strategy_on_functions():
    def add_two_ints(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def add_two_ints_with_default(a: int, b: int = 1) -> int:
        """Add two integers with a default value."""
        return a + b

    def add_two_ints_with_default_and_kwarg(*, a: int, c: int = 2) -> int:
        """Add two integers with a default value."""
        return a + c

    llm = FakeLLM(
        texts={
            'Execute the following function that is described via a doc string:\n\nAdd two integers.\n\n# '
            'Task\n\nExecute the function with the inputs that follow in the next section and finally return the '
            'output using the output type\nas YAML document in an # Output section. (If the value is a literal, '
            'then just write the value. We parse the text in the\n# Output section using `yaml.safe_load` in '
            'Python.)\n\n# Input Types\n\na: int\nb: int\n\n\n# Inputs\n\na: 1\nb: 2\n\n\n# Output Type\n\nint\n\n# '
            'Execution Scratch-Pad (Think Step by Step)\n\n\n# Output\n\n---\nresult: 3',
            'Execute the following function that is described via a doc string:\n\nAdd two integers with a default '
            'value.\n\n# Task\n\nExecute the function with the inputs that follow in the next section and finally '
            'return the output using the output type\nas YAML document in an # Output section. (If the value is a '
            'literal, then just write the value. We parse the text in the\n# Output section using `yaml.safe_load` in '
            'Python.)\n\n# Input Types\n\na: int\nc: int\n\n\n# Inputs\n\na: 1\nc: 2\n\n\n# Output Type\n\nint\n\n# '
            'Execution Scratch-Pad (Think Step by Step)\n\n\ndef add_two_integers(a, c=0):\n    """Add two integers '
            'with a default value."""\n    return a + c \n\n\n# Output\n\n---\nresult: 3',
            'Execute the following function that is described via a doc string:\n\nAdd two integers with a default '
            'value.\n\n# Task\n\nExecute the function with the inputs that follow in the next section and finally '
            'return the output using the output type\nas YAML document in an # Output section. (If the value is a '
            'literal, then just write the value. We parse the text in the\n# Output section using `yaml.safe_load` in '
            'Python.)\n\n# Input Types\n\na: int\nb: int\n\n\n# Inputs\n\na: 1\nb: 1\n\n\n# Output Type\n\nint\n\n# '
            'Execution Scratch-Pad (Think Step by Step)\n\n\n# Output\n\n---\nresult: 2',
        },
        # external_llm=OpenAI(),
    )

    assert llm_strategy(llm)(add_two_ints)(1, 2) == 3
    assert llm_strategy(llm)(add_two_ints_with_default)(1) == 2
    assert llm_strategy(llm)(add_two_ints_with_default_and_kwarg)(a=1) == 3


@dataclass
class Customer:
    first_name: str
    last_name: str
    birthday: str
    city: str


@dataclass
class CustomerDatabase:
    customers: list[Customer]

    def find_customer_index(self, query: str) -> int:
        """Find the index of the customer that matches natural language query best.

        We support semantic queries instead of SQL, so we can search for things like
        "the customer that was born in 1990".

        Args:
            query: Natural language query

        Returns:
            The index of the best matching customer in the database.
        """
        raise NotImplementedError()

    @staticmethod
    def create_mock_customers(num_customers: int = 1) -> list[Customer]:
        """
        Create mock customers with believable data (our customers are world citizens).
        """
        raise NotImplementedError()

    @staticmethod
    def create_mock_queries(customer_database: "CustomerDatabase", num_queries: int = 1) -> list[str]:
        """
        Create mock queries that are match one of the mock customer better than the others.

        We support semantic queries instead of SQL, so we can search for things like
        "the customer that was born in 1990".
        """
        raise NotImplementedError()


def test_llm_strategy():
    llm = FakeLLM(
        texts={
            'Execute the following function that is described via a doc string:\n\nFind the index of the customer '
            'that matches natural language query best.\n\n        We support semantic queries instead of SQL, '
            'so we can search for things like\n        "the customer that was born in 1990".\n\n        Args:\n       '
            '     query: Natural language query\n\n        Returns:\n            The index of the best matching '
            'customer in the database.\n        \n\n# Task\n\nExecute the function with the inputs that follow in the '
            'next section and finally return the output using the output type\nas YAML document in an # Output '
            'section. (If the value is a literal, then just write the value. We parse the text in the\n# Output '
            'section using `yaml.safe_load` in Python.)\n\n# Dataclasses Schema\n\ntypes:\n  Customer:\n    '
            'birthday:\n      type: str\n    city:\n      type: str\n    first_name:\n      type: str\n    '
            'last_name:\n      type: str\n  CustomerDatabase:\n    customers:\n      type: \'[Customer]\'\n  '
            'CustomerDatabase[FakeLLM]:\n    bases:\n    - CustomerDatabase\n    customers:\n      type: \'['
            'Customer]\'\n\n\n# Input Types\n\nquery: str\nself: CustomerDatabase[FakeLLM]\n\n\n# Inputs\n\nquery: '
            'The customer with the last name of Miller\nself:\n  customers:\n  - birthday: 05/20/1993\n    city: '
            'Tokyo\n    first_name: John\n    last_name: Doe\n  - birthday: 04/15/1989\n    city: London\n    '
            'first_name: Jane\n    last_name: Smith\n  - birthday: 03/10/1996\n    city: New York\n    first_name: '
            'Mark\n    last_name: Miller\n\n\n# Output Type\n\nint\n\n# Execution Scratch-Pad (Think Step by '
            'Step)\n\n\n# Output\n\n---\nresult: 2',
            "Execute the following function that is described via a doc string:\n\n\n        Create mock customers "
            "with believable data (our customers are world citizens).\n        \n\n# Task\n\nExecute the function "
            "with the inputs that follow in the next section and finally return the output using the output type\nas "
            "YAML document in an # Output section. (If the value is a literal, then just write the value. We parse "
            "the text in the\n# Output section using `yaml.safe_load` in Python.)\n\n# Dataclasses Schema\n\ntypes:\n "
            " Customer:\n    birthday:\n      type: str\n    city:\n      type: str\n    first_name:\n      type: "
            "str\n    last_name:\n      type: str\n\n\n# Input Types\n\nnum_customers: int\n\n\n# "
            "Inputs\n\nnum_customers: 3\n\n\n# Output Type\n\n[Customer]\n\n# Execution Scratch-Pad (Think Step by "
            "Step)\n\n\n# Output\n\n---\nresult:\n- birthday: '05/20/1993'\n  city: Tokyo\n  first_name: John\n  "
            "last_name: Doe\n- birthday: '04/15/1989'\n  city: London\n  first_name: Jane\n  last_name: Smith\n- "
            "birthday: '03/10/1996'\n  city: New York\n  first_name: Mark\n  last_name: Miller",
            'Execute the following function that is described via a doc string:\n\nFind the index of the customer '
            'that matches natural language query best.\n\n        We support semantic queries instead of SQL, '
            'so we can search for things like\n        "the customer that was born in 1990".\n\n        Args:\n       '
            '     query: Natural language query\n\n        Returns:\n            The index of the best matching '
            'customer in the database.\n        \n\n# Task\n\nExecute the function with the inputs that follow in the '
            'next section and finally return the output using the output type\nas YAML document in an # Output '
            'section. (If the value is a literal, then just write the value. We parse the text in the\n# Output '
            'section using `yaml.safe_load` in Python.)\n\n# Dataclasses Schema\n\ntypes:\n  Customer:\n    '
            'birthday:\n      type: str\n    city:\n      type: str\n    first_name:\n      type: str\n    '
            'last_name:\n      type: str\n  CustomerDatabase:\n    customers:\n      type: \'[Customer]\'\n  '
            'CustomerDatabase[FakeLLM]:\n    bases:\n    - CustomerDatabase\n    customers:\n      type: \'['
            'Customer]\'\n\n\n# Input Types\n\nquery: str\nself: CustomerDatabase[FakeLLM]\n\n\n# Inputs\n\nquery: '
            'The customer that lives in Tokyo\nself:\n  customers:\n  - birthday: 05/20/1993\n    city: Tokyo\n    '
            'first_name: John\n    last_name: Doe\n  - birthday: 04/15/1989\n    city: London\n    first_name: Jane\n '
            '   last_name: Smith\n  - birthday: 03/10/1996\n    city: New York\n    first_name: Mark\n    last_name: '
            'Miller\n\n\n# Output Type\n\nint\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\n# '
            'Output\n\n---\nresult: 0',
            'Execute the following function that is described via a doc string:\n\nFind the index of the customer '
            'that matches natural language query best.\n\n        We support semantic queries instead of SQL, '
            'so we can search for things like\n        "the customer that was born in 1990".\n\n        Args:\n       '
            '     query: Natural language query\n\n        Returns:\n            The index of the best matching '
            'customer in the database.\n        \n\n# Task\n\nExecute the function with the inputs that follow in the '
            'next section and finally return the output using the output type\nas YAML document in an # Output '
            'section. (If the value is a literal, then just write the value. We parse the text in the\n# Output '
            'section using `yaml.safe_load` in Python.)\n\n# Dataclasses Schema\n\ntypes:\n  Customer:\n    '
            'birthday:\n      type: str\n    city:\n      type: str\n    first_name:\n      type: str\n    '
            'last_name:\n      type: str\n  CustomerDatabase:\n    customers:\n      type: \'[Customer]\'\n  '
            'CustomerDatabase[FakeLLM]:\n    bases:\n    - CustomerDatabase\n    customers:\n      type: \'['
            'Customer]\'\n\n\n# Input Types\n\nquery: str\nself: CustomerDatabase[FakeLLM]\n\n\n# Inputs\n\nquery: '
            'The customer that was born in 1989\nself:\n  customers:\n  - birthday: 05/20/1993\n    city: Tokyo\n    '
            'first_name: John\n    last_name: Doe\n  - birthday: 04/15/1989\n    city: London\n    first_name: Jane\n '
            '   last_name: Smith\n  - birthday: 03/10/1996\n    city: New York\n    first_name: Mark\n    last_name: '
            'Miller\n\n\n# Output Type\n\nint\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\n# '
            'Output\n\n---\nresult: 1',
            'Execute the following function that is described via a doc string:\n\n\n        Create mock queries that '
            'are match one of the mock customer better than the others.\n\n        We support semantic queries '
            'instead of SQL, so we can search for things like\n        "the customer that was born in 1990".\n        '
            '\n\n# Task\n\nExecute the function with the inputs that follow in the next section and finally return '
            'the output using the output type\nas YAML document in an # Output section. (If the value is a literal, '
            'then just write the value. We parse the text in the\n# Output section using `yaml.safe_load` in '
            'Python.)\n\n# Dataclasses Schema\n\ntypes:\n  Customer:\n    birthday:\n      type: str\n    city:\n     '
            ' type: str\n    first_name:\n      type: str\n    last_name:\n      type: str\n  CustomerDatabase:\n    '
            'customers:\n      type: \'[Customer]\'\n  CustomerDatabase[FakeLLM]:\n    bases:\n    - '
            'CustomerDatabase\n    customers:\n      type: \'[Customer]\'\n\n\n# Input Types\n\ncustomer_database: '
            'CustomerDatabase[FakeLLM]\nnum_queries: int\n\n\n# Inputs\n\ncustomer_database:\n  customers:\n  - '
            'birthday: 05/20/1993\n    city: Tokyo\n    first_name: John\n    last_name: Doe\n  - birthday: '
            '04/15/1989\n    city: London\n    first_name: Jane\n    last_name: Smith\n  - birthday: 03/10/1996\n    '
            'city: New York\n    first_name: Mark\n    last_name: Miller\nnum_queries: 3\n\n\n# Output Type\n\n['
            'str]\n\n# Execution Scratch-Pad (Think Step by Step)\n\n\n# Output\n\n---\nresult:\n- "The customer that '
            'was born in 1989"\n- "The customer that lives in Tokyo"\n- "The customer with the last name of Miller"',
        },
        external_llm=OpenAI(),
    )

    LLMCustomerDatabase = llm_strategy(llm)(CustomerDatabase)

    customers = LLMCustomerDatabase.create_mock_customers(3)

    customer_database = LLMCustomerDatabase(customers)

    mock_queries = LLMCustomerDatabase.create_mock_queries(customer_database, 3)

    print(customer_database)

    for query in mock_queries:
        index = customer_database.find_customer_index(query)
        print(query)
        print(customer_database.customers[index])
        assert index >= 0
        assert index < len(customers)
