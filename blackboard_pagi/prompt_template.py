from dataclasses import dataclass

import parse


def prompt_template(prompt_template: str, as_dataclass=True):
    """
    Wraps
    """

    def prompt_wrapper(class_definition):
        def to_prompt(self, *args, **kwargs) -> str:
            """
            Returns a prompt string with the given arguments.
            """
            return self.__class__.prompt_template.format(*args, **kwargs, **vars(self))

        @classmethod
        def from_prompt(clazz, text: str):
            result = parse.parse(clazz.prompt_template, text)
            if result is None:
                raise ValueError(f"Could not parse prompt {text}")
            # TODO: Handle positional arguments (to to_prompt)
            # (WANT: If the result has fixed-position values from the string (e.g. {0} or {1}),
            # then we return them separately.)
            assert len(result.fixed) == 0
            prompt_instance = clazz(**result.named)
            return prompt_instance

        if as_dataclass:
            class_definition = dataclass(class_definition)

        new_type = type(
            class_definition.__name__,
            (class_definition,),
            dict(to_prompt=to_prompt, from_prompt=from_prompt),
        )
        new_type.prompt_template = prompt_template

        return new_type

    return prompt_wrapper
