from blackboard_pagi.prompt_optimizer.ai_function import (
    get_json_schema_hyperparameters,
    update_json_schema_hyperparameters,
)


def test_get_json_schema_hyperparameters():
    schema = {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"title": "test", "description": "test", "type": "string"},
            "test2": {"title": "test", "description": "test", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }
    assert get_json_schema_hyperparameters(schema) == {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"title": "test", "description": "test"},
            "test2": {"title": "test", "description": "test"},
        },
    }


def test_update_json_schema_hyperparameters():
    schema = {
        "title": "test",
        "description": "test",
        "properties": {
            "test": {"title": "test", "description": "test", "type": "string"},
            "test2": {"title": "test", "description": "test", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }
    hyperparameters = {
        "title": "test2",
        "description": "test2",
        "properties": {
            "test": {"title": "test2", "description": "test2"},
            "test2": {"title": "test2", "description": "test2"},
        },
    }
    update_json_schema_hyperparameters(schema, hyperparameters)
    assert schema == {
        "title": "test2",
        "description": "test2",
        "properties": {
            "test": {"title": "test2", "description": "test2", "type": "string"},
            "test2": {"title": "test2", "description": "test2", "type": "string"},
        },
        "type": "object",
        "required": ["test", "test2"],
    }
