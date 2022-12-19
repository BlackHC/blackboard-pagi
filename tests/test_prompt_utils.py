from blackboard_pagi.controller import approximate_token_count, extract_last_yaml_block, wrap_yaml_blocks_as_source


def test_wrap_yaml_blocks_as_source():
    """Test wrap_yaml_blocks_as_source."""
    text = """
Some text
---
tool: search-google
query: OpenAI
---
Some more text

---
tool: search-google
query: OpenAI GPT-3
---

Some more text"""
    expected = """Some text
```yaml
---
tool: search-google
query: OpenAI
---
```
Some more text

```yaml
---
tool: search-google
query: OpenAI GPT-3
---
```

Some more text"""
    assert wrap_yaml_blocks_as_source(text) == expected


def test_approximate_token_count():
    """Test approximate_token_count."""
    assert approximate_token_count("") == 0
    assert approximate_token_count(" ") == 0
    assert approximate_token_count("a") == 1
    assert approximate_token_count("ab") == 1
    assert approximate_token_count("abc") == 1
    assert approximate_token_count("abcd") == 1
    assert approximate_token_count("abcde") == 2
    assert approximate_token_count("abcdef") == 2
    assert approximate_token_count("abcdefg") == 2
    assert approximate_token_count("abcdefgh") == 2

    # Now test with whitespace
    assert approximate_token_count("a b") == 2
    assert approximate_token_count("a b c") == 3
    assert approximate_token_count("a b c d") == 4

    # Now test with words which are longer than 4 characters
    assert approximate_token_count("abcd efgh") == 2
    assert approximate_token_count("abcd efgh ijkl") == 3
    assert approximate_token_count("abcde fghij klmno") == 6


# def extract_last_yaml_block(prompt: str) -> Optional[str]:
#     """Extract the last YAML block from the prompt. Fail gracefully if there is no YAML block."""
#     yaml_blocks = prompt.split("\n---\n")
#     if len(yaml_blocks) == 1:
#         return None
#     else:
#         return yaml_blocks[-1]


def test_extract_last_yaml_block():
    """Test extract_last_yaml_block."""
    text = """
Some text
---
tool: search-google
query: OpenAI
---
Some more text

---
tool: search-google
query: OpenAI GPT-3
"""
    expected = """tool: search-google
query: OpenAI GPT-3
"""
    assert extract_last_yaml_block(text) == expected
