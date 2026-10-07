from agent.llm_tools import ASK_USER_TOOL_NAME, build_anthropic_tools
from agent.schemas import TOOL_REGISTRY


def test_includes_every_registry_tool_plus_ask_user():
    tools = build_anthropic_tools()
    names = {t["name"] for t in tools}
    assert names == set(TOOL_REGISTRY) | {ASK_USER_TOOL_NAME}


def test_every_tool_has_the_anthropic_shape():
    for tool in build_anthropic_tools():
        assert tool["description"].strip()
        assert tool["input_schema"]["type"] == "object"
        assert tool["strict"] is True


def test_additional_properties_false_is_enforced():
    for tool in build_anthropic_tools():
        assert tool["input_schema"]["additionalProperties"] is False


def test_ask_user_schema_requires_question_and_reason():
    tools = {t["name"]: t for t in build_anthropic_tools()}
    schema = tools[ASK_USER_TOOL_NAME]["input_schema"]
    assert "question" in schema["required"]
    assert "reason" in schema["required"]
