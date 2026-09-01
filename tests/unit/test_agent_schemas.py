from agent.schemas import TOOL_REGISTRY


def test_registry_contains_all_data_detective_tools():
    expected = {
        "inspect_dataset",
        "profile_columns",
        "detect_target",
        "check_data_quality",
        "detect_leakage",
        "create_split",
    }
    assert set(TOOL_REGISTRY) == expected


def test_each_tool_has_a_non_empty_description():
    for tool in TOOL_REGISTRY.values():
        assert tool.description.strip()


def test_parameters_and_returns_are_valid_json_schema_objects():
    for tool in TOOL_REGISTRY.values():
        assert tool.parameters.get("type") == "object"
        assert tool.returns.get("type") == "object"


def test_detect_leakage_args_require_target():
    schema = TOOL_REGISTRY["detect_leakage"].parameters
    assert "target" in schema["required"]
    assert "dataset_version_id" in schema["required"]


def test_detect_target_args_objective_is_optional():
    schema = TOOL_REGISTRY["detect_target"].parameters
    assert "objective" not in schema.get("required", [])
