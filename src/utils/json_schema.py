from __future__ import annotations


def make_strict(obj: dict) -> None:
    """Mutate a JSON schema object in-place for OpenAI strict mode.

    Strict mode requires:
    - additionalProperties: false on every object
    - required lists every key in properties (including optional/defaulted ones)
    """
    if obj.get("type") == "object" and "properties" in obj:
        obj["additionalProperties"] = False
        obj["required"] = list(obj["properties"].keys())
        for prop in obj["properties"].values():
            if isinstance(prop, dict):
                make_strict(prop)
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in obj.get(key, []):
            if isinstance(sub, dict):
                make_strict(sub)
    for def_schema in obj.get("$defs", {}).values():
        if isinstance(def_schema, dict):
            make_strict(def_schema)


def build_response_format(name: str, schema: dict, *, strict: bool = True) -> dict:
    """Build an OpenAI-compatible ``response_format`` dict for structured output."""
    make_strict(schema)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": strict,
        },
    }
