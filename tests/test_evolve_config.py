"""EvolveSettings: defaults, parsing, validation (C-18 / M4b §九)."""

from __future__ import annotations

import pytest

from memory_module.evolve.config import EvolveSettings


def test_defaults_match_the_contract():
    settings = EvolveSettings()
    assert settings.enabled is False
    assert settings.nudge_interval == 10
    assert settings.flush_min_turns == 6
    assert settings.session_char_limit == 4200
    assert settings.summary_token_ratio == 0.05
    assert settings.lesson_confidence_floor == 0.6
    assert settings.failure_learning is True
    assert settings.daily_token_budget == 200_000
    assert settings.daily_budget_skills == 5
    assert settings.session_budget_skills == 1
    assert settings.redact == ("tool_args", "file_content")
    assert settings.model == "cheap"
    assert settings.allow_transcript is False
    assert settings.allowlist == ()


def test_from_mapping_accepts_the_whole_evolve_table_and_coerces_strings():
    settings = EvolveSettings.from_mapping(
        {
            "enabled": "true",
            "allow_transcript": "off",
            "allowlist": ["com.example.memory", "sha256:ab"],
            "nudge_interval": "15",
            "flush_min_turns": "4",
            "session_char_limit": "8000",
            "summary_token_ratio": "0.1",
            "lesson_confidence_floor": "0.5",
            "failure_learning": 0,
            "daily_token_budget": "1000",
            "daily_budget_skills": 2,
            "session_budget_skills": 1,
            "redact": ["tool_args"],
            "model": "cheap-2",
        }
    )
    assert settings.enabled is True
    assert settings.allow_transcript is False
    assert settings.failure_learning is False
    assert settings.nudge_interval == 15
    assert settings.session_char_limit == 8000
    assert settings.summary_token_ratio == 0.1
    assert settings.redact == ("tool_args",)
    assert settings.allowlist == ("com.example.memory", "sha256:ab")
    assert settings.model == "cheap-2"


def test_from_mapping_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown \\[evolve\\] key"):
        EvolveSettings.from_mapping({"enabled": True, "nudge_intervall": 10})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("nudge_interval", 4),
        ("nudge_interval", 51),
        ("flush_min_turns", 1),
        ("flush_min_turns", 21),
        ("session_char_limit", 100),
        ("summary_token_ratio", 0.0),
        ("summary_token_ratio", 0.9),
        ("lesson_confidence_floor", -0.1),
        ("daily_token_budget", -1),
        ("daily_budget_skills", 21),
        ("session_budget_skills", 4),
    ],
)
def test_out_of_range_values_are_rejected(key, value):
    with pytest.raises(ValueError):
        EvolveSettings(**{key: value})


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("enabled", "yes please"),
        ("nudge_interval", True),
        ("nudge_interval", 10.5),
        ("summary_token_ratio", "lots"),
        ("redact", "tool_args"),
        ("redact", [""]),
        ("redact", [1]),
        ("model", ""),
        ("model", 3),
    ],
)
def test_bad_types_are_rejected(key, value):
    with pytest.raises(ValueError):
        EvolveSettings.from_mapping({key: value})


def test_redact_is_deduplicated_in_order():
    settings = EvolveSettings.from_mapping({"redact": ["tool_args", "tool_args", "file_content"]})
    assert settings.redact == ("tool_args", "file_content")


def test_summary_budget_is_ctx_times_ratio_in_chars():
    settings = EvolveSettings(summary_token_ratio=0.05)
    assert settings.summary_token_budget(32_000) == 1600
    assert settings.summary_char_budget(32_000) == 6400
    assert settings.summary_token_budget(0) == 1

    small = EvolveSettings(summary_token_ratio=0.01)
    assert small.summary_char_budget(32_000) == 1280


def test_with_overrides_keeps_validation():
    settings = EvolveSettings(enabled=True)
    changed = settings.with_overrides(nudge_interval=20)
    assert changed.nudge_interval == 20 and changed.enabled is True
    with pytest.raises(ValueError):
        settings.with_overrides(nudge_interval=1)


# --------------------------------------------------- connection settings
# These live on `MemorySettings`, not `EvolveSettings`: C-18's `[evolve]` table
# has 14 keys and no upstream address, and core rejects unknown keys at startup.
# Adding one here would make a valid config fail to boot.
def test_evolve_settings_still_rejects_connection_keys():
    for key in ("base_url", "api_key_env", "llm_base_url", "model_base_url"):
        with pytest.raises(ValueError, match="unknown \\[evolve\\] key"):
            EvolveSettings.from_mapping({key: "http://127.0.0.1:8080/v1"})


def test_evolve_settings_key_set_matches_c18_exactly():
    """Guards the contract: 14 keys, and none of them a connection setting."""
    from memory_module.evolve.config import _KNOWN_KEYS

    assert _KNOWN_KEYS == {
        "enabled",
        "allow_transcript",
        "allowlist",
        "nudge_interval",
        "flush_min_turns",
        "session_char_limit",
        "summary_token_ratio",
        "lesson_confidence_floor",
        "failure_learning",
        "daily_token_budget",
        "daily_budget_skills",
        "session_budget_skills",
        "redact",
        "model",
    }
    assert len(_KNOWN_KEYS) == 14


def test_evolve_connection_env_vars_default_to_off():
    """Unconfigured means off — the M4a-only deployment must not change."""
    from memory_module.settings import MemorySettings

    settings = MemorySettings.from_env({})
    assert settings.evolve_llm_base_url == ""
    assert settings.evolve_llm_api_key_env == "MEMORY_API_KEY"
    assert settings.evolve_llm_timeout_s == 30.0
    assert settings.evolve_llm_model == ""
    # With no explicit model, the C-18 label is the documented fallback.
    assert settings.effective_evolve_model() == "cheap"
    assert settings.effective_evolve_model("custom-label") == "custom-label"


def test_evolve_connection_env_vars_are_read_and_validated():
    from memory_module.settings import MemorySettings

    settings = MemorySettings.from_env(
        {
            "MEMORY_EVOLVE_LLM_BASE_URL": "https://router.example/v1",
            "MEMORY_EVOLVE_LLM_API_KEY_ENV": "MY_EVOLVE_KEY",
            "MEMORY_EVOLVE_LLM_TIMEOUT_S": "12.5",
            "MEMORY_EVOLVE_LLM_MODEL": "gpt-4o-mini",
        }
    )
    assert settings.evolve_llm_base_url == "https://router.example/v1"
    assert settings.evolve_llm_api_key_env == "MY_EVOLVE_KEY"
    assert settings.evolve_llm_timeout_s == 12.5
    assert settings.effective_evolve_model() == "gpt-4o-mini"

    with pytest.raises(ValueError, match="MEMORY_EVOLVE_LLM_BASE_URL"):
        MemorySettings(evolve_llm_base_url="router.example/v1")
    with pytest.raises(ValueError, match="evolve_llm_timeout_s"):
        MemorySettings(evolve_llm_timeout_s=0)
    with pytest.raises(ValueError):
        MemorySettings.from_env({"MEMORY_EVOLVE_LLM_TIMEOUT_S": "999"})
