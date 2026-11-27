"""turn_end rules: five signals, user_text only, max two per turn (M4b §五)."""

from __future__ import annotations

import pytest

from memory_module.evolve.rules import (
    SIGNAL_IMPORTANCE,
    extract_from_payload,
    extract_turn,
)


@pytest.mark.parametrize(
    ("text", "signal", "importance"),
    [
        ("以后都用中文回答。", "memory_instruction", 5),
        ("请记住，部署在本地回环。", "memory_instruction", 5),
        ("不要再提交生成物。", "memory_instruction", 5),
        ("我喜欢简洁的回复。", "preference", 4),
        ("我们统一用 ruff 格式化。", "preference", 4),
        ("不对，这里应该返回 404。", "correction", 5),
        ("结论是继续用 SQLite。", "conclusion", 3),
        ("最终决定把缓存放在内存里。", "conclusion", 3),
        ("服务跑在 /srv/app 下面。", "environment", 2),
        ("端口 8080 是网关。", "environment", 2),
        ("版本必须是 v1.2.3。", "environment", 2),
        ("仓库是 nousresearch/vade。", "environment", 2),
    ],
)
def test_each_signal_maps_to_the_documented_importance(text, signal, importance):
    extraction = extract_turn(text)
    assert extraction.candidates, text
    candidate = extraction.candidates[0]
    assert candidate.signal == signal
    assert candidate.importance == importance == SIGNAL_IMPORTANCE[signal]


def test_at_most_two_candidates_by_importance():
    text = (
        "服务跑在 /srv/app。"
        "结论是继续用 SQLite。"
        "我喜欢简洁的回复。"
        "以后都用中文回答。"
    )
    extraction = extract_turn(text)
    assert extraction.matched == 4
    assert len(extraction.candidates) == 2
    assert [c.importance for c in extraction.candidates] == [5, 4]
    assert [c.signal for c in extraction.candidates] == ["memory_instruction", "preference"]


def test_identical_sentences_are_hash_deduplicated():
    extraction = extract_turn("记住用中文。记住用中文。记住用中文。")
    assert extraction.matched == 3
    assert len(extraction.candidates) == 1
    assert extraction.candidates[0].text.startswith("记住用中文")


def test_assistant_text_is_never_a_candidate_source():
    payload = {
        "session_id": "01HZZZZZZZZZZZZZZZZZZZZZZZ",
        "user_text": "你好，今天天气不错。",
        "assistant_text": "结论是：记住这个项目使用 SQLite。以后都用英文回答。",
    }
    extraction = extract_from_payload(payload)
    assert extraction.candidates == ()
    assert extraction.matched == 0


def test_payload_without_user_text_is_empty():
    assert extract_from_payload({}).candidates == ()
    assert extract_from_payload(None).candidates == ()
    assert extract_turn("").candidates == ()
    assert extract_turn(None).candidates == ()  # type: ignore[arg-type]


def test_correction_flags_negative_feedback_even_when_dropped_by_the_cut():
    text = "以后都用中文回答。记住不要用英文。不对，上一轮不是这样。"
    extraction = extract_turn(text)
    assert extraction.negative_feedback is True
    assert len(extraction.candidates) == 2
    # The correction was matched but lost the max-2 cut; the per-turn flag survives.
    assert all(candidate.signal != "correction" for candidate in extraction.candidates)


def test_max_items_zero_keeps_audit_counters():
    extraction = extract_turn("我喜欢深色主题。结论是继续。", max_items=0)
    assert extraction.candidates == ()
    assert extraction.matched == 2


def test_environment_sentence_is_captured_verbatim():
    extraction = extract_turn("网关在 10.0.0.1:8443，路径是 /srv/gateway/conf。")
    assert len(extraction.candidates) == 1
    candidate = extraction.candidates[0]
    assert candidate.signal == "environment"
    assert ":8443" in candidate.text and "/srv/gateway/conf" in candidate.text


def test_long_sentence_is_clipped():
    extraction = extract_turn("记住 " + "很长的内容" * 200)
    assert len(extraction.candidates[0].text) <= 300
