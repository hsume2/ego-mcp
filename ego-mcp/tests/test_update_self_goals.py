"""Tests for update_self short-term goal fields."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ego_mcp._server_backend_handlers import _handle_update_self
from ego_mcp.config import EgoConfig
from ego_mcp.self_model import SelfModelStore


@pytest.fixture
def config(tmp_path: Path) -> EgoConfig:
    return EgoConfig(
        embedding_provider="gemini",
        embedding_model="gemini-embedding-001",
        api_key="test-key",
        data_dir=tmp_path,
        companion_name="TestUser",
        workspace_dir=None,
        timezone="UTC",
    )


def _update(config: EgoConfig, field: str, value: object) -> str:
    return _handle_update_self(config, {"field": field, "value": value})


class TestUpdateSelfGoals:
    def test_new_goal_accepts_bare_string_and_parent(self, config: EgoConfig) -> None:
        store = SelfModelStore(config.data_dir / "self_model.json")
        parent_id = store.set_long_term_goals([{"goal": "Learn Japanese"}])[0]["id"]

        result = _update(config, "new_goal", {"goal": "Practice daily", "parent": parent_id})

        assert "Holding short-term goal g_" in result
        assert f"toward {parent_id} (Learn Japanese)" in result
        assert "Holding short-term goal g_" in _update(
            config, "new_goal", "Review vocabulary"
        )
        assert SelfModelStore(config.data_dir / "self_model.json").get().current_goals == [
            "Practice daily",
            "Review vocabulary",
        ]

    def test_new_goal_rejects_invalid_parent_without_adding_goal(
        self, config: EgoConfig
    ) -> None:
        store = SelfModelStore(config.data_dir / "self_model.json")
        parent_id = store.set_long_term_goals([{"goal": "Learn Japanese"}])[0]["id"]
        before = store.get_goal_log()

        result = _update(config, "new_goal", {"goal": "Practice", "parent": "lt_missing"})

        assert "Invalid parent lt_missing" in result
        assert parent_id in result
        assert store.get_goal_log() == before

    def test_goal_progress_resolve_and_drop(self, config: EgoConfig) -> None:
        goal_id = SelfModelStore(config.data_dir / "self_model.json").add_goal(
            "Practice daily"
        )

        assert _update(
            config,
            "goal_progress",
            {"id": goal_id, "note": "Practiced for 20 minutes"},
        ) == f"Recorded progress for goal {goal_id}."
        empty = _update(config, "goal_progress", {"id": goal_id, "note": "  "})
        assert "non-empty note" in empty
        assert "not found or not active" in _update(
            config, "goal_progress", {"id": "g_missing", "note": "Done"}
        )
        assert "Resolved" in _update(
            config, "resolve_goal", {"id": goal_id, "note": "Completed"}
        )
        assert "not found or already closed" in _update(config, "resolve_goal", goal_id)

        drop_id = SelfModelStore(config.data_dir / "self_model.json").add_goal(
            "Try another method"
        )
        assert "Dropped" in _update(config, "drop_goal", drop_id)
        assert "not found or already closed" in _update(config, "drop_goal", drop_id)

    def test_long_term_progress_and_fixed_goal_rejections(
        self, config: EgoConfig
    ) -> None:
        path = config.data_dir / "self_model.json"
        store = SelfModelStore(path)
        long_term = store.set_long_term_goals([{"goal": "Build strength"}])[0]
        fixed_message = (
            f"Goal {long_term['id']} is a long-term goal. Long-term goals are fixed; "
            "only the operator changes them."
        )

        assert _update(
            config,
            "goal_progress",
            {"id": long_term["id"], "note": "Practiced for 20 minutes"},
        ) == f"Recorded progress for goal {long_term['id']}."
        for field in ("resolve_goal", "drop_goal"):
            assert _update(config, field, long_term["id"]) == fixed_message
            assert SelfModelStore(path).get_long_term_goals() == [long_term]

        assert (
            _update(
                config,
                "new_goal",
                {"goal": "Replace the fixed goal", "supersedes": long_term["id"]},
            )
            == fixed_message
        )
        assert store.get_goal_log() == []
        assert SelfModelStore(path).get_long_term_goals() == [long_term]

    def test_new_goal_supersedes_and_inherits_parent(self, config: EgoConfig) -> None:
        store = SelfModelStore(config.data_dir / "self_model.json")
        parent_id = store.set_long_term_goals([{"goal": "Build strength"}])[0]["id"]
        old_id = store.add_goal("Walk regularly", parent=parent_id)

        result = _update(
            config,
            "new_goal",
            {"goal": "Walk four times weekly", "supersedes": old_id},
        )

        assert f"supersedes {old_id}" in result
        updated_store = SelfModelStore(config.data_dir / "self_model.json")
        new_entry = next(
            item
            for item in updated_store.get_goal_log()
            if item["goal"] == "Walk four times weekly"
        )
        old_entry = next(
            item for item in updated_store.get_goal_log() if item["id"] == old_id
        )
        assert new_entry["parent"] == parent_id
        assert f"toward {parent_id} (Build strength)" in result
        assert new_entry["lineage"] == [old_id]
        assert old_entry["status"] == "dropped"
        assert old_entry["resolved_note"] == f"superseded by {new_entry['id']}"

    def test_goal_limit_lists_active_ids_and_allows_superseding(
        self, config: EgoConfig
    ) -> None:
        store = SelfModelStore(config.data_dir / "self_model.json")
        goal_ids = [store.add_goal(f"Goal {index}") for index in range(5)]

        result = _update(config, "new_goal", "Too many")
        assert "resolve_goal, drop_goal, or supersedes" in result
        assert all(goal_id in result for goal_id in goal_ids)

        replacement = _update(
            config,
            "new_goal",
            {"goal": "Replacement", "supersedes": goal_ids[0]},
        )
        assert "Holding short-term goal" in replacement
        assert len(store.get().current_goals) == 5

    @pytest.mark.parametrize("field", ["long_term_goals"])
    def test_long_term_goals_are_operator_only(
        self, config: EgoConfig, field: str
    ) -> None:
        result = _update(config, field, [{"goal": "Operator goal"}])
        assert (
            result
            == "Long-term goals are fixed; they are set by the operator. "
            "To hold a short-term goal of your own, use new_goal "
            "(optionally with parent: <long-term id>)."
        )

    @pytest.mark.parametrize("field", ["current_goals", "goals"])
    def test_derived_goal_lists_are_not_writable(
        self, config: EgoConfig, field: str
    ) -> None:
        result = _update(config, field, ["not writable"])
        assert "derived list" in result
        assert "new_goal" in result
        assert "resolve_goal" in result
        assert "drop_goal" in result

    def test_unknown_field_reflection_hints_include_goal_fields(
        self, config: EgoConfig
    ) -> None:
        result = _update(config, "unrecognized", None)
        assert "new_goal" in result
        assert "goal_progress" in result
        assert "resolve_goal" in result
        assert "drop_goal" in result
        assert "current_goals" not in result

    def test_invalid_goal_updates_leave_goal_log_unchanged(
        self, config: EgoConfig
    ) -> None:
        store = SelfModelStore(config.data_dir / "self_model.json")
        store.add_goal("Existing")
        before = json.loads(
            (config.data_dir / "self_model.json").read_text(encoding="utf-8")
        )
        result = _update(config, "resolve_goal", "g_unknown")
        after = json.loads(
            (config.data_dir / "self_model.json").read_text(encoding="utf-8")
        )
        assert "not found or already closed" in result
        assert after["goal_log"] == before["goal_log"]
