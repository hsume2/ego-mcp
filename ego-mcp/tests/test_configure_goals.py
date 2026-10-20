"""Tests for the operator-only configure_goals tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from ego_mcp._server_backend_configure_goals import _handle_configure_goals
from ego_mcp.self_model import SelfModelStore


@pytest.fixture
def self_model_path(tmp_path: Path) -> Path:
    return tmp_path / "self_model.json"


def _configure(path: Path, **args: object) -> str:
    return _handle_configure_goals(str(path), args)


class TestConfigureGoals:
    def test_show_returns_long_term_and_active_short_term_goals(
        self, self_model_path: Path
    ) -> None:
        store = SelfModelStore(self_model_path)
        parent = store.set_long_term_goals([{"goal": "Learn Japanese"}])[0]
        child_id = store.add_goal("Practice daily", parent=parent["id"])
        store.add_goal("Review vocabulary")

        result = _configure(self_model_path, action="show")

        assert f"[{parent['id']}] Learn Japanese" in result
        assert f"[{child_id}] Practice daily — toward [{parent['id']}]" in result
        assert "Review vocabulary — parentless" in result

    def test_set_long_term_preserves_existing_ids_and_generates_new_ids(
        self, self_model_path: Path
    ) -> None:
        store = SelfModelStore(self_model_path)
        existing = store.set_long_term_goals(
            [{"goal": "Learn Japanese"}, {"goal": "Build strength"}]
        )

        result = _configure(
            self_model_path,
            action="set_long_term",
            goals=[
                {"id": existing[0]["id"], "goal": "Study Japanese"},
                {"goal": "Write a book"},
            ],
        )

        configured = SelfModelStore(self_model_path).get_long_term_goals()
        assert existing[0]["id"] in result
        assert configured[0]["id"] == existing[0]["id"]
        assert configured[0]["goal"] == "Study Japanese"
        assert configured[0]["created_at"] == existing[0]["created_at"]
        assert configured[1]["id"].startswith("lt_")
        assert len(configured[1]["id"]) == 13
        assert existing[1]["id"] not in {item["id"] for item in configured}

    def test_removed_parent_keeps_child_active_and_marks_it_retired(
        self, self_model_path: Path
    ) -> None:
        store = SelfModelStore(self_model_path)
        parent = store.set_long_term_goals([{"goal": "Build strength"}])[0]
        child_id = store.add_goal("Walk regularly", parent=parent["id"])

        assert "Long-term goals cleared" in _configure(
            self_model_path, action="set_long_term", goals=[]
        )
        result = _configure(self_model_path, action="show")

        assert child_id in result
        assert "Walk regularly" in result
        assert f"toward [{parent['id']}] (retired)" in result
        assert store.get().current_goals == ["Walk regularly"]

    def test_rejects_unknown_supplied_long_term_id(
        self, self_model_path: Path
    ) -> None:
        result = _configure(
            self_model_path,
            action="set_long_term",
            goals=[{"id": "lt_missing", "goal": "Unknown"}],
        )
        assert "Unknown long-term goal id: lt_missing" in result

    def test_rejects_non_list_goals(self, self_model_path: Path) -> None:
        result = _configure(
            self_model_path, action="set_long_term", goals="not a list"
        )
        assert "requires goals" in result

    def test_unknown_action_is_described(self, self_model_path: Path) -> None:
        result = _configure(self_model_path, action="replace")
        assert "Use show or set_long_term" in result
