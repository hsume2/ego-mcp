"""Tests for SelfModelStore."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ego_mcp.self_model import SelfModelStore, _calculate_salience


class TestSelfModelStore:
    def test_get_default(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        model = store.get()
        assert model.confidence_calibration == 0.5
        assert model.unresolved_questions == []
        assert model.last_updated != ""

    def test_update_partial(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        store.update({"confidence_calibration": 0.8})
        store.add_goal("learn")
        model = store.get()
        assert model.confidence_calibration == 0.8
        assert model.current_goals == ["learn"]
        assert model.last_updated != ""

    def test_update_rejects_unknown_field(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        with pytest.raises(ValueError, match="Invalid self-model field"):
            store.update({"nonexistent": "value"})

    def test_update_rejects_wrong_type_for_dict_field(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        with pytest.raises(TypeError, match="preferences.*dict.*str"):
            store.update({"preferences": "likes quiet"})

    def test_update_rejects_wrong_type_for_list_field(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        with pytest.raises(TypeError, match="current_goals.*list.*str"):
            store.update({"current_goals": "ship the patch"})

    def test_update_rejects_wrong_type_for_numeric_field(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        with pytest.raises(TypeError, match="confidence_calibration"):
            store.update({"confidence_calibration": "high"})

    def test_add_question_and_resolve(self, tmp_path: Path) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        qid = store.add_question("What should I prioritize?")
        model = store.get()
        assert qid.startswith("q_")
        assert model.unresolved_questions == ["What should I prioritize?"]

        resolved = store.resolve_question(qid)
        model_after = store.get()
        assert resolved is True
        assert model_after.unresolved_questions == []

    def test_add_question_stores_importance_and_created_at(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        store = SelfModelStore(path)
        qid = store.add_question("How should I pace learning?", importance=5)

        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = next(item for item in payload["question_log"] if item["id"] == qid)
        assert entry["importance"] == 5
        assert entry["created_at"]
        assert entry["person_id"] is None
        assert entry["companions"] == []
        assert entry["lineage"] == []
        assert entry["last_fed_at"] == ""

    def test_question_log_normalizes_ripening_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        path.write_text(
            json.dumps(
                {
                    "question_log": [
                        {
                            "id": "q_with_fields",
                            "question": "Who shares this edge?",
                            "resolved": False,
                            "importance": 4,
                            "created_at": "2026-07-01T12:00:00+00:00",
                            "person_id": "alice",
                            "companions": [
                                {"memory_id": "m1", "kind": "companion"},
                                "bad",
                            ],
                            "lineage": ["q_old", 123],
                            "last_fed_at": "2026-07-02T12:00:00+00:00",
                        },
                        {
                            "id": "q_bad_fields",
                            "question": "What survives bad metadata?",
                            "person_id": 123,
                            "companions": "bad",
                            "lineage": "bad",
                            "last_fed_at": 456,
                        },
                    ],
                    "unresolved_questions": ["q_with_fields", "q_bad_fields"],
                }
            ),
            encoding="utf-8",
        )

        entries = SelfModelStore(path).get_question_log()

        assert entries[0]["person_id"] == "alice"
        assert entries[0]["companions"] == [{"memory_id": "m1", "kind": "companion"}]
        assert entries[0]["lineage"] == ["q_old"]
        assert entries[0]["last_fed_at"] == "2026-07-02T12:00:00+00:00"
        assert entries[1]["person_id"] is None
        assert entries[1]["companions"] == []
        assert entries[1]["lineage"] == []
        assert entries[1]["last_fed_at"] == ""

    def test_add_question_clamps_importance(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        store = SelfModelStore(path)
        qid = store.add_question("How urgent is this?", importance=99)
        payload = json.loads(path.read_text(encoding="utf-8"))
        entry = next(item for item in payload["question_log"] if item["id"] == qid)
        assert entry["importance"] == 5

    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        store = SelfModelStore(path)
        store.add_goal("maintain continuity")
        store.add_question("How can I improve focus?")

        reloaded = SelfModelStore(path)
        model = reloaded.get()
        assert model.current_goals == ["maintain continuity"]
        assert "How can I improve focus?" in model.unresolved_questions

    def test_goal_lifecycle_add_progress_resolve_and_drop(
        self, tmp_path: Path
    ) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        long_term_id = store.set_long_term_goals([{"goal": "Learn Japanese"}])[0]["id"]
        goal_id = store.add_goal("Practice daily", parent=long_term_id)
        assert goal_id.startswith("g_") and len(goal_id) == 12
        assert store.get().current_goals == ["Practice daily"]
        assert store.add_goal_progress(goal_id, "Reviewed five words")
        entry = next(item for item in store.get_goal_log() if item["id"] == goal_id)
        assert entry["progress"][0]["note"] == "Reviewed five words"
        assert entry["progress"][0]["at"]
        assert store.close_goal(goal_id, status="done", note="Finished the week")
        assert store.get().current_goals == []
        closed = next(item for item in store.get_goal_log() if item["id"] == goal_id)
        assert closed["status"] == "done"
        assert closed["resolved_note"] == "Finished the week"
        assert closed["resolved_at"]

        drop_id = store.add_goal("Try another routine")
        assert store.close_goal(drop_id, status="dropped")
        dropped = next(item for item in store.get_goal_log() if item["id"] == drop_id)
        assert dropped["status"] == "dropped"
        assert dropped["resolved_note"] is None

    def test_goal_supersedes_carries_lineage_and_parent(
        self, tmp_path: Path
    ) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        parent_id = store.set_long_term_goals([{"goal": "Build strength"}])[0]["id"]
        first_id = store.add_goal("Walk regularly", parent=parent_id)
        second_id = store.add_goal("Walk four times weekly", supersedes=first_id)
        first = next(item for item in store.get_goal_log() if item["id"] == first_id)
        second = next(item for item in store.get_goal_log() if item["id"] == second_id)
        assert first["status"] == "dropped"
        assert first["resolved_note"] == f"superseded by {second_id}"
        assert second["parent"] == parent_id
        assert second["lineage"] == [first_id]

        third_id = store.add_goal("Walk every weekday", supersedes=second_id)
        third = next(item for item in store.get_goal_log() if item["id"] == third_id)
        assert third["lineage"] == [first_id, second_id]
        assert third["parent"] == parent_id

    def test_active_goal_cap_allows_superseding_at_capacity(
        self, tmp_path: Path
    ) -> None:
        store = SelfModelStore(tmp_path / "self_model.json")
        goal_ids = [store.add_goal(f"Goal {index}") for index in range(5)]
        with pytest.raises(ValueError, match="Maximum active"):
            store.add_goal("One too many")
        replacement = store.add_goal("Replacement", supersedes=goal_ids[0])
        assert len(store.get().current_goals) == 5
        assert replacement in {item["id"] for item in store.get_goal_log()}

    def test_load_rescues_legacy_current_goals_and_derives_compatibility_field(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "self_model.json"
        path.write_text(
            json.dumps(
                {
                    "current_goals": ["legacy goal", "already tracked"],
                    "goal_log": [
                        {
                            "id": "g_existing",
                            "goal": "already tracked",
                            "status": "active",
                        },
                        {
                            "id": "g_closed",
                            "goal": "closed goal",
                            "status": "done",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        store = SelfModelStore(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["current_goals"] == []
        assert [entry["goal"] for entry in payload["goal_log"]] == [
            "already tracked",
            "closed goal",
            "legacy goal",
        ]
        assert store.get().current_goals == ["already tracked", "legacy goal"]
        rescued = next(
            item for item in store.get_goal_log() if item["goal"] == "legacy goal"
        )
        assert rescued["parent"] is None

    def test_goal_normalization_tolerates_junk_entries(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        path.write_text(
            json.dumps(
                {
                    "goal_log": [
                        None,
                        {"id": "g_valid", "goal": "A valid goal", "lineage": "bad"},
                        {"id": 7, "goal": "Bad ID"},
                        {"id": "g_bad", "goal": 9},
                    ],
                    "long_term_goals": [
                        "bad",
                        {"id": "lt_valid", "goal": "A long-term goal"},
                        {"id": 3, "goal": "Bad ID"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = SelfModelStore(path)
        goals = store.get_goal_log()
        assert len(goals) == 1
        assert goals[0]["lineage"] == []
        assert goals[0]["progress"] == []
        assert goals[0]["status"] == "active"
        assert [item["id"] for item in store.get_long_term_goals()] == ["lt_valid"]

    def test_corrupt_json_fallback(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        path.write_text("{broken", encoding="utf-8")
        store = SelfModelStore(path)
        model = store.get()
        assert model.unresolved_questions == []

    def test_legacy_question_entries_fallback_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "self_model.json"
        path.write_text(
            json.dumps(
                {
                    "question_log": [
                        {
                            "id": "q_legacy",
                            "question": "Legacy question without metadata",
                            "resolved": False,
                        }
                    ],
                    "unresolved_questions": ["q_legacy"],
                }
            ),
            encoding="utf-8",
        )
        store = SelfModelStore(path)
        active, resurfacing = store.get_visible_questions()
        assert resurfacing == []
        assert active
        assert active[0]["id"] == "q_legacy"
        assert active[0]["importance"] == 3
        assert active[0]["created_at"] == ""

    def test_load_rescues_orphan_unresolved_question_strings(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "self_model.json"
        path.write_text(
            json.dumps(
                {
                    "question_log": [],
                    "unresolved_questions": [
                        "What did the old raw list forget?",
                        "Where should this be carried?",
                    ],
                }
            ),
            encoding="utf-8",
        )

        store = SelfModelStore(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rescued_ids = payload["unresolved_questions"]

        assert len(rescued_ids) == 2
        assert all(str(qid).startswith("q_") for qid in rescued_ids)
        assert [entry["question"] for entry in payload["question_log"]] == [
            "What did the old raw list forget?",
            "Where should this be carried?",
        ]
        assert all(entry["importance"] == 3 for entry in payload["question_log"])
        assert all(entry["created_at"] for entry in payload["question_log"])
        assert store.get().unresolved_questions == [
            "What did the old raw list forget?",
            "Where should this be carried?",
        ]

    def test_load_reuses_matching_unresolved_question_text(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "self_model.json"
        path.write_text(
            json.dumps(
                {
                    "question_log": [
                        {
                            "id": "q_existing",
                            "question": "How do I keep this thread?",
                            "resolved": False,
                            "importance": 4,
                            "created_at": "2026-07-01T12:00:00+00:00",
                        }
                    ],
                    "unresolved_questions": ["How do I keep this thread?"],
                }
            ),
            encoding="utf-8",
        )

        SelfModelStore(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["unresolved_questions"] == ["q_existing"]
        assert len(payload["question_log"]) == 1

    def test_get_visible_questions_classifies_active_fading_dormant(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "self_model.json"
        now = datetime.now(timezone.utc)
        path.write_text(
            json.dumps(
                {
                    "question_log": [
                        {
                            "id": "q_active",
                            "question": "Active question",
                            "resolved": False,
                            "importance": 5,
                            "created_at": (now - timedelta(days=5)).isoformat(),
                        },
                        {
                            "id": "q_fading",
                            "question": "Fading question",
                            "resolved": False,
                            "importance": 4,
                            "created_at": (now - timedelta(days=60)).isoformat(),
                        },
                        {
                            "id": "q_dormant",
                            "question": "Dormant question",
                            "resolved": False,
                            "importance": 1,
                            "created_at": (now - timedelta(days=120)).isoformat(),
                        },
                    ],
                    "unresolved_questions": ["q_active", "q_fading", "q_dormant"],
                }
            ),
            encoding="utf-8",
        )
        store = SelfModelStore(path)
        active, resurfacing = store.get_visible_questions()
        assert [q["id"] for q in active] == ["q_active"]
        assert [q["id"] for q in resurfacing] == ["q_fading"]
        assert all("salience" in q and "age_days" in q for q in active + resurfacing)


class TestQuestionSalience:
    @pytest.mark.parametrize(
        ("importance", "age_days", "expected"),
        [
            (5, 0.0, 1.0),
            (3, 0.0, 0.6),
            (1, 0.0, 0.2),
        ],
    )
    def test_salience_initial_value(
        self, importance: int, age_days: float, expected: float
    ) -> None:
        assert _calculate_salience(importance, age_days) == pytest.approx(expected)

    def test_salience_decays_with_age(self) -> None:
        fresh = _calculate_salience(5, 0.0)
        older = _calculate_salience(5, 70.0)
        much_older = _calculate_salience(5, 180.0)
        assert fresh > older > much_older
