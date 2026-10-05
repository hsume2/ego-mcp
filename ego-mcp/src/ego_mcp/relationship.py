"""Relationship model persistence and updates."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ego_mcp import timezone_utils
from ego_mcp.types import RelationshipModel

INTERACTION_LOG_MAX = 200

_UPDATABLE_FIELDS = frozenset(
    field.name
    for field in dataclasses.fields(RelationshipModel)
    if field.name not in ("person_id",)
)

# Expected types for each updatable field.
_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "name": str,
    "known_facts": list,
    "communication_style": dict,
    "preferred_topics": list,
    "sensitive_topics": list,
    "emotional_baseline": dict,
    "trust_level": (int, float),
    "shared_episode_ids": list,
    "inferred_personality": dict,
    "recent_mood_trajectory": list,
    "first_interaction": str,
    "last_interaction": str,
    "total_interactions": (int, float),
    "aliases": list,
    "relation_kind": str,
}

# List fields that hold free text and grow one item at a time through the
# update_relationship tool (add/remove) instead of being overwritten.
TEXT_LIST_FIELDS = frozenset(
    {"known_facts", "preferred_topics", "sensitive_topics", "aliases"}
)


def _normalize_item(item: str) -> str:
    return " ".join(item.split()).casefold()


class RelationshipStore:
    """JSON-backed store for per-person relationship models."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            self._data = {}
            return
        try:
            parsed = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _default_model(person_id: str) -> dict[str, Any]:
        model = RelationshipModel(person_id=person_id, name=person_id)
        data = asdict(model)
        data["interaction_log"] = []
        return data

    def _get_raw(self, person_id: str) -> dict[str, Any]:
        return dict(self._data.get(person_id, self._default_model(person_id)))

    def raw(self, person_id: str) -> dict[str, Any]:
        return self._get_raw(person_id)

    def get(self, person_id: str) -> RelationshipModel:
        raw = self._get_raw(person_id)
        return RelationshipModel(
            person_id=person_id,
            name=str(raw.get("name", person_id)),
            known_facts=list(raw.get("known_facts", [])),
            communication_style=dict(raw.get("communication_style", {})),
            preferred_topics=list(raw.get("preferred_topics", [])),
            sensitive_topics=list(raw.get("sensitive_topics", [])),
            emotional_baseline=dict(raw.get("emotional_baseline", {})),
            trust_level=float(raw.get("trust_level", 0.5)),
            shared_episode_ids=list(raw.get("shared_episode_ids", [])),
            inferred_personality=dict(raw.get("inferred_personality", {})),
            recent_mood_trajectory=list(raw.get("recent_mood_trajectory", [])),
            first_interaction=str(raw.get("first_interaction", "")),
            last_interaction=str(raw.get("last_interaction", "")),
            total_interactions=int(raw.get("total_interactions", 0)),
            aliases=list(raw.get("aliases", [])),
            relation_kind=raw.get("relation_kind", "interlocutor"),
        )

    def update(self, person_id: str, patch: dict[str, Any]) -> RelationshipModel:
        invalid_fields = sorted(key for key in patch if key not in _UPDATABLE_FIELDS)
        if invalid_fields:
            valid_fields = ", ".join(sorted(_UPDATABLE_FIELDS))
            invalid = ", ".join(invalid_fields)
            raise ValueError(
                f"Invalid relationship field(s): {invalid}. "
                f"Valid fields: {valid_fields}"
            )
        for key, value in patch.items():
            expected = _FIELD_TYPES.get(key)
            if expected is not None and not isinstance(value, expected):
                raise TypeError(
                    f"Field '{key}' expects {expected.__name__ if isinstance(expected, type) else ' or '.join(t.__name__ for t in expected)}, "
                    f"got {type(value).__name__}: {value!r}"
                )
        raw = self._get_raw(person_id)
        for key, value in patch.items():
            raw[key] = value
        self._data[person_id] = raw
        self._save()
        return self.get(person_id)

    def _text_list(self, raw: dict[str, Any], field_name: str) -> list[str]:
        if field_name not in TEXT_LIST_FIELDS:
            raise ValueError(f"'{field_name}' is not a text list field")
        current = raw.get(field_name, [])
        if not isinstance(current, list):
            return []
        return [str(item) for item in current]

    def add_list_items(
        self, person_id: str, field_name: str, items: list[str]
    ) -> tuple[list[str], list[str]]:
        """Append *items* to a text list field, skipping duplicates.

        Duplicates are matched ignoring case and whitespace, both against the
        stored list and within *items*. Returns ``(added, already_present)``.
        """
        raw = self._get_raw(person_id)
        current = self._text_list(raw, field_name)
        seen = {_normalize_item(item) for item in current}
        added: list[str] = []
        already: list[str] = []
        for item in items:
            key = _normalize_item(item)
            if key in seen:
                already.append(item)
                continue
            seen.add(key)
            current.append(item)
            added.append(item)
        if added:
            raw[field_name] = current
            self._data[person_id] = raw
            self._save()
        return added, already

    def remove_list_items(
        self, person_id: str, field_name: str, items: list[str]
    ) -> tuple[list[str], list[str]]:
        """Remove *items* from a text list field.

        Matching ignores case and whitespace. Returns ``(removed, missing)``,
        where ``removed`` holds the stored wording of each removed entry.
        """
        raw = self._get_raw(person_id)
        current = self._text_list(raw, field_name)
        targets = {_normalize_item(item): item for item in items}
        removed = [item for item in current if _normalize_item(item) in targets]
        removed_keys = {_normalize_item(item) for item in removed}
        missing = [item for key, item in targets.items() if key not in removed_keys]
        if removed:
            raw[field_name] = [
                item for item in current if _normalize_item(item) not in removed_keys
            ]
            self._data[person_id] = raw
            self._save()
        return removed, missing

    def add_interaction(
        self, person_id: str, timestamp: str, tone: str
    ) -> RelationshipModel:
        raw = self._get_raw(person_id)
        interaction_log = raw.get("interaction_log", [])
        if not isinstance(interaction_log, list):
            interaction_log = []
        raw["total_interactions"] = int(raw.get("total_interactions", 0) or 0) + 1
        interaction_log.append({"timestamp": timestamp, "tone": tone})
        raw["interaction_log"] = interaction_log[-INTERACTION_LOG_MAX:]

        if not raw.get("first_interaction"):
            raw["first_interaction"] = timestamp
        raw["last_interaction"] = timestamp

        communication_style = raw.get("communication_style", {})
        if not isinstance(communication_style, dict):
            communication_style = {}
        communication_style[tone] = float(communication_style.get(tone, 0.0)) + 1.0
        raw["communication_style"] = communication_style

        trajectory = raw.get("recent_mood_trajectory", [])
        if not isinstance(trajectory, list):
            trajectory = []
        trajectory.append({"timestamp": timestamp, "mood": tone})
        raw["recent_mood_trajectory"] = trajectory[-20:]

        self._data[person_id] = raw
        self._save()
        return self.get(person_id)

    def set_reunion_note(
        self,
        person_id: str,
        *,
        gap_days: float,
        noted_at: str,
    ) -> None:
        raw = self._get_raw(person_id)
        raw["reunion_note"] = {
            "gap_days": gap_days,
            "noted_at": noted_at,
            "wake_up_shown": False,
        }
        self._data[person_id] = raw
        self._save()

    def mark_reunion_wake_up_shown(self, person_id: str) -> None:
        raw = self._get_raw(person_id)
        note = raw.get("reunion_note")
        if not isinstance(note, dict):
            return
        note["wake_up_shown"] = True
        raw["reunion_note"] = note
        self._data[person_id] = raw
        self._save()

    def add_shared_episode(self, person_id: str, episode_id: str) -> RelationshipModel:
        raw = self._get_raw(person_id)
        episode_ids = raw.get("shared_episode_ids", [])
        if not isinstance(episode_ids, list):
            episode_ids = []
        if episode_id not in episode_ids:
            episode_ids.append(episode_id)
        raw["shared_episode_ids"] = episode_ids
        self._data[person_id] = raw
        self._save()
        return self.get(person_id)

    def apply_tom_feedback(
        self,
        person_id: str,
        dominant_tone: str,
        preferred_topics: list[str],
        sensitive_topics: list[str],
    ) -> RelationshipModel:
        """Update model fields inferred from recent ToM analysis."""
        raw = self._get_raw(person_id)
        now = timezone_utils.now().isoformat()

        if dominant_tone and dominant_tone != "unknown tone":
            trajectory = raw.get("recent_mood_trajectory", [])
            if not isinstance(trajectory, list):
                trajectory = []
            trajectory.append({"timestamp": now, "mood": dominant_tone})
            raw["recent_mood_trajectory"] = trajectory[-20:]

        if preferred_topics:
            raw["preferred_topics"] = preferred_topics
        if sensitive_topics:
            raw["sensitive_topics"] = sensitive_topics

        self._data[person_id] = raw
        self._save()
        return self.get(person_id)

    def resolve_person(self, query: str) -> str | None:
        """Exact-match alias or canonical person_id resolution.

        Returns the canonical person_id if *query* matches the canonical id
        or any entry in the person's ``aliases`` list.  Returns ``None``
        when no match is found.
        """
        if query in self._data:
            return query
        for pid, raw in self._data.items():
            aliases = raw.get("aliases", [])
            if isinstance(aliases, list) and query in aliases:
                return pid
        return None
