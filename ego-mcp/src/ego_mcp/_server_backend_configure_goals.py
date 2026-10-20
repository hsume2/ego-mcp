"""Backend handler for the operator-only configure_goals tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ego_mcp.self_model import SelfModelStore


def _handle_configure_goals(path: str, args: dict[str, Any]) -> str:
    """Show or replace operator-set long-term goals."""
    store = SelfModelStore(Path(path))
    action = str(args.get("action", "")).strip()

    if action == "show":
        long_term = store.get_long_term_goals()
        active = [
            item for item in store.get_goal_log() if item["status"] == "active"
        ]
        long_term_by_id = {item["id"]: item for item in long_term}
        lines = ["Long-term goals:"]
        if long_term:
            lines.extend(f"  [{item['id']}] {item['goal']}" for item in long_term)
        else:
            lines.append("  none set")
        lines.append("Active short-term goals:")
        if active:
            for item in active:
                parent = item["parent"]
                if parent is None:
                    parent_text = "parentless"
                elif parent in long_term_by_id:
                    parent_text = f"toward [{parent}]"
                else:
                    parent_text = f"toward [{parent}] (retired)"
                lines.append(f"  [{item['id']}] {item['goal']} — {parent_text}")
        else:
            lines.append("  none")
        return "\n".join(lines)

    if action == "set_long_term":
        goals = args.get("goals")
        if not isinstance(goals, list):
            return "set_long_term requires goals (a list of {id?, goal} objects)."
        try:
            replaced = store.set_long_term_goals(goals)
        except ValueError as exc:
            return str(exc)
        if not replaced:
            return "Long-term goals cleared."
        lines = ["Long-term goals set:"]
        lines.extend(f"  [{item['id']}] {item['goal']}" for item in replaced)
        return "\n".join(lines)

    return f"Unknown action: {action}. Use show or set_long_term."
