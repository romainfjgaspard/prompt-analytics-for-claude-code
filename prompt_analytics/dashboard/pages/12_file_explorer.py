"""File Explorer: the per-file footprint table + a drill into each file.

Migrated from the Composition page's **Files** table (search + sort), now with a
row-select **drill**: pick a file to see which prompts edited it (and the task
each belongs to) and which sessions kept it in context, with its language / kind
and the load + rent it cost -- the file's whole cost of ownership.

Other pages deep-link here by setting ``drill_file`` in ``session_state`` (the
Prompt Explorer's prompt detail links the files a prompt edited), which
preselects that file's drill -- the same principle as the *Explore →* jump to the
Prompt Explorer.

Respects the global cross-filters. Table-first (``st.dataframe`` row selection);
the drill renders no ECharts, but ``main()`` still runs only under a real server
for parity with the other filter-driven pages.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st
from streamlit import runtime

from prompt_analytics import analytics
from prompt_analytics.context import NO_LANGUAGE
from prompt_analytics.dashboard import data, filters, tables

# Set by other pages (e.g. the Prompt Explorer's prompt detail) to preselect a
# file's drill on arrival -- the file analogue of ``drill_session``. ``DRILL_FILE``
# is the project-relative path; ``DRILL_FILE_PROJECT`` disambiguates which repo's
# copy (the same path recurs across projects), so the drill scopes to that project.
DRILL_FILE = "drill_file"
DRILL_FILE_PROJECT = "drill_file_project"

# The file table's row-selection widget key. Popped when leaving a drill so the
# sticky ``st.dataframe`` selection can't re-apply the just-cleared drill on the
# next rerun (also listed in ``filters._DRILL_KEYS`` for the global Reset).
_KEY_FILES_TABLE = "fe_files"

# Raw footprint keys -> display labels for the main table.
_FILE_HEADERS = {
    "project": "Project",
    "path": "File",
    "language": "Language",
    "kind": "Kind",
    "edits": "Edits",
    "lines_added": "Lines +",
    "lines_deleted": "Lines −",
    "reads": "Reads",
    "load_usd": "Load $",
    "rent_usd": "Rent $",
    "context_usd": "Context $",
}


# Magnitude columns that carry an in-cell bar: counts in blue, costs in coral.
_FILE_COUNT_COLS = ("Edits", "Lines +", "Lines −", "Reads")
_FILE_COST_COLS = ("Context $", "Load $", "Rent $")


def _edited_by(ds: analytics.Dataset, path: str, prompts: pd.DataFrame) -> pd.DataFrame:
    """The prompts (and their task) that edited ``path``, with edit/line counts."""
    agg: dict[str, dict[str, Any]] = {}
    for row in ds.output_files:
        if (row.get("path") or "") != path:
            continue
        pid = str(row.get("prompt_id") or "")
        entry = agg.setdefault(
            pid, {"prompt_id": pid, "edits": 0, "lines_added": 0, "lines_deleted": 0}
        )
        entry["edits"] += int(row.get("edits") or 0)
        entry["lines_added"] += int(row.get("lines_added") or 0)
        entry["lines_deleted"] += int(row.get("lines_deleted") or 0)
    if not agg:
        return pd.DataFrame()

    out = pd.DataFrame(list(agg.values()))
    task_of = {str(tp.get("prompt_id")): str(tp.get("task_id")) for tp in ds.task_prompts}
    task_name = {str(t.get("task_id")): str(t.get("name") or t.get("task_id")) for t in ds.tasks}
    out["task"] = out["prompt_id"].map(lambda p: task_name.get(task_of.get(p, ""), ""))
    if not prompts.empty:
        cols = [
            c
            for c in ("prompt_id", "session_id", "prompt_index", "category")
            if c in prompts.columns
        ]
        out = out.merge(prompts[cols].drop_duplicates("prompt_id"), on="prompt_id", how="left")
    sort_cols = [c for c in ("session_id", "prompt_index") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def _read_in(
    ds: analytics.Dataset, provider: str, path: str, sessions: pd.DataFrame
) -> pd.DataFrame:
    """The sessions that kept ``path`` in context: reads, tokens, and context cost.

    Context rows are per-session (the cache is a session-level resource), so the
    read side is summarized per session, not per prompt -- honest about the grain.
    """
    engine = analytics.CostEngine(provider, ds.pricing_path)
    reads: dict[str, dict[str, int]] = {}
    for row in ds.context_sources:
        if (row.get("source") or "") != "file_read" or (row.get("path") or "") != path:
            continue
        sid = str(row.get("session_id") or "")
        entry = reads.setdefault(sid, {"reads": 0, "tokens": 0})
        entry["reads"] += int(row.get("items") or 0)
        entry["tokens"] += int(row.get("tokens") or 0)

    cost: dict[str, float] = {}
    for row in ds.context_cost:
        if (row.get("source") or "") != "file_read" or (row.get("path") or "") != path:
            continue
        sid = str(row.get("session_id") or "")
        model = str(row.get("model") or "")
        cost[sid] = cost.get(sid, 0.0) + (
            engine.cost(model, "cache_read", int(row.get("rent_read_tokens") or 0))
            + engine.cost(model, "cache_write_5m", int(row.get("load_write_5m_tokens") or 0))
            + engine.cost(model, "cache_write_1h", int(row.get("load_write_1h_tokens") or 0))
        )

    sids = set(reads) | set(cost)
    if not sids:
        return pd.DataFrame()
    rows = [
        {
            "session_id": sid,
            "reads": reads.get(sid, {}).get("reads", 0),
            "tokens": reads.get(sid, {}).get("tokens", 0),
            "context_usd": round(cost.get(sid, 0.0), 4),
        }
        for sid in sids
    ]
    out = pd.DataFrame(rows)
    if not sessions.empty and {"session_id", "project"} <= set(sessions.columns):
        out = out.merge(
            sessions[["session_id", "project"]].drop_duplicates("session_id"),
            on="session_id",
            how="left",
        )
    return out.sort_values("context_usd", ascending=False).reset_index(drop=True)


def _render_file_drill(
    ds: analytics.Dataset,
    provider: str,
    raw: pd.DataFrame,
    path: str,
    prompts: pd.DataFrame,
    sessions: pd.DataFrame,
) -> None:
    """The per-file panel: header metrics + who edited it + who kept it in context."""
    head = raw.loc[raw["path"] == path]
    if head.empty:
        return
    rec = head.iloc[0]

    top = st.columns([5, 1])
    language = str(rec.get("language") or NO_LANGUAGE)
    lang_txt = language if language != NO_LANGUAGE else "—"
    top[0].subheader(f"📄 {path}")
    top[0].caption(f"{lang_txt} · {rec.get('kind', '')}")
    if top[1].button("← All files", width="stretch", key="clear_drill_file"):
        # Pop the table selection too: otherwise its sticky row re-applies the same
        # file drill on the next rerun (the "← back doesn't return to the list" bug).
        for k in (DRILL_FILE, DRILL_FILE_PROJECT, _KEY_FILES_TABLE):
            st.session_state.pop(k, None)
        st.rerun()

    cols = st.columns(4)
    cols[0].metric("Edits", f"{int(rec.get('edits', 0)):,}")
    cols[1].metric(
        "Lines",
        f"+{int(rec.get('lines_added', 0)):,}",
        delta=f"−{int(rec.get('lines_deleted', 0)):,} removed",
        delta_color="off",
    )
    cols[2].metric("Reads", f"{int(rec.get('reads', 0)):,}")
    cols[3].metric(
        f"Context cost ({provider})",
        f"${float(rec.get('context_usd', 0.0)):,.2f}",
        delta=f"load ${float(rec.get('load_usd', 0.0)):,.2f} · rent ${float(rec.get('rent_usd', 0.0)):,.2f}",
        delta_color="off",
    )

    st.markdown("**Edited by these prompts**")
    edited = _edited_by(ds, path, prompts)
    if edited.empty:
        st.caption("Never edited — this file is pure context cost (read but not written).")
    else:
        view = edited.rename(
            columns={
                "session_id": "Session",
                "prompt_index": "#",
                "category": "Category",
                "task": "Task",
                "edits": "Edits",
                "lines_added": "Lines +",
                "lines_deleted": "Lines −",
            }
        )
        order = [
            c
            for c in ("Session", "#", "Category", "Task", "Edits", "Lines +", "Lines −")
            if c in view.columns
        ]
        st.dataframe(
            tables.bar_table(view[order], count_cols=("Edits", "Lines +", "Lines −")),
            width="stretch",
            hide_index=True,
        )
        st.caption(f"{len(edited):,} prompt(s) edited this file.")

    st.markdown("**Kept in context in these sessions**")
    read_in = _read_in(ds, provider, path, sessions)
    if read_in.empty:
        st.caption("Never read into context — this file's cost is its edits only.")
    else:
        view = read_in.rename(
            columns={
                "session_id": "Session",
                "project": "Project",
                "reads": "Reads",
                "tokens": "Tokens",
                "context_usd": "Context $",
            }
        )
        order = [
            c for c in ("Session", "Project", "Reads", "Tokens", "Context $") if c in view.columns
        ]
        st.dataframe(
            tables.bar_table(view[order], count_cols=("Reads", "Tokens"), cost_cols=("Context $",)),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"{len(read_in):,} session(s) kept this file in context "
            "(cache reads are a session-level resource, so this side is per session)."
        )


def main() -> None:
    """Render the File Explorer page."""
    st.title("File Explorer")

    frames_all = data.load_all()
    filters.render_sidebar(frames_all)
    frames = filters.apply_filters(frames_all)
    filters.render_active_filter_badge(frames_all, explore_link=False)
    prompts = frames.get("prompts", pd.DataFrame())
    sessions = frames.get("sessions", pd.DataFrame())
    primary = data.primary_provider()

    if prompts.empty or "prompt_id" not in prompts.columns:
        st.info("No data for the current filters.")
        st.stop()

    # The composition CSVs are not part of the dashboard frames; load the raw
    # dataset and narrow it to the same prompts the global filters kept (this also
    # narrows the session-grained context rows to those sessions).
    kept_ids = set(prompts["prompt_id"])
    ds = analytics.filter_prompt_ids(data.load_dataset(), kept_ids)
    # by_project keys each row on (project, path): the same relative path recurs
    # across repos, and without the split a shared name (README.md) would merge
    # every repo's copy into one misleading line.
    table = analytics.file_footprint(ds, primary, by_project=True)
    if not table.rows:
        st.info(
            "No per-file data yet. The file identity (relative path) ships with the latest "
            "extractor — re-run `prompt-analytics extract`, then revisit this page."
        )
        st.stop()

    st.caption(
        "One row per file: **edits** + line diff (output) crossed with **reads** + the "
        "**context cost** they drove (loading + rent) — the file's whole cost of ownership. "
        "It reflects the dashboard's active filters; filter the columns, sort, then select a "
        "row to drill into a file. Metrics only — relative paths, never content."
    )

    raw = pd.DataFrame(table.rows)
    df = raw.rename(columns=_FILE_HEADERS)

    # Per-column filters above the table (offline, no grid dependency): a text
    # search plus a dropdown for each categorical column.
    controls = st.columns([3, 2, 2, 2, 2])
    query = controls[0].text_input(
        "Filter files", key="fe_file_filter", placeholder="path…", label_visibility="collapsed"
    )
    projects = ["All projects", *sorted(p for p in df["Project"].dropna().unique() if p)]
    project = controls[1].selectbox(
        "Project", projects, key="fe_project", label_visibility="collapsed"
    )
    languages = ["All languages", *sorted(x for x in df["Language"].dropna().unique() if x)]
    language = controls[2].selectbox(
        "Language", languages, key="fe_language", label_visibility="collapsed"
    )
    kinds = ["All kinds", *sorted(x for x in df["Kind"].dropna().unique() if x)]
    kind = controls[3].selectbox("Kind", kinds, key="fe_kind", label_visibility="collapsed")
    scope = controls[4].selectbox(
        "Footprint",
        ["All files", "Edited", "Read-only"],
        key="fe_file_scope",
        label_visibility="collapsed",
    )

    view = df
    if query:
        q = query.strip().lower()
        view = view[view["File"].str.lower().str.contains(q, regex=False)]
    if project != "All projects":
        view = view[view["Project"] == project]
    if language != "All languages":
        view = view[view["Language"] == language]
    if kind != "All kinds":
        view = view[view["Kind"] == kind]
    if scope == "Edited":
        view = view[view["Edits"] > 0]
    elif scope == "Read-only":
        view = view[(view["Edits"] == 0) & (view["Reads"] > 0)]
    view = view.reset_index(drop=True)

    event = st.dataframe(
        tables.bar_table(view, count_cols=_FILE_COUNT_COLS, cost_cols=_FILE_COST_COLS),
        width="stretch",
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={"File": st.column_config.TextColumn("File", width="large")},
        key=_KEY_FILES_TABLE,
    )
    rows = list(event.get("selection", {}).get("rows", []))
    if rows and rows[0] < len(view):
        picked = str(view.iloc[rows[0]]["File"])
        picked_project = str(view.iloc[rows[0]].get("Project", "") or "")
        if (picked, picked_project) != (
            st.session_state.get(DRILL_FILE),
            st.session_state.get(DRILL_FILE_PROJECT),
        ):
            st.session_state[DRILL_FILE] = picked
            st.session_state[DRILL_FILE_PROJECT] = picked_project
            st.rerun()

    edited = int((df["Edits"] > 0).sum())
    read_only = int(((df["Edits"] == 0) & (df["Reads"] > 0)).sum())
    st.caption(
        f"{len(view):,} of {len(df):,} files shown — {edited:,} edited, **{read_only:,} read "
        "but never edited** (pure context cost, the first candidates to keep out of context)."
    )

    # --- File drill --------------------------------------------------------
    focus = st.session_state.get(DRILL_FILE)
    focus_project = st.session_state.get(DRILL_FILE_PROJECT) or ""
    # Scope the drill to the selected file's project so its edits/reads aren't
    # re-merged across repos (the table is already split per project).
    scoped = analytics.filter_project(ds, focus_project) if focus_project else ds
    raw_focus = raw[raw["path"] == focus]
    if focus_project and "project" in raw.columns:
        raw_focus = raw_focus[raw_focus["project"] == focus_project]
    if focus and not raw_focus.empty:
        st.divider()
        _render_file_drill(
            scoped, primary, raw_focus.reset_index(drop=True), str(focus), prompts, sessions
        )
    elif focus:
        # The deep-linked / previously selected file fell out of the current
        # filters (or its project): say so, then drop the stale focus so the next
        # interaction is clean -- better than a silent no-op or an empty panel.
        st.divider()
        st.info(
            f"📄 **{focus}** isn't in the current selection — the active filters "
            "narrowed it out. Clear or widen the filters, or pick another file above."
        )
        st.session_state.pop(DRILL_FILE, None)
        st.session_state.pop(DRILL_FILE_PROJECT, None)


# Render only under a real Streamlit server (parity with the other filter-driven
# pages); excluded from the headless AppTest enumeration in tests/test_dashboard.py.
if runtime.exists():
    main()
