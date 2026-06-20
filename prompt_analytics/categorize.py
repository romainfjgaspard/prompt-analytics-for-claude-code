"""LLM categorization and complexity scoring of Claude Code prompts.

Default (no API key needed): heuristic regex classifier (FR + EN) → category,
with observed complexity from real effort metrics (quantile bands).

LLM mode (``--llm``): Anthropic, OpenRouter, or Ollama, with exponential
backoff, Retry-After on 429, transient/permanent error distinction, atomic
checkpoints, and Anthropic Message Batches (``--batch``, −50% cost).
"""

from __future__ import annotations

import csv
import importlib.resources
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import yaml

if TYPE_CHECKING:
    from anthropic import Anthropic as _AnthropicClient
    from openai import OpenAI as _OpenAIClient

    from .embeddings import Embedder, FloatMatrix

from .schema import CATEGORIES_COLS
from .storage import atomic_write_csv

__all__ = [
    "run_categorize",
    "build_client",
    "SYSTEM_PROMPT",
    "HEURISTIC_VERSION",
    "SEMANTIC_VERSION",
    "SemanticClassifier",
    "load_anchors",
    "build_system_prompt",
]

# ── constants ─────────────────────────────────────────────────────────────────

ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENROUTER_MODEL = "anthropic/claude-haiku-4.5"
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Azure OpenAI: the deployment name *is* the model id, so there is no default
# here -- it is read from AZURE_OPENAI_DEPLOYMENT (or --model). The API version
# is a sane recent default, overridable via AZURE_OPENAI_API_VERSION.
AZURE_API_VERSION = "2024-12-01-preview"

MAX_PROMPT_CHARS = 2000
MAX_TOKENS = 16
# Azure deployments are frequently reasoning models (gpt-5-class) that spend
# completion tokens on hidden reasoning before the visible reply, so the
# 16-token cap above would starve the "category|complexity" answer. Give the
# Azure path a generous completion budget (only tokens actually produced are
# billed; the visible reply stays tiny) and request it via
# ``max_completion_tokens`` (reasoning models reject ``max_tokens``).
AZURE_MAX_COMPLETION_TOKENS = 1024

# Bound the batch poll (3.5): Anthropic Message Batches are documented to finish
# within 24h, so a `while True` that never ends means something is wrong. Poll
# every 5s and give up with a clear error past the deadline.
BATCH_POLL_INTERVAL = 5.0
BATCH_POLL_TIMEOUT = 24 * 60 * 60

CATEGORIES = {
    "plan",
    "implementation",
    "debug",
    "refactor",
    "review",
    "test",
    "docs",
    "ops",
    "question",
    "followup",
    "feedback",
    "notification",
    "other",
}
COMPLEXITIES = {"1", "2", "3", "4", "5"}

# Version stamp written to ``classifier_model`` by the heuristic classifier.
# Bumping it makes the next heuristic run re-classify rows stamped with an
# older heuristic version (LLM-classified rows are never touched).
HEURISTIC_VERSION = "heuristic-v3"
_HEURISTIC_PREFIX = "heuristic-"

# Tie-break order for the heuristic classifier: on equal scores the more
# specific intent wins (a prompt that both "fixes" and mentions a "plan" is a
# debug prompt). A fixed tuple keeps classification deterministic across
# processes -- iterating over the CATEGORIES set would depend on the hash seed.
_TIE_BREAK_ORDER = (
    "debug",
    "docs",
    "test",
    "review",
    "refactor",
    "ops",
    "plan",
    "implementation",
    "question",
    "followup",
    # feedback sits last before "other": it is low-weight discourse/steering, so
    # on an equal score any concrete intent above (incl. question and the
    # stuck-assistant followup) wins -- feedback only takes a prompt when nothing
    # more specific fired.
    "feedback",
    "other",
)

# ── shared category source of truth (semantic_anchors.yml) ─────────────────────

# The anchors file is the single, editable source of truth shared by both
# classifiers (see prompt_analytics/data/semantic_anchors.yml): the LLM
# SYSTEM_PROMPT renders its definitions, the semantic classifier embeds the
# `semantic`-role examples as prototypes. Keeping them in one file is what stops
# the two modes from quietly describing the same label differently.

# Roles a category can play in the offline semantic classifier.
_ROLE_SEMANTIC = "semantic"
_ROLE_LEXICAL = "lexical"
_ROLE_SHORTCUT = "shortcut"
_ROLE_FALLBACK = "fallback"
_VALID_ROLES = {_ROLE_SEMANTIC, _ROLE_LEXICAL, _ROLE_SHORTCUT, _ROLE_FALLBACK}

# Categories whose lexical prime competes for the single label. They reuse the
# heuristic regex (high-precision git verbs, discourse markers) rather than a
# prototype, so a prompt like "implement X then commit" still wins on its
# dominant intent instead of being short-circuited to ops.
_LEXICAL_PRIME_CATEGORIES = ("ops", "feedback")

# The LLM-only tail of the prompt: the complexity scale and a few illustrative
# mappings. (Complexity is rated from observed effort, never from the LLM, so
# these stay curated here rather than in the anchors file -- the divergence-prone
# part, the category *definitions*, is what we render from the shared YAML.)
_COMPLEXITY_SCALE = """\
Complexity (1-5):
1 = trivial one-liner or yes/no ("yes", "ok", "continue")
2 = simple, single-step request
3 = moderate, requires reading a few files or reasoning
4 = complex, multi-step or cross-cutting task
5 = very complex, architectural change or deep investigation"""

_FEWSHOT_EXAMPLES = """\
Examples:
"yes" -> followup|1
"fix the null pointer in user.py" -> debug|2
"commit and push the branch, then open a PR" -> ops|2
"refactor the auth module to use the new token format" -> refactor|3
"no, rather put the legend on the right, it overlaps the bars" -> feedback|2
"analyze the project and propose an architecture for the distributed cache" -> plan|5"""

_anchors_cache: dict[str, dict[str, Any]] | None = None


def _anchors_path() -> Path:
    """Path to the bundled ``semantic_anchors.yml`` (via importlib.resources)."""
    ref = importlib.resources.files("prompt_analytics.data") / "semantic_anchors.yml"
    return Path(str(ref))


def load_anchors(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Load (and validate) the shared category anchors, keyed by category name.

    The bundled file is parsed once and cached; pass an explicit ``path`` to
    load a custom file (tests, power users) without touching the cache. Every
    category must be a known one with a valid role, and each ``semantic``
    category must carry at least one prototype example -- a malformed anchors
    file is a packaging/editing bug, so it fails loudly here rather than
    silently producing a classifier with no prototypes.
    """
    global _anchors_cache
    if path is None and _anchors_cache is not None:
        return _anchors_cache

    src = Path(path) if path is not None else _anchors_path()
    with src.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    categories = (data or {}).get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError(f"{src}: missing or empty 'categories' mapping")

    for name, spec in categories.items():
        if name not in CATEGORIES:
            raise ValueError(f"{src}: unknown category {name!r} (not in the taxonomy)")
        role = spec.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"{src}: category {name!r} has invalid role {role!r}")
        if not spec.get("definition"):
            raise ValueError(f"{src}: category {name!r} has no definition")
        if role == _ROLE_SEMANTIC and not spec.get("examples"):
            raise ValueError(f"{src}: semantic category {name!r} has no prototype examples")

    if path is None:
        _anchors_cache = categories
    return categories


def build_system_prompt(anchors: dict[str, dict[str, Any]] | None = None) -> str:
    """Render the LLM SYSTEM_PROMPT from the shared anchors (definitions block).

    The category list -- the part most prone to drifting from the offline
    classifier -- is generated from the same file the semantic prototypes come
    from. The complexity scale and few-shot examples are appended verbatim.
    """
    anchors = anchors if anchors is not None else load_anchors()
    # Collapse any folded/multi-line YAML definition into one prompt line.
    cat_lines = [
        f"- {name}: {' '.join(str(spec['definition']).split())}"
        for name, spec in anchors.items()
    ]
    return "\n".join(
        [
            "You classify developer prompts sent to an AI coding assistant.",
            'Reply with exactly "category|complexity" -- two values separated by a '
            "pipe, nothing else.",
            "",
            "Category (pick one):",
            *cat_lines,
            "",
            _COMPLEXITY_SCALE,
            "",
            _FEWSHOT_EXAMPLES,
        ]
    )


# Built once at import from the shared anchors so both modes stay in lock-step.
SYSTEM_PROMPT = build_system_prompt()


# ── heuristic classifier ──────────────────────────────────────────────────────

# v2 (4.1, audit 06 D3): rules tuned on the real "other" sample. Real prompts
# are mostly unaccented French ("probleme", "j'ai relancé mais ca marche pas"),
# so every FR pattern tolerates missing accents; the big agentic clusters the
# v1 rules missed get their own categories (review, test, docs, ops, followup).

# (category, regex_patterns, weight_per_match)
_HEURISTIC_RULES: list[tuple[str, list[str], float]] = [
    (
        "plan",
        [
            r"\barchitect(?:ure|er|ons|ural)\b",
            r"\bconcepti(?:on|ons)\b",
            r"\bpropose[rz]?\b",
            r"\bstrat[eé]gi(?:e|ques?)\b",
            r"\bcomment (?:structurer|organiser)\b",
            r"\borganis(?:er|ation)\b",
            r"\bplan(?:ning|ifier)?\b",
            r"\bdesign(?:er|ing)?\b",
            r"\bstrategy\b",
            r"\bapproach\b",
            r"\bhow should (?:i|we|the)\b",
            r"\bwhat(?:'s| is) the best way\b",
        ],
        1.2,
    ),
    (
        "implementation",
        [
            r"\bimpl[eé]ment(?:er|e|es|ez|ons|ation)\b",
            r"\bcr[eé](?:er|e|es|ez|ons|é|ée)\b",
            r"\b[eé]cri(?:re|s|t|vez|vons)\b",
            r"\b(?:r)?ajout(?:er|e|es|ez|ons)\b",
            r"\bd[eé]velopp(?:er|e|es|ez|ons)\b",
            r"\bg[eé]n[eè]r(?:e|es|ent)\b|\bg[eé]n[eé]r(?:er|ez|ons|ation)\b",
            r"\bint[eè]gr(?:e|er|es|ez)\b",
            r"\bmets? en place\b",
            r"\br[eé]alise[rz]?\b",
            r"\bfai(?:s|tes|t)[- ]moi\b",
            r"\bimplement(?:ing|ation)?\b",
            r"\bcreate\b",
            r"\bwrite\b",
            r"\badd\b",
            r"\bdevelop\b",
            r"\bgenerate\b",
            r"\bbuild\b",
        ],
        0.8,
    ),
    (
        "debug",
        [
            r"\berreurs?\b",
            r"\bcorrig(?:er|e|es|ez|eons)\b",
            r"\br[eé]sou(?:dre|s|t)\b",
            r"\bprobl[eè]m(?:e|es|atiques?)\b",
            r"\bpourquoi (?:ça |ca |cela )?(?:ne )?(?:marche|fonctionne)\b",
            r"\b(?:ne |n')?(?:marche|fonctionne) (?:pas|plus)\b",
            r"\bpas march[eé]\b",
            r"\bplant(?:e|es|ent|é|ée|er|ait)\b",
            r"\bsoucis?\b",
            r"\bcass(?:e|é|ée|és|ées)\b",
            r"\bcoquilles?\b",
            r"\bbizarre\b",
            r"\bpas normal\b",
            r"\bpas ok\b",
            r"\bpire\b",
            r"\b[eé]checs?\b",
            r"\binvestigu(?:e|er|es|ez)\b",
            r"\bcreuse[rz]?\b",
            r"\btraceback\b",
            r"\bexception\b",
            r"\bfix(?:ing)?\b",
            r"\bbug\b",
            r"\b\w*error\b",
            r"\bdebug(?:ging)?\b",
            r"\bfail(?:s|ed|ing|ure|ures)?\b",
            r"\bcrash(?:ing)?\b",
            r"\bwhy (?:is|does|isn'?t|doesn'?t)\b",
            r"\bnot working\b",
            r"\bdoesn'?t work\b",
            r"\bbroken\b",
            r"\bunable to\b",
            r"\binvalid\w*\b",
            r"\broot cause\b",
            r"\binvestigate\b",
        ],
        1.2,
    ),
    (
        "refactor",
        [
            r"\brefactoris(?:er?|es|ez|ation)\b",
            r"\bam[eé]liore[rsz]?\b",
            r"\bnetto(?:yer|yage|ie|ies|iez)\b",
            r"\bm[eé]nage\b",
            r"\brestructure[rsz]?\b",
            r"\brenomm(?:er|e|es|ez|age)\b",
            r"\bsimplifi(?:er|e|es|ez|cation)\b",
            r"\boptimis(?:er|e|es|ez|ation)\b",
            r"\brefactor(?:ing)?\b",
            r"\bimprove(?:ment)?\b",
            r"\bclean(?:up)?\b",
            r"\brestructure\b",
            r"\brename\b",
            r"\bsimplify\b",
            r"\boptimize\b",
        ],
        1.0,
    ),
    (
        "review",
        [
            r"\breview(?:s|ed|ing|er)?\b",
            r"\baudit(?:s|er|e|ez)?\b",
            r"\banalys(?:e|er|es|ez|ons|is|ée?s?)\b",
            r"\banalyz(?:e|ed|ing)\b",
            r"\bv[eé]rifi(?:e|er|es|ez|ons|cation|é|ée)s?\b",
            r"\bverify\b|\bverification\b",
            r"\bcheck(?:s|ed|ing)?\b",
            r"\brelis(?:ez)?\b|\brelire\b",
            r"\bexamin(?:e|er|es|ez|ing)\b",
            r"\binspect(?:e|er|ez|ing)?\b",
            r"\bpasse en revue\b",
            r"\bregarde[rsz]?\b|\bregardons\b",
            r"\b[eé]tudi(?:e|er|es|ez)\b",
            r"\bcontr[oô]le[rz]?\b",
            r"\bfai(?:s|re) (?:le|un) point\b",
            r"\btake a look\b|\blook (?:at|into)\b",
        ],
        1.1,
    ),
    (
        "test",
        [
            r"\btests?\b",
            r"\bteste[rz]?\b|\btest(?:ing|ed)\b",
            r"\bpytest\b|\bunittest\b|\bplaywright\b",
            r"\bcouverture\b|\bcoverage\b",
            r"\br[eé]gressions?\b",
            r"\bfixtures?\b",
            r"\bla ci\b|\bci (?:passe|verte?)\b",
        ],
        # A hair above debug: "ajoute un cas de test pour le fix" is test work
        # even when it cites the fix. Real failure reports out-vote it anyway
        # (they carry several debug words: erreur, plante, traceback...).
        1.25,
    ),
    (
        "docs",
        [
            r"\breadme\b",
            r"\bchangelog\b",
            r"\bdocstrings?\b",
            r"\bdocumentation\b",
            # Article-bound on purpose: "ajoute ça dans la doc" is docs work,
            # but a bare "docs" would fire on every cited path (docs/PLAN.md)
            # in long agentic prompts whose real ask is something else.
            r"\b(?:la |le |une |un |cette |ce |dans (?:la |le |une |un )?)docs?\b",
            r"\bmarkdown\b",
            r"\bcommentaires?\b",
            r"\bfichiers? (?:de )?statu(?:s|ts?)\b",
            r"\bstatus files?\b",
            r"\bmets? (?:bien |aussi )?[aà] jour (?:le|la|les) (?:status|statuts?|fichiers?|docs?|notes?)\b",
            r"\bsauvegarde (?:le|la|les|cette) (?:status|statuts?|fichiers?|notes?|proc[eé]dures?)\b",
        ],
        1.1,
    ),
    (
        "ops",
        [
            r"\bcommit\w*\b",
            r"\bpush\w*\b",
            r"\bpull request\b|\bpull\b|\bprs?\b",
            r"\bmerge\w*\b|\bmerg[eé]e?s?\b",
            r"\bbranch(?:e|es)?\b",
            r"\brebase\w*\b",
            r"\bgit\b",
            r"\bclon(?:e|er|es|é|ée|ed|ing)\b",
            # Past-tense "j'ai déployé" is the user narrating, not an ask.
            r"\bd[eé]ploi(?:e|er|es|ez|ement)s?\b|\bdeploy(?:s|ed|ing|ment)?\b"
            r"|(?<!j'ai )(?<!j'avais )\bd[eé]ploy[eé]e?s?\b",
            # "lance le script" is an ops ask; "j'ai (re)lancé le script" is
            # the user narrating what they already did -- skip those.
            r"(?<!j'ai )(?<!j'avais )\b(?:re)?lanc(?:e|er|es|ez|é|ée)\b[^\n.!?]{0,40}"
            r"\b(?:script|run|workflow|pipeline|job|campagne|notebook|commande|build)\b",
            r"\bpubli(?:e|er|es|ez|é|ée|cation)\b|\bpublish(?:ed|ing)?\b",
            r"\brelease\b",
            r"\binstall(?:e[rz]?|es|ation|ed|ing)?\b",
            r"\bex[eé]cut(?:e|er|es|ez|ion)\b",
        ],
        1.0,
    ),
    (
        "question",
        [
            r"\bqu'est[- ]ce que\b",
            r"\bcomment\b",
            r"\bpourquoi\b",
            r"\bc'est quoi\b|\bc'est quel(?:le)?s?\b",
            r"\b[aà] quoi (?:ça |ca )?sert\b",
            r"\best[- ]ce qu(?:e|')",
            r"\bje (?:ne )?comprends? pas\b",
            r"\bqu'en penses?[- ]tu\b",
            r"\bdis[- ]m'en plus\b",
            r"\bil est o[uù]\b|\bo[uù] (?:est|sont)\b",
            r"\bquel(?:le)?s? (?:est|sont)\b",
            r"\bexpliqu(?:er|e|es|ez)\b",
            r"\bcompren(?:dre|ds?)\b",
            r"\bd[eé]cri(?:re?|s|t|vent|vons|vez)\b",
            r"\bque fait\b",
            r"\bdiff[eé]rences?\b",
            r"\bwhat (?:is|are|does)\b|\bwhat's\b",
            r"\bhow (?:do|can|does|to)\b",
            r"\bexplain\b",
            r"\bunderstand\b",
            r"\bdescribe\b",
            r"\btell me\b",
            r"\bdifference between\b",
            r"\bcan you (?:explain|describe|tell)\b",
        ],
        1.0,
    ),
    (
        "followup",
        [
            # Nudges at a stalled assistant ("tu es bloqué?", "tu t'es encore
            # figé") and restart orders. Weighted above debug so the "planté"
            # of a stuck *assistant* does not read as a code crash. Accent-
            # tolerant ("tu etais bloque" is the common unaccented form).
            r"\btu (?:t'es|es|[eé]tais|[eé]tait|avais l'air|as l'air)\b[^.?!\n]{0,24}"
            r"(?:bloqu|fig[eé]|plant[eé]|gel[eé])",
            r"\bencore (?:bloqu[eé]|fig[eé]|plant[eé])",
            r"^\s*reprend(?:s|re)?\b",
            r"^\s*recommence[rsz]?\b",
            r"^\s*continue[rsz]?\b",
        ],
        1.5,
    ),
    (
        # Reacting to / steering the assistant's work without a fresh task:
        # critique, course-correction, preferences. Low weight on purpose -- a
        # prompt that also carries a concrete intent (debug/test/impl...) scores
        # higher and wins; feedback only takes the prompts that would otherwise
        # fall through to "other" (the bulk of that bucket on real usage). The
        # markers are accent-tolerant unaccented French + EN equivalents.
        "feedback",
        [
            # Course-correction / discourse markers heading a steering turn.
            r"\ben fait\b",
            r"\bplut[oô]t\b",
            r"\bpar contre\b",
            r"\bdu coup\b",
            r"\bfinalement\b",
            r"\b[aà] la place\b",
            r"\ben revanche\b",
            # The user weighing in / preferences (not asking, not tasking).
            r"\bje pense\b",
            r"\b[aà] mon avis\b",
            r"\bje trouve\b",
            r"\bselon moi\b",
            r"\bje pr[eé]f[eè]re\b|\bj'aurais pr[eé]f[eé]r[eé]\b",
            r"\bj'ai chang[eé] d'avis\b",
            r"\b(?:vaudrait|vaut) mieux\b|\bce serait mieux\b|\bmieux vaut\b",
            r"\bpas convaincu\b",
            r"\bj'?aime pas\b|\bje n'aime pas\b",
            # Caveated approval / critique of the assistant's output.
            r"\bc'?est pas mal\b|\bpas mal mais\b",
            r"\btout est (?:bon|ok) (?:sauf|mais)\b",
            r"\b[cç]a (?:me )?(?:semble|para[iî]t) (?:bien|bon|ok)\b",
            # "the change didn't take" -- feedback on a non-effect.
            r"\b[cç]a (?:ne )?change rien\b",
            r"\btoujours pareil\b",
            r"\bm[eê]me (?:probl[eè]me|chose|souci|qu'avant)\b",
            r"\bpas de changement\b",
            r"\brien (?:n'a |a )?chang[eé]\b",
            # EN equivalents.
            r"\binstead\b",
            r"\bactually\b",
            r"\bon second thought\b",
            r"\bi (?:think|prefer|feel)\b|\bi'd (?:prefer|rather)\b",
            r"\bchanged my mind\b",
            r"\bnot convinced\b",
            r"\b(?:looks|sounds) (?:good|fine) but\b|\bnot bad but\b",
            r"\bstill the same\b|\bno change\b",
            r"\balmost (?:there|right)\b|\bnot quite\b",
        ],
        0.5,
    ),
]

# Short pure-acknowledgement prompts ("oui", "ok go", "les deux", "1 et 3").
# The whole text must be made of steering tokens -- "ok sauvegarde le fichier"
# must NOT short-circuit here, the real instruction wins.
_ACK_TOKEN = (
    r"(?:oui|non|ok(?:ay)?|yes|no|y|n|si|go|continue[rsz]?|vas[- ]?y|allez|"
    r"reprends?|reprendre|recommence[rsz]?|stop|attends?|wait|merci|thanks?|"
    r"thank you|parfait|nickel|super|top|g[eé]nial|d'?accord|les deux|both|"
    r"c'?est (?:bon|ok|parti|fait|good)|[çc]a marche|fais[- ]le|fais|fait|do it|"
    r"done|good|great|perfect|proceed|sounds good|lgtm|approved|je valides?|"
    r"on continue|next|suite|la suite|[eé]tape suivante|valid[eé]s?|"
    # Short option picks: "ok pour A", "ok pour l'option 1", "oui partons sur
    # l'etape 2", "M7". A bare letter / letter-number ref counts only inside an
    # otherwise all-ack short string (the whole-text match guard upstream).
    r"option|l'?option|l'?[eé]tapes?|[eé]tapes?|phase|partons|sur|"
    r"pour|et|[a-z]\d+|[a-z]|\d+[a-z]?)"
)
_ACK_RE = re.compile(
    rf"^\s*{_ACK_TOKEN}(?:[\s,.!?;:…\-]+{_ACK_TOKEN})*[\s,.!?;:…\-]*$",
    re.IGNORECASE,
)
_FOLLOWUP_MAX_CHARS = 40

# Harness chrome the transcript injects into user turns: a ``<system-reminder>``
# prefix (date stamps, context notes) or a whole ``<task-notification>`` block.
# Stripped before classifying so the *real* instruction after a reminder is
# scored on its own words (and a prompt that is nothing but chrome falls through
# to "other" instead of being mis-scored on the tag soup).
_NOISE_WRAPPER_RE = re.compile(
    r"<(system-reminder|task-notification)\b[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)
_TASK_NOTIFICATION_RE = re.compile(r"<task-notification\b", re.IGNORECASE)

_compiled_rules: list[tuple[str, list[re.Pattern[str]], float]] | None = None


def _compile_rules() -> list[tuple[str, list[re.Pattern[str]], float]]:
    global _compiled_rules
    if _compiled_rules is None:
        _compiled_rules = [
            (cat, [re.compile(p, re.IGNORECASE) for p in patterns], w)
            for cat, patterns, w in _HEURISTIC_RULES
        ]
    return _compiled_rules


def _classify_heuristic(text: str) -> str:
    """Score text against FR/EN patterns; return best-matching category.

    Deterministic: ties are broken by :data:`_TIE_BREAK_ORDER`, never by set
    iteration order. Two guard rails around the scored rules:

    - a short prompt made only of steering tokens ("oui", "ok go", "1 et 3")
      is a ``followup`` before any scoring;
    - when no rule fires at all, a prompt that *ends* with "?" is filed under
      ``question`` instead of ``other`` (the bare "?" is too weak to outvote a
      real keyword match, but alone it is the clearest signal there is).

    Harness chrome (``<system-reminder>`` / ``<task-notification>`` blocks) is
    stripped first so the classification reflects the user's actual words. A turn
    that is *nothing but* a task-notification (background task / sub-agent
    finished) carries no user intent, so it gets its own ``notification`` bucket
    -- the category view can hide it while its token cost stays counted (cost is
    derived from tokens, not the category).
    """
    had_notification = _TASK_NOTIFICATION_RE.search(text) is not None
    text = _NOISE_WRAPPER_RE.sub(" ", text)
    stripped = text.strip()
    if had_notification and not stripped:
        return "notification"
    if len(stripped) <= _FOLLOWUP_MAX_CHARS and _ACK_RE.match(stripped):
        return "followup"
    scores: dict[str, float] = dict.fromkeys(_TIE_BREAK_ORDER, 0.0)
    for cat, patterns, weight in _compile_rules():
        for pat in patterns:
            if pat.search(text):
                scores[cat] += weight
    best = max(_TIE_BREAK_ORDER, key=lambda c: scores[c])
    if scores[best] > 0.0:
        return best
    return "question" if stripped.endswith("?") else "other"


# ── semantic classifier (offline embeddings, mono-label) ───────────────────────

# Version stamp written to ``classifier_model`` by the semantic classifier
# ("st" = the static model is distilled from a sentence-transformer). Bumping it
# makes the next ``--semantic`` run re-classify rows stamped with an *older*
# semantic version, mirroring the heuristic's re-classify-if-stale logic.
SEMANTIC_VERSION = "semantic-st-v1"
_SEMANTIC_PREFIX = "semantic-"

# Calibrated defaults. Tuned by ``scripts/eval_semantic.py`` (B1.3) with a
# grid search that maximises litmus accuracy, tie-broken by LLM-judge agreement,
# on the demo corpus with the real static embedder. They are overridable per-user
# via a ``semantic:`` section in config.yml or the ``--tau`` / ``--prime-weight``
# / ``--top-k`` CLI flags -- reproducible and adjustable without touching code.
DEFAULT_TAU = 0.325  # below this best score → "other" (drives the "other" volume)
DEFAULT_PRIME_WEIGHT = 0.60  # scales the lexical prime onto the cosine scale
DEFAULT_TOP_K = 1  # prototype aggregation: 1 = max, >1 = mean of the top-k


def _lexical_evidence(text: str, patterns: list[re.Pattern[str]]) -> float:
    """Saturating evidence in [0, 1) from how many lexical patterns fire.

    ``1 - 0.4**hits``: one hit → 0.6, two → 0.84, three → 0.94. A single
    unambiguous git verb already clears τ once weighted, while extra matches add
    diminishing confidence rather than running away.
    """
    hits = sum(1 for pat in patterns if pat.search(text))
    return 0.0 if hits == 0 else 1.0 - 0.4**hits


@dataclass
class _Prepared:
    """A prompt readied for semantic scoring (noise stripped, flags resolved)."""

    text: str  # harness chrome removed; used for embedding *and* lexical scoring
    had_notification: bool
    is_ack: bool


class SemanticClassifier:
    """Mono-label classifier over embeddings + a fused lexical prime (Axe B1).

    One category per prompt, chosen as follows:

    1. **Hard short-circuits** (a single intent is possible): a turn that is
       nothing but a ``<task-notification>`` block → ``notification``; a short
       pure-acknowledgement → ``followup``. These reuse the heuristic guard
       rails so the two modes agree on the unambiguous cases.
    2. **Scored categories**, combined in one logit space and resolved by argmax:
       - ``semantic`` categories score the **max (or top-k mean) cosine** to
         their *distinct* prototypes (never an averaged centroid, which would
         blur a category's sub-forms);
       - ``lexical`` categories (ops, feedback) score a reused regex prime,
         scaled by ``prime_weight`` onto the cosine scale, so they **compete**
         for the label without overriding a stronger intent.
    3. If the best score is below ``tau`` → ``other`` (which has no prototype:
       "everything else" is exactly the sub-threshold case).

    Ties are broken deterministically by :data:`_TIE_BREAK_ORDER` (more specific
    intent first), never by dict iteration order.

    The embedder is injected (the heavy external dependency): production passes a
    :class:`~prompt_analytics.embeddings.StaticEmbedder`; tests pass the
    deterministic :class:`~prompt_analytics.embeddings.HashingEmbedder` so the
    whole scoring path runs in CI without downloading the model.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        anchors: dict[str, dict[str, Any]] | None = None,
        tau: float = DEFAULT_TAU,
        prime_weight: float = DEFAULT_PRIME_WEIGHT,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.embedder = embedder
        self.tau = tau
        self.prime_weight = prime_weight
        self.top_k = max(1, top_k)
        anchors = anchors if anchors is not None else load_anchors()

        # Semantic prototypes: one matrix of all prototypes, plus the owning
        # category index per row, so a category's score is a slice + reduce.
        self._semantic_cats: list[str] = []
        proto_texts: list[str] = []
        owners: list[int] = []
        for name, spec in anchors.items():
            if spec.get("role") != _ROLE_SEMANTIC:
                continue
            examples = list(spec.get("examples") or [])
            if not examples:
                continue
            idx = len(self._semantic_cats)
            self._semantic_cats.append(name)
            proto_texts.extend(examples)
            owners.extend([idx] * len(examples))
        self._proto_owner = np.asarray(owners, dtype=np.intp)
        self._proto_matrix: FloatMatrix = (
            embedder.embed(proto_texts)
            if proto_texts
            else np.zeros((0, 0), dtype=np.float32)
        )

        # Lexical prime patterns, reused verbatim from the heuristic rules so the
        # two modes share the same regex (single source of lexical truth).
        lexical_cats = [c for c in _LEXICAL_PRIME_CATEGORIES if anchors.get(c, {}).get("role") == _ROLE_LEXICAL]
        compiled = {cat: pats for cat, pats, _w in _compile_rules()}
        self._lexical_patterns: dict[str, list[re.Pattern[str]]] = {
            cat: compiled[cat] for cat in lexical_cats if cat in compiled
        }

    @property
    def dim(self) -> int:
        """Embedding dimension (0 until any prototype has been embedded)."""
        return 0 if self._proto_matrix.size == 0 else int(self._proto_matrix.shape[1])

    def prepare(self, raw: str) -> _Prepared:
        """Strip harness chrome and resolve the short-circuit flags."""
        had_notification = _TASK_NOTIFICATION_RE.search(raw) is not None
        cleaned = _NOISE_WRAPPER_RE.sub(" ", raw)
        stripped = cleaned.strip()
        is_ack = len(stripped) <= _FOLLOWUP_MAX_CHARS and bool(_ACK_RE.match(stripped))
        return _Prepared(text=cleaned, had_notification=had_notification, is_ack=is_ack)

    def _scores(self, clean_text: str, vector: FloatMatrix) -> dict[str, float]:
        """Combined per-category logit scores (semantic cosine + lexical prime)."""
        scores: dict[str, float] = {}
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if self._proto_matrix.shape[0] and vec.shape[0] == self._proto_matrix.shape[1]:
            sims = self._proto_matrix @ vec  # vectors are L2-normalized → cosine
            for idx, cat in enumerate(self._semantic_cats):
                cat_sims = sims[self._proto_owner == idx]
                if cat_sims.size == 0:
                    continue
                k = min(self.top_k, cat_sims.size)
                scores[cat] = float(np.sort(cat_sims)[-k:].mean())
        for cat, patterns in self._lexical_patterns.items():
            evidence = _lexical_evidence(clean_text, patterns)
            if evidence > 0.0:
                scores[cat] = self.prime_weight * evidence
        return scores

    def label(self, prep: _Prepared, vector: FloatMatrix) -> str:
        """Resolve the single category from a prepared prompt + its embedding."""
        if prep.had_notification and not prep.text.strip():
            return "notification"
        if prep.is_ack:
            return "followup"
        scores = self._scores(prep.text, vector)
        best_cat, best_score = "other", float("-inf")
        for cat in _TIE_BREAK_ORDER:  # deterministic tie-break (specific first)
            score = scores.get(cat)
            if score is not None and score > best_score:
                best_score, best_cat = score, cat
        return best_cat if best_score >= self.tau else "other"

    def classify(self, raw: str) -> str:
        """Classify one prompt end to end (prepare → embed → label)."""
        prep = self.prepare(raw)
        if prep.had_notification and not prep.text.strip():
            return "notification"
        if prep.is_ack:
            return "followup"
        vector = self.embedder.embed([prep.text])
        return self.label(prep, vector[0] if vector.shape[0] else vector)


# ── observed complexity ───────────────────────────────────────────────────────


def _quantile_band(val: float, sorted_vals: list[float]) -> int:
    """Map val to 1-5 based on its quintile in sorted_vals.

    Returns 3 when all values are equal (no differentiation possible).
    """
    if not sorted_vals:
        return 3
    if sorted_vals[0] == sorted_vals[-1]:  # degenerate: all equal
        return 3
    rank = sum(1 for v in sorted_vals if v <= val) / len(sorted_vals)
    if rank <= 0.2:
        return 1
    if rank <= 0.4:
        return 2
    if rank <= 0.6:
        return 3
    if rank <= 0.8:
        return 4
    return 5


def _observed_complexity_scores(
    prompts: list[dict[str, Any]],
    prompt_costs: dict[str, float],
) -> dict[str, str]:
    """Return prompt_id -> complexity_str (1-5) from observed effort metrics."""
    rows = [r for r in prompts if r.get("prompt_id")]
    if not rows:
        return {}

    def _fv(row: dict[str, Any], key: str) -> float:
        try:
            return float(row.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    at_s = sorted(_fv(r, "assistant_turns") for r in rows)
    tc_s = sorted(_fv(r, "tool_calls") for r in rows)
    cc_s = sorted(_fv(r, "char_count") for r in rows)
    co_s = sorted(prompt_costs.get(r["prompt_id"], 0.0) for r in rows)

    result: dict[str, str] = {}
    for row in rows:
        pid = row["prompt_id"]
        avg = (
            _quantile_band(_fv(row, "assistant_turns"), at_s)
            + _quantile_band(_fv(row, "tool_calls"), tc_s)
            + _quantile_band(_fv(row, "char_count"), cc_s)
            + _quantile_band(prompt_costs.get(pid, 0.0), co_s)
        ) / 4
        result[pid] = str(round(avg))
    return result


# ── LLM infrastructure ────────────────────────────────────────────────────────


class Classifier(Protocol):
    model: str

    def classify(self, text: str) -> tuple[str, str]: ...


class _PermanentError(Exception):
    """Non-retryable: invalid key, bad request."""


class _TransientError(Exception):
    """Retryable: rate limit, server error, network."""

    def __init__(self, msg: str, *, retry_after: float = 0.0) -> None:
        super().__init__(msg)
        self.retry_after = retry_after


def _parse_reply(raw: str) -> tuple[str, str]:
    """Parse ``category|complexity`` from an LLM reply."""
    cleaned = (raw or "").strip().lower()
    parts = cleaned.split("|")
    cat = parts[0].strip() if parts else "other"
    comp = parts[1].strip() if len(parts) >= 2 else ""
    if cat not in CATEGORIES:
        cat = "other"
    if comp not in COMPLEXITIES:
        comp = "3"
    return cat, comp


def _call_with_retry(
    fn: Callable[[], tuple[str, str]],
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> tuple[str, str] | None:
    """Call fn() with exponential backoff; returns None when all retries exhausted."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except _PermanentError:
            raise
        except _TransientError as exc:
            if attempt == max_retries:
                print(f"  [warn] giving up after {max_retries} retries: {exc}", file=sys.stderr)
                return None
            delay = max(base_delay * (2**attempt), exc.retry_after)
            print(f"  [retry] {attempt + 1}/{max_retries} in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)
    return None  # unreachable


class _AnthropicClassifier:
    """Single-call Anthropic messages API with retry logic."""

    def __init__(self, client: _AnthropicClient, model: str = ANTHROPIC_MODEL) -> None:
        self._client = client
        self.model = model

    def _call(self, text: str) -> tuple[str, str]:
        import anthropic

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": text[:MAX_PROMPT_CHARS]}],
            )
            # content[0] is a union of block types; only text blocks carry
            # ``.text``. We always request a plain-text reply, so read it
            # defensively (keeps mypy honest against the SDK's block union).
            return _parse_reply(getattr(response.content[0], "text", ""))
        except anthropic.AuthenticationError as exc:
            raise _PermanentError(f"Invalid API key: {exc}") from exc
        except anthropic.RateLimitError as exc:
            import contextlib

            retry_after = 0.0
            resp = getattr(exc, "response", None)
            if resp is not None:
                with contextlib.suppress(TypeError, ValueError):
                    retry_after = float(resp.headers.get("retry-after", 0))
            raise _TransientError(str(exc), retry_after=retry_after) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code >= 500:
                raise _TransientError(f"server error {exc.status_code}") from exc
            raise _PermanentError(f"API error {exc.status_code}: {exc}") from exc
        except anthropic.APIConnectionError as exc:
            raise _TransientError(f"connection error: {exc}") from exc

    def classify(self, text: str) -> tuple[str, str]:
        result = _call_with_retry(lambda: self._call(text))
        return result if result is not None else ("", "")


class _AnthropicBatchClassifier(_AnthropicClassifier):
    """Extends _AnthropicClassifier with classify_many() for the Batches API."""

    def classify_many(self, items: list[tuple[str, str]]) -> dict[str, tuple[str, str]]:
        """Submit items as an Anthropic Message Batch; return pid -> (cat, comp)."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [
            Request(
                custom_id=pid,
                params=MessageCreateParamsNonStreaming(
                    model=self.model,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": text[:MAX_PROMPT_CHARS]}],
                ),
            )
            for pid, text in items
        ]
        batch = self._client.messages.batches.create(requests=requests)

        deadline = time.monotonic() + BATCH_POLL_TIMEOUT
        while True:
            status = self._client.messages.batches.retrieve(batch.id)
            if status.processing_status == "ended":
                break
            if time.monotonic() >= deadline:
                raise _PermanentError(
                    f"batch {batch.id} still {status.processing_status} after "
                    f"{BATCH_POLL_TIMEOUT // 3600}h; giving up (results stay retrievable for "
                    "29 days -- re-run categorize to resume)"
                )
            time.sleep(BATCH_POLL_INTERVAL)

        out: dict[str, tuple[str, str]] = {}
        for result in self._client.messages.batches.results(batch.id):
            if result.result.type == "succeeded":
                raw = getattr(result.result.message.content[0], "text", "")
                out[result.custom_id] = _parse_reply(raw)
        return out


class _OpenRouterClassifier:
    """OpenAI-compatible classifier via OpenRouter."""

    def __init__(self, client: _OpenAIClient, model: str = OPENROUTER_MODEL) -> None:
        self._client = client
        self.model = model

    def _call(self, text: str) -> tuple[str, str]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:MAX_PROMPT_CHARS]},
                ],
                max_tokens=MAX_TOKENS,
            )
            return _parse_reply(response.choices[0].message.content or "")
        except Exception as exc:
            name = type(exc).__name__
            if "Authentication" in name:
                raise _PermanentError(f"Invalid key: {exc}") from exc
            if "RateLimit" in name:
                raise _TransientError(f"Rate limit: {exc}") from exc
            raise _TransientError(f"API error: {exc}") from exc

    def classify(self, text: str) -> tuple[str, str]:
        result = _call_with_retry(lambda: self._call(text))
        return result if result is not None else ("", "")


class _OllamaClassifier:
    """Local Ollama classifier via OpenAI-compatible API (no key required)."""

    def __init__(self, client: _OpenAIClient, model: str = OLLAMA_MODEL) -> None:
        self._client = client
        self.model = model

    def classify(self, text: str) -> tuple[str, str]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:MAX_PROMPT_CHARS]},
                ],
                max_tokens=MAX_TOKENS,
            )
            return _parse_reply(response.choices[0].message.content or "")
        except Exception as exc:
            print(f"  [warn] Ollama error: {exc}", file=sys.stderr)
            return "", ""


class _AzureOpenAIClassifier:
    """Azure OpenAI classifier (OpenAI-compatible chat completions).

    Added for the dev-time "silver" LLM judge of the semantic eval (B1.3), where
    Azure OpenAI is the only reachable LLM. The ``model`` is the Azure
    *deployment* name. Unlike the OpenRouter/Ollama paths, this one requests
    :data:`AZURE_MAX_COMPLETION_TOKENS` via ``max_completion_tokens`` and never
    sets ``temperature`` -- both are required by reasoning deployments
    (gpt-5-class), which would otherwise reject the call or return an empty reply
    after spending the tiny token cap on hidden reasoning.
    """

    def __init__(self, client: _OpenAIClient, model: str) -> None:
        self._client = client
        self.model = model

    def _call(self, text: str) -> tuple[str, str]:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:MAX_PROMPT_CHARS]},
                ],
                max_completion_tokens=AZURE_MAX_COMPLETION_TOKENS,
            )
            return _parse_reply(response.choices[0].message.content or "")
        except Exception as exc:
            # Duck-typed by exception class name (same approach as OpenRouter):
            # the Azure SDK shares OpenAI's error hierarchy but we avoid a hard
            # import here.
            name = type(exc).__name__
            if "Authentication" in name or "Permission" in name:
                raise _PermanentError(f"Invalid Azure key/endpoint: {exc}") from exc
            if "RateLimit" in name:
                raise _TransientError(f"Rate limit: {exc}") from exc
            raise _TransientError(f"API error: {exc}") from exc

    def classify(self, text: str) -> tuple[str, str]:
        result = _call_with_retry(lambda: self._call(text))
        return result if result is not None else ("", "")


# ── client builder ────────────────────────────────────────────────────────────


def _build_azure(model: str) -> _AzureOpenAIClassifier | None:
    """Build an Azure OpenAI classifier, or ``None`` if its env is incomplete.

    Returns ``None`` silently when the Azure keys are absent (so it can be tried
    as a fallback in ``auto`` mode without noise); prints once when the keys are
    present but the ``openai`` package is missing.
    """
    key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    if not key or not endpoint:
        return None
    print(
        "PRIVACY WARNING: prompt excerpts will be sent to Azure OpenAI (third party).",
        file=sys.stderr,
    )
    try:
        from openai import AzureOpenAI
    except ImportError:
        print("openai package required for Azure (pip install openai).", file=sys.stderr)
        return None
    return _AzureOpenAIClassifier(
        AzureOpenAI(
            api_key=key,
            azure_endpoint=endpoint,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        ),
        model=model or os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
    )


def build_client(
    *,
    provider: str = "auto",
    model: str = "",
    use_batch: bool = False,
) -> (
    _AnthropicClassifier
    | _OpenRouterClassifier
    | _OllamaClassifier
    | _AzureOpenAIClassifier
    | None
):
    """Build an LLM Classifier from environment variables.

    provider: ``"auto"`` (ANTHROPIC_API_KEY, then OPENROUTER_API_KEY, then
    AZURE_OPENAI_API_KEY), ``"anthropic"``, ``"openrouter"``, ``"ollama"``, or
    ``"azure"``.
    """
    from dotenv import load_dotenv

    load_dotenv()

    if provider == "azure":
        if not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"):
            print(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set in .env.",
                file=sys.stderr,
            )
            return None
        return _build_azure(model)

    if provider == "ollama":
        try:
            from openai import OpenAI

            return _OllamaClassifier(
                OpenAI(api_key="ollama", base_url=OLLAMA_BASE_URL),
                model=model or OLLAMA_MODEL,
            )
        except ImportError:
            print("openai package required for Ollama (pip install openai).", file=sys.stderr)
            return None

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "")
        if not key:
            print("OPENROUTER_API_KEY not set.", file=sys.stderr)
            return None
        print(
            "PRIVACY WARNING: prompt excerpts will be sent to OpenRouter (third party).",
            file=sys.stderr,
        )
        try:
            from openai import OpenAI

            return _OpenRouterClassifier(
                OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"),
                model=model or OPENROUTER_MODEL,
            )
        except ImportError:
            print("openai package required for OpenRouter (pip install openai).", file=sys.stderr)
            return None

    # Anthropic (default or explicit)
    key = os.getenv("ANTHROPIC_API_KEY", "")
    if not key and provider == "auto":
        or_key = os.getenv("OPENROUTER_API_KEY", "")
        if or_key:
            print(
                "PRIVACY WARNING: prompt excerpts will be sent to OpenRouter (third party).",
                file=sys.stderr,
            )
            try:
                from openai import OpenAI

                return _OpenRouterClassifier(
                    OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1"),
                    model=model or OPENROUTER_MODEL,
                )
            except ImportError:
                pass
        azure = _build_azure(model)
        if azure is not None:
            return azure
        print(
            "No LLM API key found. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or "
            "AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_ENDPOINT) in .env.",
            file=sys.stderr,
        )
        return None

    if not key:
        print("ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        if use_batch:
            return _AnthropicBatchClassifier(client, model=model or ANTHROPIC_MODEL)
        return _AnthropicClassifier(client, model=model or ANTHROPIC_MODEL)
    except ImportError:
        print("anthropic package required (pip install anthropic).", file=sys.stderr)
        return None


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _is_pseudo(prompt_id: str) -> bool:
    """True for _continuation and other system pseudo-prompts."""
    return ":_" in prompt_id


def _load_texts(text_path: Path) -> dict[str, str]:
    if not text_path.exists():
        return {}
    with text_path.open(encoding="utf-8", newline="") as fh:
        return {r["prompt_id"]: r.get("prompt_text", "") for r in csv.DictReader(fh)}


def _load_categories(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            pid = row.get("prompt_id", "")
            if pid:
                result[pid] = {col: row.get(col, "") for col in CATEGORIES_COLS}
    return result


def _load_prompt_costs(tokens_path: Path) -> dict[str, float]:
    """Compute prompt_id -> USD cost from tokens.csv (Anthropic pricing).

    Cost feeds the observed-complexity bands, not the categories themselves, so
    a missing tokens.csv degrades gracefully ({}). But a *corrupt* pricing.yml
    must not be swallowed (3.6): it would silently mis-cost every prompt. The
    :exc:`PricingError` is surfaced on stderr and re-raised so the failure is
    loud, in keeping with the rest of the pipeline's "detect bad data" stance.
    """
    if not tokens_path.exists():
        return {}
    from .analytics import CostEngine
    from .pricing import PricingError

    engine = CostEngine("anthropic")
    costs: dict[str, float] = {}
    try:
        with tokens_path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                pid = row.get("prompt_id", "")
                if not pid:
                    continue
                try:
                    count = int(row.get("token_count") or 0)
                except (TypeError, ValueError):
                    count = 0
                costs[pid] = costs.get(pid, 0.0) + engine.cost(
                    row.get("model", ""), row.get("token_type", ""), count
                )
    except PricingError as exc:
        print(f"Error: cannot price prompts (pricing.yml invalid): {exc}", file=sys.stderr)
        raise
    return costs


def _flush(categories: dict[str, dict[str, str]], path: Path) -> None:
    rows = [categories[pid] for pid in sorted(categories)]
    atomic_write_csv(path, CATEGORIES_COLS, rows)


# ── main entry point ──────────────────────────────────────────────────────────


def run_categorize(
    *,
    output_dir: str = "./output",
    use_llm: bool = False,
    use_semantic: bool = False,
    use_batch: bool = False,
    provider: str = "auto",
    model: str = "",
    batch_size: int = 50,
    delay: float = 0.1,
    limit: int = 0,
    embedder: Embedder | None = None,
    tau: float | None = None,
    prime_weight: float | None = None,
    top_k: int | None = None,
) -> int:
    """Classify prompts into categories.csv; return count newly classified.

    Three modes: heuristic regex (default), the offline semantic classifier
    (``use_semantic``, embeddings, no API key), and the LLM (``use_llm``).
    Observed complexity (quantile bands) is always recomputed for all real
    prompts.

    Category is never overwritten across a more authoritative classifier:

    * heuristic mode only redoes its own *stale*-version rows
      (:data:`HEURISTIC_VERSION`);
    * semantic mode supersedes heuristic rows (it is the richer classifier) and
      redoes its own stale-version rows (:data:`SEMANTIC_VERSION`), but never
      touches LLM-stamped rows;
    * LLM-classified rows are never overwritten by either offline mode.

    ``embedder`` is an injection seam for ``use_semantic`` (the heavy model is
    the only external dependency): it defaults to the real
    :class:`~prompt_analytics.embeddings.StaticEmbedder`; tests pass the
    deterministic ``HashingEmbedder``.
    Prompts without any stored text (``extract --no-text``) are skipped with
    a warning and left uncategorized, never silently filed under "other".

    Returns ``-1`` when nothing could even be attempted (no ``prompts.csv``
    yet, or no usable LLM client in ``--llm`` mode) so the CLI can exit
    non-zero; ``0`` means "nothing new to classify", which is a success.
    """
    out = Path(output_dir)
    prompts_path = out / "prompts.csv"

    if not prompts_path.exists():
        print(
            f"No prompts file at {prompts_path}. Run `prompt-analytics extract` first.",
            file=sys.stderr,
        )
        return -1

    with prompts_path.open(encoding="utf-8", newline="") as fh:
        all_prompts: list[dict[str, Any]] = list(csv.DictReader(fh))

    texts = _load_texts(out / "prompts_text.csv")
    categories = _load_categories(out / "categories.csv")
    prompt_costs = _load_prompt_costs(out / "tokens.csv")

    real_prompts = [r for r in all_prompts if r.get("prompt_id") and not _is_pseudo(r["prompt_id"])]

    # Always recompute observed complexity for all real prompts
    complexity_scores = _observed_complexity_scores(real_prompts, prompt_costs)
    for row in real_prompts:
        pid = row["prompt_id"]
        comp = complexity_scores.get(pid, "3")
        if pid not in categories:
            categories[pid] = {
                "prompt_id": pid,
                "category": "",
                "complexity": comp,
                "classifier_model": "",
                "classified_at": "",
            }
        else:
            categories[pid]["complexity"] = comp

    # Prompts that still need a category. LLM rows are never overwritten; in
    # heuristic mode, rows stamped with an older heuristic version are redone
    # so a rules upgrade propagates without nuking LLM classifications.
    def _supersedable(pid: str) -> bool:
        """Whether the current mode may (re)classify a row that already has one."""
        model_used = categories.get(pid, {}).get("classifier_model", "")
        if use_llm:
            return False  # the LLM never overwrites an existing category
        if use_semantic:
            if model_used.startswith(_HEURISTIC_PREFIX):
                return True  # semantic supersedes the heuristic default
            return model_used.startswith(_SEMANTIC_PREFIX) and model_used != SEMANTIC_VERSION
        return model_used.startswith(_HEURISTIC_PREFIX) and model_used != HEURISTIC_VERSION

    to_classify = [
        r
        for r in real_prompts
        if not categories.get(r["prompt_id"], {}).get("category")
        or _supersedable(r["prompt_id"])
    ]

    # No stored text (extract --no-text): classifying "" would file everything
    # under "other" -- permanently, since non-empty categories are never
    # re-classified. Skip those prompts loudly and leave their category empty
    # so a later text-enabled extract + categorize picks them up (N2).
    def _text_of(row: dict[str, Any]) -> str:
        return (texts.get(row["prompt_id"]) or row.get("prompt_preview") or "").strip()

    skipped_no_text = sum(1 for r in to_classify if not _text_of(r))
    if skipped_no_text:
        to_classify = [r for r in to_classify if _text_of(r)]
        print(
            f"[warn] {skipped_no_text} prompt(s) have no stored text (did extract run "
            "with --no-text?): skipped, left uncategorized. Re-run "
            "`prompt-analytics extract` without --no-text, then categorize again.",
            file=sys.stderr,
        )

    if limit:
        to_classify = to_classify[:limit]

    total = len(to_classify)
    if total == 0:
        if skipped_no_text:
            print("Nothing classifiable: every pending prompt is missing its text.")
        else:
            print(f"Nothing new to classify ({len(real_prompts)} prompts already done).")
        _flush(categories, out / "categories.csv")
        return 0

    print(f"Prompts to classify: {total} / {len(real_prompts)}")
    classified = 0

    # ── semantic mode (offline embeddings, mono-label) ──────────────────────────
    if use_semantic and not use_llm:
        from .config import load_config
        from .embeddings import EmbeddingCache, StaticEmbedder

        emb = embedder if embedder is not None else StaticEmbedder()
        cfg = load_config(out).get("semantic") or {}
        # Precedence: explicit CLI flag > config.yml > calibrated default.
        resolved_tau = tau if tau is not None else float(cfg.get("tau", DEFAULT_TAU))
        resolved_pw = (
            prime_weight if prime_weight is not None
            else float(cfg.get("prime_weight", DEFAULT_PRIME_WEIGHT))
        )
        resolved_top_k = top_k if top_k is not None else int(cfg.get("top_k", DEFAULT_TOP_K))
        clf = SemanticClassifier(
            emb,
            tau=resolved_tau,
            prime_weight=resolved_pw,
            top_k=resolved_top_k,
        )
        ids = [r["prompt_id"] for r in to_classify]
        raws = [
            texts.get(pid) or r.get("prompt_preview", "")
            for r, pid in zip(to_classify, ids, strict=True)
        ]
        preps = [clf.prepare(raw) for raw in raws]
        # Embed the noise-stripped text once, cached on disk by (prompt_id, text)
        # so a re-run never re-embeds unchanged prompts (the cache is namespaced
        # by embedder identity, so static and hashing vectors never mix).
        cache = EmbeddingCache(out / "embeddings.npz", emb)
        vectors = cache.embed(ids, [prep.text for prep in preps])
        now = datetime.now(timezone.utc).isoformat()
        for row, prep, vector in zip(to_classify, preps, vectors, strict=True):
            categories[row["prompt_id"]].update(
                {
                    "category": clf.label(prep, vector),
                    "classifier_model": SEMANTIC_VERSION,
                    "classified_at": now,
                }
            )
            classified += 1
        _flush(categories, out / "categories.csv")
        print(f"Classified {classified}  ->  {(out / 'categories.csv').resolve()}")
        return classified

    # ── heuristic mode ────────────────────────────────────────────────────────
    if not use_llm:
        now = datetime.now(timezone.utc).isoformat()
        for row in to_classify:
            pid = row["prompt_id"]
            text = texts.get(pid) or row.get("prompt_preview", "")
            categories[pid].update(
                {
                    "category": _classify_heuristic(text),
                    "classifier_model": HEURISTIC_VERSION,
                    "classified_at": now,
                }
            )
            classified += 1
        _flush(categories, out / "categories.csv")
        print(f"Classified {classified}  ->  {(out / 'categories.csv').resolve()}")
        return classified

    # ── LLM mode ──────────────────────────────────────────────────────────────
    llm = build_client(provider=provider, model=model, use_batch=use_batch)
    if llm is None:
        return -1

    categories_path = out / "categories.csv"

    if use_batch and isinstance(llm, _AnthropicBatchClassifier):
        chunk = batch_size or 50
        for start in range(0, total, chunk):
            batch_rows = to_classify[start : start + chunk]
            items = [
                (r["prompt_id"], texts.get(r["prompt_id"]) or r.get("prompt_preview", ""))
                for r in batch_rows
            ]
            print(f"  Batch {start // chunk + 1}: submitting {len(items)} prompts...")
            try:
                results = llm.classify_many(items)
            except _PermanentError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                _flush(categories, categories_path)
                return classified
            now = datetime.now(timezone.utc).isoformat()
            for pid, (cat, _) in results.items():
                if cat:
                    categories[pid].update(
                        {
                            "category": cat,
                            "classifier_model": llm.model,
                            "classified_at": now,
                        }
                    )
                    classified += 1
            _flush(categories, categories_path)
            print(f"  [checkpoint] {classified}/{total}")
    else:
        for i, row in enumerate(to_classify, start=1):
            pid = row["prompt_id"]
            text = texts.get(pid) or row.get("prompt_preview", "")
            try:
                cat, _ = llm.classify(text)
            except _PermanentError as exc:
                print(f"Error: aborting -- {exc}", file=sys.stderr)
                _flush(categories, categories_path)
                return classified
            now = datetime.now(timezone.utc).isoformat()
            if cat:
                categories[pid].update(
                    {
                        "category": cat,
                        "classifier_model": llm.model,
                        "classified_at": now,
                    }
                )
                classified += 1

            if i % 10 == 0 or i == total:
                print(f"  {i}/{total}")
            if delay:
                time.sleep(delay)
            if batch_size and i % batch_size == 0:
                _flush(categories, categories_path)
                print(f"  [checkpoint] saved at {i}")

    _flush(categories, categories_path)
    print(f"Classified {classified}  ->  {categories_path.resolve()}")
    return classified
