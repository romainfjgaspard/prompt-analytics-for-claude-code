"""Model pricing lookup utilities."""

from __future__ import annotations

import importlib.resources
import re
from pathlib import Path
from typing import Any, cast

import yaml

__all__ = [
    "load_pricing",
    "get_model_pricing",
    "get_per_request",
    "load_plans",
    "is_long_context",
    "clear_cache",
    "PricingError",
]

_REQUIRED_KEYS = frozenset({"input", "output", "cache_read", "cache_write_5m", "cache_write_1h"})

# In-memory cache keyed by resolved file path.
_PRICING_CACHE: dict[Path, dict[str, Any]] = {}

# Matches optional [1m] / [long-context] / [any-suffix] appended to model IDs.
_SUFFIX_RE = re.compile(r"\[.*?\]$")

# A stripped suffix that denotes the >200K long-context tier (3.2). Current
# Claude models (Opus 4.6-4.8, Sonnet 4.6, Fable 5) bill their 1M window at the
# base rate -- no premium -- so pricing at base is correct; older models had a
# >200K premium. Either way the stripping must be *loud*, not silent.
_LONG_CONTEXT_RE = re.compile(r"\[\s*(1m|200k\+?|long[\s-]?context|long)\s*\]$", re.IGNORECASE)


class PricingError(Exception):
    """Raised when pricing.yml is invalid or cannot be loaded."""


def _default_pricing_path() -> Path:
    """Return the path to the bundled pricing.yml via importlib.resources."""
    ref = importlib.resources.files("prompt_analytics.data") / "pricing.yml"
    return Path(str(ref))


def clear_cache() -> None:
    """Invalidate the in-memory pricing cache.

    Call between tests or after writing a custom pricing file so that the next
    :func:`load_pricing` re-reads from disk.
    """
    _PRICING_CACHE.clear()


def _validate(data: dict[str, Any], path: Path) -> None:
    """Raise :exc:`PricingError` if the YAML structure is invalid."""
    if not isinstance(data, dict):
        raise PricingError(f"{path}: pricing YAML must be a mapping at the top level")
    if "updated_at" not in data:
        raise PricingError(f"{path}: missing required field 'updated_at'")
    providers = data.get("providers")
    if providers is None:
        raise PricingError(f"{path}: missing required field 'providers'")
    if not isinstance(providers, dict):
        raise PricingError(f"{path}: 'providers' must be a mapping")

    plans = data.get("plans")
    if plans is not None:
        if not isinstance(plans, dict):
            raise PricingError(f"{path}: 'plans' must be a mapping")
        for plan_name, plan in plans.items():
            if not isinstance(plan, dict) or "monthly_usd" not in plan:
                raise PricingError(
                    f"{path}: plans.{plan_name} must be a mapping with a 'monthly_usd' key"
                )
            if not isinstance(plan["monthly_usd"], int | float):
                raise PricingError(f"{path}: plans.{plan_name}.monthly_usd must be a number")

    for pname, pdata in providers.items():
        if not isinstance(pdata, dict):
            raise PricingError(f"{path}: provider '{pname}' must be a mapping")
        per_request = pdata.get("per_request")
        if per_request is not None:
            if not isinstance(per_request, dict):
                raise PricingError(f"{path}: providers.{pname}.per_request must be a mapping")
            for key, value in per_request.items():
                if not isinstance(value, int | float):
                    raise PricingError(
                        f"{path}: providers.{pname}.per_request.{key} must be a number"
                    )
        for section in ("models", "fallbacks"):
            section_data = pdata.get(section, {})
            if not isinstance(section_data, dict):
                raise PricingError(f"{path}: providers.{pname}.{section} must be a mapping")
            for mname, mdata in section_data.items():
                if not isinstance(mdata, dict):
                    raise PricingError(
                        f"{path}: providers.{pname}.{section}.{mname} must be a mapping"
                    )
                missing = _REQUIRED_KEYS - mdata.keys()
                if missing:
                    raise PricingError(
                        f"{path}: providers.{pname}.{section}.{mname} is missing keys: "
                        f"{sorted(missing)}"
                    )


def load_pricing(path: Path | None = None) -> dict[str, Any]:
    """Load and validate pricing.yml.

    Uses the bundled ``prompt_analytics/data/pricing.yml`` by default.
    Results are cached per resolved path; call :func:`clear_cache` to invalidate.

    Args:
        path: Optional path to a custom pricing YAML file.

    Returns:
        The validated mapping containing a ``providers`` key.

    Raises:
        PricingError: If the file cannot be read, is not valid YAML, or fails
            schema validation.
    """
    resolved = (path or _default_pricing_path()).resolve()
    cached = _PRICING_CACHE.get(resolved)
    if cached is not None:
        return cached

    try:
        with resolved.open("r", encoding="utf-8") as fh:
            data: dict[str, Any] = yaml.safe_load(fh) or {}
    except OSError as exc:
        raise PricingError(f"Cannot read {resolved}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PricingError(f"Invalid YAML in {resolved}: {exc}") from exc

    _validate(data, resolved)
    _PRICING_CACHE[resolved] = data
    return data


def get_model_pricing(
    model: str,
    provider: str,
    pricing_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the pricing entry for a given model and provider.

    The returned mapping has the keys ``input``, ``output``, ``cache_read``,
    ``cache_write_5m``, and ``cache_write_1h`` (all in USD per 1 M tokens).

    Resolution order:
        1. Strip any trailing ``[...]`` suffix (e.g. ``[1m]``, ``[long-context]``).
        2. Exact match in ``providers[provider]["models"]``.
        3. Longest-prefix match in ``providers[provider]["fallbacks"]``.
        4. Return ``None`` — the caller is responsible for surfacing the warning.

    Args:
        model: Model identifier; a trailing ``[...]`` suffix is stripped before
            lookup.
        provider: Provider key from the pricing file (e.g. ``"anthropic"`` or
            ``"copilot"``).  Unknown providers return ``None``.
        pricing_path: Optional path to a custom pricing YAML (passed to
            :func:`load_pricing`).

    Returns:
        The pricing dict, or ``None`` if no entry is found for this
        model/provider combination.
    """
    clean = _SUFFIX_RE.sub("", model).rstrip()

    pricing = load_pricing(pricing_path)
    providers: dict[str, Any] = pricing.get("providers", {})
    provider_data = providers.get(provider)
    if not isinstance(provider_data, dict):
        return None

    # Exact match
    models: dict[str, Any] = provider_data.get("models", {})
    if clean in models:
        return cast("dict[str, Any]", models[clean])

    # Longest-prefix fallback
    fallbacks: dict[str, Any] = provider_data.get("fallbacks", {})
    best: str | None = None
    for prefix in fallbacks:
        if clean.startswith(prefix) and (best is None or len(prefix) > len(best)):
            best = prefix
    if best is not None:
        return cast("dict[str, Any]", fallbacks[best])

    return None


def get_per_request(
    key: str,
    provider: str,
    pricing_path: Path | None = None,
) -> float | None:
    """USD charged per request for a server-side tool (3.3), or ``None``.

    ``server_tool_use`` counts server-side tool *requests* (e.g. web search),
    not tokens; ``providers[provider]["per_request"][key]`` gives the price per
    request when configured. Unknown provider/key returns ``None`` so the caller
    treats it as uncosted (the historical behaviour).
    """
    pricing = load_pricing(pricing_path)
    provider_data = pricing.get("providers", {}).get(provider)
    if not isinstance(provider_data, dict):
        return None
    per_request = provider_data.get("per_request")
    if not isinstance(per_request, dict) or key not in per_request:
        return None
    return float(per_request[key])


def load_plans(pricing_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The flat-rate subscription plans (3.1), in file order.

    Each value carries at least ``monthly_usd`` (USD/month) and usually a
    ``label``. Empty when the pricing file declares no ``plans`` section.
    """
    plans = load_pricing(pricing_path).get("plans", {})
    return plans if isinstance(plans, dict) else {}


def load_copilot_plans(pricing_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """GitHub Copilot subscription tiers (usage-based AI-credit billing).

    Each value carries ``monthly_usd`` (the subscription price) and ``included_usd``
    (the bundled GitHub AI-credit allowance in USD); usage beyond it is per-token
    overage on the ``copilot`` grid. A *different* axis from :func:`load_plans`
    (Claude flat-rate). Empty when the file declares no ``copilot_plans`` section.
    """
    plans = load_pricing(pricing_path).get("copilot_plans", {})
    return plans if isinstance(plans, dict) else {}


def is_long_context(model: str) -> bool:
    """True when ``model`` carries a >200K long-context suffix (3.2).

    e.g. ``claude-opus-4-8[1m]``. The suffix is stripped before pricing lookup,
    so without this signal the long-context tier would be priced at the base
    rate *silently*; callers use it to surface a warning instead.
    """
    return bool(_LONG_CONTEXT_RE.search(model.strip()))
