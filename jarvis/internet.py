"""Internet access layer: multi-provider web search, URL scraping,
authenticated API calls, and real-time data feeds.

All network I/O uses httpx (already a required dep). No additional packages
are required — every feature degrades gracefully when optional API keys are absent.

Search provider fallback chain (auto mode):
  1. Brave Search API       — free tier (2 000 req/month); set BRAVE_API_KEY
  2. Serper.dev             — freemium (2 500 free); set SERPER_API_KEY
  3. Tavily AI              — freemium (1 000 free); set TAVILY_API_KEY
  4. DuckDuckGo HTML scrape — free, no key, always available
  5. DuckDuckGo Instant API — free factual fallback (no key)

Real-time data feeds (all free unless noted):
  weather      Open-Meteo (no key) + Nominatim geocoder (no key)
  crypto       CoinGecko public API (no key)
  news         RSS feeds: BBC, Reuters, Hacker News, TechCrunch
  exchange     Frankfurter (ECB data; no key)
  ip           ipinfo.io (no key for basic info)
  stocks       Alpha Vantage (free tier; set ALPHA_VANTAGE_KEY)

API profile system:
  Store named API profiles (base URL + auth config) in ~/.jarvis/api_profiles.json.
  Secrets (tokens, keys) are resolved from environment variables or the vault —
  never stored in the profile file itself.
"""

from __future__ import annotations

import html as _html_lib
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as _ET
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlencode, urljoin, urlparse

import httpx


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_UA = "Mozilla/5.0 (compatible; JARVIS-local-assistant/1.0)"
_DEFAULT_TIMEOUT = 15.0
_MAX_RESPONSE_CHARS = 12_000

_WMO_CODES: dict[int, str] = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light showers", 81: "Showers", 82: "Violent showers",
    85: "Light snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm + hail", 99: "Thunderstorm + heavy hail",
}

_RSS_FEEDS: dict[str, str] = {
    "bbc":        "https://feeds.bbci.co.uk/news/rss.xml",
    "reuters":    "https://feeds.reuters.com/reuters/topNews",
    "hn":         "https://news.ycombinator.com/rss",
    "techcrunch": "https://techcrunch.com/feed/",
    "wired":      "https://www.wired.com/feed/rss",
    "arstechnica":"https://feeds.arstechnica.com/arstechnica/index",
    "tech":       "https://techcrunch.com/feed/",
    "general":    "https://feeds.bbci.co.uk/news/rss.xml",
    "science":    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "world":      "https://feeds.reuters.com/reuters/worldNews",
    "business":   "https://feeds.reuters.com/reuters/businessNews",
    "top":        "https://feeds.bbci.co.uk/news/rss.xml",
}


# ---------------------------------------------------------------------------
# HTML text extractor (stdlib only)
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip HTML to readable plain text without external deps."""

    _SKIP = {"script", "style", "nav", "header", "footer", "aside",
              "noscript", "iframe", "form", "button", "svg", "path"}
    _BLOCK = {"p", "div", "section", "article", "main", "li", "h1",
               "h2", "h3", "h4", "h5", "h6", "br", "tr", "td", "th"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self.title: str = ""
        self.meta_description: str = ""
        self._in_title = False
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        self._tag_stack.append(tag)
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            attr_dict = dict(attrs)
            if attr_dict.get("name", "").lower() == "description":
                self.meta_description = attr_dict.get("content", "")
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._tag_stack and self._tag_stack[-1] == tag:
            self._tag_stack.pop()
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title = data.strip()
        text = data.strip()
        if text:
            self._parts.append(text)

    def get_text(self, max_chars: int = _MAX_RESPONSE_CHARS) -> str:
        raw = " ".join(p if p == "\n" else p for p in self._parts)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        return raw.strip()[:max_chars]


def _html_to_text(html: str, max_chars: int = _MAX_RESPONSE_CHARS) -> tuple[str, str, str]:
    """Return (title, description, body_text) from an HTML string."""
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    return parser.title, parser.meta_description, parser.get_text(max_chars)


# ---------------------------------------------------------------------------
# Search result model
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""

    def as_text(self) -> str:
        lines = [f"[{self.title}]", self.url]
        if self.snippet:
            lines.append(self.snippet)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Web searcher — multi-provider with fallback
# ---------------------------------------------------------------------------

def _env_key(*names: str) -> str:
    for name in names:
        v = os.environ.get(name, "")
        if v:
            return v
    return ""


class WebSearcher:
    """Multi-provider web search with automatic fallback."""

    def __init__(self, vault=None) -> None:
        self._vault = vault

    def _get_key(self, service: str, *env_names: str) -> str:
        key = _env_key(*env_names)
        if not key and self._vault:
            try:
                key = self._vault.get(service, "api_key") or ""
            except Exception:
                pass
        return key

    # ---- provider implementations ------------------------------------------

    def _ddg_html(self, query: str, n: int) -> list[SearchResult]:
        resp = httpx.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": _UA, "Content-Type": "application/x-www-form-urlencoded"},
            timeout=_DEFAULT_TIMEOUT,
            follow_redirects=True,
        )
        results: list[SearchResult] = []
        titles_urls: list[tuple[str, str]] = []

        for m in re.finditer(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            resp.text, re.DOTALL,
        ):
            url = _html_lib.unescape(m.group(1))
            redirect = re.search(r"[?&]uddg=([^&]+)", url)
            if redirect:
                from urllib.parse import unquote
                url = unquote(redirect.group(1))
            title = _html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(2))).strip()
            if url.startswith("http"):
                titles_urls.append((title, url))

        snippets = [
            _html_lib.unescape(re.sub(r"<[^>]+>", "", m.group(1))).strip()
            for m in re.finditer(
                r'class="result__snippet"[^>]*>(.*?)</(?:a|div|span)>', resp.text, re.DOTALL
            )
        ]
        for i, (title, url) in enumerate(titles_urls[:n]):
            snippet = snippets[i] if i < len(snippets) else ""
            results.append(SearchResult(title=title, url=url, snippet=snippet, source="ddg"))
        return results

    def _ddg_api(self, query: str, n: int) -> list[SearchResult]:
        """DuckDuckGo Instant Answer API — great for factual queries."""
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers={"User-Agent": _UA},
            timeout=_DEFAULT_TIMEOUT,
        )
        data = resp.json()
        results: list[SearchResult] = []

        # Abstract (Wikipedia-style answer)
        if data.get("AbstractText") and data.get("AbstractURL"):
            results.append(SearchResult(
                title=data.get("Heading", query),
                url=data["AbstractURL"],
                snippet=data["AbstractText"][:400],
                source="ddg_api",
            ))

        # Direct answer
        if data.get("Answer") and not results:
            results.append(SearchResult(
                title="Direct Answer",
                url=data.get("AnswerURL", ""),
                snippet=str(data["Answer"])[:400],
                source="ddg_api",
            ))

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if len(results) >= n:
                break
            text = topic.get("Text", "")
            url = topic.get("FirstURL", "")
            if text and url:
                title_part = text.split(" - ")[0][:80]
                snippet = text[:300]
                results.append(SearchResult(title=title_part, url=url,
                                            snippet=snippet, source="ddg_api"))

        # Results section
        for r in data.get("Results", []):
            if len(results) >= n:
                break
            results.append(SearchResult(
                title=r.get("Text", "")[:80],
                url=r.get("FirstURL", ""),
                snippet=r.get("Text", "")[:300],
                source="ddg_api",
            ))

        return results[:n]

    def _brave(self, query: str, n: int, api_key: str) -> list[SearchResult]:
        resp = httpx.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(n, 20)},
            headers={"Accept": "application/json", "X-Subscription-Token": api_key},
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("web", {}).get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
                source="brave",
            ))
        return results[:n]

    def _serper(self, query: str, n: int, api_key: str) -> list[SearchResult]:
        resp = httpx.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": min(n, 20)},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=_DEFAULT_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for r in data.get("organic", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("link", ""),
                snippet=r.get("snippet", ""),
                source="serper",
            ))
        return results[:n]

    def _tavily(self, query: str, n: int, api_key: str) -> list[SearchResult]:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": query,
                  "max_results": min(n, 10), "include_answer": True},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        if data.get("answer"):
            results.append(SearchResult(
                title="AI Answer",
                url=data.get("query", ""),
                snippet=str(data["answer"])[:600],
                source="tavily",
            ))
        for r in data.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", "")[:400],
                source="tavily",
            ))
        return results[:n]

    # ---- public API --------------------------------------------------------

    def search(
        self,
        query: str,
        provider: str = "auto",
        num_results: int = 10,
    ) -> list[SearchResult]:
        """Search the web, trying providers in preference order."""

        if provider != "auto":
            return self._search_with(provider, query, num_results)

        # Try keyed providers first (better quality)
        for provider_name, key_env, key_vault in [
            ("brave",  "BRAVE_API_KEY", "brave"),
            ("serper", "SERPER_API_KEY", "serper"),
            ("tavily", "TAVILY_API_KEY", "tavily"),
        ]:
            api_key = self._get_key(key_vault, key_env)
            if api_key:
                try:
                    results = self._search_with(provider_name, query, num_results, api_key)
                    if results:
                        return results
                except Exception:
                    continue

        # Free fallbacks
        for p in ("ddg_html", "ddg_api"):
            try:
                results = self._search_with(p, query, num_results)
                if results:
                    return results
            except Exception:
                continue

        return []

    def _search_with(
        self, provider: str, query: str, n: int, api_key: str = ""
    ) -> list[SearchResult]:
        if provider == "ddg_html":
            return self._ddg_html(query, n)
        if provider == "ddg_api":
            return self._ddg_api(query, n)
        if provider == "brave":
            key = api_key or self._get_key("brave", "BRAVE_API_KEY")
            if not key:
                raise ValueError("BRAVE_API_KEY not set")
            return self._brave(query, n, key)
        if provider == "serper":
            key = api_key or self._get_key("serper", "SERPER_API_KEY")
            if not key:
                raise ValueError("SERPER_API_KEY not set")
            return self._serper(query, n, key)
        if provider == "tavily":
            key = api_key or self._get_key("tavily", "TAVILY_API_KEY")
            if not key:
                raise ValueError("TAVILY_API_KEY not set")
            return self._tavily(query, n, key)
        raise ValueError(f"unknown provider '{provider}'")

    def available_providers(self) -> list[str]:
        providers = ["ddg_html", "ddg_api"]
        for name, env in [("brave", "BRAVE_API_KEY"), ("serper", "SERPER_API_KEY"),
                           ("tavily", "TAVILY_API_KEY")]:
            if os.environ.get(env):
                providers.insert(0, name)
        return providers


# ---------------------------------------------------------------------------
# Web fetcher — URL → clean readable text
# ---------------------------------------------------------------------------

class WebFetcher:
    """Fetch any URL and return clean readable text."""

    def fetch(
        self,
        url: str,
        extract_text: bool = True,
        max_chars: int = _MAX_RESPONSE_CHARS,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> str:
        if not url.startswith(("http://", "https://")):
            return "ERROR: URL must start with http:// or https://"

        try:
            resp = httpx.get(
                url,
                headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
                timeout=timeout,
                follow_redirects=True,
            )
        except httpx.TimeoutException:
            return f"ERROR: request timed out after {timeout}s"
        except httpx.ConnectError as exc:
            return f"ERROR: connection failed — {exc}"
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

        content_type = resp.headers.get("content-type", "").lower()
        status = resp.status_code

        # JSON response — pretty-print
        if "application/json" in content_type:
            try:
                data = resp.json()
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if len(text) > max_chars:
                    text = text[:max_chars] + "\n... [truncated]"
                return f"HTTP {status} application/json\n{text}"
            except Exception:
                pass

        # Binary content — return metadata only
        if any(t in content_type for t in ("image/", "video/", "audio/", "application/pdf",
                                             "application/zip", "application/octet")):
            size = len(resp.content)
            return (
                f"HTTP {status}  content-type: {content_type}  "
                f"size: {size:,} bytes\n"
                f"(binary content — use a dedicated tool to process this file)"
            )

        # HTML — extract text
        raw = resp.text
        if extract_text and "html" in content_type:
            title, description, body = _html_to_text(raw, max_chars)
            parts = [f"HTTP {status}  {url}"]
            if title:
                parts.append(f"Title: {title}")
            if description:
                parts.append(f"Description: {description[:200]}")
            parts.append("")
            parts.append(body)
            return "\n".join(parts)

        # Plain text / other
        if len(raw) > max_chars:
            raw = raw[:max_chars] + "\n... [truncated]"
        return f"HTTP {status}  {content_type or 'text/plain'}\n{raw}"


# ---------------------------------------------------------------------------
# API profiles — named auth configs for external services
# ---------------------------------------------------------------------------

@dataclass
class ApiProfile:
    name: str
    base_url: str
    auth_type: str = "none"      # none | bearer | apikey | basic | header
    auth_header: str = ""        # e.g. "Authorization" or "X-API-Key"
    auth_prefix: str = ""        # e.g. "Bearer " or "Token "
    auth_env_var: str = ""       # env var name holding the secret
    default_headers: dict = field(default_factory=dict)
    notes: str = ""


class ApiProfileStore:
    """Persist API profiles as JSON. Secrets never stored here — use env vars or vault."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def set(self, profile: ApiProfile) -> None:
        with self._lock:
            data = self._load()
            data[profile.name] = {
                "base_url": profile.base_url,
                "auth_type": profile.auth_type,
                "auth_header": profile.auth_header,
                "auth_prefix": profile.auth_prefix,
                "auth_env_var": profile.auth_env_var,
                "default_headers": profile.default_headers,
                "notes": profile.notes,
            }
            self._save(data)

    def get(self, name: str) -> ApiProfile | None:
        with self._lock:
            data = self._load()
            raw = data.get(name)
            if raw is None:
                return None
            return ApiProfile(name=name, **raw)

    def list_all(self) -> list[ApiProfile]:
        with self._lock:
            data = self._load()
        return [ApiProfile(name=k, **v) for k, v in data.items()]

    def delete(self, name: str) -> bool:
        with self._lock:
            data = self._load()
            if name not in data:
                return False
            del data[name]
            self._save(data)
            return True


# ---------------------------------------------------------------------------
# API client — make authenticated HTTP calls
# ---------------------------------------------------------------------------

@dataclass
class ApiResponse:
    status: int
    content_type: str
    body: str
    elapsed_ms: float
    url: str

    def as_text(self) -> str:
        body_preview = self.body
        if len(body_preview) > 3000:
            body_preview = body_preview[:3000] + "\n... [truncated]"
        return (
            f"HTTP {self.status}  {self.content_type}  ({self.elapsed_ms:.0f} ms)\n"
            f"URL: {self.url}\n\n"
            + body_preview
        )


class ApiClient:
    """Make authenticated HTTP calls, resolving credentials from env or vault."""

    def __init__(self, profile_store: ApiProfileStore, vault=None) -> None:
        self.profiles = profile_store
        self._vault = vault

    def _resolve_secret(self, profile: ApiProfile) -> str:
        """Look up the auth secret — env var, then vault, then empty."""
        if profile.auth_env_var:
            val = os.environ.get(profile.auth_env_var, "")
            if val:
                return val
        if self._vault:
            try:
                val = self._vault.get(profile.name, "api_key") or ""
                if val:
                    return val
            except Exception:
                pass
        return ""

    def _build_headers(self, profile: ApiProfile, extra: dict) -> dict:
        headers = {"User-Agent": _UA, "Accept": "application/json"}
        headers.update(profile.default_headers)
        headers.update(extra)

        secret = self._resolve_secret(profile)
        if not secret:
            return headers

        if profile.auth_type == "bearer":
            headers[profile.auth_header or "Authorization"] = f"Bearer {secret}"
        elif profile.auth_type == "apikey":
            headers[profile.auth_header or "X-API-Key"] = (
                profile.auth_prefix + secret if profile.auth_prefix else secret
            )
        elif profile.auth_type == "header":
            headers[profile.auth_header] = profile.auth_prefix + secret
        elif profile.auth_type == "basic":
            import base64
            encoded = base64.b64encode(secret.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        return headers

    def call(
        self,
        method: str,
        url: str,
        profile_name: str = "",
        extra_headers: dict | None = None,
        body: dict | str | None = None,
        params: dict | None = None,
        timeout: float = 30.0,
    ) -> ApiResponse:
        method = method.upper()
        profile = self.profiles.get(profile_name) if profile_name else None

        # Resolve full URL (join with base_url if relative)
        if profile and not url.startswith(("http://", "https://")):
            url = urljoin(profile.base_url.rstrip("/") + "/", url.lstrip("/"))

        if not url.startswith(("http://", "https://")):
            raise ValueError(f"cannot resolve URL '{url}' without a profile base_url")

        headers = self._build_headers(profile, extra_headers or {}) if profile else {
            "User-Agent": _UA, **(extra_headers or {})
        }

        kwargs: dict = {"headers": headers, "timeout": timeout,
                        "follow_redirects": True, "params": params or {}}

        if body is not None:
            if isinstance(body, dict):
                kwargs["json"] = body
            else:
                kwargs["content"] = body.encode() if isinstance(body, str) else body

        t0 = time.time()
        try:
            resp = httpx.request(method, url, **kwargs)
        except Exception as exc:
            raise RuntimeError(f"HTTP {method} {url} failed: {exc}") from exc

        elapsed = (time.time() - t0) * 1000
        ct = resp.headers.get("content-type", "")

        try:
            if "application/json" in ct:
                body_text = json.dumps(resp.json(), indent=2, ensure_ascii=False)
            else:
                body_text = resp.text
        except Exception:
            body_text = resp.text

        return ApiResponse(
            status=resp.status_code,
            content_type=ct,
            body=body_text,
            elapsed_ms=elapsed,
            url=str(resp.url),
        )


# ---------------------------------------------------------------------------
# Real-time data hub
# ---------------------------------------------------------------------------

class RealtimeDataHub:
    """Free real-time data endpoints requiring no API keys by default."""

    # ---- Weather (Open-Meteo + Nominatim) -----------------------------------

    def weather(self, location: str) -> str:
        # Geocode
        try:
            geo_resp = httpx.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": location, "format": "json", "limit": "1"},
                headers={"User-Agent": _UA},
                timeout=10.0,
            )
            geo_data = geo_resp.json()
        except Exception as exc:
            return f"ERROR: geocoding failed for '{location}': {exc}"

        if not geo_data:
            return f"ERROR: location '{location}' not found"

        place = geo_data[0]
        lat, lon = place["lat"], place["lon"]
        display_name = place.get("display_name", location).split(",")[0]

        try:
            wx_resp = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat, "longitude": lon, "timezone": "auto",
                    "current": (
                        "temperature_2m,apparent_temperature,relative_humidity_2m,"
                        "precipitation,weather_code,wind_speed_10m,wind_direction_10m,"
                        "uv_index,is_day"
                    ),
                    "daily": (
                        "temperature_2m_max,temperature_2m_min,"
                        "weather_code,precipitation_probability_max,uv_index_max"
                    ),
                    "forecast_days": "4",
                },
                timeout=10.0,
            )
            wx_data = wx_resp.json()
        except Exception as exc:
            return f"ERROR: weather fetch failed: {exc}"

        cur = wx_data.get("current", {})
        daily = wx_data.get("daily", {})
        tz = wx_data.get("timezone_abbreviation", "")

        temp = cur.get("temperature_2m", "?")
        feels = cur.get("apparent_temperature", "?")
        humidity = cur.get("relative_humidity_2m", "?")
        wind = cur.get("wind_speed_10m", "?")
        code = cur.get("weather_code", 0)
        condition = _WMO_CODES.get(code, f"code {code}")
        uv = cur.get("uv_index", "?")
        precip = cur.get("precipitation", 0)

        lines = [
            f"Weather in {display_name}  ({tz})",
            f"  {condition}",
            f"  Temperature : {temp}°C  (feels like {feels}°C)",
            f"  Humidity    : {humidity}%",
            f"  Wind        : {wind} km/h",
            f"  UV index    : {uv}",
            f"  Precipitation: {precip} mm",
            "",
            "3-day forecast:",
        ]
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        lows = daily.get("temperature_2m_min", [])
        codes = daily.get("weather_code", [])
        rain_prob = daily.get("precipitation_probability_max", [])
        for i in range(1, min(4, len(dates))):
            day_condition = _WMO_CODES.get(codes[i] if i < len(codes) else 0, "")
            rp = f"{rain_prob[i]}% rain" if i < len(rain_prob) else ""
            lines.append(
                f"  {dates[i]}  {lows[i] if i < len(lows) else '?'}–"
                f"{highs[i] if i < len(highs) else '?'}°C  {day_condition}  {rp}"
            )
        return "\n".join(lines)

    # ---- Crypto (CoinGecko) ------------------------------------------------

    def crypto(self, symbols: str) -> str:
        """symbols: comma-separated coin names/ids, e.g. 'bitcoin,ethereum,solana'"""
        ids = [s.strip().lower().replace(" ", "-") for s in symbols.split(",") if s.strip()]
        if not ids:
            ids = ["bitcoin", "ethereum"]

        try:
            resp = httpx.get(
                "https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": ",".join(ids),
                    "vs_currencies": "usd,eur",
                    "include_24hr_change": "true",
                    "include_market_cap": "true",
                },
                headers={"User-Agent": _UA},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"ERROR: CoinGecko request failed: {exc}"

        if not data:
            return f"no price data for: {', '.join(ids)}"

        lines = ["Crypto prices (CoinGecko):"]
        for coin_id, prices in data.items():
            usd = prices.get("usd", "?")
            eur = prices.get("eur", "?")
            change = prices.get("usd_24h_change", None)
            cap = prices.get("usd_market_cap", None)
            change_str = ""
            if change is not None:
                sign = "+" if change >= 0 else ""
                change_str = f"  ({sign}{change:.1f}% 24h)"
            cap_str = f"  mktcap ${cap / 1e9:.1f}B" if cap else ""
            lines.append(
                f"  {coin_id.upper():15}  ${usd:>14,.2f}  €{eur:>12,.2f}{change_str}{cap_str}"
            )
        return "\n".join(lines)

    # ---- News (RSS) --------------------------------------------------------

    def news(self, category: str = "top", max_items: int = 8) -> str:
        feed_url = _RSS_FEEDS.get(category.lower(), _RSS_FEEDS["top"])

        try:
            resp = httpx.get(
                feed_url,
                headers={"User-Agent": _UA},
                timeout=10.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
        except Exception as exc:
            return f"ERROR: RSS fetch failed ({feed_url}): {exc}"

        try:
            root = _ET.fromstring(resp.content)
        except _ET.ParseError:
            # Atom feed or malformed XML fallback
            return f"ERROR: could not parse RSS from {feed_url}"

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        channel = root.find("channel")
        items = (channel or root).findall("item") or root.findall("atom:entry", ns)

        lines = [f"News ({category}) from {feed_url.split('/')[2]}:"]
        for item in items[:max_items]:
            title_el = item.find("title")
            link_el = item.find("link")
            desc_el = item.find("description") or item.find("atom:summary", ns)
            pub_el = item.find("pubDate") or item.find("atom:published", ns)

            title = (title_el.text or "").strip() if title_el is not None else ""
            link = (link_el.text or link_el.get("href", "")).strip() if link_el is not None else ""
            desc_raw = (desc_el.text or "").strip() if desc_el is not None else ""
            desc = re.sub(r"<[^>]+>", "", desc_raw).strip()[:160]
            pub = (pub_el.text or "").strip()[:25] if pub_el is not None else ""

            lines.append(f"\n  [{title}]")
            if pub:
                lines.append(f"  {pub}")
            if link:
                lines.append(f"  {link}")
            if desc:
                lines.append(f"  {desc}")

        return "\n".join(lines)

    # ---- Exchange rates (Frankfurter / ECB) --------------------------------

    def exchange_rate(self, from_currency: str, to_currencies: str = "USD,EUR,GBP,JPY") -> str:
        base = from_currency.upper().strip()
        targets = [t.strip().upper() for t in to_currencies.split(",") if t.strip()]

        try:
            resp = httpx.get(
                "https://api.frankfurter.dev/v1/latest",
                params={"from": base, "to": ",".join(targets)},
                headers={"User-Agent": _UA},
                timeout=10.0,
                follow_redirects=True,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"ERROR: exchange rate fetch failed: {exc}"

        rates = data.get("rates", {})
        date = data.get("date", "")
        if not rates:
            return f"no rate data for {base}"

        lines = [f"Exchange rates for {base}  (ECB, {date}):"]
        for currency, rate in sorted(rates.items()):
            lines.append(f"  1 {base} = {rate:.4f} {currency}")
        return "\n".join(lines)

    # ---- IP info ------------------------------------------------------------

    def ip_info(self, ip: str = "") -> str:
        url = f"https://ipinfo.io/{ip}/json" if ip else "https://ipinfo.io/json"
        try:
            resp = httpx.get(url, headers={"User-Agent": _UA}, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            return f"ERROR: IP lookup failed: {exc}"

        fields = ["ip", "hostname", "city", "region", "country", "org", "timezone"]
        lines = [f"IP info ({data.get('ip', ip or 'this machine')}):"]
        for f in fields:
            if f in data:
                lines.append(f"  {f:12} {data[f]}")
        return "\n".join(lines)

    # ---- Stocks (Alpha Vantage) ---------------------------------------------

    def stocks(self, symbols: str, api_key: str = "") -> str:
        key = api_key or os.environ.get("ALPHA_VANTAGE_KEY", "")
        if not key:
            return (
                "Stock data requires a free Alpha Vantage key.\n"
                "Get one at https://www.alphavantage.co/support/#api-key\n"
                "Then: export ALPHA_VANTAGE_KEY=your_key"
            )
        syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        lines = []
        for sym in syms[:5]:
            try:
                resp = httpx.get(
                    "https://www.alphavantage.co/query",
                    params={"function": "GLOBAL_QUOTE", "symbol": sym, "apikey": key},
                    headers={"User-Agent": _UA},
                    timeout=10.0,
                )
                data = resp.json().get("Global Quote", {})
                price = data.get("05. price", "?")
                change = data.get("10. change percent", "?")
                lines.append(f"  {sym:8}  ${price}  {change}")
            except Exception as exc:
                lines.append(f"  {sym}: ERROR {exc}")
        return "Stock quotes (Alpha Vantage):\n" + "\n".join(lines) if lines else "no data"


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

def register_internet_tools(
    registry,
    config_home: Path,
    vault=None,
) -> None:
    from .tools import Tier, Tool

    searcher = WebSearcher(vault=vault)
    fetcher = WebFetcher()
    profile_store = ApiProfileStore(config_home / "api_profiles.json")
    api_client = ApiClient(profile_store, vault=vault)
    rt = RealtimeDataHub()

    # ---- Upgraded web_search (replaces built-in) ----------------------------

    def web_search(
        query: str,
        provider: str = "auto",
        num_results: str = "10",
    ) -> str:
        """Multi-provider web search with automatic fallback chain."""
        try:
            n = max(1, min(20, int(num_results)))
        except ValueError:
            n = 10
        results = searcher.search(query, provider=provider, num_results=n)
        if not results:
            return f"no results for '{query}'"
        providers_used = list({r.source for r in results})
        header = f"{len(results)} result(s) via {', '.join(providers_used)}:\n"
        return header + "\n\n".join(r.as_text() for r in results)

    # ---- Upgraded web_fetch (replaces web_get) ------------------------------

    def web_fetch(
        url: str,
        extract_text: str = "true",
        max_chars: str = "8000",
    ) -> str:
        """Fetch a URL and return clean readable text (HTML stripped by default)."""
        do_extract = extract_text.lower() not in ("false", "0", "no")
        try:
            chars = int(max_chars)
        except ValueError:
            chars = 8000
        return fetcher.fetch(url, extract_text=do_extract, max_chars=chars)

    # ---- API calls ----------------------------------------------------------

    def api_call(
        method: str,
        url: str,
        body_json: str = "",
        headers_json: str = "",
        params_json: str = "",
        auth_profile: str = "",
        timeout: str = "30",
    ) -> str:
        """Make an authenticated HTTP API call (GET/POST/PUT/PATCH/DELETE)."""
        body = None
        if body_json.strip():
            try:
                body = json.loads(body_json)
            except json.JSONDecodeError as exc:
                return f"ERROR: invalid body JSON — {exc}"

        extra_headers: dict = {}
        if headers_json.strip():
            try:
                extra_headers = json.loads(headers_json)
            except json.JSONDecodeError as exc:
                return f"ERROR: invalid headers JSON — {exc}"

        params: dict = {}
        if params_json.strip():
            try:
                params = json.loads(params_json)
            except json.JSONDecodeError as exc:
                return f"ERROR: invalid params JSON — {exc}"

        try:
            t = float(timeout)
        except ValueError:
            t = 30.0

        try:
            response = api_client.call(
                method=method, url=url, profile_name=auth_profile,
                extra_headers=extra_headers, body=body, params=params, timeout=t,
            )
            return response.as_text()
        except Exception as exc:
            return f"ERROR: {type(exc).__name__}: {exc}"

    def api_profile_set(
        name: str,
        base_url: str,
        auth_type: str = "none",
        auth_header: str = "",
        auth_prefix: str = "",
        auth_env_var: str = "",
        notes: str = "",
    ) -> str:
        """Save a named API profile. The auth secret goes in an env var — never stored here."""
        profile = ApiProfile(
            name=name, base_url=base_url, auth_type=auth_type,
            auth_header=auth_header, auth_prefix=auth_prefix,
            auth_env_var=auth_env_var, notes=notes,
        )
        profile_store.set(profile)
        secret_hint = f" (secret from env var {auth_env_var})" if auth_env_var else ""
        return f"API profile '{name}' saved → {base_url}{secret_hint}"

    def api_profile_list() -> str:
        """List all saved API profiles."""
        profiles = profile_store.list_all()
        if not profiles:
            return "no API profiles saved. Use api_profile_set to add one."
        lines = [f"{len(profiles)} profile(s):"]
        for p in profiles:
            secret = f"  secret: ${p.auth_env_var}" if p.auth_env_var else ""
            lines.append(
                f"  {p.name:<20} {p.base_url}  auth={p.auth_type}{secret}"
            )
        return "\n".join(lines)

    def api_profile_delete(name: str) -> str:
        """Delete a named API profile."""
        return (
            f"profile '{name}' deleted."
            if profile_store.delete(name)
            else f"no profile named '{name}'"
        )

    # ---- Real-time data tools -----------------------------------------------

    def realtime_weather(location: str) -> str:
        """Get current weather and 3-day forecast for any location (Open-Meteo, free)."""
        return rt.weather(location)

    def realtime_crypto(symbols: str = "bitcoin,ethereum") -> str:
        """Get live crypto prices in USD/EUR with 24h change (CoinGecko, free)."""
        return rt.crypto(symbols)

    def realtime_news(category: str = "top", max_items: str = "8") -> str:
        """Fetch latest news headlines via RSS.
        Categories: top, tech, world, business, science, bbc, reuters, hn, techcrunch, wired
        """
        try:
            n = int(max_items)
        except ValueError:
            n = 8
        return rt.news(category, max_items=n)

    def realtime_exchange(
        from_currency: str = "USD",
        to_currencies: str = "EUR,GBP,JPY,AUD,CAD",
    ) -> str:
        """Get live exchange rates from the European Central Bank (free)."""
        return rt.exchange_rate(from_currency, to_currencies)

    def realtime_ip(ip: str = "") -> str:
        """Look up IP geolocation and network info (ipinfo.io, free)."""
        return rt.ip_info(ip)

    def realtime_stocks(symbols: str) -> str:
        """Get stock quotes (requires free ALPHA_VANTAGE_KEY env var)."""
        return rt.stocks(symbols)

    def search_providers() -> str:
        """List available search providers and which API keys are configured."""
        available = searcher.available_providers()
        lines = ["Search providers:"]
        all_providers = {
            "ddg_html":  ("DuckDuckGo HTML scrape", "no key needed"),
            "ddg_api":   ("DuckDuckGo Instant Answer", "no key needed"),
            "brave":     ("Brave Search", "set BRAVE_API_KEY"),
            "serper":    ("Serper.dev (Google)", "set SERPER_API_KEY"),
            "tavily":    ("Tavily AI Search", "set TAVILY_API_KEY"),
        }
        for pid, (label, note) in all_providers.items():
            status = "✓ active" if pid in available else f"  needs: {note}"
            lines.append(f"  {pid:<12} {label:<28} {status}")
        return "\n".join(lines)

    # ---- Register all tools -------------------------------------------------

    # Upgraded replacements of built-in tools (same names — they overwrite)
    registry.register(Tool(
        "web_search",
        "Search the web using a multi-provider fallback chain "
        "(Brave → Serper → Tavily → DuckDuckGo). Returns titles, URLs, and snippets.",
        {
            "query":       "search query",
            "provider":    "(optional) auto | brave | serper | tavily | ddg_html | ddg_api",
            "num_results": "(optional) number of results, 1–20 (default: 10)",
        },
        web_search,
    ))
    registry.register(Tool(
        "web_fetch",
        "Fetch any URL and return clean readable text. HTML is stripped to prose by default. "
        "JSON APIs are pretty-printed. Binary content returns metadata only.",
        {
            "url":          "full URL to fetch (must start with http:// or https://)",
            "extract_text": "(optional) true (default) to strip HTML, false for raw",
            "max_chars":    "(optional) max characters to return (default: 8000)",
        },
        web_fetch,
    ))

    registry.register(Tool(
        "api_call",
        "Make an authenticated HTTP API call (GET/POST/PUT/PATCH/DELETE) to any endpoint. "
        "Supports named API profiles for auth. Returns status, headers, and body.",
        {
            "method":      "HTTP method: GET POST PUT PATCH DELETE HEAD",
            "url":         "full URL or path relative to auth_profile base_url",
            "body_json":   "(optional) JSON request body",
            "headers_json":"(optional) JSON object of additional headers",
            "params_json": "(optional) JSON object of query parameters",
            "auth_profile":"(optional) named profile from api_profile_set",
            "timeout":     "(optional) timeout in seconds (default: 30)",
        },
        api_call, tier=Tier.CONFIRM,
    ))
    registry.register(Tool(
        "api_profile_set",
        "Save a named API profile (base URL + auth config). The actual secret goes in an "
        "environment variable referenced by auth_env_var — it is never stored in the profile.",
        {
            "name":        "profile name (e.g. 'github', 'openai', 'my_crm')",
            "base_url":    "API base URL (e.g. https://api.github.com)",
            "auth_type":   "none | bearer | apikey | basic | header",
            "auth_header": "(optional) header name to set (e.g. Authorization, X-API-Key)",
            "auth_prefix": "(optional) prefix for the value (e.g. 'Bearer ' or 'Token ')",
            "auth_env_var":"(optional) env var that holds the secret (e.g. GITHUB_TOKEN)",
            "notes":       "(optional) human notes about this profile",
        },
        api_profile_set,
    ))
    registry.register(Tool(
        "api_profile_list",
        "List all saved API profiles with their base URLs and auth configuration.",
        {},
        api_profile_list,
    ))
    registry.register(Tool(
        "api_profile_delete",
        "Delete a named API profile.",
        {"name": "profile name to delete"},
        api_profile_delete,
    ))
    registry.register(Tool(
        "realtime_weather",
        "Get current weather and 3-day forecast for any city or location worldwide. "
        "Uses Open-Meteo (free, no API key required).",
        {"location": "city, country, or place name (e.g. 'London', 'Tokyo, Japan', '48.8566,2.3522')"},
        realtime_weather,
    ))
    registry.register(Tool(
        "realtime_crypto",
        "Get live cryptocurrency prices in USD/EUR with 24h change and market cap. "
        "Uses CoinGecko public API (free, no key required).",
        {"symbols": "comma-separated coin names (e.g. 'bitcoin,ethereum,solana')"},
        realtime_crypto,
    ))
    registry.register(Tool(
        "realtime_news",
        "Fetch the latest news headlines via RSS feeds. "
        "Categories: top, tech, world, business, science, bbc, reuters, hn, techcrunch, wired",
        {
            "category":  "(optional) news category/source (default: top)",
            "max_items": "(optional) number of headlines to return (default: 8)",
        },
        realtime_news,
    ))
    registry.register(Tool(
        "realtime_exchange",
        "Get live foreign exchange rates sourced from the European Central Bank (free).",
        {
            "from_currency": "base currency code (e.g. USD, GBP, EUR)",
            "to_currencies": "(optional) comma-separated target currencies (default: EUR,GBP,JPY,AUD,CAD)",
        },
        realtime_exchange,
    ))
    registry.register(Tool(
        "realtime_ip",
        "Look up geolocation and network info for an IP address (ipinfo.io, free).",
        {"ip": "(optional) IP address to look up; leave empty to look up this machine's IP"},
        realtime_ip,
    ))
    registry.register(Tool(
        "realtime_stocks",
        "Get live stock quotes. Requires a free Alpha Vantage API key "
        "(set ALPHA_VANTAGE_KEY env var). Get a key at alphavantage.co.",
        {"symbols": "comma-separated stock ticker symbols (e.g. 'AAPL,MSFT,NVDA')"},
        realtime_stocks,
    ))
    registry.register(Tool(
        "search_providers",
        "List all available search providers and which API keys are configured.",
        {},
        search_providers,
    ))
