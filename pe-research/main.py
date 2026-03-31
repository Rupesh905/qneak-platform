"""
Private Equity Research Engine — v8.0
FastAPI + LangGraph ReAct + Gemini 2.5 Pro + Exa + TinyFish + Companies House
v8: Intelligent researcher agent — thinks freely, no rigid phases
"""

import os
import json
import logging
import asyncio
import uuid
import re
import mimetypes
import requests as req_lib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from google import genai as google_genai
from exa_py import Exa
from tinyfish import TinyFish, BrowserProfile, ProxyConfig, ProxyCountryCode, CompleteEvent, RunStatus
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("pe_research")

# ── Clients ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")
EXA_API_KEY         = os.getenv("EXA_API_KEY", "")
TINYFISH_API_KEY    = os.getenv("TINYFISH_API_KEY", "")
COMPANIES_HOUSE_KEY = os.getenv("COMPANIES_HOUSE_API_KEY", "")
SUPABASE_URL        = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "research-reports")
SUPABASE_JOBS_TABLE = os.getenv("SUPABASE_JOBS_TABLE", "research_jobs")

genai_client    = google_genai.Client(api_key=GOOGLE_API_KEY)
exa_client      = Exa(api_key=EXA_API_KEY) if EXA_API_KEY else None
tinyfish_client = TinyFish(api_key=TINYFISH_API_KEY) if TINYFISH_API_KEY else None

# ── Tuning ────────────────────────────────────────────────────────────────────
TINYFISH_TIMEOUT     = 180
MAX_STREAM_EVENTS    = 150
MAX_STEPS_PER_EXTR   = 600
MAX_AGENT_ITERATIONS = 60  # Increased: 14 phases with multiple searches each

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="PE Research Engine", version="8.0.0", docs_url=None, redoc_url=None)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

os.makedirs("reports", exist_ok=True)
os.makedirs("static",  exist_ok=True)

jobs: Dict[str, Dict] = {}


def supabase_enabled() -> bool:
    return bool(SUPABASE_URL and SUPABASE_SERVICE_KEY)


def supabase_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    }
    if extra:
        headers.update(extra)
    return headers


def serialize_job(job: Dict[str, Any]) -> Dict[str, Any]:
    result = job.get("result") or {}
    return {
        "id": job["job_id"],
        "company_name": job.get("company"),
        "country_code": job.get("country_code"),
        "website_url": job.get("website_url"),
        "status": job.get("status"),
        "progress": job.get("progress", 0),
        "message": job.get("message"),
        "error": job.get("error"),
        "result": result,
        "report_storage_path": result.get("report_storage_path"),
        "created_at": job.get("created"),
    }


def hydrate_job(row: Dict[str, Any]) -> Dict[str, Any]:
    result = row.get("result") or {}
    return {
        "job_id": row.get("id"),
        "status": row.get("status"),
        "progress": row.get("progress", 0),
        "message": row.get("message"),
        "result": result,
        "error": row.get("error"),
        "company": row.get("company_name"),
        "country_code": row.get("country_code"),
        "website_url": row.get("website_url"),
        "created": row.get("created_at"),
    }


def persist_job(job: Dict[str, Any]) -> None:
    if not supabase_enabled():
        return
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_JOBS_TABLE}"
        headers = supabase_headers({
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        })
        response = req_lib.post(url, headers=headers, params={"on_conflict": "id"}, json=serialize_job(job), timeout=15)
        response.raise_for_status()
    except Exception as exc:
        logger.warning(f"Supabase job persist failed for {job.get('job_id')}: {exc}")


def fetch_job_from_supabase(job_id: str) -> Optional[Dict[str, Any]]:
    if not supabase_enabled():
        return None
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_JOBS_TABLE}"
        response = req_lib.get(
            url,
            headers=supabase_headers({"Accept": "application/json"}),
            params={"id": f"eq.{job_id}", "select": "*", "limit": 1},
            timeout=15,
        )
        response.raise_for_status()
        rows = response.json()
        if rows:
            return hydrate_job(rows[0])
    except Exception as exc:
        logger.warning(f"Supabase job fetch failed for {job_id}: {exc}")
    return None


def list_jobs_from_supabase() -> Optional[List[Dict[str, Any]]]:
    if not supabase_enabled():
        return None
    try:
        url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_JOBS_TABLE}"
        response = req_lib.get(
            url,
            headers=supabase_headers({"Accept": "application/json"}),
            params={"select": "*", "order": "created_at.desc"},
            timeout=20,
        )
        response.raise_for_status()
        return [hydrate_job(row) for row in response.json()]
    except Exception as exc:
        logger.warning(f"Supabase job list failed: {exc}")
        return None


def upload_file_to_supabase(local_path: str, object_name: str) -> Optional[str]:
    if not supabase_enabled():
        return None
    try:
        content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
        with open(local_path, "rb") as handle:
            response = req_lib.post(
                f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{object_name}",
                headers=supabase_headers({
                    "Content-Type": content_type,
                    "x-upsert": "true",
                }),
                data=handle.read(),
                timeout=60,
            )
        response.raise_for_status()
        return object_name
    except Exception as exc:
        logger.warning(f"Supabase upload failed for {local_path}: {exc}")
        return None


def download_file_from_supabase(object_name: str) -> Optional[bytes]:
    if not (supabase_enabled() and object_name):
        return None
    try:
        response = req_lib.get(
            f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_STORAGE_BUCKET}/{object_name}",
            headers=supabase_headers(),
            timeout=60,
        )
        response.raise_for_status()
        return response.content
    except Exception as exc:
        logger.warning(f"Supabase download failed for {object_name}: {exc}")
        return None


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    company_name: str
    country_code: str = "auto"
    website_url:  str = ""  # Optional: if provided, anchors research to this specific company

class CompanyDiscoveryRequest(BaseModel):
    company_name: str
    country_code: str = "auto"

class CompanyNameInput(BaseModel):
    company_name: str = Field(description="The company name to look up")

class SearchInput(BaseModel):
    query:       str = Field(description="Natural language search query — be specific and descriptive")
    search_type: str = Field(
        default="general",
        description="""Type of search — pick the most specific one:
'funding'   — funding rounds, investors, amounts, Series A/B/C
'company'   — company pages, LinkedIn company profiles, business info
'people'    — executives, founders, LinkedIn profiles
'news'      — recent news, press releases, announcements
'general'   — anything that doesn't fit above"""
    )

class UrlInput(BaseModel):
    url: str = Field(description="The full URL to extract data from")

class TargetedExtractInput(BaseModel):
    url:             str = Field(description="The full URL to extract data from")
    extraction_goal: str = Field(
        default="general",
        description="""What specific data to extract from this page:
'company_overview'  — description, founding year, HQ, mission, key stats
'business_model'    — revenue streams, pricing tiers, fees, B2B vs B2C, partnerships
'products'          — product names, features, API offerings, integrations, white-label capabilities
'funding'           — funding rounds, amounts, investor names, dates, use of funds
'management'        — executive names, titles, backgrounds, appointment dates
'legal_entity'      — legal name, company number, registered address, copyright footer text
'geography'         — office locations, markets served, regulatory jurisdictions
'operations'        — metrics, volumes, employee count, growth indicators
'partnerships'      — named partners, customer logos, case studies, integration partners
'general'           — extract everything available"""
    )

class DeepSearchInput(BaseModel):
    query:       str = Field(description="Natural language search query")
    search_type: str = Field(default="general", description="Type: funding/company/people/news/general")
    domain:      str = Field(default="", description="Limit to specific domain e.g. techcrunch.com")


# ── Helpers ───────────────────────────────────────────────────────────────────
def gemini_json(prompt: str) -> Any:
    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        logger.error(f"Gemini JSON parse error: {e}")
        return {}


def filter_relevant_sources(sources: List[str], company_name: str, website_url: str = "") -> List[str]:
    """
    Filter out irrelevant URLs from the source list.
    Keeps: company website pages, registry pages, news articles about the company,
           relevant industry sources.
    Removes: random LinkedIn profiles, Reddit threads, unrelated domains,
             generic articles not about the target company.
    """
    from urllib.parse import urlparse

    # Extract company domain if website provided
    company_domain = ""
    if website_url:
        company_domain = urlparse(website_url).netloc.lower().replace("www.", "")

    # Normalize company name for matching
    name_lower = company_name.lower().strip()
    name_parts = [p for p in name_lower.split() if len(p) > 2]  # words > 2 chars

    # Always-relevant domains (registries, known data sources)
    trusted_domains = {
        "company-information.service.gov.uk", "find-and-update.company-information.service.gov.uk",
        "opencorporates.com", "crunchbase.com", "pitchbook.com", "cbinsights.com",
        "techcrunch.com", "sifted.eu", "finsmes.com", "eu-startups.com",
        "uktech.news", "finextra.com", "businesswire.com", "prnewswire.com",
        "growjo.com", "trustpilot.co.uk", "trustpilot.com",
        "startupmag.co.uk", "seedlegals.com", "vestbee.com",
        "magnitt.com", "wamda.com", "zawya.com",
    }

    # Always-irrelevant patterns
    noise_patterns = [
        "linkedin.com/in/",       # Individual LinkedIn profiles (not company pages)
        "reddit.com",             # Reddit threads
        "facebook.com",           # Social media
        "twitter.com", "x.com",   # Social media
        "youtube.com",            # Video
        "wikipedia.org",          # Encyclopedia
        "instagram.com",          # Social media
        "pinterest.com",          # Social media
        "tiktok.com",             # Social media
    ]

    # LinkedIn company pages ARE relevant
    linkedin_company_ok = "linkedin.com/company/"

    filtered = []
    for url in sources:
        url_lower = url.lower()
        parsed = urlparse(url_lower)
        domain = parsed.netloc.replace("www.", "")

        # Skip noise patterns (but allow LinkedIn company pages)
        if any(pat in url_lower for pat in noise_patterns):
            if linkedin_company_ok not in url_lower:
                continue

        # Always keep trusted domains
        if any(td in domain for td in trusted_domains):
            filtered.append(url)
            continue

        # Always keep company's own domain
        if company_domain and company_domain in domain:
            filtered.append(url)
            continue

        # Keep if URL or domain contains company name parts
        url_text = (parsed.netloc + parsed.path).lower()
        name_match = any(part in url_text for part in name_parts if len(part) > 3)
        if name_match:
            filtered.append(url)
            continue

        # Keep industry/news domains that likely have articles about the company
        news_domains = [
            "propertyinvestortoday", "theintermediary", "propertyindustryeye",
            "proptech", "inventoryhive", "propflo", "residently",
        ]
        if any(nd in domain for nd in news_domains):
            filtered.append(url)
            continue

        # Skip everything else — it's probably noise
        logger.debug(f"Source filtered out: {url[:80]}")

    logger.info(f"Source filter: {len(sources)} → {len(filtered)} ({len(sources) - len(filtered)} removed)")
    return filtered


# ── Language-Aware Search Terms ──────────────────────────────────────────────
# English-primary countries where translation is not needed
ENGLISH_PRIMARY = {"gb", "us", "ca", "au", "ie", "sg", "in", "bh", "ae", "qa", "kw", "om"}

# Country → primary language mapping for translation
COUNTRY_LANGUAGES = {
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "nl": "Dutch",
    "pt": "Portuguese",
    "br": "Portuguese",
    "jp": "Japanese",
    "kr": "Korean",
    "cn": "Chinese (Simplified)",
    "tw": "Chinese (Traditional)",
    "sa": "Arabic",
    "eg": "Arabic",
    "tr": "Turkish",
    "ru": "Russian",
    "pl": "Polish",
    "se": "Swedish",
    "no": "Norwegian",
    "dk": "Danish",
    "fi": "Finnish",
    "cz": "Czech",
    "at": "German",
    "ch": "German",
    "be": "French",
    "mx": "Spanish",
    "ar": "Spanish",
    "cl": "Spanish",
    "co": "Spanish",
    "th": "Thai",
    "vn": "Vietnamese",
    "id": "Indonesian",
    "my": "Malay",
    "ph": "Filipino",
}


def get_localized_search_terms(company_name: str, country_code: str) -> Dict[str, str]:
    """
    Generate localized search terms for non-English countries.
    Uses Gemini to translate key PE research phrases into the target language.
    Returns a dict of translated search phrases the agent can use.
    Called ONCE per research job — cheap and fast.
    """
    if country_code in ENGLISH_PRIMARY or country_code not in COUNTRY_LANGUAGES:
        return {}  # No translation needed

    language = COUNTRY_LANGUAGES[country_code]

    try:
        result = gemini_json(f"""
Translate these PE research search phrases into {language} for researching "{company_name}".
Keep the company name as-is (do not translate it). Return ONLY JSON.

{{
  "funding_query": "{company_name} funding round investors raised" translated to {language},
  "business_model_query": "{company_name} business model revenue pricing" translated to {language},
  "products_query": "{company_name} products services platform features" translated to {language},
  "news_query": "{company_name} news announcement 2024 2025" translated to {language},
  "management_query": "{company_name} CEO founder management team" translated to {language},
  "partnerships_query": "{company_name} partners customers collaboration" translated to {language},
  "competitors_query": "{company_name} competitors alternatives market" translated to {language},
  "operations_query": "{company_name} growth revenue employees metrics" translated to {language},
  "language": "{language}"
}}""")
        if result and result.get("funding_query"):
            logger.info(f"Localized search terms generated for {language}: {list(result.keys())}")
            return result
    except Exception as e:
        logger.warning(f"Search term translation failed for {language}: {e}")

    return {}


# ── Country Registry ──────────────────────────────────────────────────────────
COUNTRY_REGISTRY = {
    "gb": {"label": "United Kingdom", "registries": ["companieshouse.gov.uk"], "use_ch": True},
    "us": {"label": "United States",  "registries": ["sec.gov"],               "use_ch": False},
    "sa": {"label": "Saudi Arabia",   "registries": ["mc.gov.sa"],             "use_ch": False},
    "ae": {"label": "UAE",            "registries": ["dc.gov.ae"],             "use_ch": False},
    "qa": {"label": "Qatar",          "registries": ["mec.gov.qa"],            "use_ch": False},
    "kw": {"label": "Kuwait",         "registries": ["moci.gov.kw"],           "use_ch": False},
    "bh": {"label": "Bahrain",        "registries": ["sijilat.com.bh"],        "use_ch": False},
    "om": {"label": "Oman",           "registries": ["mocioman.gov.om"],       "use_ch": False},
    "de": {"label": "Germany",        "registries": ["handelsregister.de"],    "use_ch": False},
    "sg": {"label": "Singapore",      "registries": ["acra.gov.sg"],           "use_ch": False},
    "au": {"label": "Australia",      "registries": ["asic.gov.au"],           "use_ch": False},
    "in": {"label": "India",          "registries": ["mca.gov.in"],            "use_ch": False},
    "ie": {"label": "Ireland",        "registries": ["cro.ie"],                "use_ch": False},
    "ca": {"label": "Canada",         "registries": ["canadasbusinessregistries.ca"], "use_ch": False},
}

SUFFIX_HINTS = {
    "ltd": "gb", "plc": "gb", "llp": "gb",
    "inc": "us", "corp": "us", "llc": "us",
    "gmbh": "de", "ag": "de",
    "pty": "au", "pte": "sg", "pvt": "in",
    "jsc": "sa", "wll": "bh", "saoc": "om",
    "qpsc": "qa", "kscp": "kw",
}


# ── Jurisdiction Agent ────────────────────────────────────────────────────────
class JurisdictionAgent:
    def detect(self, company_name: str, country_code: str) -> Dict[str, Any]:
        if country_code != "auto" and country_code in COUNTRY_REGISTRY:
            info = COUNTRY_REGISTRY[country_code]
            return {"country_code": country_code, "label": info["label"],
                    "registries": info["registries"], "use_ch": info["use_ch"],
                    "method": "user_selected"}

        name_lower = company_name.lower()
        for suffix, code in SUFFIX_HINTS.items():
            if name_lower.endswith(suffix) or f" {suffix} " in name_lower:
                info = COUNTRY_REGISTRY.get(code, COUNTRY_REGISTRY["gb"])
                return {"country_code": code, "label": info["label"],
                        "registries": info["registries"], "use_ch": info["use_ch"],
                        "method": "suffix_hint"}

        try:
            result = gemini_json(f"""
Determine country of registration for: "{company_name}"
Available codes: {list(COUNTRY_REGISTRY.keys())}
Return ONLY: {{"country_code": "", "reason": ""}}""")
            code = result.get("country_code", "gb")
            if code not in COUNTRY_REGISTRY:
                code = "gb"
            info = COUNTRY_REGISTRY[code]
            return {"country_code": code, "label": info["label"],
                    "registries": info["registries"], "use_ch": info["use_ch"],
                    "method": "gemini_detection", "reason": result.get("reason")}
        except Exception:
            pass

        info = COUNTRY_REGISTRY["gb"]
        return {"country_code": "gb", "label": info["label"],
                "registries": info["registries"], "use_ch": info["use_ch"],
                "method": "fallback"}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL FUNCTIONS — Enhanced with domain-aware extraction
# ══════════════════════════════════════════════════════════════════════════════

def _companies_house_sync(company_name: str) -> str:
    """UK Companies House registry. Only works for UK companies."""
    if not COMPANIES_HOUSE_KEY:
        return json.dumps({"error": "Companies House API key not configured",
                           "note": "This tool only works for UK-registered companies. For other countries, use search_web and find_related_entities instead."})

    BASE = "https://api.company-information.service.gov.uk"
    auth = (COMPANIES_HOUSE_KEY, "")

    try:
        search = req_lib.get(
            f"{BASE}/search/companies",
            params={"q": company_name, "items_per_page": 10},
            auth=auth, timeout=15,
        )
        items = search.json().get("items", [])
        if not items:
            return json.dumps({"found": False, "message": f"No record for '{company_name}'"})

        best_match = items[0]
        for item in items:
            if company_name.lower() in item.get("title", "").lower():
                best_match = item
                break

        number = best_match["company_number"]
        profile_r  = req_lib.get(f"{BASE}/company/{number}", auth=auth, timeout=15)
        officers_r = req_lib.get(f"{BASE}/company/{number}/officers", auth=auth, timeout=15)
        filings_r  = req_lib.get(f"{BASE}/company/{number}/filing-history",
                                  params={"items_per_page": 10}, auth=auth, timeout=15)
        charges_r  = req_lib.get(f"{BASE}/company/{number}/charges", auth=auth, timeout=15)
        pscs_r     = req_lib.get(f"{BASE}/company/{number}/persons-with-significant-control",
                                  auth=auth, timeout=15)

        profile  = profile_r.json()
        officers = officers_r.json()
        filings  = filings_r.json()
        charges  = charges_r.json() if charges_r.status_code == 200 else {}
        pscs     = pscs_r.json() if pscs_r.status_code == 200 else {}

        # Separate active directors from resigned
        active_directors = []
        resigned_directors = []
        all_officers = []
        for o in officers.get("items", []):
            officer_info = {
                "name": o.get("name", ""),
                "role": o.get("officer_role", ""),
                "appointed": o.get("appointed_on", ""),
                "nationality": o.get("nationality", ""),
                "occupation": o.get("occupation", ""),
                "country_of_residence": o.get("country_of_residence", ""),
            }
            if o.get("resigned_on"):
                officer_info["resigned"] = o["resigned_on"]
                resigned_directors.append(officer_info)
            else:
                active_directors.append(officer_info)
            all_officers.append(officer_info)

        # PSCs (persons with significant control) — reveals ownership
        psc_list = []
        for p in pscs.get("items", []):
            psc_list.append({
                "name": p.get("name", p.get("name_elements", {}).get("surname", "")),
                "kind": p.get("kind", ""),
                "natures_of_control": p.get("natures_of_control", []),
                "notified_on": p.get("notified_on", ""),
                "nationality": p.get("nationality", ""),
                "country_of_residence": p.get("country_of_residence", ""),
            })

        addr = profile.get("registered_office_address", {})
        address_str = ", ".join(filter(None, [
            addr.get("address_line_1"), addr.get("address_line_2"),
            addr.get("locality"), addr.get("region"), addr.get("postal_code"),
            addr.get("country"),
        ]))

        # Accounts info
        accounts = profile.get("accounts", {})
        confirmation_statement = profile.get("confirmation_statement", {})

        result = {
            "source": "Companies House (authoritative UK registry)",
            "legal_name": profile.get("company_name"),
            "company_number": profile.get("company_number"),
            "status": profile.get("company_status"),
            "company_type": profile.get("type"),
            "incorporation_date": profile.get("date_of_creation"),
            "dissolution_date": profile.get("date_of_cessation"),
            "registered_address": address_str,
            "sic_codes": profile.get("sic_codes", []),
            "active_directors": active_directors,
            "resigned_directors": resigned_directors[:5],
            "persons_with_significant_control": psc_list,
            "accounts": {
                "overdue": accounts.get("overdue", False),
                "next_due": accounts.get("next_due", ""),
                "last_made_up_to": accounts.get("last_accounts", {}).get("made_up_to", ""),
                "type": accounts.get("last_accounts", {}).get("type", ""),
            },
            "confirmation_statement_overdue": confirmation_statement.get("overdue", False),
            "charges_count": len(charges.get("items", [])),
            "active_charges": [
                {
                    "classification": c.get("classification", {}).get("description", ""),
                    "created_on": c.get("created_on", ""),
                    "delivered_on": c.get("delivered_on", ""),
                    "status": c.get("status", ""),
                    "persons_entitled": [p.get("name", "") for p in c.get("persons_entitled", [])],
                }
                for c in charges.get("items", [])[:5]
                if c.get("status") != "fully-satisfied"
            ],
            "recent_filings": [
                {"date": f.get("date"), "type": f.get("type"), "description": f.get("description")}
                for f in filings.get("items", [])[:10]
            ],
            "all_search_results": [
                {"name": item.get("title"), "number": item.get("company_number"),
                 "status": item.get("company_status"), "created": item.get("date_of_creation"),
                 "address": item.get("address_snippet", "")}
                for item in items[:5]
            ],
            "ANALYSIS_NOTES": [
                "If status is 'dormant' or 'dissolved', the brand may trade via a DIFFERENT legal entity.",
                "Check PSCs (persons with significant control) for ownership structure.",
                "Active charges indicate secured debt — note the lender names.",
                "If multiple search results, identify which is the ACTIVE trading entity.",
                "Cross-reference directors with LinkedIn data from people search.",
            ],
        }
        logger.info(f"Companies House: {number} — {result['legal_name']} ({result['status']})")
        return json.dumps(result, default=str)

    except Exception as e:
        logger.error(f"Companies House error: {e}")
        return json.dumps({"error": "Companies House lookup failed"})


def _exa_deep_search_sync(query: str, search_type: str = "general", domain: str = "") -> str:
    """
    Deep Exa search with optional domain filtering — reduces entity confusion.
    When domain is specified, results are limited to that domain.
    """
    if not exa_client:
        return json.dumps({"error": "Exa not configured", "results": []})

    try:
        kwargs = {
            "type": "deep",
            "num_results": 10,
        }

        # Domain filtering to reduce entity confusion
        if domain:
            kwargs["include_domains"] = [domain]

        if search_type == "news":
            kwargs["category"] = "news"
            kwargs["start_published_date"] = "2023-06-01"

        results = exa_client.search_and_contents(
            query,
            **kwargs,
            highlights={
                "max_characters": 5000,
                "query": query,
            },
            text={"max_characters": 8000},
        )

        if not results or not hasattr(results, "results") or not results.results:
            return json.dumps({"query": query, "search_type": search_type,
                               "domain": domain, "results": [], "note": "No results found"})

        items = []
        for r in results.results:
            highlights = getattr(r, "highlights", []) or []
            text       = getattr(r, "text", "")    or ""
            items.append({
                "url":        getattr(r, "url", ""),
                "title":      getattr(r, "title", ""),
                "date":       getattr(r, "published_date", ""),
                "highlights": highlights,
                "content":    text[:6000],
            })

        logger.info(f"Exa deep [{search_type}] domain={domain or 'all'}: '{query[:55]}' → {len(items)} results")

        return json.dumps({
            "query":       query,
            "search_type": search_type,
            "domain":      domain,
            "results":     items,
        }, default=str)

    except Exception as e:
        logger.error(f"Exa deep search error: {e}")
        return json.dumps({"error": "Deep search failed", "results": []})


def _exa_search_sync(query: str, search_type: str = "general") -> str:
    """
    Exa search — domain-aware with PE-specific extraction strategies.
    """
    if not exa_client:
        return json.dumps({"error": "Exa not configured", "results": []})

    try:
        if search_type == "funding":
            results = exa_client.search(
                query,
                type="deep",
                num_results=10,
                category="news",
                start_published_date="2023-01-01",
                system_prompt=(
                    "You are researching this company for private equity due diligence. "
                    "Find ALL funding information: exact amounts with currency (e.g. £8M, $15M), "
                    "round type (seed, Series A, Series B, venture debt, JV capital), "
                    "named investors (lead investor AND participating investors), exact dates, "
                    "pre/post-money valuation if available, use of funds, and any debt facilities. "
                    "Also look for revenue figures, ARR, MRR, or growth metrics mentioned alongside funding. "
                    "Prefer PRIMARY sources: official press releases, company blogs, TechCrunch, "
                    "Crunchbase, PitchBook, BusinessWire, PRNewswire. "
                    "If the company has strategic investors or JV partners, name them and describe the structure."
                ),
                output_schema={
                    "type": "object",
                    "properties": {
                        "total_raised":      {"type": "string", "description": "Total funding raised with currency e.g. £13M or $50M"},
                        "latest_round":      {"type": "string", "description": "Most recent round type e.g. Series A, Seed, Venture Debt"},
                        "latest_amount":     {"type": "string", "description": "Most recent round amount with currency"},
                        "latest_date":       {"type": "string", "description": "Most recent round date YYYY-MM or YYYY-MM-DD"},
                        "lead_investors":    {"type": "string", "description": "Lead investor name(s) for most recent round"},
                        "all_investors":     {"type": "string", "description": "ALL known investors across all rounds, comma-separated"},
                        "use_of_funds":      {"type": "string", "description": "What the company plans to use the money for"},
                        "valuation":         {"type": "string", "description": "Pre or post-money valuation if mentioned"},
                        "revenue_metrics":   {"type": "string", "description": "Any revenue, ARR, MRR, or financial metrics mentioned"},
                        "debt_facilities":   {"type": "string", "description": "Any debt lines, credit facilities, or lending partnerships"},
                    }
                },
                contents={
                    "highlights": {
                        "max_characters": 5000,
                        "query": "funding round amount raised investors valuation revenue Series seed venture capital",
                    },
                },
            )

        elif search_type == "company":
            results = exa_client.search_and_contents(
                query,
                type="deep",
                num_results=8,
                category="company",
                highlights={
                    "max_characters": 5000,
                    "query": (
                        "business model revenue streams pricing how company makes money "
                        "products services platform customers target market B2B B2C "
                        "partnerships integrations white-label API technology stack "
                        "founded year headquarters employees team size"
                    ),
                },
                text={"max_characters": 8000},
            )

        elif search_type == "people":
            results = exa_client.search_and_contents(
                query,
                type="auto",
                num_results=10,
                category="people",
                highlights={
                    "max_characters": 4000,
                    "query": (
                        "CEO founder co-founder CTO COO CFO VP Director "
                        "title role background experience previous companies "
                        "education university board member advisor"
                    ),
                },
            )

        elif search_type == "news":
            results = exa_client.search_and_contents(
                query,
                type="deep",
                num_results=12,
                category="news",
                start_published_date="2023-06-01",
                highlights={
                    "max_characters": 5000,
                    "query": query,
                },
                text={"max_characters": 6000},
            )

        else:
            results = exa_client.search_and_contents(
                query,
                type="deep",
                num_results=10,
                highlights={
                    "max_characters": 5000,
                    "query": query,
                },
                text={"max_characters": 6000},
            )

        if not results or not hasattr(results, "results") or not results.results:
            return json.dumps({"query": query, "search_type": search_type,
                               "results": [], "note": "No results found"})

        # Extract structured output from deep search
        structured_output = None
        grounding_summary = None
        if hasattr(results, "output") and results.output:
            structured_output = getattr(results.output, "content", None)
            grounding = getattr(results.output, "grounding", None)
            if grounding:
                grounding_summary = [
                    {
                        "field":      g.field if hasattr(g, "field") else str(g),
                        "confidence": g.confidence if hasattr(g, "confidence") else "unknown",
                        "sources":    len(g.citations) if hasattr(g, "citations") else 0,
                    }
                    for g in grounding
                ] if isinstance(grounding, list) else str(grounding)

        # Build result items — extract MORE data per result
        items = []
        for r in results.results:
            highlights = getattr(r, "highlights", []) or []
            text       = getattr(r, "text", "")    or ""
            items.append({
                "url":        getattr(r, "url", ""),
                "title":      getattr(r, "title", ""),
                "date":       getattr(r, "published_date", ""),
                "highlights": highlights,
                "content":    text[:5000],  # Increased from 3000
            })

        logger.info(
            f"Exa {search_type}: '{query[:55]}' → {len(items)} results"
            + (" + structured" if structured_output else "")
        )

        return json.dumps({
            "query":             query,
            "search_type":       search_type,
            "structured_output": structured_output,
            "grounding":         grounding_summary,
            "results":           items,
        }, default=str)

    except Exception as e:
        logger.error(f"Exa {search_type} error: {e}")
        try:
            fallback = exa_client.search(query, num_results=5)
            items = [{"url": r.url, "title": getattr(r, "title", "")}
                     for r in (fallback.results or [])]
            logger.info(f"Exa fallback: {len(items)} results")
            return json.dumps({"query": query, "results": items, "note": "fallback"})
        except Exception as e2:
            logger.error(f"Exa fallback error: {e2}")
            return json.dumps({"error": "Search failed", "results": []})


# ── TARGETED TinyFish extraction — different prompts per goal ────────────────
TINYFISH_PROMPTS = {
    "company_overview": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load (wait for main content text to render).
3. Scroll down the entire page once to trigger any lazy-loaded content.
4. Extract the following from this page:

Primary location: Hero section, about section, or main content area.
Fallback 1: Footer area for legal name and address.
Fallback 2: Meta description or page title.

Return JSON matching this exact structure:
{
  "company_name": "Acme Corp",
  "tagline": "One-line company description or slogan",
  "description": "2-3 sentence description of what the company does, who it serves, and how.",
  "founded_year": "2015",
  "headquarters": "City, Country",
  "mission_statement": "Company mission or vision statement if shown",
  "employee_count": "150",
  "key_stats": ["1M customers", "50 countries", "$100M revenue"],
  "industry_keywords": ["fintech", "SaaS", "enterprise"],
  "legal_footer_name": "Acme Corp Ltd",
  "certifications": ["B Corp", "ISO 27001"],
  "office_addresses": ["123 Main St, London, EC1A 1BB"]
}

If a field is not found on the page, set it to null.
Do not click any external links or navigate away from this page.""",

    "business_model": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll down the entire page to see all pricing and business model information.
4. If there are expandable sections or "Learn more" buttons about pricing, click them.

Extract ALL information about how this company makes money.

Primary location: Pricing page, business/partner page, or "How it works" page.
Fallback 1: Footer or FAQ section.
Fallback 2: Any "Partner with us" or "Join our network" section.

Return JSON matching this exact structure:
{
  "revenue_model": "Description of how the company generates revenue (e.g. SaaS subscription, transaction fees, commissions, licensing, AUM fees)",
  "who_pays": "Description of who the paying customers are",
  "pricing_tiers": [
    {"name": "Free / Basic", "price": "$0", "includes": "Core features"},
    {"name": "Pro / Enterprise", "price": "$99/mo or custom", "includes": "Advanced features"}
  ],
  "commission_details": "Any commission, referral fee, or take-rate details found",
  "partner_revenue_share": "Any revenue sharing arrangement with partners or channels",
  "number_of_providers_or_clients": "Number of clients, providers, or partners if mentioned",
  "target_customers_b2b": "Business customer segments",
  "target_customers_b2c": "Consumer customer segments if applicable",
  "free_tier": true,
  "enterprise_pricing": "Custom pricing or contact sales",
  "fee_structure": "Management fees, transaction fees, subscription fees — any specific rates mentioned"
}

If pricing amounts are shown (e.g. "$99/mo", "0.5% AUM fee", "£15 per referral"), extract the exact figures.
If pricing shows "Contact us" or "Custom", set the field to "contact_sales".
If a field is not found, set it to null.""",

    "products": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll down the ENTIRE page to see all products and services listed.
4. If there are tabs, expandable sections, or "View all" buttons, click them to reveal hidden content.
5. Look for product cards, feature lists, service descriptions, and any navigation menu items.

Extract EVERY distinct product, service, and feature offered.

Primary location: Main content area, product cards, service sections.
Fallback 1: Navigation menu items (often reveals product names).
Fallback 2: Footer links (often lists all services/products).

Return JSON matching this exact structure:
{
  "products": [
    {
      "name": "Product or Service Name",
      "description": "What it does in 2-3 sentences, who uses it, and why it matters.",
      "key_features": ["Feature 1", "Feature 2", "Feature 3"],
      "target_users": "Who uses this product (enterprises, consumers, developers, etc.)",
      "integrations": ["Partner platform 1", "API 1"],
      "technology": "Any technology mentioned (AI, ML, blockchain, cloud, etc.)"
    }
  ],
  "api_offerings": "Any API or developer tools mentioned",
  "white_label": true,
  "mobile_apps": "iOS, Android, or null",
  "total_products_found": 5
}

IMPORTANT: List every distinct service as a separate product entry. A platform may have sub-products — list each one separately.
If fewer than 3 products found, re-scan the page including navigation menu and footer links.
Do not navigate away from this page.""",

    "funding": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll through the entire article/page to find ALL funding information.
4. Look for: headline amounts, investor names in quotes or bold, dates, round types.

Extract ALL funding and financial information mentioned on this page.

Primary location: Article body text, press release content.
Fallback 1: Sidebar or "About" section.
Fallback 2: Any embedded tweets, quotes, or pull-quotes.

Return JSON matching this exact structure:
{
  "funding_rounds": [
    {
      "date": "2024-06",
      "round_type": "Series A",
      "amount": "$10M",
      "amount_local_currency": "£8M",
      "lead_investor": "Lead VC Firm Name",
      "other_investors": ["Co-investor 1", "Co-investor 2"],
      "use_of_funds": "What the money will be used for"
    }
  ],
  "total_raised": "$25M",
  "valuation": "$100M post-money",
  "revenue_figures": "Any ARR, MRR, or revenue figures mentioned",
  "angel_investors": ["Named angel 1", "Named angel 2"],
  "debt_facilities": "Any debt, credit lines, or venture debt mentioned",
  "source_url": "URL of this article",
  "article_title": "Exact headline of the article",
  "article_date": "2024-06-15",
  "article_source": "Publication name (e.g. TechCrunch, Reuters, Bloomberg)"
}

CRITICAL: Extract EXACT amounts with currency symbols. Do not round or approximate.
If the page mentions "undisclosed" amount, still extract investor names, dates, and round type.
If no funding info found, return {"funding_rounds": [], "error": "No funding information found on this page"}.
Do not click any external links.""",

    "management": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll down the entire page to see all team members.
4. If there is a "View all team" or "Load more" button, click it ONCE.
5. Look for team cards with photos, names, and titles.

Extract ALL people shown on this page.

Primary location: Team section with cards/photos.
Fallback 1: "About us" section with leadership mentions.
Fallback 2: Footer or sidebar with key contact names.

Return JSON matching this exact structure:
{
  "team_members": [
    {
      "name": "Jane Smith",
      "title": "Chief Executive Officer",
      "bio": "Background info, previous companies, education, years of experience.",
      "linkedin_url": "https://linkedin.com/in/janesmith",
      "photo_present": true
    }
  ],
  "board_members": [
    {
      "name": "John Doe",
      "title": "Non-Executive Director",
      "bio": "Former CEO of Major Corp"
    }
  ],
  "team_size": "Number of employees if shown",
  "hiring_indication": "Any mentions of growth, hiring, or open roles"
}

If fewer than 2 people found, look in the page navigation for an /about or /team link.
If a person has no bio, set bio to null but still extract name and title.
Do not navigate away from this page.""",

    "legal_entity": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll to the very bottom of the page to read the footer.
4. Look for legal text in these specific locations:
   - Page footer (copyright notice)
   - Terms of service section
   - Privacy policy header/introduction
   - "About" section with company registration details

Extract the LEGAL ENTITY information. This is different from the brand name.

Primary location: Footer copyright text (e.g. "© 2025 Company Name Ltd").
Fallback 1: Opening paragraph of privacy policy or terms (e.g. "operated by...").
Fallback 2: Contact page with registered address.

Return JSON matching this exact structure:
{
  "legal_entity_name": "Full Legal Company Name Ltd",
  "brand_name": "Trading/Brand Name",
  "company_number": "12345678",
  "vat_number": "GB123456789",
  "regulatory_numbers": "FCA 123456 or SEC CRD# etc",
  "registered_address": "Full registered office address",
  "raw_legal_text": "Exact sentence from the page where legal entity info was found",
  "copyright_text": "Exact copyright notice text from footer"
}

If no legal entity name found, return {"legal_entity_name": null, "error": "No legal entity found on page"}.
Do not click any external links or navigate away.""",

    "geography": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll through the page looking for addresses, location mentions, and maps.

Extract ALL geographic and location information.

Return JSON matching this exact structure:
{
  "headquarters": "City, Country",
  "office_locations": [
    {"city": "London", "address": "123 Main St, EC1A 1BB", "type": "HQ"},
    {"city": "New York", "address": "456 Park Ave, NY 10022", "type": "office"}
  ],
  "markets_served": ["United Kingdom", "United States", "Europe"],
  "expansion_plans": "Any mentioned geographic expansion plans",
  "regulatory_jurisdictions": "FCA, SEC, or other regulators mentioned"
}

If no geographic info found, set fields to null.""",

    "operations": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll the entire page. Look for numbers, statistics, metrics, awards.

Extract ALL operational metrics and achievements shown on this page.

Return JSON matching this exact structure:
{
  "transaction_volumes": "Any transaction or processing volume metrics",
  "customer_count": "Number of customers or users",
  "aum_or_revenue": "Assets under management, revenue, or financial scale metrics",
  "market_share": "Any market share or ranking claims",
  "employee_count": "Number of employees",
  "growth_metrics": "Any YoY growth, CAGR, or growth rate figures",
  "certifications": ["ISO 27001", "SOC 2", "B Corp"],
  "awards": ["Award name 1", "Award name 2"],
  "trustpilot_or_rating": "Customer rating if shown",
  "case_studies": ["Named client 1", "Named client 2"]
}

If no metrics found, return {"error": "No operational metrics found on this page"}.""",

    "partnerships": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Scroll the entire page. Look for partner logos, "Trusted by" sections, client lists.
4. If there is a "View all partners" button, click it ONCE.

Extract ALL partnership and customer information.

Primary location: Partner logo section, "Trusted by" banner, client list.
Fallback 1: Case study section with named clients.
Fallback 2: Integration/compatibility section.

Return JSON matching this exact structure:
{
  "named_partners": [
    {"name": "Partner Company Name", "type": "strategic/technology/distribution/financial", "detail": "Nature of the partnership"},
    {"name": "Another Partner", "type": "technology", "detail": "Integration or API partnership"}
  ],
  "customer_logos": ["Company A", "Company B", "Company C"],
  "integration_partners": ["Platform 1", "Platform 2"],
  "total_partners_shown": 10
}

IMPORTANT: Extract REAL company names only. Do not guess or fabricate partner names.
If no partners found, return {"named_partners": [], "error": "No partners found on this page"}.
Do not navigate away from this page.""",

    "general": """
1. If a cookie or consent banner appears, close it first.
2. Wait for the page to fully load.
3. Wait 1 second for dynamic content to render.
4. If a Cloudflare or security check page appears, wait for it to complete automatically.
5. Scroll down the entire page once.

Extract ALL of the following from this webpage:

Return JSON matching this exact structure:
{
  "company_name": "string",
  "description": "What the company does in 2-3 sentences",
  "products": ["List of product/service names"],
  "business_model": "How they make money",
  "named_partners": ["Real company names only"],
  "team_members": [{"name": "string", "title": "string"}],
  "funding": {"total_raised": "string", "latest_round": "string"},
  "employee_count": "string or null",
  "locations": ["string"],
  "legal_footer_name": "string from copyright notice or null",
  "key_stats": ["Any statistics or metrics shown"]
}

If a field is not found, set it to null.
Do not click any purchase, download, or external link buttons.
Do not navigate away from this page.""",
}


# Track domains where TinyFish consistently fails (anti-bot protection)
_tinyfish_failed_domains: Dict[str, int] = {}
TINYFISH_DOMAIN_FAIL_THRESHOLD = 2  # After 2 failures on same domain, skip it

# Track URLs to prevent duplicate processing
_urls_already_processed: set = set()

def _is_url_seen(url: str) -> bool:
    """Track URLs to prevent duplicate TinyFish/Exa processing."""
    normalized = url.lower().rstrip("/").replace("www.", "")
    if normalized in _urls_already_processed:
        return True
    _urls_already_processed.add(normalized)
    return False


def _get_domain(url: str) -> str:
    """Extract domain from URL for failure tracking."""
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _tinyfish_extract_targeted_sync(url: str, extraction_goal: str = "general") -> str:
    """TinyFish extraction with goal-specific prompts for deeper data extraction."""
    if not tinyfish_client:
        return json.dumps({"error": "TinyFish not configured"})

    # Check if this domain has failed too many times (anti-bot sites)
    domain = _get_domain(url)
    if domain and _tinyfish_failed_domains.get(domain, 0) >= TINYFISH_DOMAIN_FAIL_THRESHOLD:
        logger.info(f"TinyFish SKIPPED {url[:60]} — domain '{domain}' blocked ({_tinyfish_failed_domains[domain]} prior failures). Use search_web or deep_search instead.")
        return json.dumps({
            "url":             url,
            "extraction_goal": extraction_goal,
            "data":            {},
            "success":         False,
            "skipped":         True,
            "reason":          f"Domain '{domain}' has anti-bot protection. TinyFish failed {_tinyfish_failed_domains[domain]} times. Use search_web or deep_search with domain='{domain}' instead.",
        })

    prompt = TINYFISH_PROMPTS.get(extraction_goal, TINYFISH_PROMPTS["general"])
    
    # Append anti-bot fallback instruction to every prompt (from TinyFish Anti-Bot Guide Step 3)
    anti_bot_suffix = """

If a Cloudflare "Checking your browser" page appears, wait up to 10 seconds for it to complete automatically.
If a loading screen or redirect page appears, wait for it to complete before extracting.
If an "Access Denied", 403, or CAPTCHA page appears, return {"error": "access_denied", "blocked": true}.
If the page is blank or shows an infinite spinner after 10 seconds, return {"error": "page_failed_to_load"}.
"""
    prompt = prompt + anti_bot_suffix

    try:
        # Auto-detect proxy country from URL domain for better success rates
        proxy_country = ProxyCountryCode.GB  # default
        url_lower = url.lower()
        if any(tld in url_lower for tld in [".com", ".us", ".io", ".co"]):
            proxy_country = ProxyCountryCode.US
        elif any(tld in url_lower for tld in [".co.uk", ".uk", ".gov.uk"]):
            proxy_country = ProxyCountryCode.GB
        elif any(tld in url_lower for tld in [".de", ".eu"]):
            proxy_country = ProxyCountryCode.DE
        elif any(tld in url_lower for tld in [".sg"]):
            proxy_country = ProxyCountryCode.SG
        elif any(tld in url_lower for tld in [".au", ".com.au"]):
            proxy_country = ProxyCountryCode.AU
        elif any(tld in url_lower for tld in [".in", ".co.in"]):
            proxy_country = ProxyCountryCode.IN

        with tinyfish_client.agent.stream(
            url=url,
            goal=prompt,
            browser_profile=BrowserProfile.STEALTH,
            proxy_config=ProxyConfig(
                enabled=True,
                country_code=proxy_country,
            ),
        ) as stream:
            event_count  = 0
            result_data  = None
            text_content = []

            for event in stream:
                event_count += 1
                if event_count > MAX_STREAM_EVENTS:
                    logger.warning(f"TinyFish: event ceiling at {event_count} for {url[:50]}")
                    break
                if (event_count * 5) > MAX_STEPS_PER_EXTR:
                    logger.warning(f"TinyFish: step ceiling at {event_count} for {url[:50]}")
                    break

                if isinstance(event, CompleteEvent):
                    if event.result_json:
                        result_data = event.result_json
                    break

                if hasattr(event, "result_json") and event.result_json:
                    result_data = event.result_json
                    break

                if hasattr(event, "text") and event.text:
                    text_content.append(str(event.text))
                if hasattr(event, "content") and event.content:
                    text_content.append(str(event.content))

        if not result_data and text_content:
            combined = " ".join(text_content)[:4000]
            logger.info(f"TinyFish [{extraction_goal}]: {url[:60]} — got text ({len(combined)} chars)")
            return json.dumps({
                "url":             url,
                "extraction_goal": extraction_goal,
                "data":            {"raw_text": combined},
                "success":         True,
                "type":            "text_extraction",
            }, default=str)

        logger.info(f"TinyFish [{extraction_goal}]: {url[:60]} — success={bool(result_data)}")
        
        # Track domain failures for anti-bot detection
        if not result_data:
            domain = _get_domain(url)
            if domain:
                _tinyfish_failed_domains[domain] = _tinyfish_failed_domains.get(domain, 0) + 1
                fail_count = _tinyfish_failed_domains[domain]
                if fail_count >= TINYFISH_DOMAIN_FAIL_THRESHOLD:
                    logger.warning(f"TinyFish: domain '{domain}' marked as BLOCKED after {fail_count} failures. Future calls will be skipped.")
        
        return json.dumps({
            "url":             url,
            "extraction_goal": extraction_goal,
            "data":            result_data or {},
            "success":         bool(result_data),
            "note":            f"If success=false, this site may have anti-bot protection. Use deep_search(query, domain='{_get_domain(url)}') as alternative." if not result_data else "",
        }, default=str)

    except Exception as e:
        logger.error(f"TinyFish error {url}: {e}")
        return json.dumps({"url": url, "error": "Page extraction failed"})


def _tinyfish_extract_sync(url: str) -> str:
    """Backward-compatible general extraction."""
    return _tinyfish_extract_targeted_sync(url, "general")


def _tinyfish_privacy_sync(url: str) -> str:
    """Extract legal entity from privacy/terms pages."""
    return _tinyfish_extract_targeted_sync(url, "legal_entity")


# ── OpenCorporates ────────────────────────────────────────────────────────────
def _opencorporates_sync(company_name: str) -> str:
    """OpenCorporates global registry — searches across 140+ jurisdictions."""
    try:
        # Search globally without jurisdiction filter
        resp = req_lib.get(
            "https://api.opencorporates.com/v0.4/companies/search",
            params={"q": company_name, "per_page": 8},
            timeout=15,
        )
        data      = resp.json()
        companies = data.get("results", {}).get("companies", [])

        if not companies:
            return json.dumps({"found": False, "message": f"No OpenCorporates record for '{company_name}'"})

        results = []
        best_match = None

        for item in companies[:5]:
            c = item.get("company", {})
            entry = {
                "name":               c.get("name"),
                "company_number":     c.get("company_number"),
                "jurisdiction":       c.get("jurisdiction_code"),
                "status":             c.get("current_status"),
                "incorporation_date": c.get("incorporation_date"),
                "company_type":       c.get("company_type"),
                "registered_address": c.get("registered_address_in_full"),
                "opencorporates_url": c.get("opencorporates_url"),
            }
            results.append(entry)

            # Track the first active match as best candidate for officer lookup
            if not best_match and c.get("current_status", "").lower() in ("active", "good standing", ""):
                best_match = c

        # Try to get officers for the best match (works globally — not just UK)
        officers = []
        if best_match:
            jur_code = best_match.get("jurisdiction_code", "")
            comp_num = best_match.get("company_number", "")
            if jur_code and comp_num:
                try:
                    officers_resp = req_lib.get(
                        f"https://api.opencorporates.com/v0.4/companies/{jur_code}/{comp_num}/officers",
                        timeout=15,
                    )
                    if officers_resp.status_code == 200:
                        officer_items = officers_resp.json().get("results", {}).get("officers", [])
                        for o in officer_items[:15]:
                            off = o.get("officer", {})
                            officers.append({
                                "name":      off.get("name", ""),
                                "position":  off.get("position", ""),
                                "start_date": off.get("start_date", ""),
                                "end_date":  off.get("end_date"),
                                "nationality": off.get("nationality", ""),
                            })
                        logger.info(f"OpenCorporates: found {len(officers)} officers for {best_match.get('name')}")
                except Exception as e:
                    logger.warning(f"OpenCorporates officers lookup failed: {e}")

        logger.info(f"OpenCorporates: found {len(results)} companies for '{company_name}'")
        return json.dumps({
            "source":    "OpenCorporates (global registry — 140+ jurisdictions)",
            "query":     company_name,
            "companies": results,
            "officers":  officers,
            "note":      "Officers are from the best-matching active company. If multiple results, identify which is the ACTIVE trading entity.",
        }, default=str)

    except Exception as e:
        logger.error(f"OpenCorporates error: {e}")
        return json.dumps({"error": "Registry lookup failed", "found": False})


# ══════════════════════════════════════════════════════════════════════════════
# ENTITY RESOLUTION — Cross-reference all sources to confirm correct legal entity
# ══════════════════════════════════════════════════════════════════════════════

def resolve_entity(tool_logs: list, company_name: str) -> dict:
    """
    After research agent finishes, cross-reference ALL collected data
    to confirm the correct legal entity. This replicates what Claude does
    implicitly — seeing all search results at once and pattern-matching.
    """
    early_data = []
    for log in tool_logs:
        try:
            data = json.loads(log.get("result", ""))
            early_data.append(data)
        except Exception:
            pass

    prompt = f"""You are an entity resolution system for PE due diligence.
Below is raw data from multiple sources about "{company_name}".

Your job:
1. Identify ALL distinct legal entities mentioned (different company numbers = different entities)
2. Determine which is the ACTIVE TRADING entity that operates the "{company_name}" brand
3. Cross-reference: do director names from registry match team members from website?
4. If the website footer shows a legal name, that IS the trading entity
5. A dormant company with the same name is likely a shell — look for the active one

Return ONLY JSON:
{{
    "canonical_legal_name": "The correct legal entity name",
    "canonical_company_number": "Official registration number",
    "canonical_directors": ["Name 1", "Name 2"],
    "confidence": "high/medium/low",
    "reasoning": "2-3 sentences explaining why this is the right entity",
    "rejected_entities": [
        {{"name": "Wrong Company Ltd", "number": "12345678", "reason": "Dormant since 2019"}}
    ]
}}

RAW DATA:
{json.dumps(early_data, default=str)[:25000]}"""

    result = gemini_json(prompt)
    if result.get("canonical_legal_name"):
        logger.info(f"Entity resolved: {result['canonical_legal_name']} "
                    f"(confidence: {result.get('confidence')})")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FINANCIAL EXTRACTION — Fetch and parse Companies House filed accounts
# ══════════════════════════════════════════════════════════════════════════════

def extract_financial_accounts(company_number: str) -> dict:
    """
    Fetch latest filed accounts from Companies House and extract
    balance sheet data using TinyFish.
    """
    if not COMPANIES_HOUSE_KEY or not tinyfish_client:
        return {"error": "CH or TinyFish not configured", "financials": {}}

    BASE = "https://api.company-information.service.gov.uk"
    auth = (COMPANIES_HOUSE_KEY, "")

    try:
        filings_r = req_lib.get(
            f"{BASE}/company/{company_number}/filing-history",
            params={"items_per_page": 20, "category": "accounts"},
            auth=auth, timeout=15,
        )
        if filings_r.status_code != 200:
            return {"error": "Filing history fetch failed", "financials": {}}

        filings = filings_r.json().get("items", [])
        accounts_filing = None
        for f in filings:
            desc = (f.get("description", "") or "").lower()
            if "account" in desc:
                accounts_filing = f
                break

        if not accounts_filing:
            return {"error": "No accounts filing found", "financials": {}}

        transaction_id = accounts_filing.get("transaction_id", "")
        filing_date = accounts_filing.get("date", "")
        filing_desc = accounts_filing.get("description", "")
        made_up_to = accounts_filing.get("description_values", {}).get("made_up_date", "")

        doc_url = (
            f"https://find-and-update.company-information.service.gov.uk"
            f"/company/{company_number}/filing-history/{transaction_id}/document"
        )

        logger.info(f"Financial extraction: {company_number} — {filing_desc}")

        # Use TinyFish to extract financial data
        prompt = """Close any cookie banner. Wait for page to load fully.
If this is a PDF viewer, wait for the document to render.
Scroll through the ENTIRE document looking for financial tables.
Return JSON: {"accounts_period": "Year ending YYYY-MM-DD",
"balance_sheet": {"fixed_assets": "", "current_assets": "", "total_assets": "",
"current_liabilities": "", "total_liabilities": "", "net_assets": "",
"share_capital": "", "retained_earnings": "", "total_equity": ""},
"profit_and_loss": {"turnover": "", "gross_profit": "", "operating_profit": "",
"profit_before_tax": "", "profit_after_tax": ""},
"cash_at_bank": "", "employee_count_in_accounts": "", "filing_type": ""}
Extract EXACT figures with currency. Set missing fields to null.
If blocked, return {"error": "access_denied"}."""

        result_data = None
        text_parts = []

        try:
            from urllib.parse import urlparse
            domain = urlparse(doc_url).netloc.lower().replace("www.", "")
            if _tinyfish_failed_domains.get(domain, 0) >= TINYFISH_DOMAIN_FAIL_THRESHOLD:
                return {"doc_url": doc_url, "filing_date": filing_date,
                        "note": "Domain blocked — manual review required", "financials": {}}

            with tinyfish_client.agent.stream(
                url=doc_url, goal=prompt,
                browser_profile=BrowserProfile.STEALTH,
                proxy_config=ProxyConfig(enabled=True, country_code=ProxyCountryCode.GB),
            ) as stream:
                event_count = 0
                for event in stream:
                    event_count += 1
                    if event_count > MAX_STREAM_EVENTS:
                        break
                    if isinstance(event, CompleteEvent):
                        if event.result_json:
                            result_data = event.result_json
                        break
                    if hasattr(event, "result_json") and event.result_json:
                        result_data = event.result_json
                        break
                    if hasattr(event, "text") and event.text:
                        text_parts.append(str(event.text))
        except Exception as tf_err:
            logger.warning(f"TinyFish financial extraction failed: {tf_err}")

        # If raw text, use Gemini to parse
        if not result_data and text_parts:
            raw_text = " ".join(text_parts)[:8000]
            result_data = gemini_json(f"""Extract financial data from these Companies House accounts.
Return JSON with balance_sheet and profit_and_loss fields. Use exact £ figures. Set missing to null.

RAW TEXT:
{raw_text}""")

        return {
            "company_number": company_number,
            "doc_url": doc_url,
            "filing_date": filing_date,
            "made_up_to": made_up_to,
            "financials": result_data or {},
            "success": bool(result_data and result_data.get("balance_sheet")),
        }

    except Exception as e:
        logger.error(f"Financial extraction error: {e}")
        return {"error": "Financial extraction failed", "financials": {}}


# ══════════════════════════════════════════════════════════════════════════════
# CAP TABLE ESTIMATION — Estimate ownership from PSCs + funding rounds
# ══════════════════════════════════════════════════════════════════════════════

## estimate_cap_table removed — research reports should present facts, not assumptions


# ══════════════════════════════════════════════════════════════════════════════
# REACT AGENT — Intelligent researcher (thinks freely, no rigid phases)
# ══════════════════════════════════════════════════════════════════════════════

def build_react_agent(company_name: str, country_label: str):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )

    tools = [
        StructuredTool(
            name="search_web",
            description="""Exa semantic search — finds and reads web pages.
Specify search_type for best results: 'funding', 'company', 'people', 'news', 'general'.
Read ALL fields: structured_output, highlights, content, url, date.""",
            func=_exa_search_sync,
            args_schema=SearchInput,
        ),
        StructuredTool(
            name="deep_search",
            description="""Deep Exa search with optional domain filtering.
Use domain to search WITHIN a specific website (e.g. domain="techcrunch.com").
Also search crowdfunding sites: domain="crowdcube.com" for valuation data.""",
            func=_exa_deep_search_sync,
            args_schema=DeepSearchInput,
        ),
        StructuredTool(
            name="get_registry_data",
            description="UK Companies House official registry. Returns legal name, directors, PSCs, charges, filings. ONLY for UK companies.",
            func=_companies_house_sync,
            args_schema=CompanyNameInput,
        ),
        StructuredTool(
            name="find_related_entities",
            description="OpenCorporates global registry — 140+ jurisdictions. Use for non-UK companies or to find subsidiaries.",
            func=_opencorporates_sync,
            args_schema=CompanyNameInput,
        ),
        StructuredTool(
            name="extract_page",
            description="""Live browser extraction with targeted goals.
Goals: company_overview, business_model, products, funding, management, legal_entity, operations, partnerships, general.
If extraction fails, use deep_search with that domain instead.""",
            func=_tinyfish_extract_targeted_sync,
            args_schema=TargetedExtractInput,
        ),
    ]

    is_uk = country_label == "United Kingdom"
    registry_note = "Use get_registry_data for Companies House data." if is_uk else f"get_registry_data is UK-only. For {country_label}, use find_related_entities and search_web."

    system_prompt = f"""You are a senior PE research analyst. Research "{company_name}" ({country_label}).

YOUR JOB: Build a complete investment research profile. You decide what to search, when to dig deeper, and when you have enough.

A COMPLETE REPORT ANSWERS ALL OF THESE:
1. What does the company do and how does it make money? (exact revenue mechanics, any fees charged to consumers, commission structures, pricing)
2. What is the legal entity? (registry data, company number, status, registered address, ownership/PSCs, charges/debt)
3. Who leads the company? (every director/officer — search each person individually for their PREVIOUS companies and roles)
4. How much funding has it raised? (EVERY round — seed, angel, crowdfunding, venture, Series A/B — exact amounts, dates, named investors)
5. What are ALL its products and services? (list every distinct one)
6. Who are its real partners and customers? (named companies only)
7. What operational metrics exist? (users, volume, growth, ratings, employee count)
8. What valuation data exists? (from crowdfunding pitches, Tracxn, PitchBook — factual data only, not estimates)
9. What's happened recently? (news, hires, partnerships, product launches)
10. What are the risks? (channel dependency, competition, regulation, key-person, execution)

RULES:
- {registry_note}
- ALWAYS include "{company_name}" in your search queries. Never search generic phrases like "company overview" — always search "{company_name} company overview".
- Each source may answer MULTIPLE questions — extract EVERYTHING useful from each result.
- A news article about funding often mentions team, metrics, strategy — capture all of it.
- When extract_page fails, immediately try deep_search with that domain.
- For each executive/director, run a SEPARATE people search for their background. Only include facts you find — NEVER guess previous employers.
- If funding total doesn't match the sum of individual rounds, search for the missing rounds.
- If you find a funding round but don't have investor names, search specifically for that round (e.g. "Just Move In 2021 crowdcube" or "Just Move In seed round investors").
- Search crowdfunding platforms — deep_search with domain="crowdcube.com" and domain="seedrs.com". Crowdcube pages contain valuation data (share price, equity offered, pre-money valuation) — this is factual, not estimated.
- You must use at least 15 tool calls before writing your analysis.

WHEN TO STOP: Ask yourself "If I were a PE associate reading this, what would I immediately ask that isn't answered?" If you can think of an unanswered question, search for it. When you can't, write your analysis.

WRITE YOUR ANALYSIS covering all 10 areas above. Use specific numbers, names, dates. Flag what you couldn't find."""

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent



# ══════════════════════════════════════════════════════════════════════════════
# SYNTHESIS — Enhanced with deeper extraction from tool logs
# ══════════════════════════════════════════════════════════════════════════════

class GeminiSynthesiser:
    def synthesise(
        self,
        company_name: str,
        agent_output: str,
        tool_logs:    List[Dict],
        sources_used: List[str],
        jurisdiction: Dict,
    ) -> Dict[str, Any]:

        # ── Extract ALL valuable data from tool logs ─────────────────────────
        all_data         = []
        news_content     = []
        registry_data    = []
        structured_finds = []
        page_extractions = []

        for log in tool_logs:
            try:
                content = log.get("result", "")
                if not content:
                    continue
                data = json.loads(content)

                # Registry data (Companies House, OpenCorporates) — always include fully
                if data.get("company_number") or data.get("legal_name") or data.get("source") == "Companies House (authoritative UK registry)":
                    registry_data.append(data)
                    continue

                if data.get("source") == "OpenCorporates":
                    registry_data.append(data)
                    continue

                # TinyFish page extractions — include with goal context
                if data.get("extraction_goal") or (data.get("url") and data.get("success") is not None):
                    page_extractions.append({
                        "url":  data.get("url", ""),
                        "goal": data.get("extraction_goal", "general"),
                        "data": data.get("data", {}),
                    })
                    continue

                # Structured output from Exa deep search — highest value
                if data.get("structured_output"):
                    structured_finds.append({
                        "type":   data.get("search_type", "unknown"),
                        "query":  data.get("query", ""),
                        "output": data["structured_output"],
                        "grounding": data.get("grounding"),
                    })

                # News articles — separate for better extraction
                if data.get("search_type") == "news" and "results" in data:
                    for r in data["results"]:
                        if r.get("highlights") or r.get("content"):
                            news_content.append({
                                "url":        r.get("url", ""),
                                "title":      r.get("title", ""),
                                "date":       r.get("date", ""),
                                "highlights": r.get("highlights", []),
                                "summary":    str(r.get("content", ""))[:1200],
                            })

                # Funding search results — extract MORE content for funding articles
                if data.get("search_type") == "funding" and "results" in data:
                    for r in data.get("results", [])[:8]:
                        if r.get("highlights") or r.get("content"):
                            all_data.insert(0, {  # Funding data first
                                "search_type": "funding",
                                "query":       data.get("query", ""),
                                "url":         r.get("url", ""),
                                "title":       r.get("title", ""),
                                "date":        r.get("date", ""),
                                "highlights":  r.get("highlights", []),
                                "content":     str(r.get("content", ""))[:3000],  # More content for funding
                            })

                # All other search results with highlights
                if "results" in data and data.get("search_type") != "funding":
                    for r in data.get("results", [])[:6]:
                        if r.get("highlights") or r.get("content"):
                            all_data.append({
                                "search_type": data.get("search_type", ""),
                                "query":       data.get("query", ""),
                                "url":         r.get("url", ""),
                                "title":       r.get("title", ""),
                                "date":        r.get("date", ""),
                                "highlights":  r.get("highlights", []),
                                "content":     str(r.get("content", ""))[:2000],
                            })

            except Exception:
                pass

        # ── Build the synthesis prompt ───────────────────────────────────────
        prompt = f"""You are a senior Private Equity research analyst writing a professional investment brief.
Synthesize ALL of the following research on "{company_name}" into a comprehensive, PE-grade brief.

CRITICAL INSTRUCTIONS:
1. "Has raised funding" or "Undisclosed" is UNACCEPTABLE if the data is in the research below. READ EVERY PIECE OF DATA.
2. Scan ALL highlights, content, and page extractions for funding amounts — they may be buried in article text.
3. For products: list EVERY distinct product, feature, and service found — not just the main one.
4. For business model: describe the EXACT revenue mechanics, not just "commission-based."
5. If a TechCrunch, Sifted, or news article mentions funding — extract the EXACT figures.

═══════════════════════════════════════════════════════════════
RAW RESEARCH DATA — USE ALL OF THIS. SCAN EVERY LINE.
═══════════════════════════════════════════════════════════════

AGENT ANALYSIS (the research agent's own findings and narrative):
{agent_output[:12000]}

STRUCTURED SEARCH OUTPUTS (Exa's synthesised answers — highest confidence):
{json.dumps(structured_finds, indent=2, default=str)[:6000]}

REGISTRY DATA (Companies House / OpenCorporates — authoritative legal data):
{json.dumps(registry_data, indent=2, default=str)[:6000]}

PAGE EXTRACTIONS (TinyFish browser data from official pages — check for products, features, business model):
{json.dumps(page_extractions, indent=2, default=str)[:8000]}

SEARCH RESULTS WITH HIGHLIGHTS (web research — check for funding amounts, partner names, product features):
{json.dumps(all_data[:25], indent=2, default=str)[:12000]}

NEWS ARTICLES (check for funding announcements, partnership news, product launches):
{json.dumps(news_content[:15], indent=2, default=str)[:5000]}

═══════════════════════════════════════════════════════════════
OUTPUT REQUIREMENTS
═══════════════════════════════════════════════════════════════

WRITING STYLE & DEPTH CALIBRATION:
- executive_summary: 5-7 sentences. Investment thesis paragraph that reads like a Morgan Stanley research note.
  Start with what the company does and its market position. Include key metrics (revenue, growth, market share).
  State the investment thesis (why this is interesting). Note primary risk factors. End with valuation context.
  This should be the RICHEST section of the entire report — a PE partner should be able to read only this
  and decide whether to take the next meeting.
- about_company: 3-5 sentences. Founding story, what they do, who they serve, technology approach, market position.
- business_model narrative: Full paragraph. Who pays whom, exact revenue mechanics, commission rates if found,
  B2B vs B2C dynamics, partner economics, unit economics if available.
- geographic_reach: Full paragraph with named addresses, not just "UK based".
- Each product: Name + 2-3 sentence description with features, users, integrations, and technology.
- Each executive: Name + title + 2-3 sentence bio with NAMED previous companies and specific roles.
  CRITICAL: Only include biographical details (previous companies, education, roles) that appear in the research data below.
  If the research data does not contain specific background info for a person, write "Background details not available in public sources" — do NOT guess or infer previous employers from your training data.
  Getting a bio WRONG (e.g. wrong previous company) is far worse than saying "not available".
- Each partner: Real company name + type + what they do together.
- operational_highlights: Each item MUST be a specific metric with a number, not a vague statement.
  Convert any objects to simple strings like "Processes 300,000 home moves annually (c. 10% UK market share)".

DATA STANDARDS:
- named_partners: ONLY real company names — "Knight Frank", "Hamptons", "Openrent". NOT "various partners."
- funding: Exact amounts with currency symbols — "£6M Series A led by Eos Ventures, Mar 2025". NOT "has raised funding."
  CRITICAL: Include EVERY funding round found in the data — seed, angel, crowdfunding, venture, Series A/B/C, debt.
  The total_raised MUST equal the sum of all individual rounds. If there's a gap, note it in data_gaps.
  Crowdcube/Seedrs rounds count as real funding rounds — list them with platform name and amount.
- executives: Name + exact title + background with NAMED companies from the research data only. If background not found, say "not available in public sources" — NEVER guess.
- operational_highlights: MUST be simple strings, never objects. e.g. "300,000 moves annually" not {{"metric": "300k", "detail": "..."}}
- news: MUST include at least 5 articles if found in the data. Include funding news, partnership announcements, product launches, hiring news. NOT empty arrays.
- legal entity: If brand trades under different name from Companies House, explain clearly.
- due_diligence_flags: At least 5 flags required. MUST include all of: (1) channel/customer concentration risk, (2) competitive risk, (3) financial transparency, (4) key-person dependency risk, (5) regulatory or execution risk. Add more if relevant.

Return ONLY this JSON (no markdown, no backticks):
{{
  "executive_summary": "5-7 sentence investment-grade narrative: what the company does, market position, investment thesis, key metrics, primary risk factors, valuation context. This should read like the opening paragraph of a Morgan Stanley research note — a PE partner should be able to read only this and decide whether to take the next meeting.",

  "company_identity": {{
    "legal_name": "Full legal entity name from official registry or website",
    "trading_name": "Brand/trading name if different",
    "company_number": "Official registration number if found",
    "status": "active/dormant/dissolved",
    "incorporation_date": "YYYY-MM-DD",
    "jurisdiction": "{jurisdiction.get('label', 'Unknown')}",
    "registered_address": "Full registered office address",
    "operating_address": "Where they actually work from if different",
    "sic_codes": ["industry classification codes if available"],
    "company_type": "Corporation type (Ltd, Inc, GmbH, LLC, etc.)",
    "trading_entity_note": "Explanation if brand trades under different entity from registered name",
    "ownership_structure": "Who controls the company — from registry data, SEC filings, or other sources",
    "active_charges": "Summary of any secured debt, liens, or encumbrances"
  }},

  "about_company": "3-5 sentence narrative paragraph: founding story, what they do, who they serve, how the platform works, market position, technology approach",

  "business_model": {{
    "narrative": "Full paragraph: how the company makes money, who pays, pricing mechanics (include any direct consumer fees or charges found), B2B vs B2C, partner economics, unit economics if available. CRITICAL: if the research mentions a specific price or fee charged to consumers, include it.",
    "revenue_streams": [
      {{"name": "Stream name", "description": "Full sentence explaining this revenue stream, who pays, and how it scales"}}
    ],
    "unit_economics_note": "Any per-transaction, per-customer, or per-unit economics available"
  }},

  "products_services": [
    {{"name": "Product name", "description": "2-3 sentence description: what it does, key features, target users, integrations, technology"}}
  ],

  "geographic_reach": "Full paragraph: primary markets, named office locations (with addresses if available), distribution channels, regulatory jurisdictions, expansion plans",

  "operational_highlights": [
    "Each item should be a specific metric or achievement with numbers — e.g. 'Projected to process over 300,000 home moves in 2025'"
  ],

  "named_partners": [
    {{"name": "Real company name", "type": "strategic/technology/distribution/financial", "detail": "Nature of partnership and what they do together"}}
  ],

  "management": {{
    "key_people": [
      {{"name": "Full name", "title": "Exact current title", "bio": "2-3 sentence bio: previous experience at named companies, education, years in industry, focus area at this company"}}
    ],
    "directors": ["List of all registered directors/officers with appointment dates if available"],
    "board_notes": "Any board composition notes, advisors, non-exec directors"
  }},

  "funding": {{
    "total_raised": "Total with currency e.g. £13M",
    "rounds": [
      {{"date": "YYYY-MM", "round": "Seed/Series A/B/Debt", "amount": "£Xm", "lead_investor": "Named lead", "other_investors": "Other named investors", "purpose": "What the money was for"}}
    ],
    "investors": ["Complete list of all named investors across all rounds"],
    "debt_facilities": "Any credit lines, lending facilities, or secured debt from registry charges or public filings",
    "valuation_notes": "Any valuation data available"
  }},

  "recent_news": [
    {{"date": "YYYY-MM-DD", "title": "Exact article title", "source": "Publication name", "summary": "1-2 sentence summary of what the article reports", "url": "Full URL"}}
  ],

  "valuation_history": [
    {{"date": "YYYY-MM", "valuation": "£Xm pre/post-money", "context": "Source and basis e.g. Crowdcube pitch at £8.70/share, Tracxn estimate", "source": "Where this came from"}}
  ],

  "investment_highlights": [
    "Each item: a concise investment thesis bullet — e.g. 'Market leader with ~10% UK share', 'Capital-light commission model with high operating leverage'"
  ],

  "due_diligence_flags": [
    {{"severity": "high/medium/low", "flag": "Short flag name", "detail": "2-3 sentence explanation of the risk, its implications, and suggested next steps for diligence"}}
  ],

  "data_confidence": "high/medium/low",
  "confidence_notes": "Explain what data is well-sourced vs uncertain vs missing. Be specific about gaps.",
  "data_gaps": ["List specific data points that could not be found and why"],
  "data_sources": [],
  "research_timestamp": ""
}}"""

        try:
            result = gemini_json(prompt)
            result["research_timestamp"] = datetime.now(timezone.utc).isoformat()
            result["data_sources"]       = list(set(sources_used))[:30]
            return result
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "executive_summary": "Synthesis error — raw data available",
                "error":             str(e),
                "data_confidence":   "low",
            }


# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR (kept from v4.2 — works well)
# ══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:

    def generate(self, data: Dict, company_name: str, output_path: str) -> str:
        script      = self._build_script(data, company_name, output_path)
        script_path = output_path.replace(".docx", "_gen.js")
        with open(script_path, "w") as f:
            f.write(script)

        import subprocess
        result = subprocess.run(["node", script_path], capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.error(f"Report gen error: {result.stderr}")
            json_path = output_path.replace(".docx", ".json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            return json_path

        os.unlink(script_path)
        logger.info(f"Report saved: {output_path}")
        return output_path

    def _build_script(self, data: Dict, company_name: str, output_path: str) -> str:
        identity  = data.get("company_identity",       {}) or {}
        biz_model = data.get("business_model",         {}) or {}
        products  = data.get("products_services",       []) or []
        mgmt      = data.get("management",              {}) or {}
        funding   = data.get("funding",                 {}) or {}
        flags     = data.get("due_diligence_flags",     []) or []
        news      = data.get("recent_news",             []) or []
        sources   = data.get("data_sources",            []) or []
        partners  = data.get("named_partners",          []) or []
        op_hl     = data.get("operational_highlights",  []) or []
        ts        = (data.get("research_timestamp") or datetime.now(timezone.utc).isoformat())[:10]

        def js(v):
            if v is None or v == "": return '""'
            return json.dumps(str(v))

        def jsa(lst):
            if not lst: return "[]"
            return json.dumps([str(x) for x in lst if x])

        # Flag rows
        flag_rows = ""
        for f in flags:
            sev   = (f.get("severity") or "low").lower()
            color = {"high": "C00000", "medium": "ED7D31", "low": "70AD47"}.get(sev, "70AD47")
            flag_rows += f"""
    new TableRow({{children:[
      new TableCell({{borders,width:{{size:1500,type:WidthType.DXA}},shading:{{fill:"{color}",type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(sev.upper())},bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:2800,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(f.get("flag",""))},bold:true,font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:5060,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(f.get("detail",""))},font:"Arial",size:20}})]}})]}}),
    ]}}),"""

        # Funding rows
        rounds = funding.get("rounds", []) or []
        funding_rows = ""
        for fr in rounds:
            if isinstance(fr, dict):
                investors_text = fr.get("lead_investor") or ""
                other_inv = fr.get("other_investors") or ""
                # other_investors might be a list from Gemini
                if isinstance(other_inv, list):
                    other_inv = ", ".join(str(x) for x in other_inv if x)
                if other_inv:
                    investors_text = f"{investors_text}; {other_inv}" if investors_text else other_inv
                funding_rows += f"""
    new TableRow({{children:[
      new TableCell({{borders,width:{{size:1400,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(fr.get("date",""))},font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:1600,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(fr.get("round",""))},font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:1800,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(fr.get("amount",""))},bold:true,font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:4560,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(investors_text)},font:"Arial",size:20}})]}})]}}),
    ]}}),"""

        # Revenue stream bullets
        rev_streams = biz_model.get("revenue_streams", []) or []
        rev_paras = ""
        for rs in rev_streams:
            if isinstance(rs, dict):
                name = rs.get("name",""); desc = rs.get("description","")
                rev_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},children:[
      new TextRun({{text:{js(name+": " if name else "")},bold:true,font:"Arial",size:20}}),
      new TextRun({{text:{js(desc)},font:"Arial",size:20}}),
    ]}}),"""
            elif isinstance(rs, str) and rs:
                rev_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},
      children:[new TextRun({{text:{js(rs)},font:"Arial",size:20}})]}}),"""

        # Product bullets
        product_paras = ""
        for p in (products if isinstance(products, list) else []):
            if isinstance(p, dict):
                name = p.get("name",""); desc = p.get("description","")
                product_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},children:[
      new TextRun({{text:{js(name+": " if name else "")},bold:true,font:"Arial",size:20}}),
      new TextRun({{text:{js(desc)},font:"Arial",size:20}}),
    ]}}),"""
            elif isinstance(p, str) and p:
                product_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},
      children:[new TextRun({{text:{js(p)},font:"Arial",size:20}})]}}),"""

        # Operational highlights
        op_paras = ""
        for o in (op_hl if isinstance(op_hl, list) else []):
            if o:
                op_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},
      children:[new TextRun({{text:{js(str(o))},font:"Arial",size:20}})]}}),"""

        # Partner bullets
        partner_paras = ""
        for p in (partners if isinstance(partners, list) else []):
            if isinstance(p, dict):
                name = p.get("name",""); detail = p.get("detail",""); ptype = p.get("type","")
                line = f"{name} ({ptype}): {detail}" if ptype else f"{name}: {detail}"
                partner_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},
      children:[new TextRun({{text:{js(line)},font:"Arial",size:20}})]}}),"""
            elif isinstance(p, str) and p:
                partner_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},
      children:[new TextRun({{text:{js(p)},font:"Arial",size:20}})]}}),"""

        # Key people bios
        key_people = mgmt.get("key_people", []) or []
        people_paras = ""
        for person in key_people:
            if isinstance(person, dict):
                name = person.get("name",""); title = person.get("title",""); bio = person.get("bio","")
                people_paras += f"""
    new Paragraph({{spacing:{{before:120,after:40}},children:[
      new TextRun({{text:{js(name+" — "+title if title else name)},bold:true,font:"Arial",size:22}}),
    ]}}),
    new Paragraph({{spacing:{{before:0,after:100}},children:[
      new TextRun({{text:{js(bio)},font:"Arial",size:20,color:"404040"}}),
    ]}}),"""
            elif isinstance(person, str) and person:
                people_paras += f"""
    new Paragraph({{spacing:{{before:80,after:40}},
      children:[new TextRun({{text:{js(person)},bold:true,font:"Arial",size:20}})]}}),"""

        # Corporate Structure key-value table (like Claude's report)
        corp_struct_rows = ""
        corp_fields = [
            ("Legal Name", identity.get("legal_name", "")),
            ("Trading Name", identity.get("trading_name", "") or identity.get("legal_name", "")),
            ("Company Number", identity.get("company_number", "")),
            ("Status", identity.get("status", "")),
            ("Incorporated", identity.get("incorporation_date", "")),
            ("Jurisdiction", identity.get("jurisdiction", "")),
            ("Company Type", identity.get("company_type", "")),
            ("Registered Address", identity.get("registered_address", "")),
            ("Operating Address", identity.get("operating_address", "")),
            ("SIC Codes", ", ".join(identity.get("sic_codes") or [])),
            ("Ownership Structure", identity.get("ownership_structure", "")),
            ("Active Charges", identity.get("active_charges", "")),
        ]
        for i, (label, value) in enumerate(corp_fields):
            if value:
                shade = "true" if i % 2 == 0 else "false"
                corp_struct_rows += f"""
    new TableRow({{children:[
      new TableCell({{borders,width:{{size:3200,type:WidthType.DXA}},shading:{shade}?{{fill:LGRAY,type:ShadingType.CLEAR}}:undefined,margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(label)},bold:true,font:"Arial",size:20,color:DGRAY}})]}})]}}),
      new TableCell({{borders,width:{{size:6160,type:WidthType.DXA}},shading:{shade}?{{fill:LGRAY,type:ShadingType.CLEAR}}:undefined,margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(value)},font:"Arial",size:20}})]}})]}}),
    ]}}),"""

        # Key Leadership table (like Claude's report)
        leadership_rows = ""
        for person in key_people:
            if isinstance(person, dict):
                name = person.get("name", "")
                title = person.get("title", "")
                bio = person.get("bio", "")
                if name:
                    leadership_rows += f"""
    new TableRow({{children:[
      new TableCell({{borders,width:{{size:2400,type:WidthType.DXA}},margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(name)},bold:true,font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:2400,type:WidthType.DXA}},margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(title)},font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:4560,type:WidthType.DXA}},margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(bio)},font:"Arial",size:20,color:DGRAY}})]}})]}}),
    ]}}),"""

        # Financial accounts data
        fin_accounts = data.get("financial_accounts", {}) or {}
        fin_data = fin_accounts.get("financials", {}) or {}
        # Pre-extract nested fields to avoid f-string escaping issues
        _pnl = fin_data.get("profit_and_loss") or {}
        _bs = fin_data.get("balance_sheet") or {}
        _fin_turnover = _pnl.get("turnover", "")
        _fin_pbt = _pnl.get("profit_before_tax", "")
        _fin_net_assets = _bs.get("net_assets", "")
        _fin_cash = fin_data.get("cash_at_bank", "")
        _fin_employees = fin_data.get("employee_count_in_accounts", "")
        _fin_period = fin_data.get("accounts_period", "")

        # Valuation history rows
        val_history = data.get("valuation_history", []) or []
        val_rows = ""
        for v in val_history:
            if isinstance(v, dict) and v.get("valuation"):
                val_rows += f"""
    new TableRow({{children:[
      new TableCell({{borders,width:{{size:2000,type:WidthType.DXA}},margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(v.get("date",""))},font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:2800,type:WidthType.DXA}},margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(v.get("valuation",""))},bold:true,font:"Arial",size:20}})]}})]}}),
      new TableCell({{borders,width:{{size:4560,type:WidthType.DXA}},margins:{{top:60,bottom:60,left:120,right:120}},
        children:[new Paragraph({{children:[new TextRun({{text:{js(v.get("context",""))},font:"Arial",size:18,color:"666666"}})]}})]}}),
    ]}}),"""

        # Investment highlights
        inv_highlights = data.get("investment_highlights", []) or []
        inv_paras = ""
        for h in inv_highlights:
            if h:
                inv_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},
      children:[new TextRun({{text:{js(str(h))},font:"Arial",size:20}})]}}),"""

        # News paragraphs
        news_paras = ""
        for article in news[:10]:
            if isinstance(article, dict):
                title = article.get("title",""); source = article.get("source","")
                date = article.get("date",""); summary = article.get("summary","")
                if title:
                    news_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:60,after:60}},children:[
      new TextRun({{text:{js((date+" — "+source+": ") if (date or source) else "")},bold:true,font:"Arial",size:20}}),
      new TextRun({{text:{js(title)},font:"Arial",size:20}}),
      {f'new TextRun({{text:{js(" — "+summary)},color:"666666",font:"Arial",size:18}}),' if summary else ''}
    ]}}),"""

        # Source paras
        source_paras = "".join(
            f"\n    new Paragraph({{numbering:{{reference:'numbers',level:0}},children:[new TextRun({{text:{js(u[:100])},font:'Arial',size:18}})]}}),"
            for u in sources[:20]
        )

        # Data gaps
        data_gaps = data.get("data_gaps", []) or []
        gaps_paras = ""
        for gap in data_gaps:
            if gap:
                gaps_paras += f"""
    new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:40,after:40}},
      children:[new TextRun({{text:{js(str(gap))},font:"Arial",size:18,color:"888888"}})]}}),"""

        return f"""
const fs = require('fs');
const {{
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, LevelFormat, TabStopType,
}} = require('docx');

const NAVY="1F3864",BLUE="2E75B6",LGRAY="F2F2F2",DGRAY="404040";
const b={{style:BorderStyle.SINGLE,size:1,color:"CCCCCC"}};
const borders={{top:b,bottom:b,left:b,right:b}};
const nb={{style:BorderStyle.NONE,size:0,color:"FFFFFF"}};
const noBorders={{top:nb,bottom:nb,left:nb,right:nb}};

const h1=t=>new Paragraph({{heading:HeadingLevel.HEADING_1,shading:{{fill:NAVY,type:ShadingType.CLEAR}},spacing:{{before:400,after:160}},children:[new TextRun({{text:t,bold:true,color:"FFFFFF",font:"Arial",size:28}})]}});
const h2=t=>new Paragraph({{heading:HeadingLevel.HEADING_2,spacing:{{before:240,after:80}},border:{{bottom:{{style:BorderStyle.SINGLE,size:4,color:BLUE,space:1}}}},children:[new TextRun({{text:t,bold:true,color:BLUE,font:"Arial",size:24}})]}});
const kv=(label,value)=>{{if(!value)return null;return new Paragraph({{spacing:{{before:60,after:60}},children:[new TextRun({{text:label+": ",bold:true,font:"Arial",size:20,color:DGRAY}}),new TextRun({{text:String(value),font:"Arial",size:20}})]}});}};
const bullet=t=>t?new Paragraph({{numbering:{{reference:"bullets",level:0}},spacing:{{before:40,after:40}},children:[new TextRun({{text:String(t),font:"Arial",size:20}})]}}):null;
const para=t=>t?new Paragraph({{spacing:{{before:80,after:100}},children:[new TextRun({{text:String(t),font:"Arial",size:20}})]}}):null;
const spacer=()=>new Paragraph({{children:[new TextRun("")],spacing:{{before:80,after:80}}}});
const clean=arr=>arr.filter(Boolean);

const doc=new Document({{
  numbering:{{config:[
    {{reference:"bullets",levels:[{{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,style:{{paragraph:{{indent:{{left:720,hanging:360}}}}}}}}]}},
    {{reference:"numbers",levels:[{{level:0,format:LevelFormat.DECIMAL,text:"%1.",alignment:AlignmentType.LEFT,style:{{paragraph:{{indent:{{left:720,hanging:360}}}}}}}}]}},
  ]}},
  styles:{{default:{{document:{{run:{{font:"Arial",size:20}}}}}}}},
  sections:[{{
    properties:{{page:{{size:{{width:12240,height:15840}},margin:{{top:1440,right:1080,bottom:1440,left:1080}}}}}},
    headers:{{default:new Header({{children:[
      new Paragraph({{tabStops:[{{type:TabStopType.RIGHT,position:9360}}],border:{{bottom:{{style:BorderStyle.SINGLE,size:6,color:BLUE,space:1}}}},
        children:[
          new TextRun({{text:"PRIVATE & CONFIDENTIAL — PE Research Brief",bold:true,font:"Arial",size:18,color:BLUE}}),
          new TextRun({{text:"\\t{company_name}",font:"Arial",size:18,color:DGRAY}}),
        ]}}),
    ]}})  }},
    footers:{{default:new Footer({{children:[
      new Paragraph({{tabStops:[{{type:TabStopType.RIGHT,position:9360}}],border:{{top:{{style:BorderStyle.SINGLE,size:4,color:"CCCCCC",space:1}}}},
        children:[
          new TextRun({{text:"Research Date: {ts}",font:"Arial",size:16,color:"888888"}}),
          new TextRun({{text:"\\tPage ",font:"Arial",size:16,color:"888888"}}),
          new TextRun({{children:[PageNumber.CURRENT],font:"Arial",size:16,color:"888888"}}),
        ]}}),
    ]}})  }},
    children:clean([
      new Paragraph({{alignment:AlignmentType.CENTER,spacing:{{before:720,after:120}},children:[new TextRun({{text:"PRIVATE EQUITY RESEARCH BRIEF",bold:true,font:"Arial",size:52,color:NAVY}})]}},),
      new Paragraph({{alignment:AlignmentType.CENTER,spacing:{{before:0,after:60}},children:[new TextRun({{text:{js(company_name)},bold:true,font:"Arial",size:40,color:DGRAY}})]}},),
      new Paragraph({{alignment:AlignmentType.CENTER,spacing:{{before:0,after:720}},children:[new TextRun({{text:"Prepared: {ts}  |  Confidential  |  For Internal Use Only",font:"Arial",size:20,color:"888888"}})]}},),
      new Table({{width:{{size:9360,type:WidthType.DXA}},columnWidths:[4680,4680],rows:[new TableRow({{children:[
        new TableCell({{borders:noBorders,shading:{{fill:LGRAY,type:ShadingType.CLEAR}},margins:{{top:120,bottom:120,left:200,right:200}},children:[new Paragraph({{children:[new TextRun({{text:"Data Confidence: ",bold:true,font:"Arial",size:20,color:DGRAY}}),new TextRun({{text:{js((data.get("data_confidence") or "").upper())},bold:true,font:"Arial",size:20,color:BLUE}})]}})]}},),
        new TableCell({{borders:noBorders,shading:{{fill:LGRAY,type:ShadingType.CLEAR}},margins:{{top:120,bottom:120,left:200,right:200}},children:[new Paragraph({{children:[new TextRun({{text:{js(data.get("confidence_notes") or "")},font:"Arial",size:20,color:DGRAY}})]}})]}},),
      ]}})]}}),
      spacer(),

      h1("1. EXECUTIVE SUMMARY"),
      para({js(data.get("executive_summary") or "")}),
      spacer(),

      h1("2. CORPORATE STRUCTURE"),
      {f'''new Table({{width:{{size:9360,type:WidthType.DXA}},columnWidths:[3200,6160],rows:[
        {corp_struct_rows}
      ]}})''' if corp_struct_rows else ''},
      spacer(),

      h1("3. ABOUT THE COMPANY"),
      para({js(data.get("about_company") or "")}),
      spacer(),

      h1("4. BUSINESS MODEL"),
      para({js(biz_model.get("narrative") or "")}),
      {f'h2("Revenue Streams"),{rev_paras}' if rev_paras else ''},
      {f'kv("Unit Economics", {js(biz_model.get("unit_economics_note"))}),' if biz_model.get("unit_economics_note") else ''},
      spacer(),

      h1("5. PRODUCTS & SERVICES"),
      {product_paras if product_paras else 'para("No product data available."),'},
      spacer(),

      h1("6. GEOGRAPHIC REACH"),
      para({js(data.get("geographic_reach") or "")}),
      spacer(),

      h1("7. FUNDING & FINANCIALS"),
      kv("Total Raised", {js(funding.get("total_raised"))}),
      {f'kv("Debt Facilities", {js(funding.get("debt_facilities"))}),' if funding.get("debt_facilities") else ''},
      {f'kv("Valuation", {js(funding.get("valuation_notes"))}),' if funding.get("valuation_notes") else ''},
      spacer(),
      {f"""new Table({{width:{{size:9360,type:WidthType.DXA}},columnWidths:[1400,1600,1800,4560],rows:[
        new TableRow({{tableHeader:true,children:[
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Date",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Round",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Amount",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Lead Investors",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
        ]}}),
        {funding_rows}
      ]}})""" if rounds else ""},
      spacer(),
      {f'h2("All Investors"),...clean({jsa(funding.get("investors") or [])}.map(bullet)),' if funding.get("investors") else ""},
      spacer(),

      {f'h1("8. OPERATIONAL HIGHLIGHTS"),{op_paras}spacer(),' if op_paras else ""},
      {f'h1("9. KEY PARTNERS & CUSTOMERS"),{partner_paras}spacer(),' if partner_paras else ""},

      h1("10. KEY LEADERSHIP"),
      {f'''h2("Leadership Team"),
      new Table({{width:{{size:9360,type:WidthType.DXA}},columnWidths:[2400,2400,4560],rows:[
        new TableRow({{tableHeader:true,children:[
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Name",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Role",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Background",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
        ]}}),
        {leadership_rows}
      ]}})''' if leadership_rows else people_paras if people_paras else 'para("No management data available.")'},
      {f'h2("Board of Directors"),...clean({jsa(mgmt.get("directors") or [])}.map(bullet)),' if mgmt.get("directors") else ""},
      {f'kv("Board Notes", {js(mgmt.get("board_notes"))}),' if mgmt.get("board_notes") else ""},
      spacer(),

      {f'h1("11. RECENT NEWS & DEVELOPMENTS"),{news_paras}spacer(),' if news_paras else ""},

      {f'''h1("12. VALUATION HISTORY"),
      new Table({{width:{{size:9360,type:WidthType.DXA}},columnWidths:[2000,2800,4560],rows:[
        new TableRow({{tableHeader:true,children:[
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:60,bottom:60,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Date",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:60,bottom:60,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Valuation",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:60,bottom:60,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"Context / Source",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
        ]}}),
        {val_rows}
      ]}}),
      spacer(),''' if val_rows else ""}

      {f'''h1("13. FINANCIAL ACCOUNTS"),
      kv("Filing Date", {js(fin_accounts.get("filing_date", ""))}),
      kv("Accounts Period", {js(_fin_period)}),
      kv("Turnover", {js(_fin_turnover)}),
      kv("Profit Before Tax", {js(_fin_pbt)}),
      kv("Net Assets", {js(_fin_net_assets)}),
      kv("Cash at Bank", {js(_fin_cash)}),
      kv("Employees (per accounts)", {js(_fin_employees)}),
      kv("Document", {js(fin_accounts.get("doc_url", ""))}),
      spacer(),''' if fin_accounts.get("doc_url") or fin_data else f'''h1("13. FINANCIAL ACCOUNTS"),
      para("Filed accounts document not retrieved. Check Companies House directly for the latest filing."),
      spacer(),'''}

      {f'h1("14. INVESTMENT HIGHLIGHTS"),{inv_paras}spacer(),' if inv_paras else ""},

      h1("15. DUE DILIGENCE FLAGS"),
      new Table({{width:{{size:9360,type:WidthType.DXA}},columnWidths:[1500,2800,5060],rows:[
        new TableRow({{tableHeader:true,children:[
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"SEVERITY",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"FLAG",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
          new TableCell({{borders,shading:{{fill:NAVY,type:ShadingType.CLEAR}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"DETAIL",bold:true,color:"FFFFFF",font:"Arial",size:20}})]}})]}},),
        ]}}),
        {flag_rows if flag_rows else """new TableRow({{children:[new TableCell({{borders,width:{{size:9360,type:WidthType.DXA}},margins:{{top:80,bottom:80,left:120,right:120}},children:[new Paragraph({{children:[new TextRun({{text:"No material flags identified.",font:"Arial",size:20,color:"888888"}})]}})]}})]}}),"""}
      ]}}),
      spacer(),

      {f'h1("16. DATA GAPS"),{gaps_paras}spacer(),' if gaps_paras else ""},

      h1("17. DATA SOURCES"),
      para("The following sources were searched and analysed:"),
      {source_paras if source_paras else ""},
      spacer(),

    ].filter(Boolean)),
  }}],
}});

Packer.toBuffer(doc).then(buf=>{{fs.writeFileSync({js(output_path)},buf);console.log("OK");}}).catch(err=>{{console.error(err);process.exit(1);}});
"""


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — Enhanced with better data capture
# ══════════════════════════════════════════════════════════════════════════════

class ResearchOrchestrator:
    def __init__(self):
        self.jurisdiction_agent = JurisdictionAgent()
        self.synthesiser        = GeminiSynthesiser()
        self.reporter           = ReportGenerator()

    async def run(self, job_id: str, company_name: str, country_code: str = "auto", website_url: str = ""):
        def update(progress: int, message: str):
            jobs[job_id].update({"progress": progress, "message": message})
            persist_job(jobs[job_id])
            logger.info(f"[{job_id}] {progress}%  {message}")

        try:
            jobs[job_id]["status"] = "running"
            persist_job(jobs[job_id])
            loop = asyncio.get_event_loop()

            # Reset TinyFish domain failure cache for new job
            _tinyfish_failed_domains.clear()
            _urls_already_processed.clear()

            # 1. Jurisdiction
            update(5, "Detecting jurisdiction...")
            jur = await loop.run_in_executor(
                None, self.jurisdiction_agent.detect, company_name, country_code
            )
            jobs[job_id]["jurisdiction"] = jur
            persist_job(jobs[job_id])
            logger.info(f"Jurisdiction: {jur['label']} via {jur['method']}")

            # 1b. Generate localized search terms for non-English countries
            localized_terms = {}
            detected_code = jur.get("country_code", "gb")
            if detected_code not in ENGLISH_PRIMARY and detected_code in COUNTRY_LANGUAGES:
                update(7, f"Generating {COUNTRY_LANGUAGES[detected_code]} search terms...")
                localized_terms = await loop.run_in_executor(
                    None, get_localized_search_terms, company_name, detected_code
                )

            # 2. ReAct Agent
            update(10, f"ReAct agent researching {company_name} (14-phase deep research)...")

            # Build user message with localized terms if available
            localized_hint = ""
            if localized_terms:
                lang = localized_terms.get("language", COUNTRY_LANGUAGES.get(detected_code, "local language"))
                localized_hint = (
                    f"\n\nIMPORTANT: This company is based in {jur['label']} where the primary language is {lang}. "
                    f"For EACH search phase, run the search TWICE: once in English AND once in {lang}. "
                    f"Here are pre-translated search queries you should use alongside the English ones:\n"
                    + "\n".join(f"- {k}: {v}" for k, v in localized_terms.items() if k != "language")
                )

            # Build website anchor — if user provided a URL, the agent must use it
            website_anchor = ""
            if website_url:
                from urllib.parse import urlparse
                domain = urlparse(website_url).netloc.replace("www.", "")
                website_anchor = (
                    f"\n\nCRITICAL DISAMBIGUATION: The user has confirmed the company's website is {website_url} (domain: {domain}). "
                    f"You MUST use this as the official website. Do NOT research a different company. "
                    f"In Phase 1, skip the discovery search — go directly to extract_page(\"{website_url}\", extraction_goal=\"company_overview\"). "
                    f"For all subsequent phases, prefer deep_search with domain=\"{domain}\" to stay on the correct company. "
                    f"If Exa returns results from other domains about a different company with a similar name, DISCARD those results."
                )

            def run_agent():
                agent        = build_react_agent(company_name, jur["label"])
                step_count   = 0
                final_output = ""
                tool_logs    = []
                all_sources  = []

                for chunk in agent.stream(
                    {"messages": [{"role": "user", "content":
                        f"Research {company_name} for PE due diligence. "
                        f"You MUST complete ALL 14 research phases before writing your analysis. "
                        f"DO NOT stop early. Every phase produces critical data for the investment brief. "
                        f"After each phase, note what you found and what gaps remain. "
                        f"Use targeted extraction_goal parameters when calling extract_page."
                        f"{website_anchor}"
                        f"{localized_hint}"}]},
                    {"recursion_limit": MAX_AGENT_ITERATIONS * 3},
                ):
                    if "agent" in chunk:
                        for m in chunk["agent"].get("messages", []):
                            if hasattr(m, "content") and m.content:
                                final_output = str(m.content)
                                logger.info(f"Agent: {str(m.content)[:120]}")

                    if "tools" in chunk:
                        step_count += 1
                        for m in chunk["tools"].get("messages", []):
                            full_content = str(getattr(m, "content", ""))

                            # Extract sources from FULL content before truncating
                            try:
                                data = json.loads(full_content)

                                if "results" in data and isinstance(data["results"], list):
                                    for r in data["results"]:
                                        if isinstance(r, dict) and r.get("url"):
                                            all_sources.append(r["url"])

                                if data.get("company_number"):
                                    all_sources.append(
                                        f"https://find-and-update.company-information.service.gov.uk/company/{data['company_number']}"
                                    )

                                if "companies" in data and isinstance(data["companies"], list):
                                    for c in data["companies"]:
                                        if isinstance(c, dict) and c.get("opencorporates_url"):
                                            all_sources.append(c["opencorporates_url"])

                                if data.get("url") and data.get("success") is not None:
                                    all_sources.append(data["url"])

                            except Exception:
                                pass

                            # Store MORE content for synthesis — funding articles need full text
                            tool_logs.append({"step": step_count, "result": full_content[:8000]})
                            logger.info(f"Tool step {step_count}: {full_content[:150]}")

                        pct = min(10 + step_count * 2, 75)
                        jobs[job_id].update({"progress": pct, "message": f"Research phase {step_count}..."})
                        persist_job(jobs[job_id])

                return {
                    "output":      final_output,
                    "steps":       step_count,
                    "tool_logs":   tool_logs,
                    "all_sources": all_sources,
                }

            agent_result = await loop.run_in_executor(None, run_agent)
            steps_done   = agent_result.get("steps", 0)
            logger.info(f"Agent completed {steps_done} steps")

            unique_sources = list(dict.fromkeys(agent_result.get("all_sources", [])))
            logger.info(f"Captured {len(unique_sources)} unique sources from {steps_done} steps")

            # Filter out irrelevant sources (random LinkedIn profiles, Reddit, unrelated sites)
            unique_sources = filter_relevant_sources(unique_sources, company_name, website_url)

            # 3. Synthesis
            update(80, f"Synthesising {len(unique_sources)} sources with Gemini 2.5 Pro...")
            synthesis = await loop.run_in_executor(
                None,
                self.synthesiser.synthesise,
                company_name,
                agent_result["output"],
                agent_result["tool_logs"],
                unique_sources,
                jur,
            )
            synthesis["jurisdiction_detected"] = jur

            # 3b. Entity Resolution
            update(82, "Resolving entity identity...")
            entity_resolution = await loop.run_in_executor(
                None, resolve_entity, agent_result["tool_logs"], company_name
            )
            synthesis["entity_resolution"] = entity_resolution

            # 3c. Financial Extraction (UK companies only)
            financial_data = {}
            company_number = entity_resolution.get("canonical_company_number", "")
            if company_number and jur.get("use_ch"):
                update(85, f"Extracting financial accounts for {company_number}...")
                financial_data = await loop.run_in_executor(
                    None, extract_financial_accounts, company_number
                )
                # Always store financial data (even partial) so report shows doc link
                if financial_data.get("doc_url") or financial_data.get("success"):
                    synthesis["financial_accounts"] = financial_data
                    logger.info(f"Financial extraction: success={financial_data.get('success')}")

            # 4. Word report
            update(93, "Generating Word report...")
            safe     = re.sub(r'[^a-zA-Z0-9_\-]', '', company_name.replace(" ", "_"))[:50]
            ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("reports", f"{safe}_{ts}.docx")

            await loop.run_in_executor(
                None, self.reporter.generate, synthesis, company_name, out_path
            )

            storage_path = None
            if supabase_enabled():
                storage_name = f"{job_id}/{os.path.basename(out_path)}"
                storage_path = await loop.run_in_executor(None, upload_file_to_supabase, out_path, storage_name)

            jobs[job_id].update({
                "status":   "complete",
                "progress": 100,
                "message":  "Research complete",
                "result": {
                    "synthesis":    synthesis,
                    "report_path":  out_path,
                    "report_storage_path": storage_path,
                    "agent_steps":  steps_done,
                    "jurisdiction": jur,
                },
            })
            persist_job(jobs[job_id])

        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            jobs[job_id].update({
                "status":  "failed",
                "message": "Research failed",
                "error":   "An internal error occurred during research",
            })
            persist_job(jobs[job_id])


# ── API Routes ────────────────────────────────────────────────────────────────

@app.post("/api/discover")
async def discover_companies(req: CompanyDiscoveryRequest):
    """
    STEP 1: Quick search to find matching companies.
    Returns a list of candidates with name, website, description, country.
    User picks the right one, then calls /api/research with the website_url.
    This prevents researching the wrong 'Amal Invest' when there are 5 similar names.
    """
    candidates = []

    # Search Exa for company matches
    if exa_client:
        try:
            results = exa_client.search_and_contents(
                req.company_name,
                type="auto",
                num_results=8,
                category="company",
                highlights={"max_characters": 500, "query": f"{req.company_name} company about"},
                text={"max_characters": 300},
            )
            if results and hasattr(results, "results"):
                for r in results.results:
                    url = getattr(r, "url", "")
                    title = getattr(r, "title", "")
                    text = getattr(r, "text", "") or ""
                    highlights = getattr(r, "highlights", []) or []

                    # Skip obvious non-company pages (LinkedIn, news articles, etc)
                    if any(skip in url.lower() for skip in
                           ["linkedin.com/in/", "twitter.com", "facebook.com",
                            "reddit.com", "youtube.com", "wikipedia.org/wiki/",
                            "crunchbase.com", "bloomberg.com/quote"]):
                        continue

                    # Extract domain as a proxy for company identity
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc.replace("www.", "")

                    description = highlights[0] if highlights else text[:200]

                    candidates.append({
                        "name":        title,
                        "website":     url,
                        "domain":      domain,
                        "description": description.strip(),
                    })

            logger.info(f"Discovery: found {len(candidates)} candidates for '{req.company_name}'")
        except Exception as e:
            logger.error(f"Discovery Exa error: {e}")

    # Also check OpenCorporates
    oc_candidates = []
    try:
        resp = req_lib.get(
            "https://api.opencorporates.com/v0.4/companies/search",
            params={"q": req.company_name, "per_page": 5},
            timeout=10,
        )
        if resp.status_code == 200:
            companies = resp.json().get("results", {}).get("companies", [])
            for item in companies[:5]:
                c = item.get("company", {})
                oc_candidates.append({
                    "name":            c.get("name", ""),
                    "jurisdiction":    c.get("jurisdiction_code", ""),
                    "company_number":  c.get("company_number", ""),
                    "status":          c.get("current_status", ""),
                    "incorporation":   c.get("incorporation_date", ""),
                    "address":         c.get("registered_address_in_full", ""),
                    "opencorporates_url": c.get("opencorporates_url", ""),
                })
    except Exception as e:
        logger.warning(f"Discovery OpenCorporates error: {e}")

    # De-duplicate by domain
    seen_domains = set()
    unique_candidates = []
    for c in candidates:
        d = c.get("domain", "")
        if d and d not in seen_domains:
            seen_domains.add(d)
            unique_candidates.append(c)

    return {
        "query":            req.company_name,
        "web_results":      unique_candidates[:6],
        "registry_results": oc_candidates,
        "message":          f"Found {len(unique_candidates)} web matches and {len(oc_candidates)} registry matches. "
                            f"Please select the correct company and pass its website_url to /api/research.",
        "usage":            "POST /api/research with {company_name, website_url, country_code} to start research on the confirmed company.",
    }


@app.post("/api/research")
async def start_research(req: ResearchRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "job_id":       job_id,
        "status":       "queued",
        "progress":     0,
        "message":      "Queued",
        "result":       None,
        "error":        None,
        "company":      req.company_name,
        "country_code": req.country_code,
        "website_url":  req.website_url,
        "created":      datetime.now(timezone.utc).isoformat(),
    }
    persist_job(jobs[job_id])
    background_tasks.add_task(
        ResearchOrchestrator().run, job_id, req.company_name, req.country_code, req.website_url
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/research/{job_id}")
async def get_job(job_id: str):
    if job_id in jobs:
        return jobs[job_id]
    job = fetch_job_from_supabase(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/research/{job_id}/download")
async def download_report(job_id: str):
    job = jobs.get(job_id) or fetch_job_from_supabase(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "complete":
        raise HTTPException(400, "Report not ready yet")
    result = job.get("result") or {}
    path = result.get("report_path")
    # Security: validate path stays inside reports directory
    safe_company = re.sub(r'[^a-zA-Z0-9_\- ]', '', job.get('company', 'Report'))[:80]
    storage_path = result.get("report_storage_path")
    if storage_path:
        from fastapi.responses import Response
        payload = download_file_from_supabase(storage_path)
        if payload:
            return Response(
                content=payload,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="PE_Research_{safe_company}.docx"'},
            )

    if not path:
        raise HTTPException(404, "Report file missing")
    real_path = os.path.realpath(path)
    if not real_path.startswith(os.path.realpath("reports")):
        raise HTTPException(403, "Invalid report path")
    if not os.path.exists(path):
        json_path = path.replace(".docx", ".json")
        if os.path.exists(json_path):
            return FileResponse(json_path, filename=f"PE_Research_{safe_company}.json")
        raise HTTPException(404, "Report file missing")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=f"PE_Research_{safe_company}.docx",
    )


@app.get("/api/jobs")
async def list_jobs():
    remote_jobs = list_jobs_from_supabase()
    if remote_jobs is not None:
        return [{k: v for k, v in j.items() if k != "result"} for j in remote_jobs]
    return [{k: v for k, v in j.items() if k != "result"} for j in jobs.values()]


@app.get("/api/status")
async def api_status():
    return {
        "status":          "ok",
        "version":         "8.0.0",
        "storage_mode":    "supabase" if supabase_enabled() else "local",
    }


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ── Security: Global exception handler — never leak internals ────────────────
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "An internal error occurred"})

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)
