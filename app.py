# app.py — Listing-Lift (FULL) : Titles + Tags + Description + Downloads + Tag Guard + AI Assist
# Store: https://listing-lift.lemonsqueezy.com/
# Support: hromany@hotmail.com

import os
import json
import re
import csv
import io
from datetime import date
from pathlib import Path
import urllib.parse

import streamlit as st
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
APP_TITLE = "Listing-Lift — Etsy Listing Writer + Tag Guard"
STORE_URL = "https://listing-lift.lemonsqueezy.com/"
SUPPORT_EMAIL = "hromany@hotmail.com"

PRO_USERS_FILE = "pro_users.json"
USAGE_FILE = "usage_log.json"

FREE_DAILY_LIMIT = 5
MAX_TITLE_LEN = 140
ETSY_TAG_MAX_LEN = 20
ETSY_TAG_COUNT = 13

# Optional direct checkout link (/checkout/buy/...)
LEMON_CHECKOUT_URL = ""

# AI
AI_MODEL = "gpt-4o-mini"
AI_ENABLED_DEFAULT = True
AI_MAX_OUTPUT_CHARS = 7000
AI_FREE_DAILY_LIMIT = 1   # ✅ Free users can use AI Assist 1 time/day (Pro unlimited)

# =========================
# EXAMPLE
# =========================
EXAMPLE_INPUT = {
    "product": "Minimalist Necklace",
    "material": "925 sterling silver",
    "style": "minimalist",
    "color": "gold",
    "audience": "her",
    "occasion": "birthday",
    "personalization": "Add initial letter",
    "keywords": "dainty necklace, initial charm, gift for her",
    "benefit": "Elegant everyday style + gift-ready packaging",
    "season": "Spring",
    "features": "Handmade with care\nGift-ready packaging\nTimeless minimalist look",
    "materials_desc": "Sterling silver, hypoallergenic",
    "sizing": "16-18 inch chain, adjustable",
    "shipping": "Processing 1-2 days, tracked shipping available",
    "paste_tags": "",
}

FORM_KEYS = [
    "product", "material", "style", "color",
    "audience", "occasion", "personalization",
    "keywords", "benefit", "season",
    "features", "materials_desc", "sizing", "shipping",
    "paste_tags",
]

# =========================
# Reset / Example
# =========================
def apply_example():
    for k, v in EXAMPLE_INPUT.items():
        st.session_state[k] = v
    st.rerun()

def reset_inputs():
    # ✅ reliable reset
    for k in FORM_KEYS:
        st.session_state[k] = ""
    st.session_state["season"] = "None"
    st.rerun()

# =========================
# JSON storage
# =========================
def _read_json(path: str, default):
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: str, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def load_pro_users() -> set[str]:
    """
    Supports:
    1) ["a@b.com","c@d.com"]
    2) {"emails":["a@b.com","c@d.com"]}
    3) {"a@b.com": true, "c@d.com": true}
    """
    data = _read_json(PRO_USERS_FILE, default={})
    if isinstance(data, dict) and "emails" in data and isinstance(data["emails"], list):
        return {str(e).strip().lower() for e in data["emails"] if str(e).strip()}
    if isinstance(data, dict):
        out = set()
        for k, v in data.items():
            if v is True or v == 1:
                out.add(str(k).strip().lower())
        return out
    if isinstance(data, list):
        return {str(e).strip().lower() for e in data if str(e).strip()}
    return set()

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    email = email.strip().lower()
    return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email) is not None

def load_usage() -> dict:
    return _read_json(USAGE_FILE, default={})

def save_usage(usage: dict):
    _write_json(USAGE_FILE, usage)

def today_key() -> str:
    return date.today().isoformat()

def get_free_used(usage: dict, email: str) -> int:
    email = (email or "").strip().lower()
    return int(usage.get(today_key(), {}).get(email, 0))

def inc_free_used(usage: dict, email: str):
    email = (email or "").strip().lower()
    tk = today_key()
    usage.setdefault(tk, {})
    usage[tk][email] = int(usage[tk].get(email, 0)) + 1

# AI usage counters (for Free AI limit)
def get_ai_used(usage: dict, email: str) -> int:
    email = (email or "").strip().lower()
    return int(usage.get(today_key(), {}).get(f"ai::{email}", 0))

def inc_ai_used(usage: dict, email: str):
    email = (email or "").strip().lower()
    tk = today_key()
    usage.setdefault(tk, {})
    key = f"ai::{email}"
    usage[tk][key] = int(usage[tk].get(key, 0)) + 1

# =========================
# UI helpers
# =========================
def copy_button(text: str, key: str, label="Copy"):
    safe_text = (text or "").replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = f"""
    <div style="display:flex; gap:8px; align-items:center;">
      <button
        style="border:1px solid #ddd; background:#fff; padding:6px 10px; border-radius:8px; cursor:pointer; font-size:14px; width: 100%;"
        onclick="navigator.clipboard.writeText(`{safe_text}`); this.innerText='Copied ✅'; setTimeout(()=>this.innerText='{label}', 1200);"
        id="{key}">
        {label}
      </button>
    </div>
    """
    components.html(html, height=44)

def build_lemon_link(base_url: str, email: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    if not email:
        return base_url
    parsed = urllib.parse.urlparse(base_url)
    q = dict(urllib.parse.parse_qsl(parsed.query))
    q["checkout[email]"] = email
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(q, doseq=True)))

def clamp_text(s: str, max_chars: int) -> str:
    s = s or ""
    return s if len(s) <= max_chars else s[:max_chars] + "..."

# =========================
# Base SEO logic (no AI)
# =========================
SEASONAL_PACKS = {
    "None": [],
    "Spring": ["spring", "easter", "mother's day"],
    "Summer": ["summer", "beach", "vacation"],
    "Autumn": ["fall", "autumn", "halloween", "thanksgiving"],
    "Winter": ["winter", "christmas", "new year"],
}

STOPWORDS = {"a","an","the","and","or","of","for","to","with","in","on","at","by","from","this","that","your","you","is","are"}

def clean_kw(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s'-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_keywords(s: str) -> list[str]:
    parts = [clean_kw(x) for x in (s or "").split(",")]
    return [p for p in parts if p]

def title_variations(product: str, main_kws: list[str], material: str, style: str,
                     audience: str, occasion: str, personalization: str, color: str, season: str) -> list[str]:
    product = clean_kw(product)
    material = clean_kw(material)
    style = clean_kw(style)
    audience = clean_kw(audience)
    occasion = clean_kw(occasion)
    personalization = clean_kw(personalization)
    color = clean_kw(color)

    kws = [k for k in main_kws if k]
    top = kws[:3]
    seasonal = SEASONAL_PACKS.get(season, [])
    season_kw = seasonal[0] if seasonal else ""

    templates = [
        "{kw1} {product} - {material} {style} | {audience} {occasion}",
        "{product} {kw1} {kw2} | {personalization} {audience} Gift",
        "{kw1} {kw2} {product} | {color} {style} - Perfect {occasion}",
        "{season} {kw1} {product} | Unique {audience} Gift - {material}",
        "{product} | {kw1} {style} {material} - {occasion} Gift Idea",
        "{kw1} {product} | {personalization} - {audience} {occasion} Gift",
        "{kw1} {kw2} {product} | Handmade {style} - {audience}",
        "{product} {kw1} | Premium {material} - {color} {occasion}",
    ]

    kw1 = top[0] if len(top) > 0 else ""
    kw2 = top[1] if len(top) > 1 else ""
    kw3 = top[2] if len(top) > 2 else ""

    def fill(t: str) -> str:
        out = t.format(
            kw1=kw1, kw2=kw2, kw3=kw3,
            product=product, material=material, style=style,
            audience=audience, occasion=occasion, personalization=personalization,
            color=color, season=season_kw
        )
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"\|\s*\|", "|", out)
        out = out.strip(" -|")
        return out

    titles = []
    for t in templates:
        cand = fill(t)
        if cand and cand not in titles:
            titles.append(cand)
    return titles[:8]

def title_score(title: str, product: str, main_kws: list[str]) -> int:
    score = 0
    t_norm = normalize(title)
    p_norm = normalize(product)
    if p_norm and p_norm in t_norm:
        score += 20

    kw_tokens = []
    for kw in (main_kws or []):
        kw_tokens += [w for w in normalize(kw).split() if w and w not in STOPWORDS]
    kw_tokens = list(dict.fromkeys(kw_tokens))
    hits = [w for w in kw_tokens if w in t_norm]
    score += min(25, 5 * len(hits))

    n = len(title)
    if 110 <= n <= MAX_TITLE_LEN:
        score += 18
    elif 90 <= n < 110:
        score += 10
    elif n > MAX_TITLE_LEN:
        score -= 25
    elif 0 < n < 90:
        score -= 8

    if " | " in title or " - " in title:
        score += 3

    return score

def rank_titles(titles: list[str], product: str, main_kws: list[str]) -> list[dict]:
    out = []
    for t in titles:
        out.append({"title": t, "score": title_score(t, product, main_kws)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def strong_first_two_lines(product: str, main_kws: list[str], benefit: str, audience: str,
                           occasion: str, personalization: str, season: str) -> tuple[str, str]:
    product = clean_kw(product)
    benefit = clean_kw(benefit)
    audience = clean_kw(audience)
    occasion = clean_kw(occasion)
    personalization = clean_kw(personalization)

    kw = clean_kw(main_kws[0]) if main_kws else ""
    seasonal = SEASONAL_PACKS.get(season, [])
    season_kw = seasonal[0] if seasonal else ""

    line1 = f"{product}: {benefit}" if benefit else (f"{product}: {kw} designed to stand out" if kw else f"{product}: made to stand out")
    parts = []
    if audience:
        parts.append(f"Perfect for {audience}")
    if occasion:
        parts.append(f"{occasion} gifts")
    if personalization:
        parts.append(personalization)
    if season_kw:
        parts.append(season_kw)
    line2 = " • ".join(parts) if parts else "A thoughtful gift that feels premium and personal."
    return re.sub(r"\s+", " ", line1).strip(), re.sub(r"\s+", " ", line2).strip()

def full_description(product: str, main_kws: list[str], benefit: str, features: str, materials: str,
                     sizing: str, shipping: str, personalization: str, audience: str, occasion: str, season: str) -> str:
    l1, l2 = strong_first_two_lines(product, main_kws, benefit, audience, occasion, personalization, season)
    bullets = [clean_kw(x) for x in (features or "").split("\n") if clean_kw(x)][:8]
    kws_line = ", ".join([k for k in main_kws[:8] if k])

    desc = [l1, l2, "", "✅ Why you'll love it:"]
    if bullets:
        desc += [f"• {b}" for b in bullets]
    else:
        desc += ["• High quality, made with care", "• Unique look that matches multiple styles", "• Great as a gift or for everyday use"]

    if materials:
        desc += ["", f"🧵 Materials: {clean_kw(materials)}"]
    if sizing:
        desc += ["", f"📏 Size / Details: {clean_kw(sizing)}"]
    if personalization:
        desc += ["", f"✨ Personalization: {clean_kw(personalization)}"]
    if shipping:
        desc += ["", f"🚚 Shipping: {clean_kw(shipping)}"]
    if kws_line:
        desc += ["", f"🔎 Keywords: {kws_line}"]
    desc += ["", "📩 Questions? Email support anytime — I’m happy to help!"]
    return "\n".join(desc)

# =========================
# TAG GUARD + FIX
# =========================
def parse_tags(text: str) -> list[str]:
    if not text:
        return []
    raw = re.split(r"[,\n]+", text)
    return [clean_kw(x) for x in raw if clean_kw(x)]

def smart_trim_tag(tag: str, max_len: int = ETSY_TAG_MAX_LEN) -> str:
    t = normalize(tag)
    if not t:
        return ""
    if len(t) <= max_len:
        return t

    words = [w for w in t.split() if w]
    words2 = [w for w in words if w not in STOPWORDS]
    if words2:
        words = words2

    while words and len(" ".join(words)) > max_len:
        words.pop()

    t2 = " ".join(words).strip()
    if t2 and len(t2) <= max_len:
        return t2

    return t[:max_len].strip()

def dedupe_keep_order(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for x in items:
        k = x.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(x)
    return out

def token_overlap(tag: str) -> set[str]:
    return {w for w in normalize(tag).split() if w and w not in STOPWORDS}

def tag_guard_fix(tags: list[str], required_count: int = ETSY_TAG_COUNT) -> dict:
    original = tags[:]

    fixed = []
    for t in tags:
        tt = smart_trim_tag(t, ETSY_TAG_MAX_LEN)
        if tt:
            fixed.append(tt)

    fixed = dedupe_keep_order(fixed)

    kept = []
    seen_tokens = set()
    for t in fixed:
        toks = token_overlap(t)
        if toks and toks.issubset(seen_tokens) and len(kept) >= 8:
            continue
        kept.append(t)
        seen_tokens |= toks

    fixed = kept

    fillers = ["gift idea", "handmade", "custom", "unique gift", "for her", "for him", "home decor", "birthday gift"]
    i = 0
    while len(fixed) < required_count and i < len(fillers):
        ft = smart_trim_tag(fillers[i], ETSY_TAG_MAX_LEN)
        if ft and ft not in fixed:
            fixed.append(ft)
        i += 1

    fixed = fixed[:required_count]

    too_long = [t for t in fixed if len(t) > ETSY_TAG_MAX_LEN]
    dups = len(fixed) - len(set([t.lower() for t in fixed]))
    return {"original": original, "fixed": fixed, "audit": {"count": len(fixed), "dups": dups, "too_long": too_long}}

# =========================
# AI ASSIST
# =========================
def get_openai_key() -> str:
    try:
        k = st.secrets.get("OPENAI_API_KEY", "")
        if k:
            return str(k)
    except Exception:
        pass
    return os.environ.get("OPENAI_API_KEY", "")

def ai_assist_suggest(payload: dict) -> dict:
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Install 'openai>=1.0.0' in requirements.txt") from e

    client = OpenAI(api_key=api_key)
    user_text = json.dumps(payload, ensure_ascii=False)

    instructions = (
        "You are an Etsy listing assistant. Improve the user's listing inputs. "
        "Return ONLY valid JSON with keys: "
        "keywords (comma-separated), benefit (short), features (newline bullets), "
        "audience, occasion, personalization, extra_tags (comma-separated). "
        "No extra text outside JSON. Avoid trademarks."
    )

    resp = client.responses.create(
        model=AI_MODEL,
        instructions=instructions,
        input=f"User listing data JSON:\n{user_text}"
    )

    text = getattr(resp, "output_text", "") or ""
    text = clamp_text(text.strip(), AI_MAX_OUTPUT_CHARS)

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise RuntimeError("AI did not return JSON.")
    data = json.loads(m.group(0))

    return {
        "keywords": clean_kw(str(data.get("keywords", ""))),
        "benefit": clean_kw(str(data.get("benefit", ""))),
        "features": str(data.get("features", "")).strip(),
        "audience": clean_kw(str(data.get("audience", ""))),
        "occasion": clean_kw(str(data.get("occasion", ""))),
        "personalization": clean_kw(str(data.get("personalization", ""))),
        "extra_tags": clean_kw(str(data.get("extra_tags", ""))),
    }

# =========================
# UI
# =========================
st.set_page_config(page_title="Listing-Lift", page_icon="🚀", layout="centered")
st.title(APP_TITLE)

top1, top2 = st.columns([2, 1])
with top1:
    st.caption(f"Upgrade store: {STORE_URL}")
with top2:
    st.link_button("🛒 Buy / Upgrade", STORE_URL, use_container_width=True)

pro_users = load_pro_users()
usage = load_usage()

with st.sidebar:
    st.subheader("Account")
    email = st.text_input("Your email", placeholder="you@email.com").strip().lower()
    email_ok = bool(email) and is_valid_email(email)
    if email and not email_ok:
        st.warning("Please enter a valid email.")

    pro_active = email_ok and (email in pro_users)

    st.markdown("---")
    st.subheader("Plan")
    if pro_active:
        st.success("✅ Pro active (by email)")
        st.write("Unlimited generations + unlimited AI Assist.")
    else:
        st.info("Free plan")
        st.write(f"- {FREE_DAILY_LIMIT} generations/day")
        st.write(f"- AI Assist: {AI_FREE_DAILY_LIMIT}/day")
        st.link_button("Open Listing-Lift Store", STORE_URL, use_container_width=True)

    st.markdown("---")
    st.subheader("Support")
    st.write(SUPPORT_EMAIL)
    copy_button(SUPPORT_EMAIL, key="copy_support_email_sidebar", label="Copy Support Email")

if not email_ok:
    st.info("Enter your email in the sidebar to enable Generate / AI Assist.")

tab_gen, tab_upgrade = st.tabs(["🚀 Generator", "💎 Upgrade / Pricing"])

# =========================
# Generator
# =========================
with tab_gen:
    st.subheader("Listing inputs")

    b1, b2 = st.columns(2)
    with b1:
        if st.button("✨ Load Example", use_container_width=True):
            apply_example()
    with b2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_inputs()

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Product type", key="product")
        st.text_input("Material", key="material")
        st.text_input("Style", key="style")
        st.text_input("Color (optional)", key="color")
    with col2:
        st.text_input("Target audience", key="audience")
        st.text_input("Occasion", key="occasion")
        st.text_input("Personalization (optional)", key="personalization")

    st.text_input("Main keywords (comma-separated)", key="keywords")
    st.text_input("Main benefit (for first line)", key="benefit")
    st.selectbox("Seasonality", options=list(SEASONAL_PACKS.keys()), key="season", index=0)

    st.markdown("---")
    st.subheader("Description details (optional)")
    st.text_area("Key features (one per line)", key="features", height=120)
    st.text_input("Materials text", key="materials_desc")
    st.text_input("Sizing / Details", key="sizing")
    st.text_input("Shipping policy snippet", key="shipping")

    st.markdown("---")
    st.subheader("✨ AI Assist (optional)")
    ai_on = st.checkbox("Enable AI Assist", value=AI_ENABLED_DEFAULT)

    if pro_active:
        st.info("✅ Pro: Unlimited AI Assist")
    else:
        st.info(f"🆓 Free: AI Assist {AI_FREE_DAILY_LIMIT}/day")

    ai_btn = st.button("✨ AI Assist: Improve Inputs", use_container_width=True, disabled=not ai_on)

    if ai_btn:
        if not email_ok:
            st.error("Enter a valid email first.")
            st.stop()

        # Free AI daily limit
        if not pro_active:
            ai_used = get_ai_used(usage, email)
            if ai_used >= AI_FREE_DAILY_LIMIT:
                st.error(f"AI Assist Free limit reached today ({AI_FREE_DAILY_LIMIT}/day). Upgrade to Pro for unlimited.")
                st.link_button("🛒 Upgrade to Pro", STORE_URL, use_container_width=True)
                st.stop()

        payload = {k: st.session_state.get(k, "") for k in FORM_KEYS}
        payload["season"] = st.session_state.get("season", "None")

        with st.spinner("AI Assist working..."):
            try:
                sug = ai_assist_suggest(payload)
            except Exception as e:
                st.error(f"AI Assist failed: {e}")
                st.stop()

        # Apply suggestions (only overwrite if AI returned non-empty)
        if sug.get("keywords"):
            st.session_state["keywords"] = sug["keywords"]
        if sug.get("benefit"):
            st.session_state["benefit"] = sug["benefit"]
        if sug.get("features"):
            st.session_state["features"] = sug["features"]
        if sug.get("audience"):
            st.session_state["audience"] = sug["audience"]
        if sug.get("occasion"):
            st.session_state["occasion"] = sug["occasion"]
        if sug.get("personalization"):
            st.session_state["personalization"] = sug["personalization"]
        if sug.get("extra_tags"):
            st.session_state["paste_tags"] = sug["extra_tags"]

        # Count AI usage for Free users
        if not pro_active:
            inc_ai_used(usage, email)
            save_usage(usage)

        st.success("AI Assist applied ✅ (inputs updated)")
        st.rerun()

    st.markdown("---")
    st.subheader("Tag Guard (optional)")
    st.caption("Paste tags (comma or new lines). We will fix to 13 tags and 20 chars.")
    st.text_area("Paste Tags", key="paste_tags", height=90, placeholder="tag1, tag2, tag3 ...")

    gen = st.button("🚀 Generate Listing Pack", use_container_width=True)

    if gen:
        if not email_ok:
            st.error("Enter a valid email first.")
            st.stop()

        # Free generation limit
        if not pro_active:
            used = get_free_used(usage, email)
            if used >= FREE_DAILY_LIMIT:
                st.error("Free limit reached for today. Upgrade to Pro for unlimited generations.")
                st.link_button("🛒 Upgrade", STORE_URL, use_container_width=True)
                st.stop()

        # Collect inputs
        product = st.session_state.get("product", "")
        material = st.session_state.get("material", "")
        style = st.session_state.get("style", "")
        color = st.session_state.get("color", "")
        audience = st.session_state.get("audience", "")
        occasion = st.session_state.get("occasion", "")
        personalization = st.session_state.get("personalization", "")
        keywords = st.session_state.get("keywords", "")
        benefit = st.session_state.get("benefit", "")
        season = st.session_state.get("season", "None")
        features = st.session_state.get("features", "")
        materials_desc = st.session_state.get("materials_desc", "")
        sizing = st.session_state.get("sizing", "")
        shipping = st.session_state.get("shipping", "")
        paste_tags = st.session_state.get("paste_tags", "")

        main_kws = split_keywords(keywords)

        # Titles
        raw_titles = title_variations(product, main_kws, material, style, audience, occasion, personalization, color, season)
        ranked = rank_titles(raw_titles, product, main_kws)
        best_title = ranked[0]["title"] if ranked else ""

        # Tags candidates
        base_tags = []
        for t in [product, material, style, color]:
            if t:
                base_tags.append(t)
        for t in [audience, occasion, personalization]:
            if t:
                base_tags.append(t)
        base_tags += main_kws[:10]
        base_tags += ["gift", "handmade", "custom", "unique gift"]

        gen_fix = tag_guard_fix(base_tags, required_count=ETSY_TAG_COUNT)
        best_tags = gen_fix["fixed"]

        pasted_fix = None
        if str(paste_tags).strip():
            pasted_fix = tag_guard_fix(parse_tags(paste_tags), required_count=ETSY_TAG_COUNT)

        # Description
        desc = full_description(product, main_kws, benefit, features, materials_desc, sizing, shipping,
                                personalization, audience, occasion, season)

        # Count free generation usage
        if not pro_active:
            inc_free_used(usage, email)
            save_usage(usage)
            used_now = get_free_used(usage, email)
        else:
            used_now = None

        st.success("✅ Generated")

        # Quick copy payload
        payload_all = (
            f"BEST TITLE:\n{best_title}\n\n"
            f"BEST 13 TAGS:\n{', '.join(best_tags)}\n\n"
            f"DESCRIPTION:\n{desc}"
        )

        st.subheader("✅ Quick Apply")
        c1, c2, c3 = st.columns(3)
        with c1:
            copy_button(best_title, key="copy_best_title", label="Copy Best Title")
        with c2:
            copy_button(", ".join(best_tags), key="copy_best_tags", label="Copy Best 13 Tags")
        with c3:
            copy_button(payload_all, key="copy_all", label="Copy ALL ✅")

        st.markdown("---")
        st.subheader("Tag Guard + Fix (Generated Tags)")
        l, r = st.columns(2)
        with l:
            st.markdown("**Before (raw)**")
            st.write(gen_fix["original"][:20])
        with r:
            st.markdown("**After (fixed)**")
            st.write(best_tags)
            copy_button(", ".join(best_tags), key="copy_fixed_generated", label="Copy Fixed Tags")
        st.caption(f"Audit: count={gen_fix['audit']['count']} | duplicates={gen_fix['audit']['dups']} | too_long={len(gen_fix['audit']['too_long'])}")

        if pasted_fix:
            st.markdown("---")
            st.subheader("Tag Guard + Fix (Your Pasted Tags)")
            l2, r2 = st.columns(2)
            with l2:
                st.markdown("**Before**")
                st.write(pasted_fix["original"])
            with r2:
                st.markdown("**After**")
                st.write(pasted_fix["fixed"])
                copy_button(", ".join(pasted_fix["fixed"]), key="copy_fixed_pasted", label="Copy Fixed Pasted Tags")

        st.markdown("---")
        st.subheader("1) Titles (Ranked)")
        for i, item in enumerate(ranked, start=1):
            t = item["title"]
            score = item["score"]
            n = len(t)
            if i == 1:
                st.success(f"🏆 Best Title (Score {score}) — {n}/{MAX_TITLE_LEN}")
            else:
                if n > MAX_TITLE_LEN:
                    st.error(f"Title {i} (Score {score}) — {n}/{MAX_TITLE_LEN} (OVER)")
                elif n >= 130:
                    st.warning(f"Title {i} (Score {score}) — {n}/{MAX_TITLE_LEN} (close)")
                else:
                    st.caption(f"Title {i} (Score {score}) — {n}/{MAX_TITLE_LEN}")

            a, b = st.columns([8, 2])
            with a:
                st.text_area(f"Title {i}", value=t, height=68, key=f"title_area_{i}")
            with b:
                copy_button(t, key=f"copy_title_{i}", label="Copy")

        st.markdown("---")
        st.subheader("2) Description")
        lines = desc.splitlines()
        if len(lines) >= 2:
            st.markdown("**Etsy preview (first 2 lines):**")
            st.info(f"{lines[0]}\n\n{lines[1]}")
        st.text_area("Full description", value=desc, height=260, key="desc_area")
        copy_button(desc, key="copy_desc", label="Copy Description")

        # Downloads
        export_data = {
            "best_title": best_title,
            "ranked_titles": ranked,
            "best_13_tags": best_tags,
            "description": desc,
            "meta": {
                "product": product,
                "material": material,
                "style": style,
                "color": color,
                "audience": audience,
                "occasion": occasion,
                "personalization": personalization,
                "keywords": main_kws,
                "season": season,
                "generated_on": date.today().isoformat(),
            }
        }
        json_bytes = json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8")

        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer)
        writer.writerow(["SECTION", "KEY", "VALUE"])
        writer.writerow(["BEST", "Best Title", best_title])
        writer.writerow(["BEST", "Best 13 Tags", ", ".join(best_tags)])
        writer.writerow(["BEST", "Description", desc])
        writer.writerow([])
        writer.writerow(["RANKED_TITLES", "Title", "Score"])
        for x in ranked:
            writer.writerow(["RANKED_TITLES", x["title"], x["score"]])

        txt_bytes = payload_all.encode("utf-8")
        csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")

        st.markdown("---")
        st.subheader("⬇️ Downloads")
        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("Download JSON", data=json_bytes, file_name="listinglift_pack.json",
                               mime="application/json", use_container_width=True)
        with d2:
            st.download_button("Download CSV", data=csv_bytes, file_name="listinglift_pack.csv",
                               mime="text/csv", use_container_width=True)
        with d3:
            st.download_button("Download TXT", data=txt_bytes, file_name="listinglift_pack.txt",
                               mime="text/plain", use_container_width=True)

        if not pro_active and used_now is not None:
            st.caption(f"Free usage today: {used_now}/{FREE_DAILY_LIMIT}")

# =========================
# Upgrade tab
# =========================
with tab_upgrade:
    st.title("💎 Upgrade to Pro (Listing-Lift)")
    st.write("Buy from LemonSqueezy — Pro gives unlimited generations + unlimited AI Assist.")
    st.link_button("🛒 Open Listing-Lift Store", STORE_URL, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Free")
        st.markdown(f"- {FREE_DAILY_LIMIT} generations/day\n- AI Assist: {AI_FREE_DAILY_LIMIT}/day\n- Downloads")
    with c2:
        st.markdown("### Pro")
        st.markdown("- ✅ Unlimited generations\n- ✅ Unlimited AI Assist\n- ✅ Downloads JSON/CSV/TXT\n- ✅ Support by email")

    st.markdown("---")
    st.subheader("Support")
    st.write(SUPPORT_EMAIL)
    copy_button(SUPPORT_EMAIL, key="copy_support_email_upgrade", label="Copy Support Email")

st.markdown("---")
st.caption("Listing-Lift • Pro activation by email • No WhatsApp")