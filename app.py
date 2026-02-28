import streamlit as st
import json
import re
import csv
import io
from datetime import date
from pathlib import Path
import urllib.parse
import streamlit.components.v1 as components

# =========================
# CONFIG
# =========================
APP_TITLE = "Listing-Lift (Templates Pro)"
STORE_URL = "https://listing-lift.lemonsqueezy.com/"
SUPPORT_EMAIL = "hromany@hotmail.com"

PRO_USERS_FILE = "pro_users.json"
USAGE_FILE = "usage_log.json"

FREE_DAILY_LIMIT = 5  # free generations per day per email
MAX_TITLE_LEN = 140
ETSY_TAG_MAX_LEN = 20

# Optional: direct LemonSqueezy CHECKOUT link (your product's /checkout/buy/... link)
# Leave empty to use STORE_URL only.
LEMON_CHECKOUT_URL = ""


# =========================
# ONBOARDING EXAMPLE
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
    "shipping": "Processing 1-2 days, tracked shipping available"
}

def apply_example():
    for k, v in EXAMPLE_INPUT.items():
        st.session_state[k] = v

def reset_inputs():
    for k in list(EXAMPLE_INPUT.keys()):
        if k in st.session_state:
            del st.session_state[k]


# =========================
# JSON helpers
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

def load_pro_users():
    """
    Supported formats:
    1) ["a@b.com","c@d.com"]
    2) {"emails":["a@b.com","c@d.com"]}
    3) {"a@b.com": true, "c@d.com": true}
    """
    data = _read_json(PRO_USERS_FILE, default={})
    if isinstance(data, dict) and "emails" in data and isinstance(data["emails"], list):
        return set([e.strip().lower() for e in data["emails"] if isinstance(e, str)])
    if isinstance(data, dict):
        return set([k.strip().lower() for k, v in data.items() if v is True or v == 1])
    if isinstance(data, list):
        return set([e.strip().lower() for e in data if isinstance(e, str)])
    return set()

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    email = email.strip().lower()
    return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email) is not None

def load_usage():
    return _read_json(USAGE_FILE, default={})

def save_usage(usage: dict):
    _write_json(USAGE_FILE, usage)

def today_key():
    return date.today().isoformat()

def get_free_used(usage: dict, email: str) -> int:
    email = (email or "").strip().lower()
    return int(usage.get(today_key(), {}).get(email, 0))

def inc_free_used(usage: dict, email: str):
    email = email.strip().lower()
    tk = today_key()
    usage.setdefault(tk, {})
    usage[tk][email] = int(usage[tk].get(email, 0)) + 1


# =========================
# Copy button (JS)
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


# =========================
# LemonSqueezy link builder (optional: prefill email)
# =========================
def build_lemon_link(base_url: str, email: str) -> str:
    base_url = (base_url or "").strip()
    if not base_url:
        return ""
    if not email:
        return base_url

    parsed = urllib.parse.urlparse(base_url)
    q = dict(urllib.parse.parse_qsl(parsed.query))
    q["checkout[email]"] = email
    new_query = urllib.parse.urlencode(q, doseq=True)
    return urllib.parse.urlunparse(parsed._replace(query=new_query))


# =========================
# SEO logic (No AI)
# =========================
BUYER_INTENT_WORDS = [
    "gift", "personalized", "custom", "handmade", "unique", "premium",
    "best gift", "for her", "for him", "for kids", "anniversary", "birthday",
    "wedding", "bridal", "housewarming", "christmas gift", "mothers day", "fathers day",
    "printable", "digital download"
]

SEASONAL_PACKS = {
    "None": [],
    "Spring": ["spring", "easter", "mother's day"],
    "Summer": ["summer", "beach", "vacation"],
    "Autumn": ["fall", "autumn", "halloween", "thanksgiving"],
    "Winter": ["winter", "christmas", "new year"],
}

STOPWORDS = {"a","an","the","and","or","of","for","to","with","in","on","at","by","from","this","that","your","you","is","are"}

def clean_kw(s: str):
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^a-z0-9\s'-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def split_keywords(s: str):
    parts = [clean_kw(x) for x in (s or "").split(",")]
    return [p for p in parts if p]

def title_variations(product: str, main_kws: list, material: str, style: str, audience: str,
                     occasion: str, personalization: str, color: str, season: str):
    product = clean_kw(product)
    material = clean_kw(material)
    style = clean_kw(style)
    audience = clean_kw(audience)
    occasion = clean_kw(occasion)
    personalization = clean_kw(personalization)
    color = clean_kw(color)

    kws = [k for k in main_kws if k]
    top = kws[:3] if len(kws) >= 3 else kws
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

    def fill(t):
        out = t.format(
            kw1=kw1, kw2=kw2, kw3=kw3,
            product=product, material=material, style=style,
            audience=audience, occasion=occasion, personalization=personalization,
            color=color, season=season_kw
        )
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"\|\s*\|", "|", out)
        out = re.sub(r"\s+\|", " |", out)
        out = re.sub(r"\|\s+", "| ", out)
        out = out.strip(" -|")
        return out

    titles = []
    for t in templates:
        cand = fill(t)
        if cand and cand not in titles:
            titles.append(cand)
    return titles[:8]

def smart_trim_tag(tag: str, max_len: int = ETSY_TAG_MAX_LEN) -> str:
    t = normalize(tag)
    if not t:
        return ""
    if len(t) <= max_len:
        return t
    words = [w for w in t.split(" ") if w]
    words2 = [w for w in words if w not in STOPWORDS]
    if words2:
        words = words2
    while words and len(" ".join(words)) > max_len:
        words.pop()
    t2 = " ".join(words).strip()
    if t2 and len(t2) <= max_len:
        return t2
    return t[:max_len].strip()

def guard_tags(tags: list, max_len: int = ETSY_TAG_MAX_LEN) -> list:
    out = []
    for t in (tags or []):
        trimmed = smart_trim_tag(t, max_len=max_len)
        if trimmed and trimmed not in out:
            out.append(trimmed)
    return out

def make_long_tail_tags(product: str, main_kws: list, material: str, style: str, audience: str, occasion: str, season: str):
    product = clean_kw(product)
    material = clean_kw(material)
    style = clean_kw(style)
    audience = clean_kw(audience)
    occasion = clean_kw(occasion)
    seasonal = SEASONAL_PACKS.get(season, [])
    kws = [k for k in main_kws if k]

    base = " ".join([p for p in [style, material, product] if p]).strip()
    candidates = []
    if kws:
        candidates.append(f"{kws[0]} {base}".strip())
    if len(kws) > 1:
        candidates.append(f"{kws[0]} {kws[1]} {product}".strip())
    if audience:
        candidates.append(f"{base} for {audience}".strip())
    if occasion:
        candidates.append(f"{occasion} {product} {style}".strip())
    if seasonal:
        candidates.append(f"{seasonal[0]} {product} gift".strip())

    out = []
    for c in candidates:
        c = normalize(c)
        if c and c not in out:
            out.append(c)
    return out[:13]

def make_buyer_intent_tags(audience: str, occasion: str):
    audience = clean_kw(audience).lower()
    occasion = clean_kw(occasion).lower()
    tags = ["gift", "personalized", "custom", "handmade", "unique", "premium"]
    if audience:
        tags.append(f"for {audience}".strip())
    if occasion:
        tags.append(f"{occasion} gift".strip())
    out = []
    for t in tags:
        t = normalize(t)
        if t and t not in out:
            out.append(t)
    return out[:13]

def make_seasonality_tags(season: str):
    seasonal = SEASONAL_PACKS.get(season, [])
    out = []
    for s in seasonal:
        s = normalize(s)
        if s and s not in out:
            out.append(s)
    return out[:13]

def strong_first_two_lines(product: str, main_kws: list, benefit: str, audience: str, occasion: str, personalization: str, season: str):
    product = clean_kw(product)
    benefit = clean_kw(benefit)
    audience = clean_kw(audience)
    occasion = clean_kw(occasion)
    personalization = clean_kw(personalization)
    kws = [k for k in main_kws if k]
    kw = kws[0] if kws else ""
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

def full_description(product: str, main_kws: list, benefit: str, features: str, materials: str,
                     sizing: str, shipping: str, personalization: str, audience: str, occasion: str, season: str):
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

def title_score(title: str, product: str, main_kws: list, audience: str, occasion: str, season: str) -> dict:
    reasons = []
    t_norm = normalize(title)
    score = 0

    product_n = normalize(product)
    if product_n and product_n in t_norm:
        score += 20
        reasons.append("Contains product (+20)")

    kw_tokens = []
    for kw in (main_kws or []):
        kw_tokens += [w for w in normalize(kw).split(" ") if w and w not in STOPWORDS]
    kw_tokens = list(dict.fromkeys(kw_tokens))
    hits = [w for w in kw_tokens if w in t_norm]
    if hits:
        add = min(25, 5 * len(hits))
        score += add
        reasons.append(f"Keyword hits ({len(hits)}) (+{add})")

    intent_hits = []
    for w in BUYER_INTENT_WORDS:
        if normalize(w) in t_norm:
            intent_hits.append(w)
    if intent_hits:
        score += 15
        reasons.append("Buyer intent (+15)")

    aud_n = normalize(audience)
    occ_n = normalize(occasion)
    if aud_n and aud_n in t_norm:
        score += 8
        reasons.append("Audience match (+8)")
    if occ_n and occ_n in t_norm:
        score += 8
        reasons.append("Occasion match (+8)")

    seasonal = SEASONAL_PACKS.get(season, [])
    if seasonal:
        s0 = normalize(seasonal[0])
        if s0 and s0 in t_norm:
            score += 6
            reasons.append("Seasonal (+6)")

    n = len(title)
    if 110 <= n <= MAX_TITLE_LEN:
        score += 18
        reasons.append("Ideal length (+18)")
    elif 90 <= n < 110:
        score += 10
        reasons.append("Good length (+10)")
    elif n > MAX_TITLE_LEN:
        score -= 25
        reasons.append("Over 140 (-25)")
    elif 0 < n < 90:
        score -= 8
        reasons.append("Too short (-8)")

    if " | " in title or " - " in title:
        score += 3
        reasons.append("Readable separators (+3)")

    return {"score": score, "reasons": reasons}

def rank_titles(titles: list, product: str, main_kws: list, audience: str, occasion: str, season: str):
    scored = []
    for t in titles:
        s = title_score(t, product, main_kws, audience, occasion, season)
        scored.append({"title": t, "score": s["score"], "reasons": s["reasons"]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# =========================
# UI
# =========================
st.set_page_config(page_title=APP_TITLE, layout="centered")
st.title(APP_TITLE)
st.caption(f"Upgrade: {STORE_URL}")

pro_users = load_pro_users()
usage = load_usage()

with st.sidebar:
    st.subheader("Account")
    email = st.text_input("Your email (required)", placeholder="you@email.com").strip().lower()
    email_ok = bool(email) and is_valid_email(email)

    if email and not email_ok:
        st.warning("Please enter a valid email.")

    pro_active = email_ok and (email in pro_users)

    st.markdown("---")
    st.subheader("Plan")
    if pro_active:
        st.success("✅ Pro active (by email)")
        st.write("Unlimited generations.")
    else:
        st.info("Free plan (daily limit)")
        st.write(f"Free limit: **{FREE_DAILY_LIMIT} generations/day** per email.")
        st.write("Upgrade to Pro via LemonSqueezy (manual activation by email).")
        st.link_button("Open Listing-Lift Store", STORE_URL, use_container_width=True)

# ✅ NEW BEHAVIOR: do NOT stop the app if email is missing.
if not email_ok:
    st.info("Enter your email in the sidebar to enable Generate (Free/Pro). You can still view the tool.")

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
            st.rerun()
    with b2:
        if st.button("🔄 Reset", use_container_width=True):
            reset_inputs()
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        product = st.text_input("Product type", key="product")
        material = st.text_input("Material", key="material")
        style = st.text_input("Style", key="style")
        color = st.text_input("Color (optional)", key="color")
    with col2:
        audience = st.text_input("Target audience", key="audience")
        occasion = st.text_input("Occasion", key="occasion")
        personalization = st.text_input("Personalization (optional)", key="personalization")

    keywords = st.text_input("Main keywords (comma-separated)", key="keywords")
    benefit = st.text_input("Main benefit (for first line)", key="benefit")
    season = st.selectbox("Seasonality", options=list(SEASONAL_PACKS.keys()), key="season", index=0)

    st.markdown("---")
    st.subheader("Description details (optional)")
    features = st.text_area("Key features (one per line)", key="features", height=120)
    materials_desc = st.text_input("Materials text", key="materials_desc")
    sizing = st.text_input("Sizing / Details", key="sizing")
    shipping = st.text_input("Shipping policy snippet", key="shipping")

    gen = st.button("🚀 Generate SEO Pack", use_container_width=True)

    if gen:
        # ✅ Require email only when generating
        if not email_ok:
            st.error("Please enter a valid email in the sidebar first.")
            st.stop()

        # ✅ Enforce free limit only when generating
        pro_active = email in pro_users  # recompute now that email_ok True

        if not pro_active:
            used = get_free_used(usage, email)
            if used >= FREE_DAILY_LIMIT:
                st.error("Free limit reached for today. Upgrade to Pro for unlimited generations.")
                st.link_button("Open Listing-Lift Store", STORE_URL, use_container_width=True)
                st.stop()

        main_kws = split_keywords(keywords)

        raw_titles = title_variations(
            product, main_kws, material, style, audience, occasion, personalization, color, season
        )
        ranked = rank_titles(raw_titles, product, main_kws, audience, occasion, season)
        best_title = ranked[0]["title"] if ranked else ""

        long_tail = guard_tags(make_long_tail_tags(product, main_kws, material, style, audience, occasion, season))
        buyer_intent = guard_tags(make_buyer_intent_tags(audience, occasion))
        seasonal_tags = guard_tags(make_seasonality_tags(season))

        best_tags = []
        for pack in [long_tail, buyer_intent, seasonal_tags]:
            for t in pack:
                if t and t not in best_tags:
                    best_tags.append(t)
        best_tags = best_tags[:13]

        desc = full_description(
            product, main_kws, benefit, features, materials_desc, sizing, shipping,
            personalization, audience, occasion, season
        )

        if not pro_active:
            inc_free_used(usage, email)
            save_usage(usage)
            used = get_free_used(usage, email)
        else:
            used = None

        st.success("✅ Generated")

        st.subheader("✅ Quick Apply (Copy / Download)")
        copy_payload = (
            f"BEST TITLE:\n{best_title}\n\n"
            f"BEST 13 TAGS (<=20 chars each):\n{', '.join(best_tags)}\n\n"
            f"DESCRIPTION:\n{desc}"
        )

        cA1, cA2, cA3 = st.columns(3)
        with cA1:
            copy_button(best_title, key="copy_best_title", label="Copy Best Title")
        with cA2:
            copy_button(", ".join(best_tags), key="copy_best_tags", label="Copy Best 13 Tags")
        with cA3:
            copy_button(copy_payload, key="copy_all", label="Copy ALL ✅")

        export_data = {
            "best_title": best_title,
            "ranked_titles": ranked,
            "best_13_tags": best_tags,
            "tags": {
                "long_tail": long_tail,
                "buyer_intent": buyer_intent,
                "seasonality": seasonal_tags
            },
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
                "generated_on": date.today().isoformat()
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
        for item in ranked:
            writer.writerow(["RANKED_TITLES", item["title"], item["score"]])

        csv_bytes = csv_buffer.getvalue().encode("utf-8-sig")
        txt_bytes = copy_payload.encode("utf-8")

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button("⬇️ Download JSON", data=json_bytes, file_name="listinglift_seo_pack.json",
                               mime="application/json", use_container_width=True)
        with d2:
            st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="listinglift_seo_pack.csv",
                               mime="text/csv", use_container_width=True)
        with d3:
            st.download_button("⬇️ Download TXT", data=txt_bytes, file_name="listinglift_seo_pack.txt",
                               mime="text/plain", use_container_width=True)

        st.divider()

        st.subheader("Titles (Ranked)")
        for i, item in enumerate(ranked, start=1):
            t = item["title"]
            n = len(t)
            score = item["score"]

            if i == 1:
                st.success(f"🏆 Best Title (Score {score}) — {n}/{MAX_TITLE_LEN}")
            else:
                if n > MAX_TITLE_LEN:
                    st.error(f"Title {i} (Score {score}) — {n}/{MAX_TITLE_LEN} (OVER limit)")
                elif n >= 130:
                    st.warning(f"Title {i} (Score {score}) — {n}/{MAX_TITLE_LEN} (close to limit)")
                else:
                    st.caption(f"Title {i} (Score {score}) — {n}/{MAX_TITLE_LEN}")

            c1, c2 = st.columns([8, 2])
            with c1:
                st.text_area(label=f"Title {i}", value=t, height=68, key=f"title_area_{i}")
                with st.expander("Why this score?"):
                    for r in item["reasons"]:
                        st.write(f"• {r}")
            with c2:
                copy_button(t, key=f"copy_title_{i}", label="Copy")

            st.divider()

        st.subheader("Tags (Etsy 20-char guard)")
        tcol1, tcol2, tcol3 = st.columns(3)
        with tcol1:
            st.markdown("**Long-tail**")
            st.write(long_tail)
            copy_button(", ".join(long_tail), key="copy_longtail", label="Copy Long-tail")
        with tcol2:
            st.markdown("**Buyer Intent**")
            st.write(buyer_intent)
            copy_button(", ".join(buyer_intent), key="copy_intent", label="Copy Intent")
        with tcol3:
            st.markdown("**Seasonality**")
            st.write(seasonal_tags if seasonal_tags else ["None"])
            copy_button(", ".join(seasonal_tags), key="copy_season", label="Copy Seasonal")

        st.markdown("**Best 13 Tags (ready to paste):**")
        st.write(best_tags)
        copy_button(", ".join(best_tags), key="copy_best_13", label="Copy Best 13")

        st.subheader("Description")
        lines = desc.splitlines()
        if len(lines) >= 2:
            st.markdown("**Etsy Search Preview (first 2 lines):**")
            st.info(f"{lines[0]}\n\n{lines[1]}")
        st.text_area("Full Description", value=desc, height=260)
        copy_button(desc, key="copy_desc", label="Copy Description")

        if not pro_active and used is not None:
            st.caption(f"Free usage today: {used}/{FREE_DAILY_LIMIT} generations")


# =========================
# Upgrade / Pricing
# =========================
with tab_upgrade:
    st.title("💎 Upgrade to Pro (Listing-Lift)")
    st.write("Pay on LemonSqueezy — then we activate Pro by your email (manual).")

    st.markdown("### Open the Store")
    st.link_button("🛒 Open Listing-Lift Store", STORE_URL, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Free")
        st.markdown(f"- {FREE_DAILY_LIMIT} generations/day\n- Copy titles/tags/description\n- Basic templates")
    with c2:
        st.markdown("### Pro")
        st.markdown("- ✅ Unlimited generations\n- ✅ Copy ALL + Download (JSON/CSV/TXT)\n- ✅ Best 13 tags + 20-char guard\n- ✅ Support by email")

    st.markdown("---")
    st.subheader("Optional: Direct Checkout Button")
    if LEMON_CHECKOUT_URL:
        email_for_prefill = email if (bool(email) and is_valid_email(email)) else ""
        pay_link = build_lemon_link(LEMON_CHECKOUT_URL, email_for_prefill)
        st.link_button("💳 Pay Now (Checkout)", pay_link, use_container_width=True)
        st.caption("If you typed your email, it will be pre-filled on checkout.")
    else:
        st.info("Direct checkout link not set. Customers can buy from the store page above.")

    st.markdown("---")
    st.subheader("Activation (by Email)")
    st.write("After payment, send the SAME email you used at checkout. We will activate Pro by your email.")
    st.markdown("**Support email:**")
    st.write(SUPPORT_EMAIL)
    copy_button(SUPPORT_EMAIL, key="copy_support_email", label="Copy Support Email")

st.markdown("---")
st.caption("Listing-Lift • Pro activation by email (no WhatsApp).")