import streamlit as st
import pandas as pd
import re
import json
import hashlib
from datetime import datetime, date

# ----------------- CONFIG -----------------
APP_NAME = "Etsy SEO Helper"
FREE_DAILY_LIMIT = 3
USAGE_FILE = "usage.json"
PRO_USERS_FILE = "pro_users.json"

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="centered")

# ----------------- HELPERS -----------------
def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def slugify(s: str) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s).strip("-")
    return s[:60] if s else "item"

def sha256_short(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()[:16]

def today_key() -> str:
    return str(date.today())

def split_list(raw: str, max_items: int = 12):
    items = []
    for x in re.split(r"[\n,]+", raw or ""):
        t = clean_text(x)
        if t:
            items.append(t)
    out = []
    seen = set()
    for i in items:
        k = i.lower()
        if k not in seen:
            out.append(i)
            seen.add(k)
    return out[:max_items]

# ----------------- JSON STORE -----------------
def load_json_file(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json_file(path: str, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# ----------------- USAGE -----------------
def get_user_id(email: str) -> str:
    email = clean_text(email).lower()
    return sha256_short(email)

def get_usage_today(user_id: str) -> int:
    usage = load_json_file(USAGE_FILE, {})
    if not isinstance(usage, dict):
        # self-heal if corrupted
        usage = {}
        save_json_file(USAGE_FILE, usage)
    return int(usage.get(user_id, {}).get(today_key(), 0))

def inc_usage_today(user_id: str):
    usage = load_json_file(USAGE_FILE, {})
    if not isinstance(usage, dict):
        usage = {}
    usage.setdefault(user_id, {})
    usage[user_id].setdefault(today_key(), 0)
    usage[user_id][today_key()] += 1
    save_json_file(USAGE_FILE, usage)

# ----------------- PRO USERS (BY EMAIL) -----------------
def load_pro_users() -> set:
    """
    pro_users.json should be a JSON list of emails:
    ["buyer@email.com", "another@domain.com"]
    """
    data = load_json_file(PRO_USERS_FILE, [])
    if isinstance(data, list):
        return set([clean_text(x).lower() for x in data if clean_text(x)])
    return set()

def is_pro_user(email: str) -> bool:
    email = clean_text(email).lower()
    if not email:
        return False
    return email in load_pro_users()

# ----------------- SEO GENERATION (TEMPLATES) -----------------
def extract_keywords(product, niche, style, occasion, file_type, features, audience):
    seed = " ".join([product, niche, style, occasion, file_type, audience] + features)
    words = re.findall(r"[a-z0-9]+", seed.lower())
    stop = set("""
        the a an and or for with to in of on at by from is are this that
        best premium high quality durable easy new
        digital instant download printable template templates
    """.split())
    uniq = []
    for w in words:
        if len(w) < 3 or w in stop:
            continue
        if w not in uniq:
            uniq.append(w)
    return uniq[:60]

def build_titles(product, primary_kw, niche, occasion, style, file_type, size, keywords):
    kw = clean_text(primary_kw or (keywords[0] if keywords else product))
    patterns = [
        "{kw} | {product} | {niche} | {occasion}",
        "{kw} {product} - {style} {file_type} ({size})",
        "{product} {kw} | {occasion} | Instant Download",
        "{kw} {product} | {niche} | {style}",
        "{product} | {niche} | {kw} | {occasion}",
        "{kw} {product} | {file_type} | Editable Template",
        "{product} {kw} | {style} | {niche}",
        "{kw} | {product} | {file_type} | {occasion}",
        "{product} | {kw} | {style} | Instant Download",
        "{kw} {product} | {niche} | {occasion} | {size}",
    ]
    variants = []
    for p in patterns:
        title = p.format(
            kw=kw,
            product=clean_text(product),
            niche=clean_text(niche),
            occasion=clean_text(occasion),
            style=clean_text(style),
            file_type=clean_text(file_type),
            size=clean_text(size),
        )
        title = clean_text(title).strip(" |-_")
        title = title[:140]
        if title and title.lower() not in [v.lower() for v in variants]:
            variants.append(title)
    return variants[:10]

def build_tags(keywords):
    tags = []
    for k in keywords:
        if len(tags) >= 13:
            break
        t = k.replace("_", " ").strip()
        if 2 <= len(t) <= 20 and t not in tags:
            tags.append(t)

    fillers = ["instant download", "digital file", "printable", "editable template", "gift idea"]
    for f in fillers:
        if len(tags) >= 13:
            break
        if len(f) <= 20 and f not in tags:
            tags.append(f)

    while len(tags) < 13:
        tags.append("digital download")

    return tags[:13]

def build_description(product, file_type, size, includes, how_to, features, niche, audience, style, occasion, tone):
    tone_map = {
        "Professional": {
            "intro": "A clear, SEO-friendly description designed to convert browsers into buyers.",
            "voice": "Clear, informative, and to-the-point."
        },
        "Friendly": {
            "intro": "A warm, buyer-friendly description that highlights benefits and makes purchasing easy.",
            "voice": "Friendly, helpful, and human."
        },
        "Luxury": {
            "intro": "A premium description with elevated language and a polished feel.",
            "voice": "Elegant, confident, and refined."
        }
    }
    t = tone_map.get(tone, tone_map["Professional"])

    feat_lines = "\n".join([f"• {x}" for x in features[:8]]) if features else "• High-quality design\n• Easy to use\n• Instant access"
    inc_lines = "\n".join([f"• {x}" for x in includes[:10]]) if includes else "• 1 x PDF file\n• 1 x PNG file (optional)\n• Instructions"

    return f"""**{product}** — {file_type} ({size or "N/A"})

{t['intro']}

**Style:** {style or "N/A"}  
**Occasion:** {occasion or "N/A"}  
**Best for:** {audience or "Etsy buyers"}

---

### ✅ What you’ll get
{inc_lines}

### ⭐ Key features
{feat_lines}

### 📌 Perfect for
• {niche or "digital downloads"}  
• {occasion or "gifting"}  
• {audience or "personal use / small business"}

### ⬇️ How it works (Instant Download)
{how_to}

### ⚠️ Notes
• This is a **digital product** — no physical item will be shipped.  
• Colors may vary slightly due to different screens/printers.  
• Usage rights depend on your shop policy.

**Tone:** {t['voice']}
"""

# ----------------- UI -----------------
st.markdown(f"## 🧠 {APP_NAME}")
st.caption("Generate Etsy-optimized titles, 13 tags, and a high-converting description — focused on digital downloads.")

tabs = st.tabs(["Generate", "Examples", "Pricing", "FAQ"])

# -------- Sidebar: access --------
with st.sidebar:
    st.header("Access")
    email = st.text_input("Email (used for limits & Pro)", placeholder="you@email.com")
    email_clean = clean_text(email).lower()

    pro = is_pro_user(email_clean) if email_clean else False

    if email_clean:
        uid = get_user_id(email_clean)
        used = get_usage_today(uid)
        if pro:
            st.success("Pro active ✅ (by email)")
            st.write(f"Today usage: {used} (unlimited)")
        else:
            st.info(f"Free plan: {FREE_DAILY_LIMIT}/day")
            st.write(f"Today used: {used}/{FREE_DAILY_LIMIT}")
            st.markdown("**Upgrade to Pro:** buy on Gumroad, then use the same email here.")
    else:
        st.warning("Enter your email to generate listings.")

# -------- Generate Tab --------
with tabs[0]:
    st.subheader("Product details")

    col1, col2 = st.columns(2)
    with col1:
        product = st.text_input("Product name (required)", placeholder="e.g., Minimalist Wedding Invitation Template")
        primary_kw = st.text_input("Primary keyword (SEO)", placeholder="e.g., wedding invitation template")
        niche = st.text_input("Niche / category", placeholder="e.g., wedding, planner, business, baby shower")
        file_type = st.selectbox(
            "Digital file type",
            ["Canva Template", "Printable PDF", "Editable Templett", "SVG/PNG Bundle", "Planner/Journal", "Resume Template", "Other"],
            index=0
        )

    with col2:
        style = st.text_input("Style", placeholder="e.g., minimalist, boho, modern, vintage")
        occasion = st.text_input("Occasion", placeholder="e.g., wedding, birthday, baby shower")
        size = st.text_input("Size / format", placeholder="e.g., 5x7, A4, US Letter")
        tone = st.selectbox("Tone", ["Professional", "Friendly", "Luxury"], index=0)

    st.subheader("Features (paste bullet points)")
    features = split_list(
        st.text_area("Top features", height=110, placeholder="e.g.\nEditable in Canva\nInstant download\nHigh-resolution print\nEasy to customize"),
        max_items=12
    )

    st.subheader("What’s included (files)")
    includes = split_list(
        st.text_area("Included files", height=90, placeholder="e.g.\n1 Canva template link\n1 PDF (A4)\n1 PDF (US Letter)\nInstructions PDF"),
        max_items=12
    )

    st.subheader("Instant download instructions")
    how_to = st.text_area(
        "How to download/use",
        height=90,
        value="After purchase, go to Etsy → Your Account → Purchases → Download files. Open the PDF or Canva link and edit/print as needed."
    )

    audience = st.text_input("Target buyer (optional)", placeholder="e.g., brides, small business owners, teachers")

    generate_btn = st.button("✨ Generate Etsy Listing", type="primary")

    if generate_btn:
        if not email_clean:
            st.error("Please enter your email in the sidebar first.")
            st.stop()

        uid = get_user_id(email_clean)
        used = get_usage_today(uid)

        if (not pro) and used >= FREE_DAILY_LIMIT:
            st.warning("Free limit reached for today. Upgrade to Pro (by email) for unlimited generations.")
            st.stop()

        if not clean_text(product):
            st.error("Product name is required.")
            st.stop()

        inc_usage_today(uid)

        keywords = extract_keywords(product, niche, style, occasion, file_type, features, audience)
        titles = build_titles(product, primary_kw, niche, occasion, style, file_type, size, keywords)
        tags = build_tags(keywords)
        description = build_description(
            product=product,
            file_type=file_type,
            size=size,
            includes=includes,
            how_to=how_to,
            features=features,
            niche=niche,
            audience=audience,
            style=style,
            occasion=occasion,
            tone=tone
        )

        st.success("Generated ✅")

        st.subheader("✅ Titles (10 options)")
        for i, t in enumerate(titles, 1):
            st.write(f"{i}. {t}")

        st.subheader("🏷️ Tags (13)")
        st.code(", ".join(tags))

        st.subheader("📝 Description")
        st.markdown(description)

        st.subheader("🔎 Keyword ideas")
        st.write(", ".join(keywords[:25]))

        out = {
            "email": email_clean,
            "plan": "pro" if pro else "free",
            "product": product,
            "primary_keyword": primary_kw,
            "niche": niche,
            "file_type": file_type,
            "style": style,
            "occasion": occasion,
            "size": size,
            "tone": tone,
            "titles": " | ".join(titles),
            "tags": ", ".join(tags),
            "keywords": ", ".join(keywords[:25]),
            "generated_at": datetime.utcnow().isoformat()
        }
        df = pd.DataFrame([out])

        st.subheader("⬇️ Export")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"etsy_listing_{slugify(product)}.csv",
            mime="text/csv"
        )

# -------- Examples Tab --------
with tabs[1]:
    st.subheader("Examples (what outputs look like)")
    st.markdown("""
**Example input**  
Product: Minimalist Wedding Invitation Template  
Primary keyword: wedding invitation template  
Style: minimalist  
Occasion: wedding  
File type: Canva Template  
Size: 5x7 + A4 + US Letter  
Features: editable, instant download, high-resolution, easy to customize  

**Example output**  
- 10 title variations  
- 13 tags (kept short)  
- Description with: what's included, features, download instructions, notes
""")

# -------- Pricing Tab --------
with tabs[2]:
    st.subheader("Pricing")
    st.markdown(f"""
**Free**  
- {FREE_DAILY_LIMIT} generations / day  
- Titles + Tags + Description  
- CSV export

**Pro (via email)**  
- Unlimited generations  
- Priority improvements  
- Bulk mode (coming soon)

**How Pro works right now:**  
1) Buy Pro on Gumroad  
2) Use the same purchase email in the app  
3) We activate Pro by adding your email to the Pro list
""")

# -------- FAQ Tab --------
with tabs[3]:
    st.subheader("FAQ")
    st.markdown("""
**Does this work for US & Europe?**  
Yes. Etsy SEO is broadly similar across regions, and digital listings are global.

**Is this AI?**  
This MVP uses professional SEO templates and rules. Once billing works, we’ll upgrade generation with a real AI model for higher quality.

**Will this guarantee sales?**  
No tool can guarantee sales, but better titles/tags/descriptions improve discoverability and conversion odds.

**How do I get Pro?**  
Buy on Gumroad, then use the same email in the app. We activate Pro by adding your email to the Pro list.
""")