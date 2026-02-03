import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup as soup
import os
import html

# ----------------------------
# 1) Setup
# ----------------------------
header = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0"
}

url = "https://charlottesville.craigslist.org/search/moa?purveyor=owner"


# Common phone manufacturers (brands list)
BRANDS = [
    "Apple", "Samsung", "Google", "OnePlus", "Motorola", "Nokia", "Sony",
    "LG", "Xiaomi", "Redmi", "Poco", "Huawei", "Honor", "Oppo", "Vivo",
    "Realme", "Asus", "ZTE", "TCL", "Alcatel", "Lenovo", "HTC",
    "BlackBerry", "Nothing"
]

# Map brand -> regex patterns that might appear in text
BRAND_PATTERNS = {
    "Apple": r"\b(apple|iphone)\b",
    "Samsung": r"\b(samsung|galaxy)\b",
    "Google": r"\b(google|pixel)\b",
    "OnePlus": r"\b(oneplus)\b",
    "Motorola": r"\b(motorola|moto)\b",
    "Nokia": r"\b(nokia)\b",
    "Sony": r"\b(sony|xperia)\b",
    "LG": r"\b(lg)\b",
    "Xiaomi": r"\b(xiaomi)\b",
    "Redmi": r"\b(redmi)\b",
    "Poco": r"\b(poco)\b",
    "Huawei": r"\b(huawei)\b",
    "Honor": r"\b(honor)\b",
    "Oppo": r"\b(oppo)\b",
    "Vivo": r"\b(vivo)\b",
    "Realme": r"\b(realme)\b",
    "Asus": r"\b(asus|rog phone|zenfone)\b",
    "ZTE": r"\b(zte)\b",
    "TCL": r"\b(tcl)\b",
    "Alcatel": r"\b(alcatel)\b",
    "Lenovo": r"\b(lenovo)\b",
    "HTC": r"\b(htc)\b",
    "BlackBerry": r"\b(blackberry)\b",
    "Nothing": r"\b(nothing phone)\b",
}

def clean_price(x):
    if x is None:
        return np.nan
    x = x.replace("$", "").replace(",", "").strip()
    try:
        return float(x)
    except Exception:
        return np.nan

def normalize_text(t):
    return re.sub(r"\s+", " ", (t or "")).strip()

def load_local_html(path: str) -> str:
    """
    Loads a saved Craigslist page.

    Handles BOTH:
      - normal saved HTML (contains real <li> tags)
      - Firefox 'View Page Source' wrapper (contains <span id="line..."> and &lt;li&gt; escaped tags)

    Returns REAL Craigslist HTML as a string.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Detect Firefox "view-source" wrapper format
    if 'id="viewsource"' in text or "viewsource.css" in text or 'id="line1"' in text:
        wrapper = soup(text, "html.parser")
        lines = wrapper.select("span[id^=line]")
        if not lines:
            raise ValueError("Detected view-source wrapper, but couldn't find span[id^=line].")

        extracted = "\n".join(ln.get_text("") for ln in lines)
        extracted = html.unescape(extracted)  # convert &lt;li&gt; -> <li>
        return extracted

    # Otherwise it's already normal HTML
    return text


# ----------------------------
# 2) Spec extraction (regex)
# ----------------------------
def detect_brand(text):
    t = (text or "").lower()
    for brand, pat in BRAND_PATTERNS.items():
        if re.search(pat, t, flags=re.IGNORECASE):
            return brand
    return None

def extract_ram_gb(text):
    # Matches: 8GB RAM, 8 GB, ram 8gb, 12gb memory
    t = (text or "").lower()
    m = re.search(r"\b(\d{1,2})\s*gb\s*(ram|memory)\b", t)
    if not m:
        m = re.search(r"\b(ram|memory)\s*[:\-]?\s*(\d{1,2})\s*gb\b", t)
        if m:
            return float(m.group(2))
        return np.nan
    return float(m.group(1))

def extract_storage_gb(text):
    # Matches: 128GB storage, 256 gb, 1tb, 64gb rom, etc.
    t = (text or "").lower()

    # Prefer explicit storage/rom keywords
    m = re.search(r"\b(\d{2,4})\s*gb\s*(storage|ssd|rom)\b", t)
    if m:
        return float(m.group(1))

    # TB -> GB
    m = re.search(r"\b(\d)\s*tb\b", t)
    if m:
        return float(m.group(1)) * 1024.0

    # Generic "###gb" often refers to storage in listings; but could be RAM.
    # We'll take the *largest* GB number mentioned as a best guess for storage
    # if nothing explicit matched.
    gbs = [int(x) for x in re.findall(r"\b(\d{2,4})\s*gb\b", t)]
    if gbs:
        return float(max(gbs))

    return np.nan

def extract_cpu_chipset(text):
    t = (text or "")
    # Common chipsets keywords
    patterns = [
        r"(snapdragon\s*\d{3,4}\+?)",
        r"(apple\s*a\d{1,2}\s*(bionic)?)",
        r"(exynos\s*\d{3,4})",
        r"(tensor\s*g\d)",
        r"(mediatek\s*(dimensity)?\s*\d{3,4})",
        r"(kirin\s*\d{3,4})"
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None

def extract_display(text):
    t = (text or "")
    # Examples: 6.1", 6.7 inch, OLED, AMOLED, 120Hz
    size = None
    m = re.search(r"\b(\d\.\d)\s*(\"|inch|in)\b", t, flags=re.IGNORECASE)
    if m:
        size = f'{m.group(1)}"'

    tech = None
    m2 = re.search(r"\b(amoled|oled|lcd|ips)\b", t, flags=re.IGNORECASE)
    if m2:
        tech = m2.group(1).upper()

    hz = None
    m3 = re.search(r"\b(\d{2,3})\s*hz\b", t, flags=re.IGNORECASE)
    if m3:
        hz = f"{m3.group(1)}Hz"

    parts = [p for p in [size, tech, hz] if p]
    return " / ".join(parts) if parts else None

def extract_camera(text):
    t = (text or "")
    # Examples: 48MP, triple camera, 3 cameras
    m = re.search(r"\b(\d{1,3})\s*mp\b", t, flags=re.IGNORECASE)
    mp = f"{m.group(1)}MP" if m else None

    m2 = re.search(r"\b(single|dual|triple|quad)\s*camera\b", t, flags=re.IGNORECASE)
    sys = m2.group(1).lower() + " camera" if m2 else None

    parts = [p for p in [mp, sys] if p]
    return " / ".join(parts) if parts else None

def extract_battery_charging(text):
    t = (text or "")
    # Examples: 4500mAh, 5000 mah, 25W, fast charging, magsafe
    mah = None
    m = re.search(r"\b(\d{4,5})\s*mah\b", t, flags=re.IGNORECASE)
    if m:
        mah = f"{m.group(1)}mAh"

    w = None
    m2 = re.search(r"\b(\d{2,3})\s*w\b", t, flags=re.IGNORECASE)
    if m2:
        w = f"{m2.group(1)}W"

    fast = None
    if re.search(r"\b(fast charging|quick charge|power delivery|pd)\b", t, flags=re.IGNORECASE):
        fast = "fast charging"

    magsafe = None
    if re.search(r"\b(magsafe)\b", t, flags=re.IGNORECASE):
        magsafe = "MagSafe"

    parts = [p for p in [mah, w, fast, magsafe] if p]
    return " / ".join(parts) if parts else None

def extract_model_name(title, brand):
    # Very lightweight heuristic: remove price-like, condition words, keep the main
    t = normalize_text(title)
    if brand:
        # don't overdo it; just return title as "model name" baseline
        return t
    return t

def parse_specs(title, description):
    blob = f"{title}\n{description or ''}"
    brand = detect_brand(blob)
    return {
        "company": brand,
        "phone_name": extract_model_name(title, brand),
        "memory_ram_gb": extract_ram_gb(blob),
        "storage_gb": extract_storage_gb(blob),
        "processor_chipset": extract_cpu_chipset(blob),
        "display": extract_display(blob),
        "camera_system": extract_camera(blob),
        "battery_charging": extract_battery_charging(blob),
    }

# ----------------------------
# 3) Scrape search results (LOCAL HTML)
# ----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# NOTE: you saved as .htm in your code, but your instructions say .html.
# This tries both so it won't silently fail.
candidates = [
    os.path.join(SCRIPT_DIR, "craigslist_moa.html"),
    os.path.join(SCRIPT_DIR, "craigslist_moa.htm"),
]

LOCAL_HTML_PATH = next((p for p in candidates if os.path.exists(p)), None)
if LOCAL_HTML_PATH is None:
    raise FileNotFoundError(
        "\nLocal Craigslist HTML file not found.\n\n"
        f"Tried:\n  " + "\n  ".join(candidates) + "\n\n"
        "Fix:\n"
        "1) Open the page in your browser\n"
        "2) Right click -> View Page Source\n"
        "3) Save it into wrangling/lab as craigslist_moa.html\n"
    )



# Load local HTML (handles Firefox view-source wrapper too)
html_text = load_local_html(LOCAL_HTML_PATH)

print("Loaded local HTML:", LOCAL_HTML_PATH)
print("Loaded local HTML bytes:", len(html_text))

bsObj = soup(html_text, "html.parser")

# In the actual Craigslist HTML you pasted, results are:
# <li class="cl-static-search-result" title="..."> ... </li>
cards = bsObj.select("li.cl-static-search-result")
print("Found listings in local HTML:", len(cards))

rows = []
for card in cards:
    # Title: attribute "title" OR inner div.title text
    title = card.get("title")
    if not title:
        tdiv = card.select_one("div.title")
        title = tdiv.get_text(strip=True) if tdiv else None

    # Price: <div class="price">$350</div>
    price_el = card.select_one("div.price")
    price = clean_price(price_el.get_text(strip=True) if price_el else None)

    # URL: <a href="..."> (this exists in your pasted HTML)
    a = card.select_one("a[href]")
    link = a["href"] if a else None

    # Location (optional)
    loc_el = card.select_one("div.location")
    meta = normalize_text(loc_el.get_text(" ", strip=True) if loc_el else None)

    rows.append({"title": title, "price": price, "url": link, "meta": meta})

df = pd.DataFrame(rows)

print("\nDF preview:")
print(df.head(10))

if len(df) > 0 and "price" in df.columns:
    print("\nPrice missingness:", df["price"].isna().mean())
else:
    print("\nNo listings were scraped from the local HTML (df is empty).")



    
# ----------------------------
# 4) OPTIONAL: visit each listing to get description for better spec extraction
#     (slower, but improves RAM/storage/etc. capture)
# ----------------------------
descriptions = []
for i, row in df.iterrows():
    link = row["url"]
    desc_text = ""
    if isinstance(link, str) and link.startswith("http"):
        try:
            r = requests.get(link, headers=header, timeout=30)
            page = soup(r.content, "html.parser")
            body = page.find("section", id="postingbody")
            if body:
                # postingbody contains boilerplate "QR Code Link to This Post" sometimes
                desc_text = body.get_text("\n", strip=True).replace("QR Code Link to This Post", "").strip()
        except Exception:
            desc_text = ""
        time.sleep(1.0)  # be polite to Craigslist
    descriptions.append(desc_text)

df["description"] = descriptions

# ----------------------------
# 5) Build the "specs" dataframe columns
# ----------------------------
spec_rows = []
for i, row in df.iterrows():
    specs = parse_specs(row.get("title"), row.get("description"))
    spec_rows.append(specs)

spec_df = pd.DataFrame(spec_rows)

# Combine base + specs
phones = pd.concat([df, spec_df], axis=1)

desired_cols = [
    "phone_name",
    "company",
    "price",
    "memory_ram_gb",
    "storage_gb",
    "processor_chipset",
    "display",
    "camera_system",
    "battery_charging",
    "url",
    "meta",
    "title",
    "description"
]

# Desired final schema
desired_cols = [
    "phone_name",
    "company",
    "price",
    "memory_ram_gb",
    "storage_gb",
    "processor_chipset",
    "display",
    "camera_system",
    "battery_charging",
    "url",
    "meta",
    "title",
    "description"
]

# Add missing columns as NaN
for col in desired_cols:
    if col not in phones.columns:
        phones[col] = np.nan

# Reorder safely
phones = phones[desired_cols]

# ----------------------------
# 6) Basic price stats + histogram
# ----------------------------
price_stats = phones["price"].dropna().describe()
print("Price summary:\n", price_stats)

plt.figure()

plt.hist(
    phones["price"].dropna(),
    bins=20,
    range=(0, 1000),   # <-- force range
    edgecolor="black"
)

plt.xlim(0, 1000)
plt.title("Phone Prices ($0–$1000)")
plt.xlabel("Price ($)")
plt.ylabel("Count")
plt.show()

# ----------------------------
# 7) Scatter: Price vs RAM (colored by company)
# ----------------------------
# Keep only rows with RAM and price
ram_df = phones.dropna(subset=["price", "memory_ram_gb"]).copy()

# Create a consistent color per company
companies = sorted([c for c in ram_df["company"].dropna().unique()])
cmap = plt.get_cmap("tab10")
color_map = {c: cmap(i % 10) for i, c in enumerate(companies)}

plt.figure()

for c in companies:
    subset = ram_df[ram_df["company"] == c]
    plt.scatter(
        subset["memory_ram_gb"],
        subset["price"],
        label=c,
        alpha=0.7
    )

unknown = ram_df[ram_df["company"].isna()]
if len(unknown) > 0:
    plt.scatter(
        unknown["memory_ram_gb"],
        unknown["price"],
        label="Unknown",
        marker="x",
        alpha=0.7
    )

plt.xlim(0, 32)          # <-- RAM range
plt.ylim(0, 1000)        # <-- price range

plt.xticks([2, 4, 6, 8, 12, 16, 24, 32])
plt.title("Price vs RAM")
plt.xlabel("RAM (GB)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ----------------------------
# 8) Scatter: Price vs Storage (colored by company)
# ----------------------------
storage_df = phones.dropna(subset=["price", "storage_gb"]).copy()
companies2 = sorted([c for c in storage_df["company"].dropna().unique()])
cmap2 = plt.get_cmap("tab10")
color_map2 = {c: cmap2(i % 10) for i, c in enumerate(companies2)}

plt.figure()

for c in companies2:
    subset = storage_df[storage_df["company"] == c]
    plt.scatter(
        subset["storage_gb"],
        subset["price"],
        label=c,
        alpha=0.7
    )

unknown2 = storage_df[storage_df["company"].isna()]
if len(unknown2) > 0:
    plt.scatter(
        unknown2["storage_gb"],
        unknown2["price"],
        label="Unknown",
        marker="x",
        alpha=0.7
    )

plt.xlim(0, 1024)        # <-- storage range
plt.ylim(0, 1000)        # <-- price range

plt.xticks([64, 128, 256, 512, 1024])
plt.title("Price vs Storage")
plt.xlabel("Storage (GB)")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# phones DataFrame now contains all requested columns
print("\nPreview:\n", phones.head(10))
