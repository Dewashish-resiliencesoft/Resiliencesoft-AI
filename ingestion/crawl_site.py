import re, sys, json, time, urllib.parse as up
import httpx
from bs4 import BeautifulSoup

"""Polite same-site crawler (honours robots.txt Disallow, 0.5s delay, sitemap + link discovery).

Usage: python -m ingestion.crawl_site https://example.com out.json
Needs: beautifulsoup4 (uv run --with beautifulsoup4 ...). Output = {url: {title, description, text}}.
"""
BASE = sys.argv[1].rstrip("/")
OUT = sys.argv[2]
HOST = up.urlparse(BASE).netloc.removeprefix("www.")
_rb = httpx.get(BASE + "/robots.txt", timeout=20, follow_redirects=True)
DISALLOW, _ua = [], False
for _l in (_rb.text.splitlines() if _rb.status_code == 200 else []):
    _l = _l.split("#")[0].strip()
    if _l.lower().startswith("user-agent:"): _ua = _l.split(":", 1)[1].strip() == "*"
    elif _ua and _l.lower().startswith("disallow:") and _l.split(":", 1)[1].strip():
        DISALLOW.append(_l.split(":", 1)[1].strip())
DISALLOW = tuple(DISALLOW)
c = httpx.Client(follow_redirects=True, timeout=30, headers={"User-Agent": "ResilienceSoftAI-Crawler/1.0 (own-site knowledge base)"})

def norm(u):
    u = up.urldefrag(up.urljoin(BASE + "/", u))[0]
    p = up.urlparse(u)
    if p.netloc.removeprefix("www.") != HOST: return None
    if any(p.path.startswith(d) for d in DISALLOW): return None
    if re.search(r"\.(png|jpe?g|svg|webp|ico|css|js|pdf|xml|txt|mp4|woff2?)$", p.path): return None
    if p.path.startswith("/_next"): return None
    return f"{p.scheme}://{p.netloc}{p.path}" + (f"?{p.query}" if p.query else "")

sm = c.get(BASE + "/sitemap.xml").text
if "<loc>" not in sm: sm = ""
queue = [norm(u) for u in re.findall(r"<loc>([^<]+)</loc>", sm)]
queue = [u for u in queue if u] or [norm(BASE + "/")]
seen, pages = set(), {}
while queue:
    u = queue.pop(0)
    if u in seen: continue
    seen.add(u)
    try:
        r = c.get(u)
    except Exception as e:
        print("ERR", u, e); continue
    if r.status_code != 200 or "html" not in r.headers.get("content-type", ""):
        print(r.status_code, u); continue
    soup = BeautifulSoup(r.text, "html.parser")
    for a in soup.find_all("a", href=True):
        n = norm(a["href"])
        if n and n not in seen: queue.append(n)
    for t in soup(["script", "style", "noscript", "svg", "iframe"]): t.decompose()
    title = (soup.title.get_text(strip=True) if soup.title else "")
    md = soup.find("meta", attrs={"name": "description"})
    desc = md["content"].strip() if md and md.get("content") else ""
    main = soup.find("main") or soup.body
    lines = []
    for el in main.find_all(["h1","h2","h3","h4","p","li","td","th","summary","button","a"]):
        if el.name == "a" and el.find(["p","li","h1","h2","h3"]): continue
        if el.name in ("li","button","a") and el.find(["p","li"]): continue
        t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
        if not t: continue
        pre = {"h1":"# ","h2":"## ","h3":"### ","h4":"#### ","li":"- "}.get(el.name, "")
        lines.append(pre + t)
    seen_l, out = set(), []
    for l in lines:
        if l not in seen_l: seen_l.add(l); out.append(l)
    pages[u] = {"title": title, "description": desc, "text": "\n".join(out)}
    print(200, u, len(pages[u]["text"]), "chars")
    time.sleep(0.5)
json.dump(pages, open(OUT, "w"), ensure_ascii=False, indent=1)
print("pages:", len(pages))
