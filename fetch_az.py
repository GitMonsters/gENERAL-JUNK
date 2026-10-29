import requests,re
r=requests.get("https://az-highways.preview.emergentagent.com/", headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
html=r.text
for m in re.findall(r"static/js/[^\"']+\.js", html):
    print("JS", m)
m=re.search(r"<title>(.*?)</title>", html, re.S)
print("TITLE", m.group(1) if m else None)
