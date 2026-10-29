import qrcode
import qrcode.image.svg
from pathlib import Path

out = Path(r"arizona-super-highways/assets/qr")
out.mkdir(parents=True, exist_ok=True)

codes = {
    "join": "https://app.arizona-super-highways.com/join",
    "medusa-checkin": "https://app.arizona-super-highways.com/c/medusa?t=checkin",
    "digital-guide": "https://arizona-super-highways.com/guide",
}
for name, url in codes.items():
    img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=12, border=2)
    p = out / f"{name}.svg"
    img.save(str(p))
    print(p, p.stat().st_size)
