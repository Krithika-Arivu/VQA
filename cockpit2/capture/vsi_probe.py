r"""
vsi_probe.py  -- diagnose the VSI needle lag found in dataset_v2/climb.

What it does (FlightGear running exactly as for capture, --httpd=5500, cockpit view):
  1. holds vs = 0 for 6 s, then steps to vs = 1000 and holds it for 8 s
  2. during the hold, every ~0.5 s it
       - screenshots the VSI (saved to vsi_probe/ as t_XX.X.png)
       - dumps EVERY property under /instrumentation/vertical-speed-indicator/
  3. prints a table: which property jumps to 1000 instantly (the one we write)
     and which one creeps up slowly (= the one the needle probably draws).

Run:  python vsi_probe.py
Then send: the printed table + vsi_probe/sheet.png
Same crop fractions and HTTP address as capture_synthetic_v2.py.
"""
import json
import os
import time

import requests
from mss import mss
from PIL import Image, ImageDraw

HTTP = "http://127.0.0.1:5500"
CROP = (0.365, 0.465, 0.690, 0.870)          # panel crop, same as capture
VSI_IN_CROP = (485, 285, 685, 475)          # VSI inside the 936x729 crop
VSI_DIR = "/instrumentation/vertical-speed-indicator"
OUT = "vsi_probe"
S = requests.Session()


def nasal(code):
    body = {"name": "", "children": [{"name": "script", "index": 0, "value": code}]}
    S.post(HTTP + "/run.cgi?value=nasal", data=json.dumps(body), timeout=5)


def dump(path, depth=3):
    """All numeric properties under path -> {full_path: value}."""
    out = {}
    try:
        node = S.get(f"{HTTP}/json{path}?d={depth}", timeout=3).json()
    except Exception as e:
        print("dump failed:", e)
        return out

    def walk(n, prefix):
        p = n.get("path") or (prefix + "/" + n.get("name", ""))
        if "value" in n and isinstance(n["value"], (int, float)) and not isinstance(n["value"], bool):
            out[p] = float(n["value"])
        for c in n.get("children", []) or []:
            walk(c, p)
    walk(node, path)
    return out


# if the capture script's writer is still running in this FlightGear session, pause it
# (otherwise it would keep overwriting the VSI with the last captured value)
nasal('setprop("/capture/gauge/on", 0);')
# writer loop: re-asserts the VS display property every frame (like the capture script)
nasal(r'''
var gen = getprop("/probe/gen"); if (gen == nil) gen = 0; gen += 1; setprop("/probe/gen", gen);
var loop = func {
    if (getprop("/probe/gen") != gen) return;
    var v = getprop("/probe/vs"); if (v != nil) setprop("/instrumentation/vertical-speed-indicator/indicated-speed-fpm", v);
    settimer(loop, 0);
};
loop();
''')

print("Click on FlightGear now -- starting in 8 s. Hands off.")
time.sleep(8)
os.makedirs(OUT, exist_ok=True)
nasal('setprop("/probe/vs", 0);')
time.sleep(6)

rows, tiles = [], []
with mss() as sct:
    mon = sct.monitors[1]
    nasal('setprop("/probe/vs", 1000);')
    t0 = time.time()
    while time.time() - t0 < 8.0:
        t = time.time() - t0
        s = sct.grab(mon)
        full = Image.frombytes("RGB", s.size, s.rgb)
        W, H = full.size
        crop = full.crop((int(CROP[0] * W), int(CROP[1] * H), int(CROP[2] * W), int(CROP[3] * H)))
        vsi = crop.crop(VSI_IN_CROP)
        vsi.save(os.path.join(OUT, f"t_{t:04.1f}.png"))
        tiles.append((t, vsi))
        rows.append((t, dump(VSI_DIR)))
        time.sleep(0.4)
nasal('setprop("/probe/vs", 0); setprop("/probe/gen", -1);')   # stop the probe loop

# contact sheet
w, h = 200, 190
sheet = Image.new("RGB", (6 * w, ((len(tiles) + 5) // 6) * (h + 18)), "white")
dr = ImageDraw.Draw(sheet)
for i, (t, im) in enumerate(tiles):
    x, y = (i % 6) * w, (i // 6) * (h + 18)
    sheet.paste(im.resize((w, h)), (x, y + 18))
    dr.text((x + 4, y + 3), f"t = {t:.1f} s after step to 1000", fill="black")
sheet.save(os.path.join(OUT, "sheet.png"))

# table: one column per property that exists under the VSI node
props = sorted({p for _, d in rows for p in d})
print("\nseconds | " + " | ".join(p.replace(VSI_DIR + "/", "") for p in props))
for t, d in rows:
    print(f"{t:7.1f} | " + " | ".join(f"{d.get(p, float('nan')):9.1f}" for p in props))
print(f"\nSaved {os.path.abspath(os.path.join(OUT, 'sheet.png'))}")
print("Also run the findstr command to find what the needle model reads.")
