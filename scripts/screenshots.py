"""Screenshots of the running console for the README and deck. Needs: uvicorn on :8780."""
import pathlib
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8780/"
OUT = pathlib.Path(__file__).resolve().parent.parent / "docs" / "screenshots"

with sync_playwright() as p:
    b = p.chromium.launch()
    for w in (1440, 400):
        pg = b.new_page(viewport={"width": w, "height": 900}, device_scale_factor=2 if w == 1440 else 1)
        pg.goto(URL)
        pg.wait_for_selector("#rev")
        pg.screenshot(path=OUT / f"console-{w}.png", full_page=True)
        if w == 1440:
            pg.screenshot(path=OUT / "console-top.png")
            pg.click("#rev")
            pg.wait_for_selector(".outcome")
            pg.wait_for_timeout(800)  # let the reveal fade finish
            pg.locator("#detail").screenshot(path=OUT / "event-revealed.png")
            pg.locator("section.trade").first.screenshot(path=OUT / "tradeoff.png")
            pg.locator("section.trade").nth(1).screenshot(path=OUT / "calibration.png")
            ov = pg.evaluate("document.documentElement.scrollWidth > innerWidth")
        else:
            ov = pg.evaluate("document.documentElement.scrollWidth > innerWidth")
        print(w, "horizontal scroll:", ov)
    b.close()
