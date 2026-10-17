import asyncio
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from playwright.async_api import async_playwright

async def main():
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            executable_path=chrome_path,
            headless=True
        )
        page = await browser.new_page()
        
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: console_messages.append(f"[PAGE ERROR] {err}"))
        
        print("[*] Navigating to http://127.0.0.1:7860/ ...", flush=True)
        await page.goto("http://127.0.0.1:7860/", wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        
        # Switch to Snake tab
        print("[*] Switching to Snake tab...", flush=True)
        await page.click("#nav-toggle-snake")
        await page.wait_for_timeout(500)
        
        # Click "1 mossa"
        print("[*] Clicking button with text '1 mossa'...", flush=True)
        await page.click("button:has-text('1 mossa')")
        
        # Wait for step to complete
        for i in range(12):
            await page.wait_for_timeout(600)
            status = await page.inner_text("#sn-status-title")
            if status != "Sto decidendo...":
                break
                
        status_title = await page.inner_text("#sn-status-title")
        status_sub = await page.inner_text("#sn-status-sub")
        gpu_time = await page.inner_text("#sn-gpu")
        score_bar = await page.inner_text("#sn-score-bar")
        
        print("=" * 60, flush=True)
        print(f"sn-status-title:   '{status_title}'", flush=True)
        print(f"sn-status-sub:     '{status_sub}'", flush=True)
        print(f"GPU latency:       '{gpu_time}'", flush=True)
        print(f"Score bar:         '{score_bar}'", flush=True)
        print("=" * 60, flush=True)
        
        screenshot_path = r"C:\Users\Acer\spark-snake\screen_snake_browser_step.png"
        await page.screenshot(path=screenshot_path)
        print(f"[+] Screenshot saved to {screenshot_path}", flush=True)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
