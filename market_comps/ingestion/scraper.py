"""
Scraper — Extraction Layer
===========================
Pure data retrieval: fetches web pages and strips HTML.
No database operations, no LLM calls.
"""

import logging
import os

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def fetch_page_text(url: str) -> str:
    """Render a URL with Playwright (headless Chromium) and return stripped text.

    Handles:
        - Auto-installing Chromium if missing (Streamlit Cloud)
        - Scrolling to trigger lazy-loaded content
        - Preserving <a> href attributes as inline [link: ...] annotations
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True)
        except Exception as e:
            if "Executable doesn't exist" in str(e) or "playwright install" in str(e):
                logger.warning("Playwright browser not found — installing chromium…")
                os.system("playwright install chromium")
                browser = p.chromium.launch(headless=True)
            else:
                raise

        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        # Scroll to bottom to trigger lazy loading
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(2000)

        html_content = page.content()
        browser.close()

    return strip_html(html_content)


def strip_html(html: str) -> str:
    """Remove script/style tags, preserve link hrefs inline, and return clean text.

    Converts <a href="/companies/acme">Acme</a> into:
        Acme [link: /companies/acme]
    so the LLM can see navigable URLs even after HTML is stripped.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()

    # Inline <a> hrefs so they survive get_text()
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        link_text = a_tag.get_text(strip=True)
        a_tag.replace_with(f"{link_text} [link: {href}]")

    return soup.get_text(separator="\n", strip=True)
