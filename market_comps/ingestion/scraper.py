"""
Scraper — Extraction Layer
===========================
Pure data retrieval: fetches web pages and strips HTML.
No database operations, no LLM calls.
"""

import logging
import os

import html2text
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
    """Clean the HTML to remove navigation, footers, scripts, and then convert to Markdown."""
    soup = BeautifulSoup(html, "html.parser")
    
    # 1. Remove non-content structural elements
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav"]):
        tag.extract()
        
    # 2. Remove common junk classes/IDs like sidebars, menus, navs, footer, cookies
    junk_patterns = ["header", "footer", "nav", "menu", "sidebar", "cookie", "banner", "ad-", "promo"]
    for tag in soup.find_all(True):
        # Check classes
        classes = tag.get("class", [])
        if any(any(pattern in str(cls).lower() for pattern in junk_patterns) for cls in classes):
            tag.extract()
            continue
            
        # Check ID
        tag_id = tag.get("id", "")
        if any(pattern in str(tag_id).lower() for pattern in junk_patterns):
            tag.extract()
            continue

    # 3. Unwrap all <a> tags and insert inline [link: ...] annotations
    for a_tag in list(soup.find_all("a", href=True)):
        href = a_tag["href"]
        if href:
            link_span = soup.new_tag("span")
            link_span.string = f" [link: {href}]"
            a_tag.append(link_span)
        a_tag.unwrap()

    # 4. Use html2text to convert the cleaned and unwrapped HTML body
    cleaned_html = str(soup)
    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = False
    h.body_width = 0  # Prevents wrapping lines
    
    return h.handle(cleaned_html)
