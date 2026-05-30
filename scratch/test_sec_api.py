from bs4 import BeautifulSoup
from market_comps.ingestion.sec_pipeline import sec_get

url = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=D&owner=include&count=100&start=0"
html = sec_get(url).text
soup = BeautifulSoup(html, "html.parser")
rows = soup.select("tr")
print(f"Total rows on page 1 (start=0): {len(rows)}")

url2 = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=D&owner=include&count=100&start=100"
html2 = sec_get(url2).text
soup2 = BeautifulSoup(html2, "html.parser")
rows2 = soup2.select("tr")
print(f"Total rows on page 2 (start=100): {len(rows2)}")

url3 = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=D&owner=include&count=100&start=200"
html3 = sec_get(url3).text
soup3 = BeautifulSoup(html3, "html.parser")
rows3 = soup3.select("tr")
print(f"Total rows on page 3 (start=200): {len(rows3)}")
