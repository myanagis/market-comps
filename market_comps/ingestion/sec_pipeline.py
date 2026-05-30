import re
import time
import hashlib
import logging
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from lxml import etree
from sqlalchemy.orm import Session

from market_comps.config import settings
from market_comps.db.models import Pipeline, PipelineRun, SourceDocument, DocumentText, Organization, FundProfile, Person, PersonOrganizationRole, PersonEmail
from market_comps.ingestion.interfaces import BaseFetcher, BasePreparer, BaseExtractor, BaseNormalizer, BaseUpdater

logger = logging.getLogger(__name__)

SEC_BASE = "https://www.sec.gov"
HEADERS = {
    "User-Agent": settings.sec_edgar_user_agent,
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

def sec_get(url: str) -> requests.Response:
    time.sleep(settings.sec_edgar_rate_limit_delay)
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response

class SECFormDFetcher(BaseFetcher):
    def fetch_data(self, db: Session, run: PipelineRun, pipeline: Pipeline) -> list[dict]:
        config = pipeline.config_json or {}
        days_back = config.get("days_back", 7)
        total_target = days_back * 50 # Approx 50 per day
        
        filings = []
        start_idx = 0
        urls_fetched = []
        
        while len(filings) < total_target:
            fetch_count = min(100, total_target - len(filings))
            url = f"{SEC_BASE}/cgi-bin/browse-edgar?action=getcurrent&type=D&owner=include&count={fetch_count}&start={start_idx}"
            urls_fetched.append(url)
            logger.info(f"Requesting recent Form D filings: {url}")

            html = sec_get(url).text
            soup = BeautifulSoup(html, "html.parser")

            rows = soup.select("tr")
            if not rows or len(rows) < 2:
                break # No more results

            added_this_page = 0
            for row in rows:
                row_text = " ".join(row.get_text(" ", strip=True).split())
                if "Accession Number:" not in row_text:
                    continue

                accession_match = re.search(r"Accession Number:\s*([0-9-]+)", row_text)
                filing_date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", row_text)

                accession_number = accession_match.group(1) if accession_match else None
                filing_date = filing_date_match.group(1) if filing_date_match else None

                filing_url = None
                for link in row.find_all("a", href=True):
                    href = link["href"]
                    if "Archives/edgar/data" in href:
                        filing_url = urljoin(SEC_BASE, href)
                        break

                form_type = "D/A" if "D/A" in row_text else "D"
                issuer_guess = row_text.split("Accession Number:")[0].strip()

                if accession_number and filing_url:
                    filings.append({
                        "accession_number": accession_number,
                        "filing_date": filing_date,
                        "form_type": form_type,
                        "issuer_guess": issuer_guess,
                        "filing_url": filing_url
                    })
                    added_this_page += 1
            
            if added_this_page == 0:
                break # We paginated but found no valid Form D rows
                
            start_idx += 100

        # dedupe
        seen = set()
        unique_filings = []
        for f in filings:
            if f["accession_number"] not in seen:
                seen.add(f["accession_number"])
                unique_filings.append(f)
                
        # Log to the run for troubleshooting
        current_logs = run.logs_json or {}
        current_logs["sec_fetch_urls"] = urls_fetched
        current_logs["sec_days_back"] = days_back
        current_logs["sec_total_fetched"] = len(unique_filings)
        run.logs_json = current_logs
        db.add(run)
        db.flush()

        return unique_filings

class SECFormDXMLPreparer(BasePreparer):
    def prepare_raw_data(self, db: Session, run: PipelineRun, pipeline: Pipeline, raw_data: list[dict]) -> list[DocumentText]:
        doc_texts = []
        # For cost/time, limit to a small number unless explicitly overriden
        limit = pipeline.config_json.get("max_filings_to_process", 10) if pipeline.config_json else 10
        
        for filing in raw_data[:limit]:
            try:
                filing_detail_url = filing["filing_url"]
                html = sec_get(filing_detail_url).text
                soup = BeautifulSoup(html, "html.parser")

                xml_url = None
                for a in soup.find_all("a", href=True):
                    href = a["href"]
                    text = a.get_text(" ", strip=True)
                    if href.lower().endswith(".xml") and text.lower().endswith("xml"):
                        xml_url = urljoin(SEC_BASE, href)
                        break

                if not xml_url:
                    continue

                raw_xml = sec_get(xml_url).text
                content_hash = hashlib.sha256(raw_xml.encode()).hexdigest()

                source_doc = SourceDocument(
                    pipeline_run_id=run.id,
                    document_type="SEC_XML",
                    source_url=xml_url,
                    content_hash=content_hash,
                )
                db.add(source_doc)
                db.flush()

                doc_text = DocumentText(
                    source_document_id=source_doc.id,
                    data_type="XML_TEXT",
                    raw_content=raw_xml,
                    content_hash=content_hash,
                )
                db.add(doc_text)
                db.flush()

                doc_texts.append(doc_text)
            except Exception as e:
                logger.error(f"Failed to prepare {filing.get('accession_number')}: {e}")
                
        return doc_texts

class SECFormDExtractor(BaseExtractor):
    def extract_attributes(self, db: Session, run: PipelineRun, pipeline: Pipeline, prepared_data: list[DocumentText]) -> list[dict]:
        results = []
        for doc in prepared_data:
            try:
                parser = etree.XMLParser(recover=True, huge_tree=True)
                root = etree.fromstring(doc.raw_content.encode("utf-8"), parser=parser)
                
                def first_text(tag_name: str) -> str | None:
                    res = root.xpath(f'//*[local-name()="{tag_name}"]/text()')
                    return str(res[0]).strip() if res else None

                def all_texts(tag_name: str) -> list[str]:
                    res = root.xpath(f'//*[local-name()="{tag_name}"]/text()')
                    return [str(x).strip() for x in res if str(x).strip()]

                # Address
                nodes = root.xpath('//*[local-name()="primaryIssuer"]')
                address = {}
                if nodes:
                    node = nodes[0]
                    def lf(tag):
                        r = node.xpath(f'.//*[local-name()="{tag}"]/text()')
                        return str(r[0]).strip() if r else None
                    address = {
                        "street1": lf("street1"),
                        "street2": lf("street2"),
                        "city": lf("city"),
                        "state_or_country": lf("stateOrCountry"),
                        "zip_code": lf("zipCode"),
                    }

                # People
                people = []
                for pnode in root.xpath('//*[local-name()="relatedPersonInfo"]'):
                    def pf(tag):
                        r = pnode.xpath(f'.//*[local-name()="{tag}"]/text()')
                        return str(r[0]).strip() if r else None
                    fname = pf("firstName")
                    lname = pf("lastName")
                    mname = pf("middleName")
                    full = " ".join(x for x in [fname, mname, lname] if x)
                    rel = pf("relationship")
                    clarif = pf("relationshipClarification")
                    rel_full = f"{rel} ({clarif})" if rel and clarif else (rel or clarif)
                    
                    people.append({
                        "name": full or pf("relatedPersonName"),
                        "first_name": fname,
                        "last_name": lname,
                        "relationship": rel_full,
                        "city": pf("city"),
                        "state": pf("stateOrCountry"),
                    })

                data = {
                    "accession_number": doc.source_document.source_url.split("/")[-1].replace(".xml", ""),
                    "issuer_name": first_text("issuerName"),
                    "industry_group": ", ".join(all_texts("investmentFundType")) or first_text("industryGroupType"),
                    "total_offering_amount": first_text("totalOfferingAmount"),
                    "total_amount_sold": first_text("totalAmountSold"),
                    "issuer_address": address,
                    "related_persons": people
                }
                results.append(data)
            except Exception as e:
                logger.error(f"Failed to extract {doc.id}: {e}")
                
        return results
from market_comps.llm_client import LLMClient
import json

class SECFormDNormalizer(BaseNormalizer):
    def normalize_data(self, db: Session, run: PipelineRun, pipeline: Pipeline, extracted_data: list[dict]) -> list[dict]:
        normalized = []
        llm = LLMClient()
        
        for filing in extracted_data:
            issuer_name = filing.get("issuer_name")
            if not issuer_name:
                continue

            addr = filing.get("issuer_address", {})
            people = filing.get("related_persons", [])
            
            # Use LLM to deduce firm name
            prompt = f"""Given the following SEC Form D filing data for an investment fund, deduce the likely name of the overarching parent Investment Management Firm.
For example, if the fund is "PUMA Venture Capital Fund II, LP", the firm is likely "PUMA Venture Capital".
If the fund is "Sequoia Capital U.S. Growth Fund VIII, L.P.", the firm is likely "Sequoia Capital".

Fund Name: {issuer_name}
Fund Address: {addr.get('city')}, {addr.get('state_or_country')}
Related People/Entities: {[p.get('name') for p in people]}

Respond ONLY with the deduced firm name. Do not include any other text."""
            
            try:
                firm_name_raw, usage = llm.chat_completion([{"role": "user", "content": prompt}], step_name="deduce_firm_name")
                firm_name = firm_name_raw.strip('"').strip()
            except Exception as e:
                logger.error(f"LLM deduction failed for firm name: {e}")
                firm_name = issuer_name.split(" Fund ")[0].split(" LLC")[0].split(" L.P.")[0].split(" LP")[0]
            
            fund_data = {
                "accession_number": filing.get("accession_number"),
                "firm_name": firm_name,
                "fund_name": issuer_name,
                "fund_type": filing.get("industry_group"),
                "fund_size_raised": filing.get("total_amount_sold"),
                "fund_size_target": filing.get("total_offering_amount"),
                "street1": addr.get("street1"),
                "street2": addr.get("street2"),
                "city": addr.get("city"),
                "state": addr.get("state_or_country"),
                "country": "US", # SEC default usually
                "zip_code": addr.get("zip_code"),
                "people": filing.get("related_persons", [])
            }
            normalized.append(fund_data)
        return normalized

class SECFormDUpdater(BaseUpdater):
    def update_records(self, db: Session, run: PipelineRun, pipeline: Pipeline, normalized_data: list[dict]) -> dict:
        stats = {"orgs_created": 0, "funds_created": 0, "people_created": 0}
        
        for data in normalized_data:
            # 1. UPSERT Organization (Firm)
            firm_name = data["firm_name"]
            firm = db.query(Organization).filter(Organization.name.ilike(firm_name)).first()
            if not firm:
                firm = Organization(
                    name=firm_name,
                    normalized_name=firm_name.lower(),
                    city=data.get("city"),
                    state=data.get("state"),
                    country=data.get("country"),
                    street1=data.get("street1"),
                    street2=data.get("street2"),
                    zip_code=data.get("zip_code"),
                )
                db.add(firm)
                db.flush()
                stats["orgs_created"] += 1

            # 2. UPSERT FundProfile
            fund_name = data["fund_name"]
            fund = db.query(FundProfile).filter(FundProfile.fund_name.ilike(fund_name)).first()
            if not fund:
                fund = FundProfile(
                    parent_organization_id=firm.id,
                    fund_name=fund_name,
                    investment_fund_type=data.get("fund_type"),
                    fund_size_raised=data.get("fund_size_raised"),
                    fund_size_target=data.get("fund_size_target"),
                    street1=data.get("street1"),
                    street2=data.get("street2"),
                    city=data.get("city"),
                    state=data.get("state"),
                    country=data.get("country"),
                    zip_code=data.get("zip_code"),
                    accession_number=data.get("accession_number")
                )
                db.add(fund)
                db.flush()
                stats["funds_created"] += 1

            # 3. UPSERT People
            for p in data.get("people", []):
                p_name = p.get("name")
                if not p_name: continue
                person = db.query(Person).filter(Person.full_name.ilike(p_name)).first()
                if not person:
                    person = Person(
                        first_name=p.get("first_name"),
                        last_name=p.get("last_name"),
                        full_name=p_name,
                        city=p.get("city"),
                        state=p.get("state"),
                        country="US"
                    )
                    db.add(person)
                    db.flush()
                    stats["people_created"] += 1
                
                # Check role
                role = db.query(PersonOrganizationRole).filter_by(person_id=person.id, organization_id=firm.id).first()
                if not role:
                    role = PersonOrganizationRole(
                        person_id=person.id,
                        organization_id=firm.id,
                        title=p.get("relationship") or "Executive",
                        source="SEC Form D"
                    )
                    db.add(role)
                    db.flush()

        db.commit()
        return stats
