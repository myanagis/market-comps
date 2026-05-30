from market_comps.ingestion.interfaces import BaseFetcher, BasePreparer, BaseExtractor, BaseNormalizer, BaseUpdater
from market_comps.ingestion.generic_pipeline import (
    GenericHTTPFetcher,
    GenericHTMLPreparer,
    GenericLLMExtractor,
    InvestorPortfolioNormalizer,
    GenericCanonicalUpdater
)
from market_comps.ingestion.sec_pipeline import (
    SECFormDFetcher,
    SECFormDXMLPreparer,
    SECFormDExtractor,
    SECFormDNormalizer,
    SECFormDUpdater
)

FETCHERS: dict[str, type[BaseFetcher]] = {
    "WEB_PAGE": GenericHTTPFetcher,
    "SEC_DAILY_INDEX": SECFormDFetcher
}

PREPARERS: dict[str, type[BasePreparer]] = {
    "HTML_TO_MARKDOWN": GenericHTMLPreparer,
    "SEC_FORM_D_XML": SECFormDXMLPreparer
}

EXTRACTORS: dict[str, type[BaseExtractor]] = {
    "GENERIC_LLM": GenericLLMExtractor,
    "HTML_TO_MARKDOWN": GenericLLMExtractor, # Fallback compatibility
    "SEC_FORM_D_XML": SECFormDExtractor
}

NORMALIZERS: dict[str, type[BaseNormalizer]] = {
    "PORTFOLIO_TO_INVESTMENTS": InvestorPortfolioNormalizer,
    "PORTFOLIO_COMPANIES_TO_INVESTMENTS": InvestorPortfolioNormalizer,
    "SEC_FORM_D_TO_ORGS": SECFormDNormalizer
}

UPDATERS: dict[str, type[BaseUpdater]] = {
    "CANONICAL_UPSERT": GenericCanonicalUpdater,
    "RECORD_UPDATER": GenericCanonicalUpdater,
    "SEC_FORM_D_UPDATER": SECFormDUpdater,
}

def get_fetcher(connector_type: str) -> BaseFetcher:
    cls = FETCHERS.get(connector_type, GenericHTTPFetcher)
    return cls()

def get_preparer(parser_type: str) -> BasePreparer:
    cls = PREPARERS.get(parser_type, GenericHTMLPreparer)
    return cls()

def get_extractor(parser_type: str) -> BaseExtractor:
    cls = EXTRACTORS.get(parser_type, GenericLLMExtractor)
    return cls()

def get_normalizer(normalizer_type: str) -> BaseNormalizer:
    cls = NORMALIZERS.get(normalizer_type, InvestorPortfolioNormalizer)
    return cls()

def get_updater(updater_type: str) -> BaseUpdater:
    cls = UPDATERS.get(updater_type, GenericCanonicalUpdater)
    return cls()
