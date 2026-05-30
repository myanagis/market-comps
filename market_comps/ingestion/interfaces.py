from abc import ABC, abstractmethod
from typing import Any
from sqlalchemy.orm import Session
from market_comps.db.models import Pipeline, PipelineRun

class BaseFetcher(ABC):
    @abstractmethod
    def fetch_data(self, db: Session, run: PipelineRun, pipeline: Pipeline) -> Any:
        """
        Fetch raw data (e.g. web pages, SEC filings).
        Returns raw fetched data (list of items, single string, or writes directly to DB).
        """
        pass

class BasePreparer(ABC):
    @abstractmethod
    def prepare_raw_data(self, db: Session, run: PipelineRun, pipeline: Pipeline, raw_data: Any) -> Any:
        """
        Convert raw data into a standard format (e.g. HTML to Markdown, or save SourceDocument).
        """
        pass

class BaseExtractor(ABC):
    @abstractmethod
    def extract_attributes(self, db: Session, run: PipelineRun, pipeline: Pipeline, prepared_data: Any) -> Any:
        """
        Extract structured entities and relationships from prepared text (LLM or parsing).
        """
        pass

class BaseNormalizer(ABC):
    @abstractmethod
    def normalize_data(self, db: Session, run: PipelineRun, pipeline: Pipeline, extracted_data: Any) -> Any:
        """
        Transform extracted attributes into Canonical formats (Organizations, People, Funds).
        """
        pass

class BaseUpdater(ABC):
    @abstractmethod
    def update_records(self, db: Session, run: PipelineRun, pipeline: Pipeline, normalized_data: Any) -> dict:
        """
        Upsert the normalized canonical records to the CRM models. Returns a stats dict.
        """
        pass
