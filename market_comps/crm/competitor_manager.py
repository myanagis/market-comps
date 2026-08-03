import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_

from market_comps.db.models import (
    Market, 
    MarketSegment, 
    CompanyMarketSegment,
    CompetitiveAnalysis,
    CompetitiveAnalysisSegment,
    CompetitiveAnalysisCompany,
    Organization
)
from market_comps.crm.company_manager import normalize_company_name

logger = logging.getLogger(__name__)

# Enums/Constants
THREAT_LEVELS = ["High", "Medium/High", "Medium", "Medium/Low", "Low", "N/A"]
RELATIONSHIP_TYPES = [
    "direct_competitor", "indirect_competitor", "substitute", 
    "incumbent", "adjacent", "potential_entrant", "partner_competitor"
]

# ==============================================================================
# MARKETS & SEGMENTS
# ==============================================================================

def get_all_markets(db: Session) -> List[Market]:
    return db.query(Market).order_by(Market.name).all()

def create_market(db: Session, name: str, description: Optional[str] = None) -> Market:
    market = Market(name=name.strip(), description=description)
    db.add(market)
    db.flush()
    return market

def get_market_segments(db: Session, market_id: int) -> List[MarketSegment]:
    return db.query(MarketSegment).filter_by(market_id=market_id).order_by(MarketSegment.sort_order, MarketSegment.name).all()

def create_market_segment(
    db: Session, 
    market_id: int, 
    name: str, 
    description: Optional[str] = None, 
    segment_type: Optional[str] = None
) -> MarketSegment:
    seg = MarketSegment(
        market_id=market_id,
        name=name.strip(),
        description=description,
        segment_type=segment_type
    )
    db.add(seg)
    db.flush()
    return seg

# ==============================================================================
# COMPANY MARKET LINKAGES
# ==============================================================================

def add_company_to_segment(
    db: Session,
    company_id: int,
    market_segment_id: int,
    differentiation: str,
    is_primary: bool = False,
    notes: Optional[str] = None
) -> CompanyMarketSegment:
    # Ensure it's not a duplicate
    existing = db.query(CompanyMarketSegment).filter_by(
        company_id=company_id,
        market_segment_id=market_segment_id
    ).first()
    
    if existing:
        existing.differentiation = differentiation
        existing.is_primary = is_primary
        existing.notes = notes
        db.flush()
        return existing
        
    link = CompanyMarketSegment(
        company_id=company_id,
        market_segment_id=market_segment_id,
        is_primary=is_primary,
        differentiation=differentiation,
        notes=notes
    )
    db.add(link)
    db.flush()
    return link

def get_company_segments(db: Session, company_id: int) -> List[CompanyMarketSegment]:
    return db.query(CompanyMarketSegment).filter_by(company_id=company_id).all()

# ==============================================================================
# COMPETITIVE ANALYSIS
# ==============================================================================

def get_or_create_competitive_analysis(
    db: Session, 
    subject_company_id: int, 
    market_id: int,
    title: str
) -> CompetitiveAnalysis:
    ca = db.query(CompetitiveAnalysis).filter_by(
        subject_company_id=subject_company_id,
        market_id=market_id
    ).first()
    
    if ca:
        return ca
        
    ca = CompetitiveAnalysis(
        subject_company_id=subject_company_id,
        market_id=market_id,
        title=title,
        status="draft"
    )
    db.add(ca)
    db.flush()
    return ca

def add_competitive_analysis_company(
    db: Session,
    competitive_analysis_id: int,
    competitor_company_id: int,
    relationship_type: str,
    threat_level: str,
    threat_level_description: Optional[str] = None,
    competitive_notes: Optional[str] = None
) -> CompetitiveAnalysisCompany:
    
    existing = db.query(CompetitiveAnalysisCompany).filter_by(
        competitive_analysis_id=competitive_analysis_id,
        competitor_company_id=competitor_company_id
    ).first()
    
    if existing:
        existing.relationship_type = relationship_type
        existing.threat_level = threat_level
        existing.threat_level_description = threat_level_description
        existing.competitive_notes = competitive_notes
        db.flush()
        return existing
        
    comp = CompetitiveAnalysisCompany(
        competitive_analysis_id=competitive_analysis_id,
        competitor_company_id=competitor_company_id,
        relationship_type=relationship_type,
        threat_level=threat_level,
        threat_level_description=threat_level_description,
        competitive_notes=competitive_notes
    )
    db.add(comp)
    db.flush()
    return comp
