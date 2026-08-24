import logging
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import or_

from market_comps.db.models import (
    Market, 
    MarketSegment, 
    MarketSegmentCompanyLink,
    CompetitiveAnalysis,
    CompetitiveAnalysisSegment,
    CompetitiveAnalysisCompany,
    Organization,
    AuditTrail
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

def create_market(db: Session, name: str, description: Optional[str] = None, sectors: Optional[List[str]] = None, created_by: str = "USER") -> Market:
    market = Market(name=name.strip(), description=description, sectors=sectors)
    db.add(market)
    db.flush()
    db.add(AuditTrail(
        canonical_entity_type="MARKET",
        canonical_entity_id=str(market.id),
        mutation_type="CREATE",
        source="CRM",
        created_by=created_by
    ))
    return market

def get_market_segments(db: Session, market_id: int) -> List[MarketSegment]:
    return db.query(MarketSegment).filter_by(market_id=market_id).order_by(MarketSegment.sort_order, MarketSegment.name).all()

def create_market_segment(
    db: Session, 
    market_id: int, 
    name: str, 
    description: Optional[str] = None, 
    segment_type: Optional[str] = None,
    created_by: str = "USER"
) -> MarketSegment:
    seg = MarketSegment(
        market_id=market_id,
        name=name.strip(),
        description=description,
        segment_type=segment_type
    )
    db.add(seg)
    db.flush()
    db.add(AuditTrail(
        canonical_entity_type="MARKET_SEGMENT",
        canonical_entity_id=str(seg.id),
        mutation_type="CREATE",
        source="CRM",
        created_by=created_by
    ))
    return seg

# ==============================================================================
# COMPANY MARKET LINKAGES
# ==============================================================================

def add_company_to_segment(
    db: Session,
    company_id: int,
    market_segment_id: int,
    differentiation: str,
    notes: Optional[str] = None,
    created_by: str = "USER"
) -> MarketSegmentCompanyLink:
    existing = db.query(MarketSegmentCompanyLink).filter_by(
        company_id=company_id,
        market_segment_id=market_segment_id
    ).first()
    
    if existing:
        existing.differentiation = differentiation
        existing.notes = notes
        db.flush()
        db.add(AuditTrail(
            canonical_entity_type="MARKET_SEGMENT_LINK",
            canonical_entity_id=str(existing.id),
            mutation_type="UPDATE",
            source="CRM",
            created_by=created_by
        ))
        return existing
        
    link = MarketSegmentCompanyLink(
        company_id=company_id,
        market_segment_id=market_segment_id,
        differentiation=differentiation,
        notes=notes
    )
    db.add(link)
    db.flush()
    db.add(AuditTrail(
        canonical_entity_type="MARKET_SEGMENT_LINK",
        canonical_entity_id=str(link.id),
        mutation_type="CREATE",
        source="CRM",
        created_by=created_by
    ))
    return link

def get_company_segments(db: Session, company_id: int) -> List[MarketSegmentCompanyLink]:
    return db.query(MarketSegmentCompanyLink).filter_by(company_id=company_id).all()

# ==============================================================================
# COMPETITIVE ANALYSIS
# ==============================================================================

def get_or_create_competitive_analysis(
    db: Session, 
    subject_company_id: int, 
    market_id: int,
    title: str,
    created_by: str = "USER"
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
    db.add(AuditTrail(
        canonical_entity_type="COMPETITIVE_ANALYSIS",
        canonical_entity_id=str(ca.id),
        mutation_type="CREATE",
        source="CRM",
        created_by=created_by
    ))
    return ca

def add_competitive_analysis_segment(
    db: Session,
    competitive_analysis_id: int,
    market_segment_id: int,
    threat_level: Optional[str] = None,
    analysis_notes: Optional[str] = None,
    created_by: str = "USER"
) -> CompetitiveAnalysisSegment:
    existing = db.query(CompetitiveAnalysisSegment).filter_by(
        competitive_analysis_id=competitive_analysis_id,
        market_segment_id=market_segment_id
    ).first()
    if existing:
        if threat_level: existing.threat_level = threat_level
        if analysis_notes: existing.analysis_notes = analysis_notes
        db.flush()
        db.add(AuditTrail(
            canonical_entity_type="COMPETITIVE_ANALYSIS_SEGMENT",
            canonical_entity_id=str(existing.id),
            mutation_type="UPDATE",
            source="CRM",
            created_by=created_by
        ))
        return existing
    
    seg = CompetitiveAnalysisSegment(
        competitive_analysis_id=competitive_analysis_id,
        market_segment_id=market_segment_id,
        threat_level=threat_level,
        analysis_notes=analysis_notes
    )
    db.add(seg)
    db.flush()
    db.add(AuditTrail(
        canonical_entity_type="COMPETITIVE_ANALYSIS_SEGMENT",
        canonical_entity_id=str(seg.id),
        mutation_type="CREATE",
        source="CRM",
        created_by=created_by
    ))
    return seg


def add_competitive_analysis_company(
    db: Session,
    competitive_analysis_id: int,
    competitor_company_id: int,
    market_segment_id: Optional[int] = None,
    relationship_type: str = "direct_competitor",
    threat_level: Optional[str] = None,
    threat_level_description: Optional[str] = None,
    competitive_notes: Optional[str] = None,
    created_by: str = "USER"
) -> CompetitiveAnalysisCompany:
    
    existing = db.query(CompetitiveAnalysisCompany).filter_by(
        competitive_analysis_id=competitive_analysis_id,
        competitor_company_id=competitor_company_id,
        market_segment_id=market_segment_id
    ).first()
    
    if existing:
        existing.relationship_type = relationship_type
        existing.threat_level = threat_level
        existing.threat_level_description = threat_level_description
        existing.competitive_notes = competitive_notes
        db.flush()
        db.add(AuditTrail(
            canonical_entity_type="COMPETITIVE_ANALYSIS_COMPANY",
            canonical_entity_id=str(existing.id),
            mutation_type="UPDATE",
            source="CRM",
            created_by=created_by
        ))
        return existing
        
    comp = CompetitiveAnalysisCompany(
        competitive_analysis_id=competitive_analysis_id,
        competitor_company_id=competitor_company_id,
        market_segment_id=market_segment_id,
        relationship_type=relationship_type,
        threat_level=threat_level,
        threat_level_description=threat_level_description,
        competitive_notes=competitive_notes
    )
    db.add(comp)
    db.flush()
    db.add(AuditTrail(
        canonical_entity_type="COMPETITIVE_ANALYSIS_COMPANY",
        canonical_entity_id=str(comp.id),
        mutation_type="CREATE",
        source="CRM",
        created_by=created_by
    ))
    return comp
