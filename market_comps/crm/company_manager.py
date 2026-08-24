import logging
import re
from typing import Optional, List, Tuple
from sqlalchemy.orm import Session

from market_comps.db.models import Organization, CompanyProfile, InvestorProfile, AuditTrail
from market_comps.ingestion.company_augmentation import run_augmentation_pipeline
from market_comps.utils import normalize_company_name

logger = logging.getLogger(__name__)

def find_existing_company(db: Session, name: str, domain: Optional[str] = None) -> Optional[Organization]:
    """
    Checks if a company already exists in the database by exact domain match or normalized name match.
    """
    if domain:
        clean_domain = domain.strip().lower().replace("http://", "").replace("https://", "").replace("www.", "").split('/')[0]
        if clean_domain:
            org = db.query(Organization).filter(Organization.primary_domain == clean_domain).first()
            if org:
                return org
                
    norm_name = normalize_company_name(name)
    if norm_name:
        org = db.query(Organization).filter(Organization.normalized_name == norm_name).first()
        if org:
            return org
            
    return None

def create_company(
    db: Session, 
    name: str, 
    domain: Optional[str] = None, 
    description: Optional[str] = None, 
    ticker_symbol: Optional[str] = None,
    stock_exchange: Optional[str] = None,
    ownership_type: Optional[str] = None,
    organization_type: str = "COMPANY",
    created_by: str = "UploaderAgent"
) -> Organization:
    """
    Creates a new Organization, its Profile, and an AuditTrail.
    """
    clean_domain = None
    if domain:
        clean_domain = domain.strip().lower().replace("http://", "").replace("https://", "").replace("www.", "").split('/')[0]

    norm_name = normalize_company_name(name)
    
    org = Organization(
        name=name.strip(),
        normalized_name=norm_name,
        primary_domain=clean_domain,
        website_url=f"https://{clean_domain}" if clean_domain else None,
        description=description,
        ticker_symbol=ticker_symbol,
        stock_exchange=stock_exchange,
        ownership_type=ownership_type,
        organization_type=organization_type.upper(),
        status="active"
    )
    db.add(org)
    db.flush()
    
    # Create profile
    if organization_type.upper() == "INVESTOR":
        profile = InvestorProfile(organization_id=org.id)
    else:
        profile = CompanyProfile(organization_id=org.id)
        
    db.add(profile)
    
    # Audit trail
    audit = AuditTrail(
        canonical_entity_type="Organization",
        canonical_entity_id=str(org.id),
        mutation_type="CREATE",
        source="UPLOADER_AGENT",
        created_by=created_by
    )
    db.add(audit)
    
    db.flush()
    return org

def process_new_company(db: Session, org_id: int):
    """
    Wrapper to run the augmentation pipeline for a newly created company.
    """
    try:
        run_augmentation_pipeline(org_id)
        logger.info(f"Successfully ran augmentation pipeline for org_id {org_id}")
    except Exception as e:
        logger.error(f"Failed to run augmentation pipeline for org_id {org_id}: {e}")
        raise e
