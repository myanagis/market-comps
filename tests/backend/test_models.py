from market_comps.db.models import Organization, Market, MarketSegment, MarketSegmentCompanyLink

def test_create_organization(test_db):
    """Test creating an organization record"""
    org = Organization(name="Test Quantum", organization_type="COMPANY")
    test_db.add(org)
    test_db.commit()
    
    fetched = test_db.query(Organization).filter_by(name="Test Quantum").first()
    assert fetched is not None
    assert fetched.name == "Test Quantum"
    assert fetched.organization_type == "COMPANY"

def test_market_segment_link(test_db):
    """Test creating a market, segment, and linking a company"""
    org = Organization(name="Quantum Corp", organization_type="COMPANY")
    market = Market(name="Quantum Computing")
    test_db.add(org)
    test_db.add(market)
    test_db.commit()
    
    segment = MarketSegment(name="Hardware", market_id=market.id)
    test_db.add(segment)
    test_db.commit()
    
    link = MarketSegmentCompanyLink(
        company_id=org.id,
        market_segment_id=segment.id,
        differentiation="Uses neutral atoms."
    )
    test_db.add(link)
    test_db.commit()
    
    fetched_link = test_db.query(MarketSegmentCompanyLink).first()
    assert fetched_link is not None
    assert fetched_link.differentiation == "Uses neutral atoms."
    assert fetched_link.company.name == "Quantum Corp"
    assert fetched_link.market_segment.name == "Hardware"
