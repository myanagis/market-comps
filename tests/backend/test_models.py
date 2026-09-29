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

def test_financing_round_sort(test_db):
    """Test sorting financing rounds with mixed datetime and date objects (regression test for sorting crash)"""
    from market_comps.db.models import FinancingRound
    from datetime import date, datetime
    org = Organization(name="Sort Test Corp", organization_type="COMPANY")
    test_db.add(org)
    test_db.commit()
    
    # 1. Round with no date (relies on created_at which is a datetime)
    r1 = FinancingRound(company_id=org.id, round_name="Seed", status="closed")
    
    # 2. Round with exact date (date object)
    r2 = FinancingRound(company_id=org.id, round_name="Series A", status="closed", announced_date=date(2023, 1, 1))
    
    # 3. Round with datetime object (from aggressive parser)
    r3 = FinancingRound(company_id=org.id, round_name="Series B", status="closed", announced_date=datetime(2024, 1, 1, 12, 0))
    
    test_db.add_all([r1, r2, r3])
    test_db.commit()
    
    # Simulate UI sorting logic
    def get_rnd_sort_key(r):
        if r.announced_date:
            if isinstance(r.announced_date, datetime): return r.announced_date.date()
            return r.announced_date
        if r.created_at:
            if isinstance(r.created_at, datetime): return r.created_at.date()
            return r.created_at
        return date.min
        
    try:
        sorted_rounds = sorted([r1, r2, r3], key=get_rnd_sort_key, reverse=True)
        assert len(sorted_rounds) == 3
    except TypeError as e:
        import pytest
        pytest.fail(f"Sorting raised TypeError due to mixed types: {e}")
