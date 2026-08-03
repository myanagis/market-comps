import os
import sys

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from market_comps.db.session import SessionLocal
from market_comps.db.models import Organization, Market, MarketSegment
from market_comps.crm.company_manager import create_company, find_existing_company
from market_comps.crm.competitor_manager import (
    create_market, get_all_markets, create_market_segment, add_company_to_segment,
    get_or_create_competitive_analysis, add_competitive_analysis_company
)

def seed_data():
    db = SessionLocal()
    try:
        print("Creating Market...")
        # Check if exists
        market = db.query(Market).filter_by(name="Utility Inspection").first()
        if not market:
            market = create_market(
                db, 
                "Utility Inspection", 
                "Technologies used to inspect and monitor utility infrastructure"
            )
            db.commit()
            print("Market 'Utility Inspection' created.")
        else:
            print("Market already exists.")
            
        print("Creating Market Segments...")
        # Dictionary of segments to create
        segments_data = [
            ("Street-Level Imaging Startups", "Companies using vehicle-mounted or road-level imagery to inspect infrastructure", "Technology", 10),
            ("Drone Inspection Companies", "Companies using drones to capture utility asset data", "Technology", 20),
            ("Fixed Sensing Companies", "Companies deploying persistent sensors on infrastructure", "Technology", 30),
            ("Satellite Inspection Companies", "Companies using satellite data to inspect assets, vegetation or environmental risks", "Technology", 40),
            ("Grid Modeling Platforms", "Companies creating digital or physics-based models of utility networks", "Technology", 50),
            ("Incumbent Inspection Providers", "Large established inspection or engineering providers", "Competitor class", 60)
        ]
        
        from market_comps.db.models import MarketSegment
        segment_map = {}
        for name, desc, seg_type, sort in segments_data:
            seg = db.query(MarketSegment).filter_by(market_id=market.id, name=name).first()
            if not seg:
                seg = create_market_segment(db, market.id, name, desc, seg_type)
                seg.sort_order = sort
            segment_map[name] = seg
        db.commit()

        print("Creating/Fetching Companies...")
        companies_data = [
            ("Noteworthy AI", "noteworthy.ai", "Collects inspection imagery through cameras mounted on utility fleet vehicles during normal driving"),
            ("Buzz Solutions", "buzzsolutions.co", "Analyzes drone imagery using computer vision for utility asset inspection"),
            ("LineVision", "linevisioninc.com", "Provides continuous monitoring of transmission lines using non-contact sensors"),
            ("Neara", "neara.com", "Builds engineering-grade digital network models using physics-based simulation"),
            ("AiDash", "aidash.com", "Uses satellite data and AI for vegetation management and infrastructure monitoring"),
            ("Traditional engineering firms", None, "Established engineering, field-service or inspection providers")
        ]
        
        company_map = {}
        for name, domain, _ in companies_data:
            org = find_existing_company(db, name, domain)
            if not org:
                org = create_company(db, name=name, domain=domain, description=None)
            company_map[name] = org
        db.commit()
        
        print("Linking Companies to Segments...")
        # Noteworthy AI
        add_company_to_segment(db, company_map["Noteworthy AI"].id, segment_map["Street-Level Imaging Startups"].id, "Collects inspection imagery through cameras mounted on utility fleet vehicles during normal driving", is_primary=True)
        # Buzz
        add_company_to_segment(db, company_map["Buzz Solutions"].id, segment_map["Drone Inspection Companies"].id, "Analyzes drone imagery using computer vision for utility asset inspection", is_primary=True)
        # LineVision
        add_company_to_segment(db, company_map["LineVision"].id, segment_map["Fixed Sensing Companies"].id, "Provides continuous monitoring of transmission lines using non-contact sensors", is_primary=True)
        # Neara
        add_company_to_segment(db, company_map["Neara"].id, segment_map["Grid Modeling Platforms"].id, "Builds engineering-grade digital network models using physics-based simulation", is_primary=True)
        # AiDash
        add_company_to_segment(db, company_map["AiDash"].id, segment_map["Satellite Inspection Companies"].id, "Uses satellite data and AI for vegetation management and infrastructure monitoring", is_primary=True)
        # Traditional
        add_company_to_segment(db, company_map["Traditional engineering firms"].id, segment_map["Incumbent Inspection Providers"].id, "Established engineering, field-service or inspection providers", is_primary=True)
        
        db.commit()

        print("Creating Competitive Analysis for Noteworthy AI...")
        ca = get_or_create_competitive_analysis(
            db, 
            company_map["Noteworthy AI"].id, 
            market.id, 
            "Noteworthy AI Competitive Landscape"
        )
        ca.status = "published"
        db.commit()
        
        # Add Competitors to Analysis
        add_competitive_analysis_company(db, ca.id, company_map["Buzz Solutions"].id, "direct_competitor", "High", "Similar AI visual-inspection capability but relies more heavily on drones")
        add_competitive_analysis_company(db, ca.id, company_map["Neara"].id, "indirect_competitor", "Medium", "Different core product but overlaps in utility asset intelligence and capital budgets")
        add_competitive_analysis_company(db, ca.id, company_map["LineVision"].id, "substitute", "Medium/Low", "Provides more continuous data but requires dedicated hardware installation")
        add_competitive_analysis_company(db, ca.id, company_map["AiDash"].id, "adjacent", "Low", "Focuses on satellite-scale vegetation and infrastructure analysis")
        add_competitive_analysis_company(db, ca.id, company_map["Traditional engineering firms"].id, "incumbent", "Medium", "Entrenched relationships and services capabilities but less scalable data collection")

        db.commit()
        print("Done!")

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
