import os
import re

file_path = r"c:\Users\micha\.gemini\antigravity\scratch\market-comps\market_comps\ingestion\company_augmentation.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
content = content.replace("Investment, ", "FinancingRound, FinancingRoundFact, RoundInvestor, ")
content = content.replace(", Investment", ", FinancingRound, FinancingRoundFact, RoundInvestor")
content = content.replace("from market_comps.db.models import Investment", "from market_comps.db.models import FinancingRound, FinancingRoundFact, RoundInvestor")

# 2. clear_augmentation_data
clear_code_old = """        # Delete investments created by WEB_AUGMENTATION for this company
        from market_comps.db.models import Investment
        from datetime import datetime
        from sqlalchemy import cast, Integer
        db.query(Investment).filter(
            Investment.company_organization_id == org_id,
            Investment.id.in_(db.query(cast(AuditTrail.canonical_entity_id, Integer)).filter_by(canonical_entity_type="INVESTMENT", source="WEB_AUGMENTATION"))
        ).update({Investment.deleted_at: datetime.utcnow(), Investment.deleted_by: 'USER'}, synchronize_session=False)"""

clear_code_new = """        # Delete financing rounds created by WEB_AUGMENTATION
        from market_comps.db.models import FinancingRound
        from datetime import datetime
        from sqlalchemy import cast, Integer
        # For simplicity, we just delete the rounds directly based on audit trail
        round_ids = db.query(cast(AuditTrail.canonical_entity_id, Integer)).filter_by(canonical_entity_type="FINANCING_ROUND", source="WEB_AUGMENTATION").all()
        round_ids = [r[0] for r in round_ids]
        if round_ids:
            db.query(FinancingRound).filter(FinancingRound.company_id == org_id, FinancingRound.id.in_(round_ids)).delete(synchronize_session=False)"""

content = content.replace(clear_code_old, clear_code_new)

content = content.replace("db.query(Investment).filter(Investment.source_document_id.in_(doc_ids)).update({Investment.deleted_at: datetime.utcnow(), Investment.deleted_by: 'USER'}, synchronize_session=False)", "")

# 3. Upsert blocks (Block 1)
block_1_old = """            investment = Investment(
                investor_organization_id=investor_org.id,
                company_organization_id=org.id,
                round_type=inv.get("round_type"),
                total_round_amount=inv.get("total_round_amount"),
                firm_investment_amount=inv.get("firm_investment_amount"),
                is_lead=inv.get("is_lead", False),
                source_document_id=source_doc_id
            )
            
            # Simple date parsing attempt
            date_str = inv.get("investment_date")
            if date_str:
                try:
                    investment.investment_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass
            
            db.add(investment)
            db.flush()
            
            db.add(AuditTrail(
                canonical_entity_type="INVESTMENT",
                canonical_entity_id=str(investment.id),
                mutation_type="CREATE",
                source="WEB_AUGMENTATION",
                created_by="SYSTEM"
            ))"""

block_1_new = """            rnd = db.query(FinancingRound).filter_by(company_id=org.id, round_name=inv.get("round_type")).first()
            if not rnd:
                rnd = FinancingRound(company_id=org.id, round_name=inv.get("round_type"), status="closed")
                db.add(rnd)
                db.flush()
                db.add(AuditTrail(canonical_entity_type="FINANCING_ROUND", canonical_entity_id=str(rnd.id), mutation_type="CREATE", source="WEB_AUGMENTATION", created_by="SYSTEM"))
                
                if inv.get("total_round_amount"):
                    fact = FinancingRoundFact(financing_round_id=rnd.id, fact_type="amount_raised", value_text=inv.get("total_round_amount"), certainty="reported", source_id=source_doc_id)
                    db.add(fact)

            date_str = inv.get("investment_date")
            inv_date = None
            if date_str:
                try:
                    inv_date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    pass

            rinv = RoundInvestor(
                financing_round_id=rnd.id,
                investor_id=investor_org.id,
                role="lead" if inv.get("is_lead") else "participant",
                status="invested",
                notes=f"Firm Amount: {inv.get('firm_investment_amount')}" if inv.get("firm_investment_amount") else None,
                source_id=source_doc_id,
                reported_at=inv_date
            )
            db.add(rinv)
            db.flush()"""

content = content.replace(block_1_old, block_1_new)

# 4. Upsert blocks (Block 2)
block_2_old = """            existing_inv = db.query(Investment).filter_by(investor_organization_id=investor_org.id, company_organization_id=org.id, round_type=r_type).first()
            
            if not existing_inv:
                investment = Investment(
                    investor_organization_id=investor_org.id,
                    company_organization_id=org.id,
                    round_type=r_type,
                    total_round_amount=inv.get("total_round_amount"),
                    firm_investment_amount=inv.get("firm_investment_amount"),
                    is_lead=inv.get("is_lead", False),
                    source_document_id=src_doc.id
                )
                date_str = inv.get("investment_date")
                if date_str:
                    try:
                        investment.investment_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except:
                        pass
                db.add(investment)"""

block_2_new = """            rnd = db.query(FinancingRound).filter_by(company_id=org.id, round_name=r_type).first()
            if not rnd:
                rnd = FinancingRound(company_id=org.id, round_name=r_type, status="closed")
                db.add(rnd)
                db.flush()
                
                if inv.get("total_round_amount"):
                    fact = FinancingRoundFact(financing_round_id=rnd.id, fact_type="amount_raised", value_text=inv.get("total_round_amount"), certainty="reported", source_id=src_doc.id)
                    db.add(fact)

            existing_rinv = db.query(RoundInvestor).filter_by(financing_round_id=rnd.id, investor_id=investor_org.id).first()
            if not existing_rinv:
                date_str = inv.get("investment_date")
                inv_date = None
                if date_str:
                    try:
                        inv_date = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        pass
                        
                rinv = RoundInvestor(
                    financing_round_id=rnd.id,
                    investor_id=investor_org.id,
                    role="lead" if inv.get("is_lead") else "participant",
                    status="invested",
                    notes=f"Firm Amount: {inv.get('firm_investment_amount')}" if inv.get("firm_investment_amount") else None,
                    source_id=src_doc.id,
                    reported_at=inv_date
                )
                db.add(rinv)
                db.flush()"""

content = content.replace(block_2_old, block_2_new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Augmentation updated")
