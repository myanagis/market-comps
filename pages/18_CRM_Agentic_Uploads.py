import streamlit as st
import pandas as pd
import json
import threading
from typing import List, Dict, Any

from market_comps.db.session import get_db_context, SessionLocal
from market_comps.db.models import Organization, Event, Source, EventOrganizationLink, OrganizationSourceLink
from market_comps.crm.company_manager import find_existing_company, create_company, process_new_company
from market_comps.ingestion.uploader_agent import UploaderChatAgent
from market_comps.ingestion.company_augmentation import run_manual_url_augmentation
from market_comps.ingestion.event_ingestion import EventScraperAgent
from datetime import datetime

st.set_page_config(page_title="Agentic Uploads", page_icon="📤", layout="wide")

col_main, col_sidebar = st.columns([3, 1])

with col_sidebar:
    st.markdown("### 🤖 Agentic Capabilities")
    st.info("""
    I can help you with:
    - **Create Companies in CRM**: Paste a list of text/companies to bulk-create them.
    - **Process Web Links**: Paste a URL (News, PRs, Articles) to ingest it.
    - **File Data to Entities**: Tell me if a link belongs to a Company, Investor, or Market Map, and I will extract facts and file them automatically.
    """)

with col_main:
    st.title("📤 Agentic Uploads")
    
    st.markdown("""
    Paste a list of companies or a web link below. The AI will extract data, ask for any missing info, and automatically update the CRM.
    """)

# Initialize Session State
if "uploader_messages" not in st.session_state:
    st.session_state.uploader_messages = [
        {"role": "assistant", "content": "Hi! Paste a list of companies or a block of text, and I'll extract them for you."}
    ]
if "pending_companies" not in st.session_state:
    st.session_state.pending_companies = []

# Display Chat History
for msg in st.session_state.uploader_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("details"):
            with st.expander("Show Execution Details", expanded=False):
                for detail in msg["details"]:
                    st.write(detail)

# Display Pending Companies (Staging Area)
if st.session_state.pending_companies:
    st.divider()
    st.subheader("📋 Pending Companies")
    df = pd.DataFrame(st.session_state.pending_companies)
    st.dataframe(df, use_container_width=True)
    
    augmentation_mode = st.radio(
        "AI Web Augmentation on new companies", 
        options=["None", "Fast (Homepage Only)", "Full (Deep Research)"], 
        horizontal=True, 
        index=1,
        key="augmentation_mode_radio"
    )
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Create & Run Pipeline", type="primary", use_container_width=True):
            # Trigger Proceed manually if button clicked
            st.session_state.manual_proceed = True
    with col2:
        if st.button("🗑️ Clear List", use_container_width=True):
            st.session_state.pending_companies = []
            st.rerun()
else:
    augmentation_mode = st.session_state.get("augmentation_mode_radio", "Fast (Homepage Only)")

# Handle Chat Input
prompt = st.chat_input("E.g., 'Add Acme Corp...' or 'Process https://...'")

if prompt or st.session_state.get("manual_proceed", False):
    
    if st.session_state.get("manual_proceed", False):
        prompt = "Proceed with creating the companies."
        st.session_state.manual_proceed = False
        
    st.session_state.uploader_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            agent = UploaderChatAgent()
            validation_rules = {
                "required_fields": [],
                "extract_parameters": [
                    "founders",
                    "founded_year",
                    "check_size_min",
                    "check_size_max",
                    "stages",
                    "specialties",
                    "themes"
                ]
            }
            
            with get_db_context() as db:
                existing_sources = [s.name for s in db.query(Source.name).distinct().all() if s.name]
                
            action_data, reply_msg = agent.process_message(
                user_message=prompt,
                pending_companies=st.session_state.pending_companies,
                chat_history=st.session_state.uploader_messages[:-1],
                validation_rules=validation_rules,
                existing_sources=existing_sources
            )
            
            action_type = action_data.get("action")
            
            # --- Domain Resolution Interceptor ---
            needs_clarification = []
            if action_type in ["extract", "update_market_map"]:
                if action_type == "extract":
                    comps = action_data.get("extracted_companies", [])
                else:
                    comps = action_data.get("market_map_update", {}).get("companies", [])
                
                from market_comps.ingestion.company_augmentation import fetch_exa_results
                from urllib.parse import urlparse
                
                for comp in comps:
                    # check if string (due to older schema versions) or dict
                    if isinstance(comp, str):
                        # Should not happen with updated schema, but just in case
                        continue
                        
                    name = comp.get("name")
                    if name and not comp.get("domain"):
                        st.write(f"🔍 Automatically searching for {name}'s website...")
                        try:
                            docs = fetch_exa_results([f"{name} official website"], company_name=name)
                            if docs and docs[0].get("url"):
                                url = docs[0]["url"]
                                domain = urlparse(url).netloc.replace("www.", "")
                                if domain:
                                    comp["domain"] = domain
                                    st.write(f"✅ Found {domain}")
                                    continue
                        except Exception:
                            pass
                        needs_clarification.append(name)
            
            if needs_clarification:
                action_type = "clarify"
                action_data["action"] = "clarify"
                reply_msg = f"I couldn't automatically find the websites for the following companies: {', '.join(needs_clarification)}. Could you please provide their domains? (e.g. 'Lennar is lennar.com')"
            # --- End Interceptor ---
            
            st.markdown(reply_msg)
            st.session_state.uploader_messages.append({"role": "assistant", "content": reply_msg})
            
            if action_type == "extract":
                extracted = action_data.get("extracted_companies", [])
                
                with get_db_context() as db:
                    for comp in extracted:
                        name = comp.get("name")
                        domain = comp.get("domain")
                        desc = comp.get("description")
                        ticker = comp.get("ticker_symbol")
                        exchange = comp.get("stock_exchange")
                        ownership = comp.get("ownership_type")
                        org_type = comp.get("organization_type") or "COMPANY"
                        parameters = comp.get("parameters", {})
                        
                        if not name:
                            continue
                            
                        # Check if exists
                        existing = find_existing_company(db, name, domain)
                        if existing:
                            st.warning(f"Company '{name}' already exists in the CRM. Skipping.")
                            continue
                            
                        # Check if already in pending list
                        already_pending = False
                        for p in st.session_state.pending_companies:
                            if p["name"].lower() == name.lower():
                                already_pending = True
                                # Update domain if newly provided
                                if not p.get("domain") and domain:
                                    p["domain"] = domain
                                break
                                
                        if not already_pending:
                            st.session_state.pending_companies.append({
                                "name": name,
                                "domain": domain,
                                "description": desc,
                                "ticker_symbol": ticker,
                                "stock_exchange": exchange,
                                "ownership_type": ownership,
                                "organization_type": org_type,
                                "parameters": parameters,
                                "canonical_source_name": comp.get("canonical_source_name")
                            })
                
                st.rerun()
                
            elif action_type == "proceed":
                if not st.session_state.pending_companies:
                    st.info("No companies in the pending list to create.")
                else:
                    st.write("### Creating Companies...")
                    
                    created_orgs = []
                    failed_orgs = []
                    
                    with get_db_context() as db:
                        for comp in st.session_state.pending_companies:
                            with st.status(f"Processing {comp['name']}...", expanded=True) as status:
                                try:
                                    st.write("Creating database record...")
                                    org = create_company(
                                        db=db,
                                        name=comp["name"],
                                        domain=comp["domain"],
                                        description=comp["description"],
                                        ticker_symbol=comp.get("ticker_symbol"),
                                        stock_exchange=comp.get("stock_exchange"),
                                        ownership_type=comp.get("ownership_type"),
                                        organization_type=comp.get("organization_type", "COMPANY"),
                                        parameters=comp.get("parameters", {})
                                    )
                                    # Must commit here so the augmentation pipeline (which uses a new session) can see the org!
                                    db.commit()
                                    created_orgs.append(comp["name"])
                                    
                                    # Link canonical source if provided
                                    source_name = comp.get("canonical_source_name")
                                    if source_name:
                                        from sqlalchemy import func
                                        source = db.query(Source).filter(func.lower(Source.name) == source_name.lower()).first()
                                        if not source:
                                            source = Source(name=source_name, source_type="webpage")
                                            db.add(source)
                                            db.flush()
                                        
                                        link = OrganizationSourceLink(organization_id=org.id, source_id=source.id)
                                        db.add(link)
                                        db.commit()
                                    
                                    if augmentation_mode != "None":
                                        is_fast = "Fast" in augmentation_mode
                                        st.write(f"Queuing AI Web Augmentation Pipeline ({'Fast' if is_fast else 'Full'}) in background...")
                                        
                                        def run_augmentation(org_id, fast):
                                            local_db = SessionLocal()
                                            try:
                                                process_new_company(local_db, org_id, fast_mode=fast)
                                            except Exception as e:
                                                import logging
                                                logging.error(f"Augmentation failed for org {org_id}: {e}")
                                            finally:
                                                local_db.close()
                                                
                                        threading.Thread(target=run_augmentation, args=(org.id, is_fast), daemon=True).start()
                                        status.update(label=f"Successfully queued {comp['name']}!", state="complete", expanded=False)
                                    else:
                                        status.update(label=f"Successfully created {comp['name']}!", state="complete", expanded=False)
                                except Exception as e:
                                    db.rollback()
                                    error_msg = f"Failed processing {comp['name']}: {str(e)}"
                                    status.update(label=error_msg, state="error", expanded=True)
                                    failed_orgs.append(error_msg)
                                    
                    st.session_state.pending_companies = []
                    
                    if created_orgs:
                        st.session_state.uploader_messages.append({"role": "assistant", "content": f"✅ Successfully created {len(created_orgs)} organization(s): {', '.join(created_orgs)}"})
                    if failed_orgs:
                        st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Failed to create {len(failed_orgs)} organization(s):\n" + "\n".join(failed_orgs)})
                        
                    st.rerun()
                    
            elif action_type == "update_market_map":
                map_update = action_data.get("market_map_update", {})
                market_name = map_update.get("market_name")
                segment_name = map_update.get("segment_name") or market_name
                companies = map_update.get("companies", [])
                notes = map_update.get("notes")
                
                if not market_name or not companies:
                    st.error("Missing market name or companies to add.")
                else:
                    with st.status(f"Updating market map '{market_name}'...", expanded=True) as status:
                        try:
                            with get_db_context() as db:
                                from market_comps.db.models import Market, ComparisonSet, ComparisonSetOrganizationLink, MarketComparisonSetLink
                                
                                # Look for market
                                market = db.query(Market).filter(Market.name.ilike(f"%{market_name}%")).first()
                                if not market:
                                    st.write(f"Could not find Market named '{market_name}'. Please ensure it exists.")
                                    status.update(label="Market not found.", state="error")
                                else:
                                    details = []
                                    def log_detail(msg):
                                        st.write(msg)
                                        details.append(msg)
                                        
                                    log_detail(f"Found Market: {market.name}")
                                    
                                    segment_type = map_update.get("segment_type", "competitors")
                                    
                                    if segment_type == "competitors":
                                        # Handle Competitors via MarketSegment
                                        seg = db.query(MarketSegment).filter(
                                            MarketSegment.market_id == market.id,
                                            MarketSegment.name.ilike(f"%{segment_name}%")
                                        ).first()
                                        
                                        if not seg:
                                            log_detail(f"Creating new Market Segment '{segment_name}' in Market '{market.name}'...")
                                            from market_comps.crm.competitor_manager import create_market_segment
                                            seg = create_market_segment(db, market.id, segment_name, notes)
                                            db.commit()
                                            
                                        comp_names_str = ", ".join([c.get("name", "") for c in companies if isinstance(c, dict) and c.get("name")] + [c for c in companies if isinstance(c, str)])
                                        log_detail(f"Adding competitors to segment: {comp_names_str}")
                                        
                                        added_count = 0
                                        for comp_obj in companies:
                                            if isinstance(comp_obj, str):
                                                comp_name = comp_obj
                                                comp_domain = None
                                                comp_ticker = None
                                                comp_exchange = None
                                                comp_ownership = None
                                            else:
                                                comp_name = comp_obj.get("name")
                                                comp_domain = comp_obj.get("domain")
                                                comp_ticker = comp_obj.get("ticker_symbol")
                                                comp_exchange = comp_obj.get("stock_exchange")
                                                comp_ownership = comp_obj.get("ownership_type")
                                                
                                            if not comp_name:
                                                continue
                                                
                                            org = db.query(Organization).filter(
                                                (Organization.name.ilike(f"%{comp_name}%")) | 
                                                (Organization.ticker.ilike(f"{comp_name}"))
                                            ).first()
                                            
                                            if not org:
                                                log_detail(f"Company '{comp_name}' not found. Creating it...")
                                                org = create_company(
                                                    db=db, 
                                                    name=comp_name, 
                                                    domain=comp_domain, 
                                                    ticker_symbol=comp_ticker,
                                                    stock_exchange=comp_exchange,
                                                    ownership_type=comp_ownership,
                                                    created_by="AgenticUploads"
                                                )
                                                db.commit()
                                                
                                                # Queue fast augmentation
                                                if augmentation_mode != "None":
                                                    def run_augmentation(org_id):
                                                        local_db = SessionLocal()
                                                        try:
                                                            process_new_company(local_db, org_id, fast_mode=True)
                                                        except Exception as e:
                                                            logging.error(f"Augmentation failed for org {org_id}: {e}")
                                                        finally:
                                                            local_db.close()
                                                    threading.Thread(target=run_augmentation, args=(org.id,), daemon=True).start()
                                                    
                                            from market_comps.crm.competitor_manager import add_company_to_segment
                                            add_company_to_segment(db, org.id, seg.id, notes, False)
                                            added_count += 1
                                            
                                        db.commit()
                                        msg = f"✅ Added {added_count} competitors to segment '{segment_name}' in Market '{market.name}'."
                                    else:
                                        # Handle other types via ComparisonSet
                                        display_set_type = "Public Comps" if segment_type == "public_comps" else ("Investors" if segment_type == "investors" else "Other")
                                        cset = db.query(ComparisonSet).join(MarketComparisonSetLink).filter(
                                            MarketComparisonSetLink.market_id == market.id,
                                            ComparisonSet.name.ilike(f"%{segment_name}%")
                                        ).first()
                                        
                                        if not cset:
                                            log_detail(f"Creating new ComparisonSet '{segment_name}' ({display_set_type}) in Market '{market.name}'...")
                                            cset = ComparisonSet(name=segment_name, set_type=display_set_type, description=notes)
                                            db.add(cset)
                                            db.flush()
                                            db.add(MarketComparisonSetLink(market_id=market.id, comparison_set_id=cset.id))
                                            db.commit()
                                            
                                        comp_names_str = ", ".join([c.get("name", "") for c in companies if isinstance(c, dict) and c.get("name")] + [c for c in companies if isinstance(c, str)])
                                        log_detail(f"Adding companies to comparison set: {comp_names_str}")
                                        
                                        added_count = 0
                                        for comp_obj in companies:
                                            if isinstance(comp_obj, str):
                                                comp_name = comp_obj
                                                comp_domain = None
                                                comp_ticker = None
                                                comp_exchange = None
                                                comp_ownership = None
                                            else:
                                                comp_name = comp_obj.get("name")
                                                comp_domain = comp_obj.get("domain")
                                                comp_ticker = comp_obj.get("ticker_symbol")
                                                comp_exchange = comp_obj.get("stock_exchange")
                                                comp_ownership = comp_obj.get("ownership_type")
                                                
                                            if not comp_name:
                                                continue
                                                
                                            org = db.query(Organization).filter(
                                                (Organization.name.ilike(f"%{comp_name}%")) | 
                                                (Organization.ticker.ilike(f"{comp_name}"))
                                            ).first()
                                            
                                            if not org:
                                                log_detail(f"Company '{comp_name}' not found. Creating it...")
                                                org = create_company(
                                                    db=db, 
                                                    name=comp_name, 
                                                    domain=comp_domain, 
                                                    ticker_symbol=comp_ticker,
                                                    stock_exchange=comp_exchange,
                                                    ownership_type=comp_ownership,
                                                    created_by="AgenticUploads"
                                                )
                                                db.commit()
                                                
                                                # Queue fast augmentation
                                                if augmentation_mode != "None":
                                                    def run_augmentation(org_id):
                                                        local_db = SessionLocal()
                                                        try:
                                                            process_new_company(local_db, org_id, fast_mode=True)
                                                        except Exception as e:
                                                            logging.error(f"Augmentation failed for org {org_id}: {e}")
                                                        finally:
                                                            local_db.close()
                                                    threading.Thread(target=run_augmentation, args=(org.id,), daemon=True).start()
                                            
                                            # Guardrail check
                                            if segment_type == "public_comps" and (org.ownership_type and org.ownership_type.upper() == "PRIVATE"):
                                                log_detail(f"⚠️ Guardrail: Skipping {org.name} because it is a PRIVATE company and cannot be added to a Public Comps bucket.")
                                                continue
                                                
                                            # Link to Comparison Set
                                            existing_link = db.query(ComparisonSetOrganizationLink).filter_by(
                                                comparison_set_id=cset.id, organization_id=org.id
                                            ).first()
                                            if not existing_link:
                                                db.add(ComparisonSetOrganizationLink(
                                                    comparison_set_id=cset.id, 
                                                    organization_id=org.id,
                                                    notes=notes
                                                ))
                                            added_count += 1
                                            
                                        db.commit()
                                        msg = f"✅ Added {added_count} companies to '{segment_name}' in Market '{market.name}'."
                                    
                                    # Append the new message but also attach the details list
                                    new_msg = {"role": "assistant", "content": msg, "details": details}
                                    st.session_state.uploader_messages.append(new_msg)
                                    status.update(label=msg, state="complete")
                        except Exception as e:
                            status.update(label=f"Failed to update market map: {str(e)}", state="error", expanded=True)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
                    st.rerun()

            elif action_type == "add_transaction":
                transactions = action_data.get("transactions", [])
                
                if not transactions:
                    st.error("No transactions found in request.")
                else:
                    with st.status(f"Recording {len(transactions)} M&A Transaction(s)...", expanded=True) as status:
                        try:
                            with get_db_context() as db:
                                from market_comps.db.models import Transaction, Market, ComparisonSet, MarketComparisonSetLink, ComparisonSetOrganizationLink
                                
                                details = []
                                def log_detail(msg):
                                    st.write(msg)
                                    details.append(msg)
                                
                                def get_or_create(db, name):
                                    org = db.query(Organization).filter(Organization.name.ilike(f"%{name}%")).first()
                                    if not org:
                                        log_detail(f"Creating missing company '{name}'...")
                                        org = create_company(db=db, name=name, created_by="AgenticUploads")
                                        db.flush()
                                    return org
                                
                                recorded_count = 0
                                for tx_details in transactions:
                                    acq_name = tx_details.get("acquirer")
                                    tgt_name = tx_details.get("target")
                                    price = tx_details.get("price")
                                    notes = tx_details.get("notes")
                                    year = tx_details.get("year")
                                    
                                    if not acq_name or not tgt_name:
                                        continue
                                        
                                    acquirer = get_or_create(db, acq_name)
                                    target = get_or_create(db, tgt_name)
                                    
                                    tx = Transaction(
                                        transaction_name=f"{acquirer.name} acquires {target.name}",
                                        transaction_type="ACQUISITION",
                                        status="ANNOUNCED",
                                        acquirer_company_id=acquirer.id,
                                        target_company_id=target.id,
                                        transaction_value_numeric=price,
                                        notes=notes
                                    )
                                    
                                    if year:
                                        from datetime import date
                                        tx.announced_date = date(year, 1, 1)
                                    db.add(tx)
                                    db.flush()
                                    
                                    if tx_details.get("market_name"):
                                        market_name = tx_details.get("market_name")
                                        segment_name = tx_details.get("segment_name") or "M&A Precedents"
                                        market = db.query(Market).filter(Market.name.ilike(f"%{market_name}%")).first()
                                        if market:
                                            cset = db.query(ComparisonSet).join(MarketComparisonSetLink).filter(
                                                MarketComparisonSetLink.market_id == market.id,
                                                ComparisonSet.name.ilike(f"%{segment_name}%")
                                            ).first()
                                            if not cset:
                                                log_detail(f"Creating M&A Precedents set '{segment_name}' in Market '{market.name}'...")
                                                cset = ComparisonSet(name=segment_name, set_type="M&A Precedents")
                                                db.add(cset)
                                                db.flush()
                                                db.add(MarketComparisonSetLink(market_id=market.id, comparison_set_id=cset.id))
                                                
                                            existing_link = db.query(ComparisonSetOrganizationLink).filter_by(
                                                comparison_set_id=cset.id, organization_id=target.id
                                            ).first()
                                            if not existing_link:
                                                db.add(ComparisonSetOrganizationLink(
                                                    comparison_set_id=cset.id, organization_id=target.id, notes=""
                                                ))
                                            log_detail(f"Added {target.name} to {segment_name} in {market.name}.")
                                    recorded_count += 1
                                    
                                db.commit()
                                
                                msg = f"✅ Recorded {recorded_count} M&A Transaction(s)."
                                new_msg = {"role": "assistant", "content": msg, "details": details}
                                st.session_state.uploader_messages.append(new_msg)
                                status.update(label=msg, state="complete")
                        except Exception as e:
                            status.update(label=f"Failed to record transaction: {str(e)}", state="error", expanded=True)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
                    st.rerun()
                    
            elif action_type == "add_financing":
                financings = action_data.get("financings", [])
                
                if not financings:
                    st.error("No financings found in request.")
                else:
                    with st.status(f"Recording {len(financings)} Financing(s)...", expanded=True) as status:
                        try:
                            with get_db_context() as db:
                                from market_comps.db.models import FinancingRound, RoundInvestor, FinancingRoundFact, Market, ComparisonSet, MarketComparisonSetLink, ComparisonSetOrganizationLink
                                
                                details = []
                                def log_detail(msg):
                                    st.write(msg)
                                    details.append(msg)
                                
                                def get_or_create(db, name):
                                    org = db.query(Organization).filter(Organization.name.ilike(f"%{name}%")).first()
                                    if not org:
                                        log_detail(f"Creating missing company '{name}'...")
                                        org = create_company(db=db, name=name, created_by="AgenticUploads")
                                        db.flush()
                                    return org
                                    
                                recorded_count = 0
                                for fin_details in financings:
                                    company_name = fin_details.get("company_name")
                                    round_name = fin_details.get("round_name")
                                    amount = fin_details.get("amount")
                                    market_name = fin_details.get("market_name")
                                    segment_name = fin_details.get("segment_name") or "Financing Comps"
                                    
                                    if not company_name or not round_name:
                                        continue
                                        
                                    comp = get_or_create(db, company_name)
                                    
                                    fin = FinancingRound(
                                        company_id=comp.id,
                                        round_name=round_name,
                                        status="announced"
                                    )
                                    db.add(fin)
                                    db.flush()
                                    
                                    if amount:
                                        fact = FinancingRoundFact(
                                            financing_round_id=fin.id,
                                            fact_type="amount_raised",
                                            value_numeric=amount,
                                            certainty="announced"
                                        )
                                        db.add(fact)
                                    
                                    lead_investors = fin_details.get("lead_investors", [])
                                    for inv_name in lead_investors:
                                        inv_org = get_or_create(db, inv_name)
                                        if not inv_org.organization_type:
                                            inv_org.organization_type = "INVESTOR"
                                        inv_link = RoundInvestor(
                                            financing_round_id=fin.id,
                                            investor_id=inv_org.id,
                                            role="lead"
                                        )
                                        db.add(inv_link)
                                    
                                    if market_name:
                                        market = db.query(Market).filter(Market.name.ilike(f"%{market_name}%")).first()
                                        if market:
                                            cset = db.query(ComparisonSet).join(MarketComparisonSetLink).filter(
                                                MarketComparisonSetLink.market_id == market.id,
                                                ComparisonSet.name.ilike(f"%{segment_name}%")
                                            ).first()
                                            if not cset:
                                                log_detail(f"Creating Financing Comps set '{segment_name}' in Market '{market.name}'...")
                                                cset = ComparisonSet(name=segment_name, set_type="Financing Comps")
                                                db.add(cset)
                                                db.flush()
                                                db.add(MarketComparisonSetLink(market_id=market.id, comparison_set_id=cset.id))
                                                
                                            existing_link = db.query(ComparisonSetOrganizationLink).filter_by(
                                                comparison_set_id=cset.id, organization_id=comp.id
                                            ).first()
                                            if not existing_link:
                                                db.add(ComparisonSetOrganizationLink(
                                                    comparison_set_id=cset.id, organization_id=comp.id, notes=""
                                                ))
                                            log_detail(f"Added {comp.name} to {segment_name} in {market.name}.")
                                    recorded_count += 1
                                    
                                db.commit()
                                msg = f"✅ Recorded {recorded_count} Financing(s)."
                                new_msg = {"role": "assistant", "content": msg, "details": details}
                                st.session_state.uploader_messages.append(new_msg)
                                status.update(label=msg, state="complete")
                        except Exception as e:
                            status.update(label=f"Failed to record financing: {str(e)}", state="error", expanded=True)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
                    st.rerun()
                url = action_data.get("url")
                target_name = action_data.get("target_entity_name")
                target_type = action_data.get("target_entity_type")
                
                if not url or not target_name:
                    st.error("Missing URL or target entity name from agent.")
                else:
                    with st.status(f"Processing link for {target_name}...", expanded=True) as status:
                        try:
                            with get_db_context() as db:
                                details = []
                                def log_detail(msg):
                                    st.write(msg)
                                    details.append(msg)
                                
                                # For now, we only support mapping to Organizations (Company/Investor)
                                log_detail(f"Looking up {target_type}: {target_name}...")
                                org = db.query(Organization).filter(Organization.name.ilike(f"%{target_name}%")).first()
                                
                                if not org:
                                    log_detail(f"Could not find {target_type} named '{target_name}'. Creating it now...")
                                    org = create_company(
                                        db=db,
                                        name=target_name,
                                        created_by="AgenticUploads"
                                    )
                                    if target_type and target_type.lower() == "investor":
                                        org.organization_type = "INVESTOR"
                                    db.commit()
                                    
                                log_detail(f"Found/Created {org.name}. Scraping and running extraction pipeline...")
                                run_manual_url_augmentation(org.id, url)
                                status.update(label=f"Successfully extracted data from link and updated {org.name}!", state="complete", expanded=False)
                                new_msg = {"role": "assistant", "content": f"✅ Successfully extracted data from the link and filed it under **{org.name}**.", "details": details}
                                st.session_state.uploader_messages.append(new_msg)
                        except Exception as e:
                            status.update(label=f"Failed to process link: {str(e)}", state="error", expanded=True)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Failed to process link: {str(e)}"})
                    st.rerun()

            elif action_type == "process_event_link":
                url = action_data.get("url")
                if not url:
                    st.error("Missing URL for event processing.")
                else:
                    with st.status(f"Processing event from {url}...", expanded=True) as status:
                        try:
                            with get_db_context() as db:
                                # Deduplicate Source
                                source = db.query(Source).filter(Source.url == url).first()
                                if source:
                                    st.write(f"Using existing canonical source: {source.name}")
                                    event = source.event
                                else:
                                    st.write("Scraping new event page with AI...")
                                    scraper = EventScraperAgent()
                                    event_data = scraper.process_event_url(url)
                                    
                                    st.write(f"Creating Event: {event_data.get('event_name')}...")
                                    def parse_date(ds):
                                        if not ds: return None
                                        try: return datetime.fromisoformat(ds.replace("Z", "+00:00"))
                                        except: return None
                                    
                                    event = Event(
                                        name=event_data.get('event_name'),
                                        start_at=parse_date(event_data.get('start_at')),
                                        end_at=parse_date(event_data.get('end_at')),
                                        location=event_data.get('location'),
                                        event_type=event_data.get('event_type'),
                                        url=url,
                                        status="discovered"
                                    )
                                    db.add(event)
                                    db.flush()
                                    
                                    source = Source(
                                        source_type="webpage",
                                        url=url,
                                        name=f"Event Page: {event.name}",
                                        occurred_at=datetime.utcnow(),
                                        event_id=event.id
                                    )
                                    db.add(source)
                                    db.flush()
                                
                                st.write("Processing Companies...")
                                # Only process if we just scraped it or if we want to run through existing?
                                # If the source existed, we don't have event_data. So we skip scraping and extracting orgs.
                                if not source.event:
                                    # Fallback in case of some weird state
                                    pass
                                elif 'event_data' in locals():
                                    for comp in event_data.get("companies", []):
                                        name = comp.get("name")
                                        domain = comp.get("domain")
                                        role = comp.get("role", "attendee")
                                        if not name: continue
                                        
                                        org = find_existing_company(db, name, domain)
                                        is_new = False
                                        if not org:
                                            org = create_company(
                                                db=db,
                                                name=name,
                                                domain=domain,
                                                description=comp.get("description"),
                                                organization_type="COMPANY"
                                            )
                                            is_new = True
                                            
                                        db.flush()
                                        # Add Link
                                        link = EventOrganizationLink(event_id=event.id, organization_id=org.id, role=role)
                                        db.merge(link)
                                        db.commit()
                                        
                                        if is_new and augmentation_mode != "None":
                                            is_fast = "Fast" in augmentation_mode
                                            threading.Thread(target=lambda o_id, fast: process_new_company(SessionLocal(), o_id, fast_mode=fast), args=(org.id, is_fast), daemon=True).start()

                                    st.write("Processing Investors...")
                                    for inv in event_data.get("investors", []):
                                        name = inv.get("name")
                                        domain = inv.get("domain")
                                        role = inv.get("role", "attendee")
                                        if not name: continue
                                        
                                        org = find_existing_company(db, name, domain)
                                        is_new = False
                                        if not org:
                                            org = create_company(
                                                db=db,
                                                name=name,
                                                domain=domain,
                                                description=inv.get("description"),
                                                organization_type="INVESTOR"
                                            )
                                            is_new = True
                                            
                                        db.flush()
                                        link = EventOrganizationLink(event_id=event.id, organization_id=org.id, role=role)
                                        db.merge(link)
                                        db.commit()
                                        
                                        if is_new and augmentation_mode != "None":
                                            is_fast = "Fast" in augmentation_mode
                                            threading.Thread(target=lambda o_id, fast: process_new_company(SessionLocal(), o_id, fast_mode=fast), args=(org.id, is_fast), daemon=True).start()
                                        
                            status.update(label=f"Successfully ingested event '{event.name}' and all participants!", state="complete", expanded=False)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"✅ Extracted event **{event.name}** and linked all parsed participants."})
                        except Exception as e:
                            status.update(label=f"Failed to process event: {str(e)}", state="error", expanded=True)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Failed to process event link: {str(e)}"})
                    st.rerun()
