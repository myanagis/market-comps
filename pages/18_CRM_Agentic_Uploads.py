import streamlit as st
import pandas as pd
import json
import threading
from typing import List, Dict, Any

from market_comps.db.session import get_db_context, SessionLocal
from market_comps.db.models import Organization
from market_comps.crm.company_manager import find_existing_company, create_company, process_new_company
from market_comps.ingestion.uploader_agent import UploaderChatAgent
from market_comps.ingestion.company_augmentation import run_manual_url_augmentation

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

# Display Pending Companies (Staging Area)
if st.session_state.pending_companies:
    st.divider()
    st.subheader("📋 Pending Companies")
    df = pd.DataFrame(st.session_state.pending_companies)
    st.dataframe(df, use_container_width=True)
    
    do_augmentation = st.toggle("Run AI Web Augmentation on new companies", value=True)
    
    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        if st.button("✅ Create & Run Pipeline", type="primary", use_container_width=True):
            # Trigger Proceed manually if button clicked
            st.session_state.manual_proceed = True
    with col2:
        if st.button("🗑️ Clear List", use_container_width=True):
            st.session_state.pending_companies = []
            st.rerun()

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
                "required_fields": ["domain"],
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
            
            action_data, reply_msg = agent.process_message(
                user_message=prompt,
                pending_companies=st.session_state.pending_companies,
                chat_history=st.session_state.uploader_messages[:-1],
                validation_rules=validation_rules
            )
            
            st.markdown(reply_msg)
            st.session_state.uploader_messages.append({"role": "assistant", "content": reply_msg})
            
            action_type = action_data.get("action")
            
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
                                "parameters": parameters
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
                                    
                                    if do_augmentation:
                                        st.write("Queuing AI Web Augmentation Pipeline in background...")
                                        
                                        def run_augmentation(org_id):
                                            local_db = SessionLocal()
                                            try:
                                                process_new_company(local_db, org_id)
                                            except Exception as e:
                                                import logging
                                                logging.error(f"Augmentation failed for org {org_id}: {e}")
                                            finally:
                                                local_db.close()
                                                
                                        threading.Thread(target=run_augmentation, args=(org.id,), daemon=True).start()
                                        status.update(label=f"Successfully queued {comp['name']}!", state="complete", expanded=False)
                                    else:
                                        status.update(label=f"Successfully created {comp['name']}!", state="complete", expanded=False)
                                except Exception as e:
                                    db.rollback()
                                    error_msg = f"Failed processing {comp['name']}: {str(e)}"
                                    status.update(label=error_msg, state="error", expanded=True)
                                    failed_orgs.append(error_msg)
                                    
                    st.session_state.pending_companies = []
                    
            elif action_type == "process_link":
                url = action_data.get("url")
                target_name = action_data.get("target_entity_name")
                target_type = action_data.get("target_entity_type")
                
                if not url or not target_name:
                    st.error("Missing URL or target entity name from agent.")
                else:
                    with st.status(f"Processing link for {target_name}...", expanded=True) as status:
                        try:
                            with get_db_context() as db:
                                # For now, we only support mapping to Organizations (Company/Investor)
                                st.write(f"Looking up {target_type}: {target_name}...")
                                org = db.query(Organization).filter(Organization.name.ilike(f"%{target_name}%")).first()
                                
                                if not org:
                                    st.write(f"Could not find {target_type} named '{target_name}'. Creating it now...")
                                    org = create_company(
                                        db=db,
                                        name=target_name,
                                        created_by="AgenticUploads"
                                    )
                                    if target_type and target_type.lower() == "investor":
                                        org.organization_type = "INVESTOR"
                                    db.commit()
                                    
                                st.write(f"Found/Created {org.name}. Scraping and running extraction pipeline...")
                                run_manual_url_augmentation(org.id, url)
                                status.update(label=f"Successfully extracted data from link and updated {org.name}!", state="complete", expanded=False)
                                st.session_state.uploader_messages.append({"role": "assistant", "content": f"✅ Successfully extracted data from the link and filed it under **{org.name}**."})
                        except Exception as e:
                            status.update(label=f"Failed to process link: {str(e)}", state="error", expanded=True)
                            st.session_state.uploader_messages.append({"role": "assistant", "content": f"❌ Failed to process link: {str(e)}"})
                    st.rerun()
