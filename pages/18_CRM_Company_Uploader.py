import streamlit as st
import pandas as pd
from typing import List, Dict, Any

from market_comps.db.session import get_db_context
from market_comps.crm.company_manager import find_existing_company, create_company, process_new_company
from market_comps.ingestion.uploader_agent import UploaderChatAgent

st.set_page_config(page_title="Company Uploader", page_icon="📤", layout="wide")
st.title("📤 Company Uploader (AI Agent)")

st.markdown("""
Paste a list of companies below. The AI will extract them, check if they exist in the CRM, ask for any missing websites, and then automatically create the records and run the web ingestion pipeline!
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
prompt = st.chat_input("E.g., 'Add Acme Corp (acme.com) and Globex...'")

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
            action_data, reply_msg = agent.process_message(
                user_message=prompt,
                pending_companies=st.session_state.pending_companies,
                chat_history=st.session_state.uploader_messages[:-1]
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
                                "description": desc
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
                                        description=comp["description"]
                                    )
                                    # Must commit here so the augmentation pipeline (which uses a new session) can see the org!
                                    db.commit()
                                    created_orgs.append(comp["name"])
                                    
                                    st.write("Running AI Web Augmentation Pipeline...")
                                    process_new_company(db, org.id)
                                    status.update(label=f"Successfully processed {comp['name']}!", state="complete", expanded=False)
                                except Exception as e:
                                    db.rollback()
                                    error_msg = f"Failed processing {comp['name']}: {str(e)}"
                                    status.update(label=error_msg, state="error", expanded=True)
                                    failed_orgs.append(error_msg)
                                    
                    st.session_state.pending_companies = []
                    
                    # Add system message to chat
                    completion_msg = f"**Successfully created and processed {len(created_orgs)} companies:**\n"
                    if created_orgs:
                        for name in created_orgs:
                            completion_msg += f"- ✅ {name}\n"
                    if failed_orgs:
                        completion_msg += "\n**Errors encountered:**\n"
                        for err in failed_orgs:
                            completion_msg += f"- ❌ {err}\n"
                            
                    st.session_state.uploader_messages.append({"role": "assistant", "content": completion_msg})
                    st.rerun()
