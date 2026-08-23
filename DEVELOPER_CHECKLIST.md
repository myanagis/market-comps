# 📋 Market Comps Development & Release Checklist

Always verify this checklist whenever implementing features or schema changes in this repository:

1. **DB Models & Alembic Migrations**:
   - If adding or modifying SQLAlchemy models in `market_comps/db/models.py`, generate or create a corresponding migration script in `alembic/versions/`.
   - Ensure column types and defaults prevent table lock deadlocks during remote app reboots.
   - **ERD Schema Validation**: Ensure all new models inherit from `Base` so that they automatically reflect in the dynamic Entity-Relationship Diagram (ERD) on the Admin Database Schema page (`pages/02_Admin_Database_Schema.py`).

2. **`app.py` Page Registration & Startup Migration**:
   - Ensure any new page in `pages/` is explicitly registered in `app.py` under `st.navigation`.
   - Run `alembic upgrade head` on startup in `app.py` so remote deployments (Streamlit Cloud, Docker, Supabase) auto-migrate database schema on boot.

3. **Audit Trail & Data Provenance**:
   - Every data mutation (creating/editing competitors, market segments, financing rounds, team roles, or companies) MUST write an `AuditTrail` entry recording `who`, `what`, `when`, `mutation_type`, and `source`.
   - Ensure audit logs are queried and rendered on the company/record lookup detail pages.

4. **External API & Search Guardrails**:
   - Wrap external search/LLM API calls (Exa, OpenRouter, Yahoo Finance) in strict quality, relevance, and junk-text guardrails.
   - Retain errored or blocked documents with error status badges for 100% data auditability without polluting LLM extraction prompts.

5. **Git & Deployment Synchronization**:
   - Verify local execution build succeeds.
   - Commit code and migration scripts cleanly and push to GitHub (`git push`).
