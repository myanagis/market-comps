import io
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Literal, Optional

import polars as pl
import requests
import streamlit as st
from pydantic import BaseModel, Field

from market_comps.config import MODEL_OPTIONS, settings
from market_comps.llm_client import LLMClient
from market_comps.models import LLMUsage
from market_comps.connections.ct_registry import CTBusinessRegistryClient

logger = logging.getLogger(__name__)

st.set_page_config(page_title="CT Business Registry", page_icon="🏢", layout="wide")


# --- Schemas for LLM Output ---
class AugmentedCompany(BaseModel):
    id: str = Field(description="The exact id of the company from the input")
    name: str = Field(description="The exact name of the company from the input")
    description: Optional[str] = Field(default=None, description="A brief 1-2 sentence description of what the company does")
    website: Optional[str] = Field(default=None, description="Company website URL")
    founders: list[str] = Field(default_factory=list, description="Names of founders")
    recent_fundraising_series: Optional[str] = Field(default=None, description="e.g. Seed, Series A, Pre-seed")
    recent_fundraising_amount: Optional[str] = Field(default=None, description="Amount raised, e.g. $5M")
    recent_fundraising_date: Optional[str] = Field(default=None, description="Approximate date of last raise")
    employee_count: Optional[str] = Field(default=None, description="Approximate employee count or range")
    is_public: bool = Field(default=False, description="True if publicly traded")
    ticker: Optional[str] = Field(default=None, description="Stock ticker if public")
    is_venture_funded: bool = Field(default=False, description="True if the company has received VC or angel funding")
    is_venture_fundable: Literal["Yes", "No", "Maybe"] = Field(default="No", description="Yes/No/Maybe if the company fits a high-growth VC-backable profile")


class AugmentedBatch(BaseModel):
    companies: list[AugmentedCompany]


# --- Helper Functions ---
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_recent_ct_businesses(start: str, end: str, prefixes: list[str], name_filter: str) -> pl.DataFrame:
    """Fetch recently registered CT businesses based on NAICS code prefixes."""
    try:
        return CTBusinessRegistryClient.fetch_recent(start, end, prefixes, name_filter)
    except Exception as e:
        st.error(f"API Error: {e}")
        return pl.DataFrame()


def generate_excel_bytes(df: pl.DataFrame) -> bytes:
    """Convert Polars DataFrame to Excel bytes."""
    output = io.BytesIO()
    df.write_excel(workbook=output)
    return output.getvalue()


def augment_businesses(companies_df: pl.DataFrame, client: LLMClient, max_rows: int = 50) -> tuple[pl.DataFrame, LLMUsage]:
    """Use the LLM to augment business data."""
    df_to_augment = companies_df.head(max_rows) if len(companies_df) > max_rows else companies_df
    
    if "business_email_address" in df_to_augment.columns:
        df_to_augment = df_to_augment.with_columns(
            pl.col("business_email_address").str.split("@").list.last().alias("email_domain")
        )
        context_cols = ["id", "name", "billingcity", "naics_code", "email_domain"]
    else:
        context_cols = ["id", "name", "billingcity", "naics_code"]
    
    # We only need the names and basic context to give to the LLM
    context_list = df_to_augment.select(context_cols).to_dicts()
    
    batch_size = 10
    total_usage = LLMUsage()
    augmented_rows = []
    
    schema = AugmentedBatch.model_json_schema()
    
    progress_bar = st.progress(0, text="Starting LLM Augmentation...")
    
    for i in range(0, len(context_list), batch_size):
        chunk = context_list[i:i+batch_size]
        prompt = f"""
        Here is a list of companies recently registered in Connecticut. 
        For each company, please find or estimate the requested information based on your knowledge base.
        If you cannot find specific information (like recent fundraising), leave it null.
        IMPORTANT: You MUST return exactly one entry in the `companies` array for every company provided in the input list. Do not skip any companies!
        NOTE: The `email_domain` (if provided) is the official domain of the company's registration email. Use it to find their exact website and disambiguate from other companies with similar names.
        
        Companies:
        {json.dumps(chunk, indent=2)}
        
        JSON SCHEMA:
        {json.dumps(schema, indent=2)}
        """
        
        try:
            progress_bar.progress(
                (i) / len(context_list), 
                text=f"Augmenting batch {i//batch_size + 1}/{(len(context_list) + batch_size - 1)//batch_size}..."
            )
            
            content, usage = client.chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert VC/PE analyst. Find data about these private companies. Return valid JSON matching the provided schema."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            stripped = content.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("```")[1]
                if stripped.startswith("json"):
                    stripped = stripped[4:]
                stripped = stripped.rsplit("```", 1)[0].strip()
                
            result = json.loads(stripped)
            
            total_usage.total_prompt_tokens += usage.total_prompt_tokens
            total_usage.total_completion_tokens += usage.total_completion_tokens
            total_usage.total_tokens += usage.total_tokens
            total_usage.estimated_cost_usd += usage.estimated_cost_usd
            total_usage.call_count += 1
            
            # Extract items
            if "companies" in result:
                augmented_rows.extend(result["companies"])
            
        except Exception as e:
            st.warning(f"Error augmenting batch {i//batch_size + 1}: {e}")
            # Insert dummy empty entries for failed rows to maintain alignment if we join by id
            for c in chunk:
                augmented_rows.append({"id": c["id"], "name": c["name"]})

    progress_bar.progress(1.0, text="Augmentation complete.")
    
    # Convert LLM results to dataframe
    if not augmented_rows:
        return companies_df, total_usage
        
    # Clean up fields in raw dictionaries before polars conversion
    for row in augmented_rows:
        if "founders" in row and isinstance(row["founders"], list):
            row["founders"] = ", ".join(str(f) for f in row["founders"] if f is not None)
            
    llm_df = pl.DataFrame(augmented_rows)
    
    if "name" in llm_df.columns:
        llm_df = llm_df.drop("name")
    
    # Ensure id is string in both to guarantee join
    companies_df = companies_df.with_columns(pl.col("id").cast(pl.String))
    llm_df = llm_df.with_columns(pl.col("id").cast(pl.String))
    
    # Join with original dataframe by id
    final_df = companies_df.join(llm_df, on="id", how="left")
    
    return final_df, total_usage


# --- Main UI ---
st.title("CT Business Registry")
st.markdown("Query the CT registry for recently formed businesses matching specific NAICS codes.")

col_a, col_b, col_c = st.columns(3)
with col_a:
    default_start = datetime.now() - timedelta(days=180)
    start_date = st.date_input("Start Date", value=default_start)
with col_b:
    end_date = st.date_input("End Date", value=datetime.now())
with col_c:
    name_filter = st.text_input("Name Filter (Optional)", placeholder="e.g. Acme Corp")
    
with st.expander("Settings (NAICS & LLM Options)", expanded=False):
    DEFAULT_NAICS = [
        "5112", "5182", "5191", "5415", "5416", "3341", "5417", "3254", "3391", 
        "5223", "5222", "5239", "5242", "4541", "5111", "7139", "4885", "3345", 
        "3339", "3364", "2211", "2213", "2371", "5173", "5239"
    ]
    
    naics_selected = st.multiselect(
        "NAICS Code Prefixes",
        options=list(set(DEFAULT_NAICS + ["All"])),
        default=DEFAULT_NAICS,
        help="Filters for Tech, Biotech, Finance, Logistics, etc."
    )
    
    st.markdown("---")
    st.markdown("**LLM Augmentation Settings**")
    
    def format_model(m: str) -> str:
        in_price, out_price = settings.get_model_pricing(m)
        return f"{m} (${in_price:.2f} / ${out_price:.2f})"
    
    selected_model = st.selectbox(
        "Model",
        options=MODEL_OPTIONS,
        index=MODEL_OPTIONS.index("google/gemini-2.5-flash") if "google/gemini-2.5-flash" in MODEL_OPTIONS else 0,
        format_func=format_model,
    )
    
    max_augment_rows = st.slider(
        "Max Rows to Augment",
        min_value=10,
        max_value=100,
        value=20,
        step=10,
        help="Limit the number of rows sent to the LLM to control costs."
    )

col_btn1, col_btn2 = st.columns([1, 5])
has_data = "recent_df" in st.session_state and not st.session_state["recent_df"].is_empty()

with col_btn1:
    fetch_clicked = st.button("Fetch Businesses", type="primary", key="btn_fetch")
with col_btn2:
    augment_clicked = st.button("✨ Augment with AI", key="btn_augment", disabled=not has_data)
    
if fetch_clicked:
    if not naics_selected:
        st.warning("Please select at least one NAICS prefix.")
    else:
        with st.spinner("Fetching from CT Open Data..."):
            start_iso = start_date.strftime("%Y-%m-%dT00:00:00")
            end_iso = end_date.strftime("%Y-%m-%dT23:59:59")
            
            recent_df = fetch_recent_ct_businesses(start_iso, end_iso, naics_selected, name_filter.strip())
            
        if recent_df.is_empty():
            st.info("No new businesses found in this date range.")
        else:
            st.session_state["recent_df"] = recent_df
            # Clear previous augmentation
            for key in ["aug_df", "aug_usage", "aug_time"]:
                st.session_state.pop(key, None)
            st.rerun()

# Display single dataframe (either augmented or raw) BEFORE augment block
if "aug_df" in st.session_state:
    aug_df = st.session_state["aug_df"]
    usage = st.session_state["aug_usage"]
    elapsed = st.session_state["aug_time"]
    
    st.success(f"Augmentation complete in {elapsed:.1f}s! Used ~{usage.total_tokens} tokens (Est. Cost: ${usage.estimated_cost_usd:.4f})")
    st.dataframe(aug_df.to_pandas(), use_container_width=True)
    
    st.download_button(
        label="📥 Export Augmented Data",
        data=generate_excel_bytes(aug_df),
        file_name="augmented_ct_businesses.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="btn_dl_aug"
    )
                
elif "recent_df" in st.session_state and not st.session_state["recent_df"].is_empty():
    df_to_show = st.session_state["recent_df"]
    st.success(f"Fetched {len(df_to_show)} recently registered businesses.")
    st.dataframe(df_to_show.to_pandas(), use_container_width=True)
    
    st.download_button(
        label="📥 Export to Excel",
        data=generate_excel_bytes(df_to_show),
        file_name="recently_registered_ct.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if augment_clicked:
    df_to_show = st.session_state["recent_df"]
    client = LLMClient(model=selected_model)
    with st.spinner(f"LLM Augmentation in progress using {selected_model}... This may take a minute."):
        start_t = time.time()
        augmented_df, usage = augment_businesses(df_to_show, client, max_rows=max_augment_rows)
        elapsed = time.time() - start_t
        
    st.session_state["aug_df"] = augmented_df
    st.session_state["aug_usage"] = usage
    st.session_state["aug_time"] = elapsed
    st.rerun()
