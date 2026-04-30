import logging
import requests
import polars as pl

logger = logging.getLogger(__name__)

class CTBusinessRegistryClient:
    """Client for connecting to the Connecticut Business Registry Open Data API."""
    
    BASE_URL = "https://data.ct.gov/resource/n7gp-d28j.json"

    @classmethod
    def search_by_name(cls, name: str, status: str, biz_type: str) -> pl.DataFrame:
        """Search CT database by name and optional filters."""
        where_clauses = []
        if name:
            # Case insensitive partial match
            clean_name = name.replace("'", "''")  # escape single quotes
            where_clauses.append(f"upper(name) LIKE upper('%{clean_name}%')")
        
        if status != "Any":
            where_clauses.append(f"status='{status}'")
        
        if biz_type != "Any":
            where_clauses.append(f"business_type='{biz_type}'")
        
        where_str = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        params = {
            "$select": "id,name,business_type,status,naics_code,billingcity,billingstate,date_registration",
            "$where": where_str,
            "$limit": 500,
            "$order": "date_registration DESC"
        }
        
        try:
            resp = requests.get(cls.BASE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return pl.DataFrame()
            return pl.DataFrame(data)
        except Exception as e:
            logger.error(f"Error fetching data from CT Registry: {e}")
            raise

    @classmethod
    def fetch_recent(cls, start: str, end: str, prefixes: list[str], name_filter: str | None = None) -> pl.DataFrame:
        """Fetch recently registered CT businesses based on NAICS code prefixes and optional name filter."""
        batches = []
        offset = 0
        limit = 50000

        naics_filter = " OR ".join([f"naics_code LIKE '{p}%'" for p in prefixes])
        
        where_clauses = [
            "business_type='Stock'",
            "status='Active'",
            f"date_registration BETWEEN '{start}' AND '{end}'",
            f"({naics_filter})"
        ]
        
        if name_filter:
            clean_name = name_filter.replace("'", "''")
            where_clauses.append(f"upper(name) LIKE upper('%{clean_name}%')")
            
        where_str = " AND ".join(where_clauses)

        while True:
            params = {
                "$select": "id,name,naics_code,billingcity,billingstate,date_registration,business_email_address",
                "$where": where_str,
                "$order": "date_registration DESC",
                "$limit": limit,
                "$offset": offset
            }

            try:
                resp = requests.get(cls.BASE_URL, params=params)
                resp.raise_for_status()
                batch = resp.json()

                if not batch:
                    break

                batches.append(pl.DataFrame(batch))
                offset += limit
            except Exception as e:
                logger.error(f"API Error fetching recent businesses from CT Registry: {e}")
                raise

        df = pl.concat(batches, how="vertical") if batches else pl.DataFrame()

        if not df.is_empty():
            df = df.with_columns([
                pl.col("date_registration").str.strptime(pl.Datetime, strict=False),
                pl.col("naics_code").cast(pl.Int64, strict=False)
            ])
        return df
