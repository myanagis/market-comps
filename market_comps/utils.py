import zoneinfo
import re

def format_est_datetime(dt):
    if not dt:
        return "Unknown Time"
    try:
        eastern = zoneinfo.ZoneInfo("America/New_York")
        tz_time = dt.replace(tzinfo=zoneinfo.ZoneInfo("UTC")).astimezone(eastern)
        return tz_time.strftime('%Y-%m-%d %I:%M %p ET')
    except Exception:
        return str(dt)

def format_currency(amount_str):
    if not amount_str:
        return ""
    
    # Try to extract the number
    amount_str = str(amount_str).strip()
    
    # If it already looks formatted like "$5M", just return it
    if "$" in amount_str and amount_str[-1].upper() in ("M", "B", "K"):
        return amount_str
        
    try:
        # Remove non-numeric characters except dot
        clean_str = re.sub(r'[^\d.]', '', amount_str)
        if clean_str:
            val = float(clean_str)
            if val.is_integer():
                return f"${int(val):,}"
            else:
                return f"${val:,.2f}"
    except Exception:
        pass
        
    return amount_str
