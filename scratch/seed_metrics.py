import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from market_comps.db.session import get_db
from market_comps.db.models import MetricType

def seed_metrics():
    db = next(get_db())
    
    default_metrics = [
        {
            "code": "revenue",
            "display_name": "Revenue",
            "value_type": "currency",
            "is_point_in_time": False,
            "description": "Total revenue generated over a period."
        },
        {
            "code": "arr",
            "display_name": "ARR",
            "value_type": "currency",
            "is_point_in_time": True,
            "description": "Annual Recurring Revenue as of a specific date."
        },
        {
            "code": "post_money_valuation",
            "display_name": "Post-money valuation",
            "value_type": "currency",
            "is_point_in_time": True,
            "description": "Valuation of the company after a financing round."
        },
        {
            "code": "employee_count",
            "display_name": "Employee count",
            "value_type": "integer",
            "is_point_in_time": True,
            "description": "Total number of employees."
        },
        {
            "code": "gross_margin",
            "display_name": "Gross margin",
            "value_type": "percentage",
            "is_point_in_time": False,
            "description": "Percentage of revenue remaining after deducting COGS."
        },
        {
            "code": "customer_count",
            "display_name": "Customer count",
            "value_type": "integer",
            "is_point_in_time": True,
            "description": "Total number of active customers."
        }
    ]
    
    for metric_data in default_metrics:
        existing = db.query(MetricType).filter_by(code=metric_data["code"]).first()
        if not existing:
            new_metric = MetricType(**metric_data)
            db.add(new_metric)
            print(f"Added {metric_data['code']}")
        else:
            print(f"Skipped {metric_data['code']} - already exists")
            
    db.commit()
    print("Seeding complete.")

if __name__ == "__main__":
    seed_metrics()
