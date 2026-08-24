from market_comps.db.session import engine
from sqlalchemy import text

def update_seq():
    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT setval('comparison_set_organization_links_id_seq', COALESCE((SELECT MAX(id)+1 FROM comparison_set_organization_links), 1), false);"))
            conn.commit()
            print("Sequence updated!")
        except Exception as e:
            print("Seq Update Error:", e)

if __name__ == "__main__":
    update_seq()
