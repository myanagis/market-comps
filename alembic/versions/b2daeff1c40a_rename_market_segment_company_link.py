"""rename market segment company link

Revision ID: b2daeff1c40a
Revises: 42114f2b3da8
Create Date: 2026-08-04 12:06:29.607847

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2daeff1c40a'
down_revision: Union[str, Sequence[str], None] = '42114f2b3da8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.rename_table('company_market_segments', 'market_segment_company_links')
    op.add_column('competitive_analysis_companies', sa.Column('market_segment_id', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'competitive_analysis_companies', 'market_segments', ['market_segment_id'], ['id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(None, 'competitive_analysis_companies', type_='foreignkey')
    op.drop_column('competitive_analysis_companies', 'market_segment_id')
    op.rename_table('market_segment_company_links', 'company_market_segments')
