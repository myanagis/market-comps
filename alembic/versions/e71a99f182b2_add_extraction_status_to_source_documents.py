"""add extraction status to source documents

Revision ID: e71a99f182b2
Revises: c20fafe36454
Create Date: 2026-08-05 20:42:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e71a99f182b2'
down_revision: Union[str, Sequence[str], None] = 'c20fafe36454'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('source_documents', sa.Column('extraction_status', sa.String(), nullable=True, server_default='SUCCESS'))
    op.add_column('source_documents', sa.Column('extraction_error', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('source_documents', 'extraction_error')
    op.drop_column('source_documents', 'extraction_status')
