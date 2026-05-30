"""Manual migration

Revision ID: 8b190512ed2b
Revises: 252886341727
Create Date: 2026-05-30 10:00:28.116126

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '8b190512ed2b'
down_revision: Union[str, Sequence[str], None] = '252886341727'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
