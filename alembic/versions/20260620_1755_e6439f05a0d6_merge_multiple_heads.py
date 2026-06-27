"""merge multiple heads

Revision ID: e6439f05a0d6
Revises: 002_add_content_hash, 002_add_conversations
Create Date: 2026-06-20 17:55:35.327678

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e6439f05a0d6'
down_revision: Union[str, None] = ('002_add_content_hash', '002_add_conversations')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
