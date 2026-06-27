"""add_content_hash

Revision ID: 002_add_content_hash
Revises: 001_initial
Create Date: 2026-06-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '002_add_content_hash'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('papers', sa.Column('content_hash', sa.String, nullable=True))
    op.create_index('ix_papers_content_hash', 'papers', ['content_hash'])


def downgrade() -> None:
    op.drop_index('ix_papers_content_hash', table_name='papers')
    op.drop_column('papers', 'content_hash')
