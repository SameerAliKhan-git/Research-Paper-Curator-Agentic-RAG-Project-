"""add_briefings_and_interests

Revision ID: 003_add_briefings_and_interests
Revises: e6439f05a0d6
Create Date: 2026-06-21

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = '003_add_briefings_and_interests'
down_revision: Union[str, None] = 'e6439f05a0d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'researcher_interests',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('keyword', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_researcher_interests_keyword', 'researcher_interests', ['keyword'], unique=True)

    op.create_table(
        'daily_briefings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('arxiv_id', sa.String(), nullable=False),
        sa.Column('title', sa.String(), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('published_date', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_daily_briefings_arxiv_id', 'daily_briefings', ['arxiv_id'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_daily_briefings_arxiv_id', table_name='daily_briefings')
    op.drop_table('daily_briefings')
    op.drop_index('ix_researcher_interests_keyword', table_name='researcher_interests')
    op.drop_table('researcher_interests')
