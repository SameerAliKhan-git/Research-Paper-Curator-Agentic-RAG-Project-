"""initial_schema

Revision ID: 001_initial
Revises: 
Create Date: 2026-06-19

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'papers',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('arxiv_id', sa.String, nullable=False, unique=True, index=True),
        sa.Column('title', sa.String, nullable=False),
        sa.Column('authors', sa.JSON, nullable=False),
        sa.Column('abstract', sa.Text, nullable=False),
        sa.Column('categories', sa.JSON, nullable=False),
        sa.Column('published_date', sa.DateTime, nullable=False),
        sa.Column('pdf_url', sa.String, nullable=False),
        sa.Column('raw_text', sa.Text, nullable=True),
        sa.Column('sections', sa.JSON, nullable=True),
        sa.Column('references', sa.JSON, nullable=True),
        sa.Column('parser_used', sa.String, nullable=True),
        sa.Column('parser_metadata', sa.JSON, nullable=True),
        sa.Column('pdf_processed', sa.Boolean, nullable=False, server_default=sa.text('false')),
        sa.Column('pdf_processing_date', sa.DateTime, nullable=True),
        sa.Column('created_at', sa.DateTime, nullable=True),
        sa.Column('updated_at', sa.DateTime, nullable=True),
    )


def downgrade() -> None:
    op.drop_table('papers')
