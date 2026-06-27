"""add_conversations

Revision ID: 002_add_conversations
Revises: 001_initial
Create Date: 2026-06-20

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = '002_add_conversations'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = inspector.get_table_names()
    if 'conversations' not in tables:
        op.create_table(
            'conversations',
            sa.Column('id', UUID(as_uuid=True), primary_key=True),
            sa.Column('session_id', sa.String, nullable=False, index=True),
            sa.Column('messages', sa.JSON, nullable=False, server_default=sa.text("'[]'")),
            sa.Column('created_at', sa.DateTime, nullable=True),
            sa.Column('updated_at', sa.DateTime, nullable=True),
        )


def downgrade() -> None:
    op.drop_table('conversations')
