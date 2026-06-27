"""add_users_collections_annotations

Revision ID: 004_add_users_collections_annotations
Revises: 003_add_briefings_and_interests
Create Date: 2026-06-27

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = '004_add_users_collections_annotations'
down_revision: Union[str, None] = '003_add_briefings_and_interests'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Users Table
    op.create_table(
        'users',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False, server_default='researcher'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

    # 2. Collections Table
    op.create_table(
        'collections',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_collections_user_id', 'collections', ['user_id'], unique=False)

    # 3. Collection Papers Many-to-Many Table
    op.create_table(
        'collection_papers',
        sa.Column('collection_id', UUID(as_uuid=True), sa.ForeignKey('collections.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), primary_key=True),
        sa.Column('added_at', sa.DateTime(), nullable=True),
    )

    # 4. Annotations Table
    op.create_table(
        'annotations',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('paper_id', UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('text_selection', sa.Text(), nullable=True),
        sa.Column('note', sa.Text(), nullable=False),
        sa.Column('tag', sa.String(), nullable=True),
        sa.Column('page', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_annotations_paper_id', 'annotations', ['paper_id'], unique=False)
    op.create_index('ix_annotations_user_id', 'annotations', ['user_id'], unique=False)
    op.create_index('ix_annotations_tag', 'annotations', ['tag'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_annotations_tag', table_name='annotations')
    op.drop_index('ix_annotations_user_id', table_name='annotations')
    op.drop_index('ix_annotations_paper_id', table_name='annotations')
    op.drop_table('annotations')
    op.drop_table('collection_papers')
    op.drop_index('ix_collections_user_id', table_name='collections')
    op.drop_table('collections')
    op.drop_index('ix_users_email', table_name='users')
    op.drop_table('users')
