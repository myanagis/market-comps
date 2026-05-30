"""Auto migration

Revision ID: 252886341727
Revises: f366c4d9af01
Create Date: 2026-05-30 10:00:17.890487

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '252886341727'
down_revision: Union[str, Sequence[str], None] = 'f366c4d9af01'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename tables
    op.rename_table('ingestion_runs', 'pipeline_runs')
    op.rename_table('canonical_mutations', 'audit_trails')

    # Rename indices
    op.drop_index('ix_ingestion_runs_id', table_name='pipeline_runs')
    op.create_index(op.f('ix_pipeline_runs_id'), 'pipeline_runs', ['id'], unique=False)
    
    op.drop_index('ix_canonical_mutations_id', table_name='audit_trails')
    op.create_index(op.f('ix_audit_trails_id'), 'audit_trails', ['id'], unique=False)
    
    op.drop_index('ix_canonical_mutations_canonical_entity_id', table_name='audit_trails')
    op.create_index(op.f('ix_audit_trails_canonical_entity_id'), 'audit_trails', ['canonical_entity_id'], unique=False)

    # Create new table pipeline_run_steps
    op.create_table('pipeline_run_steps',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('pipeline_run_id', sa.Integer(), nullable=False),
    sa.Column('step_order', sa.Integer(), nullable=False),
    sa.Column('step_name', sa.String(), nullable=False),
    sa.Column('step_type', sa.String(), nullable=True),
    sa.Column('method', sa.String(), nullable=True),
    sa.Column('started_at', sa.DateTime(), nullable=True),
    sa.Column('completed_at', sa.DateTime(), nullable=True),
    sa.Column('output_count', sa.Integer(), nullable=True),
    sa.Column('records_created', sa.Integer(), nullable=True),
    sa.Column('records_updated', sa.Integer(), nullable=True),
    sa.Column('records_failed', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(), nullable=True),
    sa.Column('error_message', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_run_id'], ['pipeline_runs.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_pipeline_run_steps_id'), 'pipeline_run_steps', ['id'], unique=False)

    # Rename foreign keys in extraction_jobs
    op.alter_column('extraction_jobs', 'ingestion_run_id', new_column_name='pipeline_run_id')
    op.drop_constraint('extraction_jobs_ingestion_run_id_fkey', 'extraction_jobs', type_='foreignkey')
    op.create_foreign_key(None, 'extraction_jobs', 'pipeline_runs', ['pipeline_run_id'], ['id'])

    # Rename foreign keys in source_documents
    op.alter_column('source_documents', 'ingestion_run_id', new_column_name='pipeline_run_id')
    op.drop_constraint('source_documents_ingestion_run_id_fkey', 'source_documents', type_='foreignkey')
    op.create_foreign_key(None, 'source_documents', 'pipeline_runs', ['pipeline_run_id'], ['id'])

    # Update pipelines
    op.add_column('pipelines', sa.Column('connector_type', sa.String(), nullable=True))
    op.add_column('pipelines', sa.Column('parser_type', sa.String(), nullable=True))
    op.add_column('pipelines', sa.Column('normalizer_type', sa.String(), nullable=True))
    op.drop_column('pipelines', 'pipeline_type')


def downgrade() -> None:
    # Reverse pipelines
    op.add_column('pipelines', sa.Column('pipeline_type', sa.VARCHAR(), autoincrement=False, nullable=False, server_default='UNKNOWN'))
    op.drop_column('pipelines', 'normalizer_type')
    op.drop_column('pipelines', 'parser_type')
    op.drop_column('pipelines', 'connector_type')

    # Reverse source_documents
    op.drop_constraint(None, 'source_documents', type_='foreignkey')
    op.alter_column('source_documents', 'pipeline_run_id', new_column_name='ingestion_run_id')
    op.create_foreign_key('source_documents_ingestion_run_id_fkey', 'source_documents', 'pipeline_runs', ['ingestion_run_id'], ['id'])

    # Reverse extraction_jobs
    op.drop_constraint(None, 'extraction_jobs', type_='foreignkey')
    op.alter_column('extraction_jobs', 'pipeline_run_id', new_column_name='ingestion_run_id')
    op.create_foreign_key('extraction_jobs_ingestion_run_id_fkey', 'extraction_jobs', 'pipeline_runs', ['ingestion_run_id'], ['id'])

    # Drop pipeline_run_steps
    op.drop_index(op.f('ix_pipeline_run_steps_id'), table_name='pipeline_run_steps')
    op.drop_table('pipeline_run_steps')

    # Reverse indices
    op.drop_index(op.f('ix_audit_trails_canonical_entity_id'), table_name='audit_trails')
    op.create_index('ix_canonical_mutations_canonical_entity_id', 'audit_trails', ['canonical_entity_id'], unique=False)
    op.drop_index(op.f('ix_audit_trails_id'), table_name='audit_trails')
    op.create_index('ix_canonical_mutations_id', 'audit_trails', ['id'], unique=False)
    op.drop_index(op.f('ix_pipeline_runs_id'), table_name='pipeline_runs')
    op.create_index('ix_ingestion_runs_id', 'pipeline_runs', ['id'], unique=False)

    # Rename tables back
    op.rename_table('audit_trails', 'canonical_mutations')
    op.rename_table('pipeline_runs', 'ingestion_runs')
