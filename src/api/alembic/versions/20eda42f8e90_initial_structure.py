"""initial structure

Revision ID: 20eda42f8e90
Revises:
Create Date: 2018-07-10 11:03:31.950684

"""
from __future__ import unicode_literals
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20eda42f8e90'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('processes',
                    sa.Column('id', sa.Integer, primary_key=True,
                              autoincrement=True),
                    sa.Column('name', sa.Unicode(64), unique=True, nullable=False),
                    )


def downgrade():
    pass
