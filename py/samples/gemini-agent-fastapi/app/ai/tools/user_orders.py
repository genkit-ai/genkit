# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""List orders for the signed-in user, read from the datastore."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.ai import ai
from app.ai.context import current_uid
from app.data import orders
from app.models import Order
from genkit import ToolRunContext


class OrderLookupInput(BaseModel):
    status: str | None = Field(default=None, description='Optional filter: active, fulfilled, cancelled')


class OrderLookupOutput(BaseModel):
    orders: list[Order] = Field(default_factory=list)


@ai.tool(name='listMyOrders', description='List orders for the currently signed-in user.')
async def list_my_orders(input: OrderLookupInput, ctx: ToolRunContext) -> OrderLookupOutput:
    uid = current_uid(ctx)
    if not uid:
        return OrderLookupOutput()

    rows = await orders.list(uid)
    if input.status:
        rows = [row for row in rows if row.status == input.status]
    return OrderLookupOutput(orders=rows)
