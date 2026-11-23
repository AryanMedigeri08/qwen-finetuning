import re
from datetime import date
from sqlite3 import Error as DatabaseError
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from .db import REFUND_DAYS

ORDER_PATTERN = r"KK-[0-9]{5}"
REFUND_PATTERN = r"RF-[0-9]{4,12}"


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Order(Arguments):
    order_id: str = Field(pattern=f"^{ORDER_PATTERN}$")


class ConfirmedOrder(Order):
    confirmed: StrictBool


class RefundRequest(ConfirmedOrder):
    reason: str = Field(min_length=3, max_length=300)


class AddressRequest(ConfirmedOrder):
    new_address: str = Field(min_length=8, max_length=240)


class RefundLookup(Arguments):
    order_id: str | None = Field(default=None, pattern=f"^{ORDER_PATTERN}$")
    refund_id: str | None = Field(default=None, pattern=f"^{REFUND_PATTERN}$")


class Search(Arguments):
    query: str = Field(min_length=2, max_length=2000)
    k: int = Field(default=3, ge=1, le=5)


class Ticket(Arguments):
    reason: str = Field(min_length=1, max_length=120)
    summary: str = Field(min_length=1, max_length=300)


INPUTS = {
    "get_order_status": Order,
    "cancel_order": ConfirmedOrder,
    "initiate_refund": RefundRequest,
    "track_refund": RefundLookup,
    "update_shipping_address": AddressRequest,
    "get_invoice": Order,
    "search_policy": Search,
    "escalate_to_human": Ticket,
}
TOOL_SCHEMAS = {name: schema.model_json_schema() for name, schema in INPUTS.items()}
MUTATIONS = {"cancel_order", "initiate_refund", "update_shipping_address"}


def ok(**data):
    return {"ok": True, "data": data}


def error(code):
    return {"ok": False, "error": code}


class Tools:
    def __init__(self, store, retriever):
        self.store = store
        self.retriever = retriever

    def call(self, name, arguments, customer_id):
        if name not in INPUTS:
            return error("UNKNOWN_TOOL")
        if not re.fullmatch(r"C[0-9]{4}", customer_id):
            return error("INVALID_CUSTOMER")
        try:
            args = INPUTS[name].model_validate(arguments).model_dump(exclude_none=True)
            if name in MUTATIONS and not args.pop("confirmed"):
                return error("CONFIRMATION_REQUIRED")
            return getattr(self, "_" + name)(customer_id, **args)
        except ValidationError:
            return error("INVALID_ARGUMENTS")
        except DatabaseError:
            return error("DATABASE_UNAVAILABLE")

    @staticmethod
    def _order(connection, customer, order_id):
        return connection.execute(
            "SELECT * FROM orders WHERE order_id=? AND customer_id=?", (order_id, customer)
        ).fetchone()

    def _get_order_status(self, customer, order_id):
        with self.store.connect() as connection:
            row = self._order(connection, customer, order_id)
            if row is None:
                return error("ORDER_NOT_FOUND")
            return ok(**{key: row[key] for key in ("order_id", "status", "amount_inr", "eta", "placed_at")})

    def _create_refund(self, connection, row, reason):
        number = int(row["order_id"].split("-")[-1])
        details = {
            "refund_id": f"RF-{number:05d}", "order_id": row["order_id"],
            "amount_inr": row["amount_inr"], "status": "initiated", "eta_days": REFUND_DAYS[row["payment_mode"]],
        }
        connection.execute(
            "INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?, ?)",
            (details["refund_id"], details["order_id"], details["amount_inr"], details["status"],
             self.store.today.isoformat(), details["eta_days"], reason),
        )
        return details

    def _cancel_order(self, customer, order_id):
        with self.store.connect(write=True) as connection:
            row = self._order(connection, customer, order_id)
            if row is None:
                return error("ORDER_NOT_FOUND")
            if row["status"] not in ("placed", "packed"):
                return error("NOT_CANCELLABLE")
            refund = self._create_refund(connection, row, "cancelled order")
            connection.execute("UPDATE orders SET status='cancelled' WHERE order_id=?", (order_id,))
            return ok(**(refund | {"status": "cancelled"}))

    def _initiate_refund(self, customer, order_id, reason):
        if len(reason.strip()) < 3:
            return error("INVALID_ARGUMENTS")
        with self.store.connect(write=True) as connection:
            row = self._order(connection, customer, order_id)
            if row is None:
                return error("ORDER_NOT_FOUND")
            if connection.execute("SELECT 1 FROM refunds WHERE order_id=?", (order_id,)).fetchone():
                return error("REFUND_EXISTS")
            if row["status"] != "delivered" or not 0 <= (
                self.store.today - date.fromisoformat(row["delivered_at"])
            ).days <= 7:
                return error("REFUND_NOT_ELIGIBLE")
            return ok(**self._create_refund(connection, row, reason))

    def _track_refund(self, customer, order_id=None, refund_id=None):
        if (order_id is None) == (refund_id is None):
            return error("INVALID_ARGUMENTS")
        column, value = ("r.order_id", order_id) if order_id else ("r.refund_id", refund_id)
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT r.refund_id, r.order_id, r.amount_inr, r.status, r.eta_days "
                f"FROM refunds r JOIN orders o USING (order_id) WHERE o.customer_id=? AND {column}=?",
                (customer, value),
            ).fetchone()
            return ok(**dict(row)) if row else error("REFUND_NOT_FOUND")

    def _update_shipping_address(self, customer, order_id, new_address):
        new_address = new_address.strip()
        if len(new_address) < 8 or any(ord(char) < 32 for char in new_address):
            return error("INVALID_ARGUMENTS")
        with self.store.connect(write=True) as connection:
            row = self._order(connection, customer, order_id)
            if row is None:
                return error("ORDER_NOT_FOUND")
            if row["status"] not in ("placed", "packed"):
                return error("ADDRESS_LOCKED")
            connection.execute("UPDATE orders SET shipping_address=? WHERE order_id=?", (new_address, order_id))
            return ok(order_id=order_id, shipping_address=new_address)

    def _get_invoice(self, customer, order_id):
        with self.store.connect() as connection:
            row = connection.execute(
                "SELECT i.* FROM invoices i JOIN orders o USING (order_id) WHERE o.customer_id=? AND i.order_id=?",
                (customer, order_id),
            ).fetchone()
            return ok(**dict(row)) if row else error("INVOICE_NOT_FOUND")

    def _search_policy(self, customer, query, k=3):
        return ok(chunks=self.retriever.search(query, k))

    def _escalate_to_human(self, customer, reason, summary):
        with self.store.connect(write=True) as connection:
            if not connection.execute("SELECT 1 FROM customers WHERE customer_id=?", (customer,)).fetchone():
                return error("INVALID_CUSTOMER")
            ticket = f"TK-{uuid4().hex[:10]}"
            connection.execute("INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?)", (
                ticket, customer, reason, summary, self.store.today.isoformat(), "open",
            ))
            return ok(ticket_id=ticket, status="open")
