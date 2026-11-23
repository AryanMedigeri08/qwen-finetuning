import argparse
import sqlite3
from contextlib import closing, contextmanager
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from . import ROOT, RUNTIME

DEMO_DATE = date(2026, 11, 23)
REFUND_DAYS = {"UPI": 5, "card": 7, "net banking": 7, "COD": 10}
DEFAULT_DB = RUNTIME / "demo.sqlite3"


class Store:
    def __init__(self, path=DEFAULT_DB, today=DEMO_DATE):
        self.path = Path(path)
        self.today = today

    @contextmanager
    def connect(self, write=False):
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                if write:
                    connection.execute("BEGIN IMMEDIATE")
                yield connection

    def seed(self, seed=10):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript((ROOT / "schema.sql").read_text(encoding="utf-8"))
        fake = Faker("en_IN")
        fake.seed_instance(seed)
        cities = ("Pune", "Mumbai", "Nagpur", "Nashik", "Satara")
        with self.connect(write=True) as connection:
            if connection.execute("SELECT 1 FROM customers LIMIT 1").fetchone():
                return
            for number in range(1, 201):
                customer = f"C{number:04d}"
                connection.execute("INSERT INTO customers VALUES (?, ?, ?, ?, ?, ?)", (
                    customer, fake.name(), f"{customer.lower()}@example.test",
                    f"00000{number:05d}", cities[(number - 1) % len(cities)], "2026-01-01",
                ))
            for number in range(1, 501):
                order = f"KK-{10000 + number}"
                customer = f"C{(number - 1) % 200 + 1:04d}"
                status = ("placed", "packed", "shipped", "delivered", "cancelled")[(number - 1) % 5]
                payment = tuple(REFUND_DAYS)[(number - 1) % 4]
                amount = fake.random_int(399, 12000)
                placed = self.today - timedelta(days=7 + number % 5)
                eta = placed + timedelta(days=6) if status == "delivered" else self.today + timedelta(days=4)
                delivered_at = eta.isoformat() if status == "delivered" else None
                connection.execute("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    order, customer, placed.isoformat(), status, amount, payment,
                    f"{number} Demo Lane, {cities[(number - 1) % len(cities)]}", eta.isoformat(), delivered_at,
                ))
                connection.execute("INSERT INTO invoices VALUES (?, ?, ?, ?, ?)", (
                    f"INV-{10000 + number}", order, amount * 18 // 118, amount, placed.isoformat(),
                ))
                if status == "cancelled":
                    connection.execute("INSERT INTO refunds VALUES (?, ?, ?, ?, ?, ?, ?)", (
                        f"RF-{number:04d}", order, amount, "processing", self.today.isoformat(),
                        REFUND_DAYS[payment], "cancelled order",
                    ))

    def customers(self):
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT customer_id, name, city FROM customers ORDER BY customer_id"
            )]

    def customer_records(self, customer_id):
        with self.connect() as connection:
            orders = connection.execute(
                "SELECT order_id, status, amount_inr, payment_mode, eta FROM orders WHERE customer_id=?",
                (customer_id,),
            )
            refunds = connection.execute(
                "SELECT r.refund_id, r.order_id, r.amount_inr, r.status, r.eta_days "
                "FROM refunds r JOIN orders o USING (order_id) WHERE o.customer_id=?", (customer_id,),
            )
            return [dict(row) for row in orders], [dict(row) for row in refunds]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the fictional KharidiKart database")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--seed", type=int, default=10)
    options = parser.parse_args()
    Store(options.db).seed(options.seed)
