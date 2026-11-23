CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    phone TEXT NOT NULL,
    city TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    placed_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('placed', 'packed', 'shipped', 'delivered', 'cancelled')),
    amount_inr INTEGER NOT NULL CHECK (amount_inr > 0),
    payment_mode TEXT NOT NULL,
    shipping_address TEXT NOT NULL,
    eta TEXT NOT NULL,
    delivered_at TEXT
);
CREATE TABLE IF NOT EXISTS refunds (
    refund_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    amount_inr INTEGER NOT NULL CHECK (amount_inr > 0),
    status TEXT NOT NULL CHECK (status IN ('initiated', 'processing', 'completed')),
    initiated_at TEXT NOT NULL,
    eta_days INTEGER NOT NULL CHECK (eta_days > 0),
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invoices (
    invoice_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL UNIQUE REFERENCES orders(order_id),
    gst_inr INTEGER NOT NULL,
    total_inr INTEGER NOT NULL,
    issued_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    reason TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS orders_by_customer ON orders (customer_id, order_id);
