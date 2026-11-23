import re
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .model import CATEGORIES, Route
from .tools import MUTATIONS, ORDER_PATTERN, REFUND_PATTERN

AFFIRM = re.compile(r"(?:yes(?: please|,? go ahead)?|confirm(?: it)?|go ahead|proceed)[.!]?", re.I)
DECLINE = re.compile(r"(?:no(?: thanks)?|stop|never mind|do not proceed|don't proceed)[.!]?", re.I)
INJECTION = re.compile(
    r"ignore (?:all |your |the |previous )*(?:rules|instructions|policies)|"
    r"(?:system|developer) prompt|bypass|skip (?:the )?confirmation|"
    r"confirmed\s*[:=]\s*true|override (?:your |the )?|pretend (?:i|you) (?:am|are) (?:admin|the owner)|"
    r"reveal .*?(?:secret|database|instructions)", re.I,
)
SCOPE = re.compile(
    r"\b(?:homework|assignment|weather|write (?:me )?(?:python|code|a poem)|stock price|medical advice|capital of|"
    r"football|solve .*equation|joke)\b", re.I,
)
PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")
REFERENCE = re.compile(fr"\b(?:{ORDER_PATTERN}|{REFUND_PATTERN})\b", re.I)
ESCALATION = {"complaint", "contact_human_agent"}
TOOL_FOR = {
    "track_order": "get_order_status", "change_order": "get_order_status",
    "cancel_order": "cancel_order", "get_refund": "initiate_refund", "track_refund": "track_refund",
    "change_shipping_address": "update_shipping_address", "set_up_shipping_address": "update_shipping_address",
    "get_invoice": "get_invoice", "check_invoice": "get_invoice",
}
ERROR_TEXT = {
    "ORDER_NOT_FOUND": "I couldn't find that order in your account.",
    "REFUND_NOT_FOUND": "I couldn't find that refund in your account.",
    "INVOICE_NOT_FOUND": "I couldn't find that invoice in your account.",
    "NOT_CANCELLABLE": "That order is no longer eligible for cancellation.",
    "ADDRESS_LOCKED": "That order's shipping address can no longer be changed.",
    "REFUND_NOT_ELIGIBLE": "That order is outside the refund eligibility rules.",
    "REFUND_EXISTS": "A refund already exists for that order. You can ask me to track it.",
    "INVALID_ARGUMENTS": "Please check the information and try again.",
    "DATABASE_UNAVAILABLE": "The order service is temporarily unavailable. Please try again.",
}
RULES = (
    (r"\b(?:fee|charge|cost)\b.*\bcancel|\bcancell?ation\b.*\bfee", "check_cancellation_fee"),
    (r"\b(?:manager|supervisor|human agent|live agent|speak to a person)\b", "contact_human_agent"),
    (r"\b(?:complaint|unacceptable|terrible service)\b", "complaint"),
    (r"\bcancel\b", "cancel_order"),
    (r"\b(?:modify|change|edit)\b.*\border\b(?!.*address)", "change_order"),
    (r"\b(?:shipping address|address)\b.*\b(?:change|update|replace|redirect)\b|\b(?:change|update|ship to|send to)\b.*\baddress\b", "change_shipping_address"),
    (r"\b(?:invoice|gst|receipt)\b", "get_invoice"),
    (r"\b(?:track|check|status|where)\b.*\b(?:refund|RF-)|\brefund\b.*\bstatus", "track_refund"),
    (r"\b(?:refund|exchange|return)\b.*\b(?:policy|days|time|window|eligible|rules|take)\b|\b(?:how long|how many days)\b.*\b(?:refund|exchange|return)\b", "check_refund_policy"),
    (r"\brefund\b", "get_refund"),
    (r"\b(?:debit(?:ed)?|failed|charged|pending|duplicate)\b.*\b(?:payment|charge|money|order)\b|\bpayment\b.*\b(?:pending|problem|failed)\b", "payment_issue"),
    (r"\b(?:payment methods?|cash on delivery|COD|credit cards?|net banking|PIN|checkout accepts)\b", "check_payment_methods"),
    (r"\b(?:shipping|delivery|express|pickup)\b.*\b(?:options|cost|fees|hold|free|price)\b|\bhow much\b.*\bshipping\b", "delivery_options"),
    (r"\b(?:shipping|delivery|express|metro)\b.*\b(?:time|days|when|long|take|arrive)\b|\bhow long\b.*\b(?:delivery|express)\b(?!.*confirmation email)", "delivery_period"),
    (r"\b(?:where|track|status|eta|arrive|look up)\b.*\b(?:order|package|parcel|shipment|KK-)|\b(?:order|KK-)\b.*\b(?:status|shipped)\b", "track_order"),
    (r"\b(?:password|reset)\b", "recover_password"),
    (r"\b(?:newsletter|unsubscribe|marketing email)\b", "newsletter_subscription"),
    (r"\b(?:delete|deletion|close)\b.*\baccount\b", "delete_account"),
    (r"\b(?:switch|different)\b.*\baccount\b", "switch_account"),
    (r"\b(?:registration is stuck|registration error)\b", "registration_problems"),
    (r"\b(?:sign up|new account|register|verification link)\b", "create_account"),
    (r"\b(?:review|moderation)\b", "review"),
    (r"\b(?:profile|email in my account)\b", "edit_account"),
    (r"\b(?:support hours|human support|contact support|helpdesk|talk to a person|file a ticket)\b", "contact_customer_service"),
    (r"\b(?:checkout|cart|new order|order confirmation email)\b", "place_order"),
)


class State(TypedDict, total=False):
    customer: str
    query: str
    history: list
    context: dict
    next_context: dict
    entities: dict
    intent: str
    category: str
    confidence: float
    alternatives: list
    branch: str
    answer: str
    approved: tuple
    draft: str
    chunks: list
    calls: list
    trace: list
    steps: int
    retry: int
    verdict: str
    checks: dict
    executed: bool
    request: dict
    routing_override: str


def demo_route(query):
    for pattern, intent in RULES:
        if re.search(pattern, query, re.I):
            return Route(intent, CATEGORIES[intent], 1.0)
    return Route("unknown", "", 0)


def render_tool(name, result):
    if not result["ok"]:
        return ERROR_TEXT.get(result["error"], "I couldn't complete the request. Please contact support.")
    d = result["data"]
    if name == "get_order_status":
        if d["status"] == "cancelled":
            return f'Order {d["order_id"]} is cancelled.'
        return f'Order {d["order_id"]} is {d["status"]}. Its recorded delivery date is {d["eta"]}.'
    if name == "get_invoice":
        return f'Invoice {d["invoice_id"]} for {d["order_id"]} totals ₹{d["total_inr"]:,}, including ₹{d["gst_inr"]:,} GST.'
    if name in {"track_refund", "initiate_refund", "cancel_order"}:
        line = f'Refund {d["refund_id"]} for {d["order_id"]} is ₹{d["amount_inr"]:,}.'
        if name == "cancel_order":
            return f'Order {d["order_id"]} is cancelled. {line} The estimated timeline is {d["eta_days"]} working days.'
        return f'{line} Status is {d["status"]}. The estimated timeline is {d["eta_days"]} working days.'
    if name == "update_shipping_address":
        return f'The shipping address for {d["order_id"]} was updated.'
    return f'I opened ticket {d["ticket_id"]} for human support.'


class Assistant:
    def __init__(self, tools, model=None, verify=True):
        self.tools = tools
        self.model = model
        graph = StateGraph(State)
        for name in ("router", "knowledge", "action", "verifier", "escalate"):
            graph.add_node(name, self.timed(name))
        graph.add_edge(START, "router")
        graph.add_conditional_edges("router", lambda s: s["branch"], {"knowledge": "knowledge", "action": "action", "end": END})
        graph.add_edge("knowledge", "verifier" if verify else END)
        graph.add_edge("action", "verifier" if verify else END)
        graph.add_conditional_edges("verifier", lambda s: s["verdict"], {"PASS": END, "REVISE": "knowledge", "ESCALATE": "escalate"})
        graph.add_edge("escalate", END)
        self.graph = graph.compile()

    def timed(self, name):
        node = getattr(self, name)

        def run(state):
            start = perf_counter()
            if state["steps"] >= 5 and name != "escalate":
                update = self.escalate(state) | {"branch": "end", "answer": "", "verdict": "PASS"}
                update["answer"] = update["draft"]
                name_recorded = "escalate"
            else:
                update = node(state)
                name_recorded = name
            if update.get("draft") and "answer" not in update:
                update["answer"] = update["draft"]
            event = {
                "node": name_recorded, "ms": round((perf_counter() - start) * 1000, 2),
                "input": state["query"] if name == "router" else state.get("intent", ""),
                "output": {key: update[key] for key in (
                    "intent", "confidence", "alternatives", "entities", "draft", "verdict", "checks", "routing_override"
                ) if key in update},
            }
            if "calls" in update:
                event["tools"] = update["calls"][len(state["calls"]):]
            return update | {"trace": [*state["trace"], event], "steps": state["steps"] + 1}

        return run

    def ask(self, customer, query, context=None, history=()):
        return self.graph.invoke({
            "customer": customer, "query": query.strip()[:2000], "history": list(history)[-12:],
            "context": context or {}, "next_context": {}, "calls": [], "trace": [], "steps": 0,
            "retry": 0, "chunks": [], "executed": False,
        })

    @staticmethod
    def end(answer, **updates):
        return {"branch": "end", "answer": answer, **updates}

    def router(self, state):
        query = state["query"]
        context = state["context"] if state["context"].get("customer") == state["customer"] else {}
        if INJECTION.search(query):
            return self.end("I can help with KharidiKart orders and policies using normal account and confirmation checks.", intent="refuse")
        if SCOPE.search(query):
            return self.end("I can help with KharidiKart orders, refunds and store policies.", intent="out_of_scope")
        if DECLINE.fullmatch(query) and context:
            return self.end("Okay, I left the order unchanged.", intent=context["intent"])
        if AFFIRM.fullmatch(query):
            if context.get("kind") == "confirm":
                return {"branch": "action", "intent": context["intent"], "entities": context["arguments"],
                        "confidence": 1.0, "request": context}
            return self.end("There is no pending action to confirm.", intent="unknown")
        ids = list(dict.fromkeys(value.upper() for value in REFERENCE.findall(query)))
        if len(ids) > 1 or re.search(r"\b(?:all|every|both|multiple) (?:my )?orders\b", query, re.I):
            return self.end("Please share one order or refund reference at a time.", intent="clarify")
        if context.get("kind") == "slot" and self.is_slot_reply(query, ids, context["slot"]):
            routed = Route(context["intent"], CATEGORIES[context["intent"]], 1.0)
            entities = dict(context["arguments"])
            if context["slot"] in {"reason", "new_address"}:
                entities[context["slot"]] = query
        else:
            entities = {}
            follow_up = re.fullmatch(r"(?:what about|and) (?:this one|that one|it)[?]?", query, re.I)
            if follow_up and context.get("kind") == "read":
                entities = dict(context["arguments"])
                routed = Route(context["intent"], CATEGORIES[context["intent"]], 1.0)
            else:
                routed = self.model.classify(query) if self.model else demo_route(query)
        if ids:
            field = "order_id" if ids[0].startswith("KK-") else "refund_id"
            entities[field] = ids[0]
        if routed.intent in {"get_refund", "change_shipping_address", "set_up_shipping_address"}:
            field, delimiter = ("reason", "because") if routed.intent == "get_refund" else ("new_address", "new address:")
            match = re.search(re.escape(delimiter) + r"\s+(.+)$", query, re.I)
            if match and len(match.group(1).strip()) >= (3 if field == "reason" else 8):
                entities[field] = match.group(1).strip()
        timing_question = re.search(r"\b(?:how long|how many (?:working )?days|refund timeline|refund policy)\b", query, re.I)
        override = routed.intent == "get_refund" and not ids and "refund" in query.lower() and timing_question
        if override:
            routed = Route("check_refund_policy", "REFUND", routed.confidence, routed.alternatives)
        details = {"intent": routed.intent, "category": routed.category, "entities": entities,
                   "confidence": routed.confidence, "alternatives": list(routed.alternatives)}
        if override:
            details["routing_override"] = "A general policy timing question cannot initiate a refund."
        if routed.confidence < 0.5:
            return self.end("Could you clarify what you need help with at KharidiKart?", **details)
        branch = "action" if routed.intent in TOOL_FOR or routed.intent in ESCALATION else "knowledge"
        return details | {"branch": branch}

    @staticmethod
    def is_slot_reply(query, ids, slot):
        if slot in {"order_id", "reference"}:
            return len(ids) == 1 and bool(re.fullmatch(fr"(?:it is |order |refund )?(?:{ORDER_PATTERN}|{REFUND_PATTERN})[.!]?", query, re.I))
        return not ids and 3 <= len(query) <= 240 and demo_route(query).intent == "unknown"

    def call(self, state, name, args, escalation=False):
        if len(state["calls"]) >= (3 if escalation else 2):
            return {"ok": False, "error": "TOOL_BUDGET"}, state["calls"]
        result = self.tools.call(name, args, state["customer"])
        return result, [*state["calls"], {"name": name, "args": dict(args), "result": result}]

    def action(self, state):
        intent = state["intent"]
        if intent in ESCALATION:
            return self.escalate(state)
        name = TOOL_FOR[intent]
        if state.get("request"):
            args = state["request"]["arguments"] | {"confirmed": True}
            result, calls = self.call(state, name, args)
            return self.draft(render_tool(name, result), calls=calls, executed=result["ok"])
        entities = state["entities"]
        if name == "track_refund" and "refund_id" in entities:
            args = {"refund_id": entities["refund_id"]}
        elif "order_id" in entities:
            args = {"order_id": entities["order_id"]}
        else:
            slot = "reference" if name == "track_refund" else "order_id"
            prompt = "Which order or refund reference should I look up?" if slot == "reference" else "What is the order ID? It looks like KK-12345."
            return self.remember(state, "slot", {}, prompt, slot=slot)
        if name in MUTATIONS:
            result, calls = self.call(state, "get_order_status", args)
            if not result["ok"]:
                return self.draft(render_tool("get_order_status", result), calls=calls)
            status = result["data"]["status"]
            if name != "initiate_refund" and status not in {"placed", "packed"}:
                code = "NOT_CANCELLABLE" if name == "cancel_order" else "ADDRESS_LOCKED"
                return self.draft(ERROR_TEXT[code], calls=calls)
            if name == "initiate_refund" and status != "delivered":
                code = "REFUND_EXISTS" if status == "cancelled" else "REFUND_NOT_ELIGIBLE"
                return self.draft(ERROR_TEXT[code], calls=calls)
            if name != "cancel_order":
                slot = "reason" if name == "initiate_refund" else "new_address"
                if slot not in entities:
                    prompt = "What is the reason for the refund?" if slot == "reason" else "What is the new shipping address?"
                    return self.remember(state, "slot", args, prompt, slot=slot, calls=calls)
                args = args | {slot: entities[slot]}
            description = {"cancel_order": "cancel order", "initiate_refund": "initiate a refund for",
                           "update_shipping_address": "update the shipping address for"}[name]
            return self.remember(state, "confirm", args,
                                 f'Please confirm that you want me to {description} {args["order_id"]}.', calls=calls)
        result, calls = self.call(state, name, args)
        answer = render_tool(name, result)
        if result["ok"]:
            return self.remember(state, "read", args, answer, calls=calls)
        return self.draft(answer, calls=calls)

    @staticmethod
    def draft(answer, **updates):
        return {"draft": answer, "approved": (answer,), **updates}

    def remember(self, state, kind, arguments, answer, slot=None, **updates):
        context = {"kind": kind, "intent": state["intent"], "customer": state["customer"], "arguments": arguments,
                   "slot": slot}
        return self.draft(answer, next_context=context, **updates)

    def knowledge(self, state):
        if state["retry"]:
            return {"draft": state["approved"][0]}
        result, calls = self.call(state, "search_policy", {"query": state["query"], "k": 3})
        chunks = result.get("data", {}).get("chunks", [])
        matches = [chunk for chunk in chunks if state["intent"] in chunk["intents"]]
        if not matches:
            return self.draft("Could you clarify which KharidiKart policy you mean?", calls=calls, chunks=chunks)
        approved = tuple(f'{chunk["answer"]} [{chunk["id"]}]' for chunk in matches)
        draft = self.model.policy_draft(state["query"], approved) if self.model else approved[0]
        return {"approved": approved, "draft": draft, "calls": calls, "chunks": chunks}

    def verifier(self, state):
        draft = state["draft"]
        checks = {"no_placeholders": not PLACEHOLDER.search(draft), "grounded": draft in state["approved"],
                  "tool_budget": len(state["calls"]) <= 3, "step_budget": state["steps"] < 6}
        if all(checks.values()):
            return {"verdict": "PASS", "checks": checks}
        if not state["retry"] and state["branch"] == "knowledge" and state["steps"] <= 3:
            return {"verdict": "REVISE", "checks": checks, "retry": 1, "answer": "", "next_context": {}}
        return {"verdict": "ESCALATE", "checks": checks, "answer": "", "next_context": {}}

    def escalate(self, state):
        result, calls = self.call(state, "escalate_to_human", {
            "reason": state.get("intent", "verification"), "summary": state["query"][:300] or "Support request",
        }, escalation=True)
        return self.draft(render_tool("escalate_to_human", result), calls=calls)
