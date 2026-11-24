import csv
import json
import re

with open('benchmarks/scenarios.json', encoding='utf-8') as f:
    scenarios = {s['id']: s for s in json.load(f)}

with open('data/rating_input.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

# Let's inspect each row in detail and build the score dicts
PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")

def score_all():
    r1_scores = []
    r2_scores = []

    for r in rows:
        rid = int(r['response_id'])
        sid = r['scenario']
        query = r['query']
        resp = r['response']
        s = scenarios[sid]
        stype = s['type']
        
        # Check if placeholders exist
        has_ph = bool(PLACEHOLDER_RE.search(resp))
        
        # Determine ratings based on thorough evaluation of the response
        # Let's write specific, explicit rules for each response id or category:
        
        # Row 1: S012 (C2, info): COD max -> leaks {{Min Amount}} to {{Max Amount}}, says varies by region
        if rid == 1:
            # Fact is ₹5,000. Leaks placeholders, wrong info.
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 2: S087 (C2, denied): Track refund for KK-10005 -> leaks {{Order Number}}, asks irrelevant questions
        elif rid == 2:
            r1 = (2, 2, 2)
            r2 = (2, 2, 3)
        # Row 3: S023 (C4, info): pending payment -> "Wait 30 minutes before retrying... [payment-issues.md #1]"
        elif rid == 3:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 4: S096 (C2, esc): speak to person -> acknowledges billing problem, asks details, doesn't escalate
        elif rid == 4:
            r1 = (3, 2, 3)
            r2 = (3, 3, 4)
        # Row 5: S054 (C2, write_confirm): customer says "yes" to cancel KK-10007, bot repeats identical placeholder steps, order NOT cancelled
        elif rid == 5:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 6: S087 (C4, denied): "I couldn't find that refund in your account."
        elif rid == 6:
            r1 = (5, 4, 4)
            r2 = (5, 4, 4)
        # Row 7: S095 (C2, esc): complaint about delays -> apologizes, asks for details, no ticket opened
        elif rid == 7:
            r1 = (3, 2, 4)
            r2 = (3, 3, 4)
        # Row 8: S098 (C2, esc): transfer to live agent -> "bear with me while I transfer you...", no actual escalation/ticket
        elif rid == 8:
            r1 = (3, 3, 4)
            r2 = (3, 3, 4)
        # Row 9: S012 (C4, info): COD max -> "Could you clarify which KharidiKart policy you mean?"
        elif rid == 9:
            r1 = (3, 2, 3)
            r2 = (3, 2, 4)
        # Row 10: S082 (C2, denied): cancel KK-10004 -> order is delivered (ineligible), but bot gives DIY cancel steps with {{Online Company Portal Info}}
        elif rid == 10:
            r1 = (1, 1, 2)
            r2 = (2, 1, 3)
        # Row 11: S017 (C2, info): password reset link validity -> asks for email address instead of stating 15 mins
        elif rid == 11:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 12: S003 (C4, info): COD refund timeline -> 10 working days [refund-timelines.md #3]
        elif rid == 12:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 13: S059 (C2, write_confirm): refund for damaged KK-10004 -> user says "confirm", bot says "I'll personally review...", refund NOT initiated
        elif rid == 13:
            r1 = (1, 1, 3)
            r2 = (1, 2, 3)
        # Row 14: S078 (C2, missing_slot): change address -> "Sure! Please provide new address..." (forgot to ask for order ID)
        elif rid == 14:
            r1 = (3, 3, 5)
            r2 = (3, 3, 4)
        # Row 15: S065 (C4, write_confirm): "The shipping address for KK-10021 was updated." (DB updated)
        elif rid == 15:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 16: S089 (C2, denied): refund for KK-10003 -> asks irrelevant transaction questions
        elif rid == 16:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 17: S063 (C4, write_confirm): "Refund RF-10024 for KK-10024 is ₹11,304. Status is initiated. The estimated timeline is 10 working days." (DB confirmed)
        elif rid == 17:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 18: S085 (C2, denied): track KK-10002 -> asks to confirm if order number is correct
        elif rid == 18:
            r1 = (2, 2, 3)
            r2 = (2, 2, 4)
        # Row 19: S072 (C2, missing_slot): cancel latest order -> gives generic DIY steps with {{Online Company Portal Info}}
        elif rid == 19:
            r1 = (2, 2, 2)
            r2 = (2, 2, 3)
        # Row 20: S059 (C4, write_confirm): "Refund RF-10004 for KK-10004 is ₹11,111. Status is initiated..." (DB confirmed)
        elif rid == 20:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 21: S006 (C4, info): standard shipping ₹49 below ₹999, free at ₹999 [delivery-options.md #1]
        elif rid == 21:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 22: S091 (C4, esc): "Could you clarify what you need help with at KharidiKart?"
        elif rid == 22:
            r1 = (3, 2, 3)
            r2 = (3, 2, 4)
        # Row 23: S096 (C4, esc): "Could you clarify which KharidiKart policy you mean?"
        elif rid == 23:
            r1 = (3, 2, 3)
            r2 = (3, 2, 3)
        # Row 24: S082 (C4, denied): "That order is no longer eligible for cancellation."
        elif rid == 24:
            r1 = (5, 4, 4)
            r2 = (5, 5, 4)
        # Row 25: S039 (C4, read): "Refund RF-0020 for KK-10020 is ₹7,435. Status is processing. The estimated timeline is 10 working days."
        elif rid == 25:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 26: S058 (C2, write_confirm): cancel KK-10017 -> user said "yes", bot repeats placeholder steps, order NOT cancelled
        elif rid == 26:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 27: S007 (C4, info): express shipping ₹99 [delivery-options.md #2]
        elif rid == 27:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 28: S089 (C4, denied): "That order is outside the refund eligibility rules."
        elif rid == 28:
            r1 = (5, 4, 4)
            r2 = (5, 5, 4)
        # Row 29: S025 (C2, info): edit review -> asks what edits to make, misses 30-day window policy
        elif rid == 29:
            r1 = (2, 2, 4)
            r2 = (3, 2, 4)
        # Row 30: S095 (C4, esc): "I opened ticket TK-d15d68aab1 for human support."
        elif rid == 30:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 31: S032 (C4, read): track KK-10007 -> "Could you clarify what you need help with at KharidiKart?"
        elif rid == 31:
            r1 = (3, 2, 3)
            r2 = (3, 2, 3)
        # Row 32: S091 (C2, esc): complaint -> recommends contacting customer support team
        elif rid == 32:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 33: S055 (C4, write_confirm): "Order KK-10011 is cancelled. Refund RF-10011 for KK-10011 is ₹5,256..." (DB confirmed)
        elif rid == 33:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 34: S065 (C2, write_confirm): change address -> gives manual steps {{Current Shipping Address}} to {{New Shipping Address}}, address NOT updated
        elif rid == 34:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 35: S044 (C2, read): send invoice -> says "Let me quickly retrieve the invoice for you..." leaks {{Invoice Number}}, no invoice
        elif rid == 35:
            r1 = (1, 1, 3)
            r2 = (1, 1, 3)
        # Row 36: S052 (C2, write_confirm): cancel KK-10002 -> user said "yes please", bot repeats steps with placeholders, NOT cancelled
        elif rid == 36:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 37: S100 (C2, esc): supervisor -> "Please allow me just a moment to connect you...", no ticket/action
        elif rid == 37:
            r1 = (3, 3, 4)
            r2 = (3, 3, 4)
        # Row 38: S039 (C2, read): track refund KK-10020 -> asks if made refund through website or app, leaks {{Order Number}}
        elif rid == 38:
            r1 = (2, 2, 2)
            r2 = (2, 2, 3)
        # Row 39: S014 (C2, info): create account -> generic registration steps, leaks {{Website URL}}
        elif rid == 39:
            r1 = (3, 3, 4)
            r2 = (4, 3, 4)
        # Row 40: S083 (C4, denied): "That order is no longer eligible for cancellation."
        elif rid == 40:
            r1 = (5, 4, 4)
            r2 = (5, 5, 4)
        # Row 41: S023 (C2, info): pending payment -> asks for payment method and reference number instead of 30 min rule
        elif rid == 41:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 42: S021 (C2, info): checkout -> 6 generic steps, leaks {{Website URL}}, {{Customer Support Hours}}
        elif rid == 42:
            r1 = (3, 3, 3)
            r2 = (3, 3, 4)
        # Row 43: S020 (C4, info): support hours -> cites ticket filing hours [contact-hours.md #2], misses operating hours 09:00-18:00
        elif rid == 43:
            r1 = (3, 3, 5)
            r2 = (3, 3, 4)
        # Row 44: S078 (C4, missing_slot): "What is the order ID? It looks like KK-12345."
        elif rid == 44:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 45: S044 (C4, read): "Invoice INV-10041 for KK-10041 totals ₹1,090, including ₹166 GST."
        elif rid == 45:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 46: S058 (C4, write_confirm): "Order KK-10017 is cancelled. Refund RF-10017 for KK-10017 is ₹5,787..." (DB confirmed)
        elif rid == 46:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 47: S007 (C2, info): express shipping -> "range of options... visit website", misses ₹99
        elif rid == 47:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 48: S030 (C2, read): look up KK-10005 -> asks to confirm order number, leaks {{Order Number}}
        elif rid == 48:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 49: S029 (C4, read): "Order KK-10004 is delivered. Its recorded delivery date is 2026-11-18."
        elif rid == 49:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 50: S004 (C2, info): cancellation fee before dispatch -> says "allow me to check", misses "no fee"
        elif rid == 50:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 51: S014 (C4, info): how to create account -> answers verification link expiration [account-creation.md #2]
        elif rid == 51:
            r1 = (3, 2, 5)
            r2 = (2, 2, 4)
        # Row 52: S076 (C4, missing_slot): "What is the order ID? It looks like KK-12345."
        elif rid == 52:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 53: S021 (C4, info): checkout -> "Add items to the cart, check delivery details... [checkout.md #1]"
        elif rid == 53:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 54: S054 (C4, write_confirm): "Order KK-10007 is cancelled. Refund RF-10007 for KK-10007 is ₹4,318..." (DB confirmed)
        elif rid == 54:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 55: S015 (C2, info): link expiry -> asks for account details to check expiry date (policy is fixed 30 mins)
        elif rid == 55:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 56: S094 (C2, esc): speak to manager -> asks for details, no escalation/ticket
        elif rid == 56:
            r1 = (3, 2, 4)
            r2 = (3, 3, 4)
        # Row 57: S018 (C2, info): missing reset email -> gives steps with {{Login Page URL}}, {{Forgot Password}}
        elif rid == 57:
            r1 = (2, 2, 2)
            r2 = (2, 2, 3)
        # Row 58: S015 (C4, info): "Could you clarify what you need help with at KharidiKart?"
        elif rid == 58:
            r1 = (3, 2, 3)
            r2 = (3, 2, 4)
        # Row 59: S080 (C2, missing_slot): refund for KK-10004 -> gives manual steps with {{Order Number}}
        elif rid == 59:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 60: S090 (C4, denied): "A refund already exists for that order. You can ask me to track it."
        elif rid == 60:
            r1 = (5, 5, 5)
            r2 = (5, 4, 4)
        # Row 61: S094 (C4, esc): "I opened ticket TK-162a7f031e for human support."
        elif rid == 61:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 62: S080 (C4, missing_slot): "What is the reason for the refund?"
        elif rid == 62:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 63: S085 (C4, denied): "I couldn't find that order in your account."
        elif rid == 63:
            r1 = (5, 4, 4)
            r2 = (5, 4, 4)
        # Row 64: S005 (C2, info): cancellation fee once shipped -> asks for order number instead of stating shipped orders cannot be cancelled
        elif rid == 64:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 65: S068 (C2, write_confirm): change address KK-10027 -> user said "go ahead", gives DIY steps, address NOT updated
        elif rid == 65:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 66: S052 (C4, write_confirm): "Order KK-10002 is cancelled. Refund RF-10002 for KK-10002 is ₹1,085..." (DB confirmed)
        elif rid == 66:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 67: S001 (C4, info): UPI refund 3-5 working days [refund-timelines.md #1]
        elif rid == 67:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 68: S006 (C2, info): standard shipping cost -> claims ${{Shipping Cost}} per item, 7-10 days
        elif rid == 68:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 69: S098 (C4, esc): "I opened ticket TK-4cac1d84d9 for human support."
        elif rid == 69:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 70: S100 (C4, esc): "I opened ticket TK-1da0e16d85 for human support."
        elif rid == 70:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 71: S070 (C4, write_confirm): "The shipping address for KK-10032 was updated." (DB confirmed)
        elif rid == 71:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 72: S036 (C4, read): "Refund RF-0005 for KK-10005 is ₹11,635. Status is processing. The estimated timeline is 5 working days."
        elif rid == 72:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 73: S028 (C2, read): check status KK-10003 -> asks to confirm order number, leaks {{Order Number}}
        elif rid == 73:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 74: S055 (C2, write_confirm): cancel KK-10011 -> user said "yes", repeated placeholder steps, NOT cancelled
        elif rid == 74:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 75: S072 (C4, missing_slot): "What is the order ID? It looks like KK-12345."
        elif rid == 75:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 76: S026 (C4, read): "Order KK-10001 is placed. Its recorded delivery date is 2026-11-27."
        elif rid == 76:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 77: S070 (C2, write_confirm): change address KK-10032 -> user said "go ahead", gives DIY steps, NOT updated
        elif rid == 77:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 78: S017 (C4, info): reset link validity -> 15 minutes [password-recovery.md #1]
        elif rid == 78:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 79: S001 (C2, info): UPI refund duration -> claims 7-10 business days (wrong, truth is 3-5 days)
        elif rid == 79:
            r1 = (1, 2, 4)
            r2 = (2, 2, 4)
        # Row 80: S028 (C4, read): "Order KK-10003 is shipped. Its recorded delivery date is 2026-11-27."
        elif rid == 80:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 81: S047 (C4, read): "Invoice INV-10044 for KK-10044 totals ₹8,314, including ₹1,268 GST."
        elif rid == 81:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 82: S029 (C2, read): has order KK-10004 shipped? -> "Let me quickly check... {{Order Number}}... have you received any tracking updates..."
        elif rid == 82:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 83: S069 (C4, write_confirm): "The shipping address for KK-10031 was updated." (DB confirmed)
        elif rid == 83:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 84: S047 (C2, read): send invoice KK-10044 -> claims "quickly retrieve... {{Invoice Number}}", no invoice provided
        elif rid == 84:
            r1 = (1, 1, 3)
            r2 = (1, 1, 3)
        # Row 85: S003 (C2, info): COD refund timeline -> claims 7-10 business days (truth: up to 10 working days)
        elif rid == 85:
            r1 = (2, 2, 4)
            r2 = (2, 2, 4)
        # Row 86: S005 (C4, info): cancellation fee shipped -> shipped orders cannot be cancelled... [cancellation.md #2]
        elif rid == 86:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 87: S069 (C2, write_confirm): change address KK-10031 -> user said "go ahead", gives DIY steps, NOT updated
        elif rid == 87:
            r1 = (1, 1, 2)
            r2 = (1, 1, 2)
        # Row 88: S004 (C4, info): cancellation fee before dispatch -> no fee... [cancellation.md #1]
        elif rid == 88:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 89: S036 (C2, read): track refund RF-0005 -> asks if return or refund, leaks {{Order Number}}
        elif rid == 89:
            r1 = (2, 2, 2)
            r2 = (2, 2, 3)
        # Row 90: S076 (C2, missing_slot): find GST receipt -> gives DIY steps for website Billing section
        elif rid == 90:
            r1 = (3, 3, 4)
            r2 = (3, 3, 4)
        # Row 91: S068 (C4, write_confirm): "The shipping address for KK-10027 was updated." (DB confirmed)
        elif rid == 91:
            r1 = (5, 5, 5)
            r2 = (5, 5, 4)
        # Row 92: S026 (C2, read): where is KK-10001 -> asks to confirm order number, leaks {{Order Number}}
        elif rid == 92:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 93: S025 (C4, info): "Could you clarify which KharidiKart policy you mean?"
        elif rid == 93:
            r1 = (3, 2, 3)
            r2 = (3, 2, 4)
        # Row 94: S063 (C2, write_confirm): refund for KK-10024 -> user said "confirm", bot says "allow me a moment...", refund NOT initiated
        elif rid == 94:
            r1 = (1, 1, 3)
            r2 = (1, 2, 3)
        # Row 95: S030 (C4, read): "Order KK-10005 is cancelled."
        elif rid == 95:
            r1 = (5, 5, 4)
            r2 = (5, 4, 4)
        # Row 96: S083 (C2, denied): cancel KK-10005 -> gives DIY steps with {{Online Company Portal Info}}, ignores cancelled status
        elif rid == 96:
            r1 = (1, 1, 2)
            r2 = (2, 1, 3)
        # Row 97: S032 (C2, read): track KK-10007 -> leaks KK-{{Tracking Number}}, asks for name on shipping label
        elif rid == 97:
            r1 = (2, 2, 2)
            r2 = (2, 2, 3)
        # Row 98: S090 (C2, denied): refund for KK-10005 -> asks irrelevant questions, leaks {{Order Number}}
        elif rid == 98:
            r1 = (2, 2, 3)
            r2 = (2, 2, 3)
        # Row 99: S018 (C4, info): reset password email missing -> 2 minutes [password-recovery.md #2]
        elif rid == 99:
            r1 = (5, 5, 5)
            r2 = (5, 5, 5)
        # Row 100: S020 (C2, info): support hours -> leaks {{Customer Support Hours}}, no actual hours given
        elif rid == 100:
            r1 = (1, 1, 2)
            r2 = (1, 1, 3)
        else:
            raise ValueError(f"Unhandled row {rid}")

        r1_row = dict(r)
        r1_row['correctness'] = r1[0]
        r1_row['helpfulness'] = r1[1]
        r1_row['tone'] = r1[2]
        r1_scores.append(r1_row)

        r2_row = dict(r)
        r2_row['correctness'] = r2[0]
        r2_row['helpfulness'] = r2[1]
        r2_row['tone'] = r2[2]
        r2_scores.append(r2_row)

    return r1_scores, r2_scores

if __name__ == '__main__':
    from pathlib import Path
    
    r1, r2 = score_all()
    print(f"Scored {len(r1)} rows for Rater 1 and {len(r2)} rows for Rater 2.")

    fieldnames = ['response_id', 'scenario', 'query', 'response', 'correctness', 'helpfulness', 'tone']

    # Root outputs
    with open('rater1.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r1)

    with open('rater2.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r2)

    # rating/ folder outputs
    rating_dir = Path('rating')
    rating_dir.mkdir(parents=True, exist_ok=True)

    with open(rating_dir / 'rating2.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r2)

    with open(rating_dir / 'rater2.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r2)

    with open(rating_dir / 'rater1.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(r1)

    print("Files written successfully:")
    print(" - rater1.csv")
    print(" - rater2.csv")
    print(" - rating/rating2.csv")
    print(" - rating/rater2.csv")
    print(" - rating/rater1.csv")

