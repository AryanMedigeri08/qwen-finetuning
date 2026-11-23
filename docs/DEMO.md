# Demo walkthrough and presentation outline

Use a fresh SQLite database. Show the runtime label in the sidebar so the audience can identify the model-backed or deterministic mode.

| Time | On screen | Discussion |
| --- | --- | --- |
| 0:00–0:40 | About and the architecture diagram | Define the support problem. Explain why a QLoRA response can contain placeholders and why a shop needs verified order and policy data. |
| 0:40–1:30 | Chat as C0001 | Ask how long a UPI refund takes. Open the citation and show the exact source sentence. |
| 1:30–2:30 | Chat and Database | Track KK-10001. Ask to cancel it. Open Database before confirmation, then approve and show the updated order and refund. |
| 2:30–3:20 | Trace viewer | Show the tool arguments, the database result, Router and Verifier output, and time for the last turn. |
| 3:20–4:00 | Chat | Try to look up KK-10002 as C0001. Explain the ownership condition and the limited information returned. |
| 4:00–5:00 | Experiments | Present C0–C2 language scores and E5 retrieval. If matched scenario runs are available, show C2/C4/C5 together with their device and precision. |

Suggested slide sequence: research question, dataset and training, architecture, live demonstration, results and limitations. Use one graph or table per results slide and attribute the Bitext dataset.

## Human ratings

Ask two independent raters to score each anonymised response from 1 to 5 on correctness, helpfulness and tone. Give them the reference policy or the scripted scenario's true state. A score of 1 indicates a serious error or poor response, 3 indicates a usable but incomplete response, and 5 indicates an accurate, actionable and professional response. Let raters score on their own before comparing ratings.

## Contribution record

Each student records their initial, task ID, summary, commit link, and verification evidence. Use the actual commit history when preparing the individual viva.
