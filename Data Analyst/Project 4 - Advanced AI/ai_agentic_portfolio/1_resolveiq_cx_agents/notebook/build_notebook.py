import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md(r"""
# ResolveIQ — Autonomous CX Agent Performance & Containment Analytics

**What this is:** an AgentOps analytics system for Ravel AI, a fictional
autonomous AI customer-support-agent platform — the "agentic AI" trend
where an LLM agent doesn't just answer a question, it takes multi-step
actions (looks up an order, issues a refund, resets a password) and
closes the ticket without a human. This notebook answers the question
every agentic-AI product now has to answer that a simple chatbot never
did: **when the agent reports a ticket "resolved," did it actually solve
the customer's problem, or did it just close the conversation?**

**The finding this uncovers:** Ravel ships an Agent v2 release that lets
the account_recovery agent take autonomous account actions instead of
always escalating. The agent's own reported containment rate for
account_recovery jumps **+20.2 percentage points** — the best-looking
number in the whole release. But once containment is adjusted for
tickets that get **reopened within 48 hours** (a proxy for "we didn't
actually fix it"), the true containment rate is **completely flat**
(50.2% → 50.2%, not remotely significant) — the entire headline win is
explained by a reopen rate that exploded from 9.8% to 33.9%. Billing and
technical, by contrast, show genuine, reopen-corroborated improvements
from the same release. Telling these apart — not just reading the
headline containment number — is the whole point of this analysis.

**On the LLM backend:** same cached/live design as the rest of this
session's AI-augmented projects — see `../shared/llm_client.py`.
""")

code(r"""
import sys, json
sys.path.insert(0, "../../shared")
sys.path.insert(0, "../pipeline")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from containment import containment_summary, containment_shift_test, classify_policy_change
from economics import cost_by_path, cost_by_category_version, reopen_cost_impact
from agent import review_category
from llm_client import get_client

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

tickets = pd.read_csv("../data/tickets.csv", parse_dates=["timestamp"])
tickets["reopened_within_48h"] = tickets["reopened_within_48h"].astype(bool)
client = get_client(cache_file="../cached_responses.json")

print(f"{len(tickets):,} tickets | {tickets.timestamp.min()} to {tickets.timestamp.max()} | "
      f"deploy at 2026-07-01 (v1 -> v2)")
tickets.head()
""")

md("## 1. Official vs. True Containment, by Category and Version")

code(r"""
rows = []
for version in ["v1", "v2"]:
    sub = tickets[tickets.agent_version == version]
    s = containment_summary(sub, groupby="category")
    s["agent_version"] = version
    rows.append(s.reset_index())
summary = pd.concat(rows).sort_values(["category", "agent_version"])
summary
""")

code(r"""
categories = ["billing", "technical", "shipping", "account_recovery"]
fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharey=True)
for ax, cat in zip(axes, categories):
    sub = summary[summary.category == cat]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, sub.official_containment, width, label="official", color="#2E86AB")
    ax.bar(x + width/2, sub.true_containment, width, label="true (reopen-adj.)", color="#C0392B")
    ax.set_xticks(x); ax.set_xticklabels(["v1", "v2"])
    ax.set_title(cat, fontsize=10)
    if cat == "billing":
        ax.set_ylabel("Containment rate")
        ax.legend(fontsize=8)
fig.suptitle("Official vs. True Containment: account_recovery Is the Outlier", y=1.03)
plt.tight_layout()
plt.savefig("../dashboard/containment_by_category.png", dpi=150)
plt.show()
""")

md(r"""billing and technical show official and true containment moving
**together** — a real improvement. `account_recovery`'s official bar
jumps dramatically from v1 to v2, but its true-containment bar barely
moves at all. That gap between the two bars *is* the finding.""")

md("## 2. Statistical Tests — Official vs. True, Side by Side")

code(r"""
results = []
for cat in categories:
    off = containment_shift_test(tickets, cat, metric="official")
    true = containment_shift_test(tickets, cat, metric="true")
    verdict = classify_policy_change(off, true)
    results.append({
        "category": cat,
        "official_lift_pp": off.lift_pp, "official_p_value": off.p_value, "official_significant": off.significant,
        "true_lift_pp": true.lift_pp, "true_p_value": true.p_value, "true_significant": true.significant,
        "verdict": verdict,
    })
results_df = pd.DataFrame(results)
results_df
""")

md(r"""`account_recovery` is the only category where `official_significant`
is True but `true_significant` is False — the exact statistical signature
of a metric being gamed by a policy change rather than genuinely
improved by one. `shipping` correctly shows no significant change on
either metric, acting as a useful control: the v2 release's real effects
are isolated to specific categories, not a broad seasonal shift.""")

md("## 3. Does the 'Win' Even Save Money?")

code(r"""
cost_summary = cost_by_category_version(tickets)
cost_summary.loc["account_recovery"]
""")

code(r"""
ar_cost = reopen_cost_impact(tickets, "account_recovery")
pd.DataFrame(ar_cost).T
""")

code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
versions = ["v1", "v2"]
avg_cost = [ar_cost[v]["avg_cost_all"] for v in versions]
reopen_rate = [ar_cost[v]["reopen_rate"] * 100 for v in versions]
ax2 = ax.twinx()
ax.bar(versions, avg_cost, color="#2E86AB", alpha=0.7, width=0.4, label="avg cost/ticket ($)")
ax2.plot(versions, reopen_rate, color="#C0392B", marker="o", linewidth=2.5, markersize=10, label="reopen rate (%)")
ax.set_ylabel("Avg cost per ticket ($)", color="#2E86AB")
ax2.set_ylabel("Reopen rate (%)", color="#C0392B")
ax.set_title("account_recovery: Cost Is Flat, Reopen Rate Tripled")
plt.tight_layout()
plt.savefig("../dashboard/account_recovery_cost.png", dpi=150)
plt.show()
""")

md(r"""Average cost per account_recovery ticket is roughly flat
($3.89 → $3.65) even as the reopen rate more than tripled — the fewer
upfront human escalations are almost exactly offset by the cost of
handling reopened tickets later. **By both the true-quality lens and the
cost lens, this policy change delivered nothing** — which makes the
official +20.2pp containment number actively misleading if reported on
its own.""")

md("## 4. Agent Release-Review Reports")

code(r"""
reports = {cat: review_category(tickets, cat, client) for cat in categories}
for cat in categories:
    print(f"=== {cat} ===")
    print(reports[cat].narrative)
    print()
""")

md("## 5. Summary")

code(r"""
print("=== ResolveIQ summary ===")
for cat in categories:
    r = reports[cat]
    print(f"{cat:20s} -> {r.verdict}")
print()
print("The one finding that matters most: account_recovery's official containment metric improved")
print("+20.2pp (p<0.001) while its TRUE, reopen-adjusted containment was completely flat (p=0.98) —")
print("a textbook case of an AI agent's self-reported success metric being gamed by a policy change,")
print("caught only by deliberately validating the metric against real downstream customer behavior.")
""")

nb["cells"] = cells
nbf.write(nb, "analysis.ipynb")
print("Notebook written.")
