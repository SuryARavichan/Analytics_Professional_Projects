"""
Executes both the GOLD and PREDICTED SQL for every benchmark question
against the real database and scores execution accuracy: does the
predicted query return the same result set as the gold query (order-
insensitive, floats rounded to 2dp for tolerance)? This is "execution
accuracy," the standard text-to-SQL evaluation metric (matching query
strings would be far too strict -- two differently-written queries can be
equally correct).
"""
import json
import sqlite3
import pandas as pd

DB = "../data/retail.db"


def run(sql):
    conn = sqlite3.connect(DB)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return cols, rows, None
    except Exception as e:
        return [], [], str(e)
    finally:
        conn.close()


def normalize(rows):
    def norm_val(v):
        return round(v, 2) if isinstance(v, float) else v
    return sorted(tuple(norm_val(v) for v in r) for r in rows)


with open("benchmark.json") as f:
    bench = json.load(f)

results = []
for q in bench:
    g_cols, g_rows, g_err = run(q["gold"])
    p_cols, p_rows, p_err = run(q["predicted"])
    correct = (g_err is None and p_err is None and normalize(g_rows) == normalize(p_rows))
    results.append({
        "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
        "correct": correct, "gold_result": g_rows[:5], "predicted_result": p_rows[:5],
        "gold_error": g_err, "predicted_error": p_err,
        "failure_note": q["note"],
    })

df = pd.DataFrame(results)
df.to_csv("eval_results.csv", index=False)

overall_acc = df["correct"].mean()
by_diff = df.groupby("difficulty")["correct"].agg(["mean", "count"])

print(f"Overall execution accuracy: {overall_acc:.1%} ({df.correct.sum()}/{len(df)})")
print("\nBy difficulty:")
print(by_diff)
print("\nFailed questions:")
for _, r in df[~df.correct].iterrows():
    print(f"  [{r['id']}] {r['question']}")
    print(f"      -> {r['failure_note']}")

with open("eval_summary.json", "w") as f:
    json.dump({
        "overall_accuracy": overall_acc,
        "n_questions": len(df),
        "n_correct": int(df.correct.sum()),
        "by_difficulty": by_diff.to_dict(),
    }, f, indent=2, default=str)
