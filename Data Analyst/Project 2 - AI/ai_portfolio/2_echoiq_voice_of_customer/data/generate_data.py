"""
Generates a synthetic Voice-of-Customer dataset for "PulseFit," a
fictional fitness-tracking subscription app: ~5,200 app-store reviews and
~1,400 support tickets over 14 months (Nov 2024-Dec 2025).

Text is generated from theme + sentiment-conditioned phrase banks (not an
LLM) -- randomized combinations of realistic opening/body/closing phrases
per theme, so the corpus has genuine, discoverable thematic structure for
unsupervised clustering to find, rather than being hand-labeled.

Story baked in (for EchoIQ to discover, not read off a label):
  - A bad app update ships in month 9 (Jul 2025), causing a spike in
    "crashes/bugs" + "sync issues" themed reviews and a visible dip in
    average rating and sentiment for ~6 weeks, recovering after a patch.
  - Star rating and text sentiment usually agree but not always (a
    deliberate ~12% "surprise" rate: a low-star review with a genuinely
    positive-sounding compliment tucked in, or a high-star review that
    still contains a real complaint) -- this is what makes validating
    sentiment-vs-rating a non-trivial, worthwhile analytical step rather
    than a formality.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(33)

START = datetime(2024, 11, 1)
END = datetime(2025, 12, 31)
INCIDENT_START = datetime(2025, 7, 5)
INCIDENT_END = datetime(2025, 8, 20)

# ---------------------------------------------------------------------
# Phrase banks: theme -> sentiment -> list of sentence fragments
# ---------------------------------------------------------------------
THEMES = {
    "workout_results": {
        "pos": [
            "I've lost 12 pounds since I started using this app three months ago.",
            "The workout plans actually work, I can see real progress every week.",
            "Best fitness app I've tried, my strength has improved so much.",
            "The personalized training plan pushed me way past my old plateau.",
        ],
        "neg": [
            "The workout plans feel generic and haven't helped me progress at all.",
            "I followed the program for two months with zero results.",
            "The exercise recommendations don't adjust to my actual fitness level.",
        ],
    },
    "usability": {
        "pos": [
            "The app is so intuitive, I figured out everything in five minutes.",
            "Clean interface, logging a workout takes seconds.",
            "Love how easy it is to navigate between workouts and progress charts.",
        ],
        "neg": [
            "The navigation is so confusing, I can never find my workout history.",
            "Way too many taps just to log a single exercise.",
            "The interface redesign made everything harder to find.",
        ],
    },
    "community": {
        "pos": [
            "The community challenges keep me motivated every single week.",
            "Love connecting with other members, it's like having a workout buddy.",
            "The leaderboard feature makes workouts genuinely fun and competitive.",
        ],
        "neg": [
            "The community feature feels dead, nobody posts anything anymore.",
            "Wish there were more active groups for my specific goals.",
        ],
    },
    "pricing": {
        "pos": [
            "Worth every penny for the value you get.",
            "The subscription price is fair compared to a real gym membership.",
        ],
        "neg": [
            "The subscription price keeps going up and the app isn't improving.",
            "Way too expensive for what's basically a workout timer.",
            "Cancelled after the price hike, not worth $25 a month.",
            "They snuck in a price increase without clearly telling existing subscribers.",
        ],
    },
    "crashes_bugs": {
        "pos": [
            "Runs smoothly on my phone, no complaints on stability.",
        ],
        "neg": [
            "The app crashes every single time I try to start a workout.",
            "Constant freezing since the last update, completely unusable.",
            "It crashed mid-workout and I lost my whole session log.",
            "App keeps force-closing on the home screen, had to reinstall twice.",
            "Since the July update the app is basically broken, crashes constantly.",
        ],
    },
    "sync_issues": {
        "pos": [
            "Syncs perfectly with my smartwatch every time.",
        ],
        "neg": [
            "My watch data won't sync anymore, heart rate is always missing.",
            "Lost a week of step data because syncing silently failed.",
            "Wearable integration has been broken since the recent update.",
            "Sync keeps timing out and I have to manually re-link my device.",
        ],
    },
    "support": {
        "pos": [
            "Support responded within an hour and fixed my billing issue right away.",
            "Customer service was genuinely helpful and patient with my questions.",
        ],
        "neg": [
            "Contacted support three times about a billing issue with no response.",
            "Customer service just sends copy-paste replies that don't address the problem.",
            "Still waiting on a refund they promised two weeks ago.",
        ],
    },
    "feature_request": {
        "pos": [
            "Would love a dark mode option but otherwise really happy with it.",
        ],
        "neg": [
            "Really needs a way to export workout data to a spreadsheet.",
            "No option to customize rest timers, which is a basic feature at this point.",
            "Missing integration with my other health apps, hoping this gets added soon.",
        ],
    },
}

OPENERS = ["", "Honestly, ", "Overall, ", "So far, ", "Update: ", "After a few months, "]
PLATFORMS = ["iOS", "Android"]


def make_review(date, theme, sentiment):
    frag = rng.choice(THEMES[theme][sentiment])
    opener = rng.choice(OPENERS)
    text = f"{opener}{frag}"
    if rng.random() < 0.4:
        # occasionally tack on a second, sometimes cross-theme fragment for realism
        other_theme = rng.choice(list(THEMES.keys()))
        other_sent = sentiment if rng.random() < 0.7 else rng.choice(["pos", "neg"])
        if THEMES[other_theme].get(other_sent):
            text += " " + rng.choice(THEMES[other_theme][other_sent])
    return text


def sample_date(is_incident_biased):
    if is_incident_biased:
        span = (INCIDENT_END - INCIDENT_START).days
        return INCIDENT_START + timedelta(days=int(rng.integers(0, span)))
    span = (END - START).days
    return START + timedelta(days=int(rng.integers(0, span)))


# ---------------------------------------------------------------------
# Generate reviews
# ---------------------------------------------------------------------
N_REVIEWS = 5200
rows = []
for i in range(N_REVIEWS):
    # ~22% of reviews land in the 6-week incident window (oversampled vs. its
    # ~11% share of the calendar, simulating a real review-volume spike),
    # and incident-window reviews skew heavily toward crashes/sync/negative
    in_incident = rng.random() < 0.22
    date = sample_date(in_incident)

    if in_incident and rng.random() < 0.55:
        theme = rng.choice(["crashes_bugs", "sync_issues"], p=[0.6, 0.4])
        sentiment = "neg"
    else:
        theme = rng.choice(list(THEMES.keys()),
                            p=[0.16, 0.14, 0.10, 0.13, 0.14, 0.11, 0.12, 0.10])
        sentiment = rng.choice(["pos", "neg"], p=[0.62, 0.38])
        if not THEMES[theme].get(sentiment):
            sentiment = "pos" if THEMES[theme].get("pos") else "neg"

    text = make_review(date, theme, sentiment)

    # base rating from sentiment/theme, with a deliberate ~12% "surprise" mismatch
    base_rating = {"pos": rng.choice([4, 5], p=[0.35, 0.65]), "neg": rng.choice([1, 2], p=[0.55, 0.45])}[sentiment]
    if rng.random() < 0.12:
        base_rating = 6 - base_rating if base_rating <= 2 else rng.choice([1, 2])
        base_rating = int(np.clip(base_rating, 1, 5))
    if in_incident and theme in ("crashes_bugs", "sync_issues"):
        base_rating = min(base_rating, 2)

    rows.append({
        "review_id": f"REV{10000+i}", "date": date.date(), "platform": rng.choice(PLATFORMS, p=[0.55, 0.45]),
        "rating": int(base_rating), "review_text": text, "true_theme": theme,
    })

reviews = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ---------------------------------------------------------------------
# Generate support tickets (smaller corpus, skews negative/operational)
# ---------------------------------------------------------------------
N_TICKETS = 1400
ticket_rows = []
for i in range(N_TICKETS):
    in_incident = rng.random() < 0.30
    date = sample_date(in_incident)
    if in_incident and rng.random() < 0.6:
        theme = rng.choice(["crashes_bugs", "sync_issues"])
    else:
        theme = rng.choice(["support", "pricing", "crashes_bugs", "sync_issues", "feature_request"],
                            p=[0.22, 0.18, 0.24, 0.20, 0.16])
    text = make_review(date, theme, "neg") if THEMES[theme].get("neg") else make_review(date, theme, "pos")
    resolved = rng.random() < (0.65 if not in_incident else 0.45)
    ticket_rows.append({
        "ticket_id": f"TICK{50000+i}", "date": date.date(), "category": theme,
        "ticket_text": text, "resolved": int(resolved),
    })
tickets = pd.DataFrame(ticket_rows).sort_values("date").reset_index(drop=True)

print(f"reviews={len(reviews)}  avg_rating={reviews.rating.mean():.2f}  tickets={len(tickets)}")
reviews.drop(columns=["true_theme"]).to_csv("app_reviews.csv", index=False)
reviews.to_csv("app_reviews_with_true_theme.csv", index=False)  # kept for internal validation only
tickets.to_csv("support_tickets.csv", index=False)
print("Done.")
