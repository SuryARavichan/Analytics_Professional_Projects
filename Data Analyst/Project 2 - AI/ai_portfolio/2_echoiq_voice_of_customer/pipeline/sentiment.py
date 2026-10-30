"""
sentiment.py — lexicon-based sentiment scoring via VADER (Valence Aware
Dictionary and sEntiment Reasoner), a real, widely-used, fully offline
NLP tool (no model download, no API) well suited to short, informal text
like app reviews and support tickets. Chosen deliberately over a heavier
transformer sentiment model since it runs instantly on CPU with zero
dependencies beyond the `vaderSentiment` package, and its compound score
gives a continuous -1..+1 signal (not just a 3-class label) that's more
useful for trend analysis over time.
"""
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> dict:
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return {"sentiment_compound": compound, "sentiment_label": label}


def score_dataframe(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    scored = df[text_col].apply(score_text).apply(pd.Series)
    return pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
