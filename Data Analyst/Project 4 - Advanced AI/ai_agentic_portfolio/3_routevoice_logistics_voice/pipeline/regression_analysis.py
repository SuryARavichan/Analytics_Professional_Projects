"""
regression_analysis.py — a logistic regression of call containment on
the three call-level telemetry signals unique to voice AI (transcription
confidence, interruption count, response latency), answering a more
precise question than "which region is worse": WHICH of these mechanics
actually drives whether a call gets contained, holding the others fixed?
This is what turns "Appalachian is worse" into "Appalachian is worse
because transcription confidence is lower there, and confidence is the
dominant driver of containment" -- an actionable mechanism, not just a
correlation.
"""
import pandas as pd
import statsmodels.api as sm


def fit_containment_model(calls: pd.DataFrame):
    df = calls.copy()
    df["contained"] = (~df.escalated_to_human).astype(int)
    X = df[["transcription_confidence", "n_interruptions", "avg_response_latency_ms"]].copy()
    X["avg_response_latency_ms"] = X["avg_response_latency_ms"] / 1000.0  # rescale to seconds for interpretable coefficients
    X = sm.add_constant(X)
    y = df["contained"]
    model = sm.Logit(y, X).fit(disp=0)
    return model


def standardized_effect_summary(model) -> pd.DataFrame:
    """Odds-ratio view of the fitted model's coefficients, easier to
    communicate than raw logit coefficients: exp(coef) is the
    multiplicative change in the odds of containment per one-unit
    increase in the predictor."""
    params = model.params
    conf = model.conf_int()
    out = pd.DataFrame({
        "coef": params, "odds_ratio": params.apply(lambda b: round(2.718281828 ** b, 3)),
        "p_value": model.pvalues, "ci_low": conf[0], "ci_high": conf[1],
    })
    return out.round(4)
