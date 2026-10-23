"""Turn-level call sentiment, in the style of Google CCAI Insights.

Each caller and agent turn gets a score in [-1, 1] (P(positive) - P(negative)) and a magnitude in [0, 1]
(1 - P(neutral)) from cardiffnlp/twitter-roberta-base-sentiment-latest, run locally so real calls never
leave the machine. Turn scores roll up into call-level numbers: overall, start, end, shift, lowest turn.

    salud-sentiment run     # score every call in the team database -> reports/sentiment/
    salud-sentiment eval    # call-level labels vs raters' sentiment tags (agreement only)

Needs the optional install: pip install -e '.[sentiment]'.
"""
