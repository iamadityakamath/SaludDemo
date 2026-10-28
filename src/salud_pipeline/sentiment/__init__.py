"""Turn-level call sentiment, in the style of Google CCAI Insights.

Each caller and agent turn gets a score in [-1, 1] (P(positive) - P(negative)) and a magnitude in [0, 1]
(1 - P(neutral)) from cardiffnlp/twitter-roberta-base-sentiment-latest (int8 ONNX), run in-process on a laptop
or the website, so transcripts are never sent to a sentiment API. Turn scores roll up into call-level
numbers: overall, start, end, shift, lowest turn. Results live in phase1.turn_sentiment / call_sentiment.

    salud-sentiment run     # score calls not scored yet -> team database
    salud-sentiment eval    # call-level labels vs planted labels, raters' tags and the LLM (agreement only)
"""
