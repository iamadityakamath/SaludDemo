"""Turn-level call sentiment (CCAI style) from cardiffnlp/twitter-roberta-base-sentiment-latest, run locally as an
int8 ONNX model: each caller and agent turn gets score = P(positive) - P(negative) in [-1, 1] and magnitude =
1 - P(neutral). Transcripts never leave the machine for sentiment."""
