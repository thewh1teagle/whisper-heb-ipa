import evaluate


def compute_metrics(pred, tokenizer):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # convert -100 (ignored) tokens into pad tokens for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    wer_score = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer_score = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score, "cer": cer_score}
