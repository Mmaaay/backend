def tajweed_score(result: dict) -> dict:
    """
    Returns a dict with scores for transcription and each word in the result.
    Scoring:
        -1: transcription wrong
         0: wrong word
         1: right word
         2: right word and tajweed
        -2: tajweed false (any rule in word is incorrect)
    """
    scores = {}

    # Transcription score
    transcription_is_correct = result.get("transcription_is_correct", False)
    scores["transcription"] = 1 if transcription_is_correct else -1

    # Word scores
    word_scores = []
    for word_info in result.get("words", []):
        word_score = 1 if word_info.get("word_is_correct", False) else 0
        tajweed_rules = word_info.get("tajweed_rules", [])
        if word_score == 1 and tajweed_rules:
            # If any tajweed rule is incorrect
            if any(not rule.get("tajweed_is_correct", False) for rule in tajweed_rules):
                word_score = -2
            else:
                word_score = 2  # All tajweed rules correct
        word_scores.append({
            "word": word_info.get("word"),
            "score": word_score
        })
    scores["words"] = word_scores

    return scores


def extract_scores(result: dict, session_id=None, user_id=None, created_at=None, updated_at=None) -> dict:
    """
    Returns a dict with:
      - session_id and user_id if provided
      - created_at and updated_at if provided
      - transcription and its score
      - words and their scores, start_idx, and end_idx
    """
    scores = tajweed_score(result)
    transcription = result.get("transcription", "")
    words_info = result.get("words", [])

    # Pair each word with its score and indices
    word_scores = []
    for word_info, score_info in zip(words_info, scores.get("words", [])):
        word_scores.append({
            "word": word_info.get("word", ""),
            "start_idx": word_info.get("start_idx"),
            "end_idx": word_info.get("end_idx"),
            "score": score_info.get("score")
        })

    summary = {
        "surah_idx": result.get("surah_idx", 1),
        "ayah_idx": result.get("ayah_idx", 1),
        "transcription": transcription,
        "transcription_score": scores.get("transcription"),
        "words": word_scores
    }
    if session_id is not None:
        summary["session_id"] = session_id
    if user_id is not None:
        summary["user_id"] = user_id
    if created_at is not None:
        summary["created_at"] = created_at
    if updated_at is not None:
        summary["updated_at"] = updated_at
    return summary