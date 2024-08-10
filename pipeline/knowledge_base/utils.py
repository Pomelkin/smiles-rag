def safe_truncate(text: str, start: int, stop: int) -> str:
    """Truncate text with safe bounds"""
    # validate right bound
    stop = stop if stop < len(text) - 1 else len(text) - 1
    start = start if start > 0 else 0

    # find first and last spaces. This need for guarantee that we will not cut word in the middle.
    while (text[start] != " " and start != 0) or (
        text[stop] != " " and stop != len(text) - 1
    ):
        start = start - 1 if text[start] != " " and start != 0 else start
        stop = stop + 1 if text[stop] != " " and stop != len(text) - 1 else stop
    return text[start : stop + 1]
