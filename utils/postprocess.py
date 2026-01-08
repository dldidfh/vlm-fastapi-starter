
def parse_category(text: str) -> str:
    if not text:
        return "unknown"
    cleaned = text.strip().splitlines()[0]
    return cleaned.strip()
