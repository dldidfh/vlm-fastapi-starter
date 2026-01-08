
def category_prompt() -> str:
    return (
        "Classify the scene category from the given images. "
        "Return only one short category label."
    )


def summarize_prompt(category: str) -> str:
    return (
        "Summarize the sequence of images in Korean. "
        f"Category: {category}. "
        "Return 2-3 sentences."
    )


def motion_recognition_prompt() -> str:
    return (
        "Recognize the main action in the image sequence. "
        "Return a short phrase."
    )


def object_detection_prompt() -> str:
    return (
        "List key objects visible in the image. "
        "Return a comma-separated list."
    )
