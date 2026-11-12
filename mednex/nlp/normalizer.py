import re


ALIASES = {
    "high temperature": "fever",
    "difficulty breathing": "shortness of breath",
    "short of breath": "shortness of breath",
    "pulse": "heart rate",
    "spo2": "oxygen saturation",
    "o2 sat": "oxygen saturation",
}


def normalize_text(text: str) -> str:
    normalized = text.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    for alias in sorted(ALIASES, key=len, reverse=True):
        normalized = re.sub(
            rf"\b{re.escape(alias)}\b",
            ALIASES[alias],
            normalized,
        )
    return normalized


def tokenize(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?", normalize_text(text))


def parse_number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        pass

    words = text.lower().replace("-", " ").split()
    values = {
        "zero": 0,
        "one": 1,
        "a": 1,
        "an": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
    }
    if not words or any(word not in values for word in words):
        return None
    if len(words) == 1:
        return float(values[words[0]])
    if words[-1] == "hundred" and len(words) == 2:
        return float(values[words[0]] * 100)
    if len(words) == 2 and values[words[0]] >= 20 and values[words[1]] < 10:
        return float(values[words[0]] + values[words[1]])
    return None
