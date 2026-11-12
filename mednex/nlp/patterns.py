import re

SYMPTOM_ALIASES = {
    "shortness of breath": (
        "shortness of breath",
        "difficulty breathing",
        "short of breath",
        "breathlessness",
        "shortness breath",
    ),
    "abdominal pain": (
        "abdominal pain",
        "stomach pain",
        "stomach ache",
        "stomachache",
        "tummy ache",
        "belly pain",
        "stomack pain",
        "stomache pain",
    ),
    "sore throat": (
        "sore throat",
        "throat pain",
        "throat ache",
        "sore throte",
    ),
    "chest pain": ("chest pain",),
    "fever": ("fever", "high temperature", "feaver"),
    "cold": ("cold", "common cold", "colld"),
    "cough": ("cough", "coughing", "caugh", "couph"),
    "headache": ("headache", "head ache", "head pain", "headake", "hedache"),
    "fatigue": ("fatigue", "tiredness"),
    "nausea": ("nausea", "feeling nauseated", "feeling sick"),
    "vomiting": ("vomiting", "vomit"),
    "diarrhea": ("diarrhea", "diarrhoea"),
    "dizziness": ("dizziness", "dizzy"),
    "rash": ("rash",),
    "chills": ("chills",),
    "wheezing": ("wheezing",),
    "congestion": (
        "congestion",
        "stuffy nose",
        "blocked nose",
        "runny nose",
        "nose running",
    ),
}

SYMPTOM_PATTERNS = [
    (
        canonical,
        re.compile(
            rf"\b(?:{'|'.join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))})\b",
            re.IGNORECASE,
        ),
    )
    for canonical, aliases in SYMPTOM_ALIASES.items()
]
SYMPTOM_PATTERNS.sort(key=lambda item: max(map(len, SYMPTOM_ALIASES[item[0]])), reverse=True)

NUMBER_WORDS = (
    r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|"
    r"eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
)
DURATION_PATTERN = re.compile(
    rf"\b(?P<amount>\d+(?:\.\d+)?|{NUMBER_WORDS})\s+"
    r"(?P<unit>minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?)\b",
    re.IGNORECASE,
)

TEMPERATURE_PATTERN = re.compile(
    r"\b(?:temperature|temp|fever)\s*(?:is|of|at|was|=|:)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>degrees?\s*(?:celsius|centigrade|fahrenheit|[cf])|"
    r"°\s*[cf]|celsius|centigrade|fahrenheit|[cf])\b",
    re.IGNORECASE,
)
HEART_RATE_PATTERN = re.compile(
    r"\b(?:heart\s+rate|pulse)\s*(?:is|of|at|was|=|:)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>bpm|beats?\s*(?:per|/)\s*minute)?\b",
    re.IGNORECASE,
)
RESPIRATORY_RATE_PATTERN = re.compile(
    r"\b(?:respiratory\s+rate|breathing\s+rate)\s*(?:is|of|at|was|=|:)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>breaths?\s*(?:per|/)\s*minute|/min)?\b",
    re.IGNORECASE,
)
OXYGEN_PATTERN = re.compile(
    r"\b(?:oxygen\s+saturation|oxygen\s+sat(?:uration)?|spo2|o2\s+sat)\s*"
    r"(?:is|of|at|was|=|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent(?:age)?)(?![a-z])",
    re.IGNORECASE,
)
BLOOD_PRESSURE_PATTERN = re.compile(
    r"\b(?:blood\s+pressure|bp)\s*(?:is|of|at|was|=|:)?\s*"
    r"(?P<systolic>\d+(?:\.\d+)?)\s*/\s*(?P<diastolic>\d+(?:\.\d+)?)"
    r"\s*(?:mm\s*hg|mmhg)?\b",
    re.IGNORECASE,
)

HEDGE_PATTERN = re.compile(
    r"\b(?:maybe|perhaps|possibly|might have|may have|could have|"
    r"uncertain(?:ly)?|unclear whether)\b",
    re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(
    r"\b(?:no|without|denies?|negative for|does not have|doesn't have|"
    r"do not have|don't have|did not have|didn't have|not experiencing|"
    r"has not had|have not had)\b[^.!?;]{0,40}$",
    re.IGNORECASE,
)
VITAL_LABELS = {
    "temperature": re.compile(r"\b(?:temperature|temp)\b", re.IGNORECASE),
    "heart_rate": re.compile(r"\b(?:heart\s+rate|pulse)\b", re.IGNORECASE),
    "respiratory_rate": re.compile(
        r"\b(?:respiratory\s+rate|breathing\s+rate)\b", re.IGNORECASE
    ),
    "oxygen_saturation": re.compile(
        r"\b(?:oxygen\s+saturation|oxygen\s+sat(?:uration)?|spo2|o2\s+sat)\b",
        re.IGNORECASE,
    ),
    "blood_pressure": re.compile(r"\b(?:blood\s+pressure|bp)\b", re.IGNORECASE),
}
