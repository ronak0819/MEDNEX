# MEDNEX

MEDNEX is a local, deterministic health-information chat demonstration. Its
custom HTML/CSS/JavaScript interface sends messages to a small Python server.
The interface offers a general-information chat and a separate reasoning-agent
mode for explicitly fictional or synthetic case descriptions.

> **Important:** MEDNEX is not a medical service and is not suitable for real
> patient decisions. It cannot diagnose, assess an individual, or recommend
> medicines or dosages. Its content is limited general educational
> information, not a clinical guideline. For personal health concerns, contact
> a qualified healthcare professional; urgent warning signs require urgent
> local help. Do not enter identifying or sensitive information.

## What this project demonstrates

The chat is designed to feel conversational and to provide a readable response
instead of exposing probability tables. It can:

- Recognize a finite, explicit list of common symptom phrases.
- Match everyday wording such as “cold,” “stomach pain,” and “throat pain,”
  plus a small tested list of common misspellings for supported symptoms.
- Preserve explicit negation instead of treating a denied symptom as present.
- Give a small set of non-medicine comfort suggestions for supported symptoms.
- Explain when to contact a health professional and identify selected urgent
  warning phrases.
- Ask for clarification when it cannot match the message to supported content.

It uses deterministic templates and bundled guidance data. It does not use or
pretend to be an LLM. The graph, symbolic, Bayesian, and classical ML engines
are used only by the separate Reasoning Agent mode described below; they do
not generate the general-information chat response.

## Architecture

```text
Browser: web/index.html + styles.css + app.js
    -> local Python standard-library HTTP server (app.py)
       -> /api/chat: parser + safety checks + curated guidance
       -> /api/agent/start and /api/agent/answer:
          MEDNEXAgent -> validation -> graph + symbolic + Bayesian + ML
          + information gain -> planner -> explainability report
```

The browser uses no external font, analytics, model, or API service. By
default, the server binds to `127.0.0.1`, and the UI makes requests only to the
same local server.

## Reasoning Agent mode

Select **Reasoning Agent** in the web interface and enter a case that
explicitly includes the word “fictional” or “synthetic.” The local server
creates a session with the existing `MEDNEXAgent`; the agent parses and
validates the case, then runs the existing knowledge graph, symbolic,
Bayesian, ML (when at least one supported feature is present and
unambiguous), information-gain, planner, and explainability components. It
may ask the single planner-selected follow-up question and reruns the agent
after an answer. The agent runs its existing educational engines internally,
but the interface turns the parsed symptoms and curated guidance into one
plain-English answer. It does not expose graph links, rule names, probability
values, or other engine diagnostics in the user-facing chat.

This mode does not turn the toy engines into medical tools. Unsupported,
ambiguous, and contradictory inputs remain uncertain; safety boundaries stop
evaluation. No candidate is a confirmed diagnosis. The agent's in-memory web
sessions are temporary, held locally, and capped at 128 sessions. The existing
general-information chat remains available as a separate mode.

## Technology

- Python 3.10+
- Standard-library `http.server` for local serving and the JSON endpoint
- Pydantic for validated response and guidance data
- Regular expressions and deterministic phrase normalization
- HTML, CSS, and vanilla JavaScript
- pytest for parser, response, and local HTTP integration tests

Streamlit has been removed from the project dependencies and app entry point.
No Gradle is used.

## Setup

From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
python -m pytest
```

## Run

```powershell
python app.py
```

Open the local address printed by the server (by default
`http://127.0.0.1:8000`). Stop the server with Ctrl+C.

The server always binds to loopback (`127.0.0.1`). Set `MEDNEX_PORT` to change
the default port. The app is not configured for remote or production
deployment.

## Example

Enter:

```text
I have a cough.
```

MEDNEX responds in a conversational format with general comfort ideas such as
rest, fluids, and a warm (not scalding) drink; it notes that honey must not be
given to children under one year. It also lists selected warning signs and
suggests contacting a healthcare professional if symptoms worsen, persist, or
cause concern. It does not identify a cause or recommend a medicine.

## Dataset

[`mednex/data/general_guidance.json`](./mednex/data/general_guidance.json)
contains ten hand-curated symptom topics. Each has deterministic aliases,
comfort suggestions, seek-help language, and selected warning signs. This
small set is intentionally explicit and reviewable. It is not a comprehensive
medical dataset. Spelling handling is also explicit and limited to tested
variants; it is not a general-purpose spell checker.

The project does not fabricate 5,000 duplicated or machine-expanded health
records. A large unreviewed set could make unsupported or unsafe information
look authoritative. Any future expansion should have reliable source
provenance, clinical review, licensing review, and tests for each entry.

## Output

The local JSON response includes:

- `status`: general information, clarification needed, or safety boundary.
- `response`: conversational text shown in the chat.
- `understood`: supported symptoms explicitly found in the message.
- `comfort_steps`, `when_to_seek_help`, and `urgent_signs`: the matched
  general-information sections.
- `disclaimer`: the scope and non-diagnostic warning.

## Safety and limitations

- The general-information response is rule-based and may fail to recognize
  different wording, miss context, or misunderstand a negation. It is not an
  emergency detector.
- It does not know a person's medical history, age unless explicitly parsed,
  allergies, pregnancy status, medicines, or other relevant context.
- Comfort ideas are not individualized care. If symptoms are concerning,
  persistent, or worsening, consult a qualified healthcare professional.
- Urgent phrase matching is a conservative boundary, not triage; it can
  over-trigger or miss unrecognized emergency language.
- The guidance set and reasoning datasets are small synthetic educational
  examples, not clinical guidelines. No clinical validation or production
  privacy/security assessment has been performed.
- The API does not store chat history intentionally, but this prototype is not
  independently audited for production privacy or security.

## Future work

- Have qualified reviewers and legal/licensing review approve every additional
  guidance entry before it is added.
- Expand aliases and negation handling with explicit tests and human review.
- Add accessibility testing, keyboard/screen-reader usability checks, and
  cross-browser verification.
- Add deployment hardening only if the project moves beyond a local classroom
  demonstration.
