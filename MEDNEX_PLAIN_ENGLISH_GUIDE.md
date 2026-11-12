# MEDNEX: A Plain-English Guide

This guide explains the MEDNEX project without assuming that you already know
Python, artificial intelligence, statistics, or web development. It describes
the files and behavior currently in the project. It is not a medical guide.

> **Important:** MEDNEX is a classroom software demonstration. It is not a
> medical service, has not been clinically validated, and must not be used to
> make real health decisions.

## 1. MEDNEX in one paragraph

MEDNEX is a local health-information demonstration built without a large
language model (LLM). Its web chat uses written matching rules to recognize
some common symptom phrases, checks a small number of safety-related phrases,
looks up matching text in a local JSON file, and assembles a response from
prewritten sentences. The web page also offers a distinct Reasoning Agent mode
that calls the existing agent orchestrator and its educational reasoning
engines for explicitly fictional or synthetic case descriptions.

## 2. What to call it

A careful description is:

> “MEDNEX is a non-LLM, rule-based conversational health-information
> demonstration. It recognizes a limited set of phrases, applies explicit
> safety and matching rules, retrieves locally stored educational text, and
> returns a structured response. Its separate Reasoning Agent mode orchestrates
> existing symbolic, graph, Bayesian, classical ML, and information-gain
> components for fictional cases.”

The general-information chat and the Reasoning Agent are two different
features. The first selects curated general text. The second calls
`MEDNEXAgent`, tracks a follow-up session, and presents one plain-language
response based on the parsed symptoms and curated guidance. Detailed engine
outputs remain internal to the program and are not shown in the chat.

## 3. What happens when someone sends a web-chat message

For example, when a user types “I have a cough”:

1. The browser's JavaScript reads the text.
2. It sends the text to the local Python server as JSON.
3. The server checks that the request has the expected format and size.
4. Python checks selected safety phrases and medication requests.
5. A deterministic parser searches for known phrases and their known aliases.
6. The chat looks for a matching topic in the local guidance file.
7. It combines the topic's prepared comfort information, advice about seeking
   professional help, urgent warning language, and disclaimer into a reply.
8. The server sends the reply back to the browser as JSON.
9. The browser displays the response as text.

There is no model generating new medical text in this process. The wording
comes from the program and the local guidance data.

### What happens in Reasoning Agent mode

1. The user selects **Reasoning Agent** and enters an explicitly fictional or
   synthetic case. The server rejects a case that does not label itself this
   way.
2. The browser sends the text to `/api/agent/start`.
3. The server calls the existing `MEDNEXAgent.start` method and saves the
   returned session temporarily in local memory.
4. The agent runs the existing parser and validator, then the knowledge
   graph, symbolic engine, Bayesian engine, supported-feature ML engine,
   information-gain engine, planner, and explainability report.
5. The server translates that actual agent turn into one short, plain-English
   answer using the parsed symptoms and matching curated guidance. Technical
   graph links, rule names, posterior values, ML labels, and entropy numbers
   are not sent to or displayed by the user-facing interface.
6. If the planner selected a follow-up, the user answers it and the browser
   sends the response and session ID to `/api/agent/answer`.
7. The agent updates its existing session, reruns the cycle, and returns a
   new plain-English response for the updated case.

The server's reasoning sessions are held temporarily in memory, are not
written to disk, and are limited to 128 retained sessions. Unsupported,
ambiguous, contradictory, or safety-boundary cases do not receive a forced
candidate conclusion.

## 4. What the current web chat understands

The guidance library has ten topics:

- Cough
- Fever
- Cold symptoms
- Stomach or abdominal pain
- Sore throat
- Headache
- Nasal congestion
- Nausea
- Diarrhea
- Fatigue

The parser recognizes additional terms, such as vomiting, dizziness, rash,
chills, wheezing, chest pain, and shortness of breath. However, recognition by
the parser does not automatically mean that there is a matching general
guidance entry. Some phrases instead trigger a safety boundary, and some
recognized symptoms have no chat guidance topic. The guidance library—not the
parser's full vocabulary—is what determines whether the chat can produce
topic-specific comfort information.

The program also includes an explicit, limited list of spelling variants.
Examples include “caugh” for cough, “feaver” for fever, “stomack pain” for
stomach pain, “sore throte” for sore throat, “headake” for headache, and
“colld” for cold. This is **not** a general spell-checker. A different typo or
unfamiliar way of describing a symptom may not be recognized.

The text matching aliases live in `mednex/nlp/patterns.py`. The guidance JSON
also has an `aliases` list for each topic, but the current chat chooses
guidance by the canonical symptom name produced by the parser; it does not
use that JSON list as a second independent matching system.

The parser can retain simple negation, such as “no fever” or “does not have a
cough.” If it cannot match a supported topic, it asks for clarification
instead of deliberately inventing topic-specific guidance.

## 5. Technologies and frameworks

- **Python** runs the server and the reasoning code.
- **Pydantic** checks that structured data has the expected fields and types.
- **Python's built-in HTTP server** serves the page and receives chat messages.
  The web server does not use Flask, Django, or Streamlit.
- **HTML** describes the page's content and controls.
- **CSS** controls layout, colors, and responsive presentation.
- **Vanilla JavaScript** means ordinary browser JavaScript without a frontend
  framework. It sends messages and updates the chat display.
- **Regular expressions** are written text patterns used to find known terms,
  numbers, units, and selected safety phrases.
- **JSON** stores the curated chat guidance and the other small demonstration
  datasets in a readable data format.
- **NetworkX** is used by the separate knowledge-graph example.
- **scikit-learn** is used by the separate classical machine-learning example.
- **Matplotlib** is listed as a project dependency for graph visualization
  support.
- **pytest** runs the automated tests.

There is no LLM package, generative API, hosted model, or Gradle build system
in the web-chat flow.

## 6. Important concepts in everyday language

### Deterministic rules

“Deterministic” means that the same input and the same program data produce
the same result. For example, if a phrase matches the cough pattern, the
program selects the cough entry. It does not create an original answer by
guessing what a writer might say.

### Phrase matching and aliases

An alias is another phrase the program treats as the same known item. For
example, the parser can treat “stomach pain” as “abdominal pain.” The lists of
aliases are finite: wording not on the lists can be missed.

### Structured data

Structured data is information placed into named fields rather than kept as
one long sentence. A symptom can have a name, duration, and a marker saying
whether it was negated. This makes information easier for other code to
validate and inspect.

### Knowledge graph

A knowledge graph is a network of named items connected by named
relationships. MEDNEX's separate example links synthetic symptoms,
observations, candidate labels, and question or investigation concepts.
A connection in this toy graph is not proof of a medical relationship.

### Production rule and forward chaining

A production rule is an “if these conditions are true, add this result”
instruction. Forward chaining means the program repeatedly checks rules and
uses newly created flags when checking later rules. MEDNEX's small, invented
rules demonstrate this programming technique; they do not establish diagnoses.

### Prior, likelihood, and posterior

The separate Bayesian example uses three types of numbers:

- A **prior** is the starting weight assigned to a candidate label.
- A **likelihood** is a hand-entered teaching number describing how the toy
  example treats an observation for that label.
- A **posterior** is the recalculated, normalized weight after applying the
  selected likelihoods to the priors.

In plain English: start with invented weights, adjust them using more invented
weights, then rescale the results so they add up to one. The numbers are not
real-world medical risks. The demonstration assumes evidence items are
conditionally independent, a simplifying assumption that is not established
for real cases.

### Classical machine learning

Machine learning is software that fits a model to example rows. MEDNEX's
separate scikit-learn example trains logistic regression, a decision tree,
and a random forest on a tiny synthetic CSV. It holds some rows out for a
basic evaluation and reports common classification measures. Its predictions
and scores reflect only that artificial dataset; they do not show how the
models perform on people.

- **Accuracy** is the fraction of test examples the model labeled correctly.
- **Precision** asks, among examples the model assigned to a label, what
  fraction truly had that label in the test data.
- **Recall** asks, among test examples with a label, what fraction the model
  found.
- **F1** combines precision and recall into one score.
- A **confusion matrix** is a table showing which test labels were correctly
  identified and which labels were mistaken for one another.

### Entropy and information gain

Entropy is a mathematical way to describe how spread out or uncertain a
probability distribution is. Information gain estimates how much a
hypothetical answer might reduce that mathematical uncertainty. MEDNEX uses
invented example probabilities for this demonstration. A high information
gain is not the same as medical importance or urgency.

### Separate evidence sources

The Bayesian and machine-learning examples produce different kinds of
outputs. MEDNEX does not simply average their numbers together. The separate
reporting code keeps those results distinguishable.

## 7. Plain-English file map

### Files used to run the web chat

- [`app.py`](./app.py) — The small starting point. When run directly, it asks
  the local web server to start.
- [`mednex/web_server.py`](./mednex/web_server.py) — Starts a server on this
  computer only, serves the approved web files, and returns JSON for the
  general `/api/chat` endpoint plus the `/api/agent/start` and
  `/api/agent/answer` endpoints. It limits request size, requires explicit
  fictional/synthetic wording for agent cases, stores limited temporary
  sessions, and offers `/health` as a simple server check.
- [`mednex/guidance.py`](./mednex/guidance.py) — Checks message length and
  basic safety boundaries, calls the parser, selects matching guidance, and
  constructs the reply. Its reply object has a status, response text,
  recognized topics, comfort steps, professional-help prompts, urgent signs,
  and a disclaimer.
- [`mednex/data/general_guidance.json`](./mednex/data/general_guidance.json)
  — The ten hand-curated web-chat topics. Each entry supplies known aliases,
  general comfort wording, when to seek professional help, and selected
  urgent warning signs. The file labels itself educational and not a clinical
  guideline.
- [`web/index.html`](./web/index.html) — The page structure: MEDNEX branding,
  warning text, mode tabs, example buttons, message area, and input box.
- [`web/styles.css`](./web/styles.css) — The visual design, including its
  responsive layout for different screen sizes.
- [`web/app.js`](./web/app.js) — Sends and clears messages, shows loading and
  error states, sends general-chat or agent requests, renders returned reports
  and follow-up prompts, and inserts message text into the page. It uses text
  insertion rather than interpreting user messages as HTML.

### Shared data models and input handling

- [`mednex/models.py`](./mednex/models.py) — Defines the Pydantic data
  structures described in the next section.
- [`mednex/state.py`](./mednex/state.py) — Defines the shared reasoning state
  and the result returned by the natural-language parser.
- [`mednex/nlp/normalizer.py`](./mednex/nlp/normalizer.py) — Normalizes a few
  phrase aliases and punctuation, splits text into tokens, and converts a
  limited set of number words into numbers.
- [`mednex/nlp/patterns.py`](./mednex/nlp/patterns.py) — Stores the symptom
  alias lists and regular-expression patterns for durations, vital signs,
  negation, and uncertain wording.
- [`mednex/nlp/parser.py`](./mednex/nlp/parser.py) — Applies those patterns to
  a description. It extracts supported demographics, symptoms, durations, and
  measurements; keeps negation; and records selected ambiguity,
  contradiction, or unsupported-text cases. Its extraction-confidence value
  describes the clarity of pattern extraction, not medical correctness.
- [`mednex/safety/validation.py`](./mednex/safety/validation.py) — Checks
  whether case text is empty or too long, whether known extraction issues
  occurred, and whether any supported facts were found.
- [`mednex/safety/safety_checks.py`](./mednex/safety/safety_checks.py) —
  Searches for selected personal-care, urgent, and direct-care phrases. This
  is phrase matching, not comprehensive emergency detection or triage.
- [`config.py`](./config.py) — Holds the project name, version, and default
  educational notice.
- [`mednex/__init__.py`](./mednex/__init__.py) — Makes selected data models,
  state objects, and agent classes available through the package's top-level
  import.

### The shared data fields

The models in `mednex/models.py` define these plain-English items:

- **Patient information:** optional identifier, age in years, sex, and a list
  of relevant-history notes. The parser does not fill in facts that were not
  stated.
- **Symptom duration:** a positive amount and one of minutes, hours, days,
  weeks, months, or years.
- **Symptom:** a required name, optional description and location, optional
  duration, and a yes/no marker for explicit negation.
- **Vital signs:** optional temperature, heart rate, breathing rate, blood
  pressure, and oxygen saturation. Values must be finite; positive constraints
  apply to rates and pressures, oxygen saturation is limited to 0–100, and
  blood pressure needs both top and bottom values.
- **Extracted observation:** a name, value, original source text, source type,
  and negation marker.
- **Missing information:** a field name and the reason it is missing.
- **Reasoning evidence:** a statement, the observation it came from, and
  whether it supports, contradicts, or provides context.
- **Investigation suggestion:** a name and its educational purpose. This is a
  data shape, not an instruction for a user to perform an investigation.
- **Final assessment report:** a summary, evidence, missing information,
  investigation suggestions, and an educational notice. It is a structured
  type; it does not mean the web chat currently creates clinical assessments.

The common base model rejects unexpected fields and trims surrounding
whitespace. This helps catch malformed data rather than silently accepting
unknown fields.

### Educational reasoning components used by Reasoning Agent mode

- [`mednex/agent.py`](./mednex/agent.py) — Coordinates the reasoning
  components for Reasoning Agent mode. It parses and validates a case, calls
  the engines, manages one-at-a-time follow-up answers, and prepares an
  explainable report.
- [`mednex/engines/knowledge_graph.py`](./mednex/engines/knowledge_graph.py)
  — Validates and searches a small directed NetworkX graph. It returns linked
  candidates, visited graph items, and explanations of the links. The bundled
  graph contains 22 nodes and 19 edges; graph connectivity alone is not a
  conclusion or a clinical score.
- [`mednex/data/knowledge_base.json`](./mednex/data/knowledge_base.json) —
  The synthetic nodes and relationships loaded into the graph.
- [`mednex/engines/symbolic_engine.py`](./mednex/engines/symbolic_engine.py)
  — Evaluates the explicit educational “if/then” rules and returns fired-rule
  traces and evidence flags.
- [`mednex/data/symbolic_rules.json`](./mednex/data/symbolic_rules.json) —
  Four hand-written synthetic rules, including a forward-chaining example.
- [`mednex/engines/bayesian_engine.py`](./mednex/engines/bayesian_engine.py)
  — Applies the small probability calculation to the evidence keys it knows
  about, reports its evidence and limitations, and keeps missing information
  separate from absent information.
- [`mednex/data/bayesian_parameters.json`](./mednex/data/bayesian_parameters.json)
  — Invented priors and likelihoods for three synthetic labels. The file
  explains their non-empirical origin and simplifying assumptions.
- [`mednex/engines/ml_engine.py`](./mednex/engines/ml_engine.py) — Trains and
  evaluates three scikit-learn classifiers on a small, synthetic dataset.
  Its output is kept separate from Bayesian output.
- [`mednex/data/synthetic_ml_cases.csv`](./mednex/data/synthetic_ml_cases.csv)
  — Fifty invented rows for the machine-learning exercise, not real patient
  records.
- [`mednex/engines/information_engine.py`](./mednex/engines/information_engine.py)
  — Calculates entropy and ranks fixed hypothetical questions by synthetic
  expected information gain.
- [`mednex/data/information_questions.json`](./mednex/data/information_questions.json)
  — Seven sample questions with hand-authored hypothetical answer
  likelihoods.
- [`mednex/engines/planner.py`](./mednex/engines/planner.py) — Applies
  deterministic gates and decides whether that separate agent should ask a
  follow-up, stop as uncertain, or return another state.
- [`mednex/engines/explainability.py`](./mednex/engines/explainability.py) —
  Collects the separate engine outputs, evidence, uncertainty, limitations,
  and next action into an explainable report.
- [`mednex/reporting.py`](./mednex/reporting.py) — Turns an agent result into
  readable plain text. The browser currently renders the structured report
  directly rather than calling this formatter.
- [`mednex/demo_cases.py`](./mednex/demo_cases.py) — Loads and validates
  six fictional, synthetic demonstration descriptions. It checks the cases;
  it does not itself run them through the agent.
- [`mednex/data/demo_cases.json`](./mednex/data/demo_cases.json) — The six
  fictional exercise descriptions and their teaching goals.
- [`mednex/engines/__init__.py`](./mednex/engines/__init__.py) — Re-exports
  selected engine types and functions for convenient imports.

### Tests and project documents

- [`tests/test_models.py`](./tests/test_models.py) — Checks model validation.
- [`tests/test_parser.py`](./tests/test_parser.py) — Checks supported
  extraction, normalization, missing details, negation, and contradictions.
- [`tests/test_guidance.py`](./tests/test_guidance.py) — Checks chat replies,
  safety boundaries, dataset validation, and local HTTP behavior.
- [`tests/test_web_agent.py`](./tests/test_web_agent.py) — Exercises the live
  local HTTP routes for reasoning-agent start, follow-up, safety boundaries,
  unsupported and contradictory inputs, and same-origin browser requests.
- [`tests/test_knowledge_graph.py`](./tests/test_knowledge_graph.py) —
  Exercises graph construction and retrieval.
- [`tests/test_symbolic_engine.py`](./tests/test_symbolic_engine.py) — Checks
  rule matching and reasoning traces.
- [`tests/test_bayesian_engine.py`](./tests/test_bayesian_engine.py) — Checks
  probability calculations and invalid or missing evidence cases.
- [`tests/test_ml_engine.py`](./tests/test_ml_engine.py) — Checks model
  training, evaluation, and prediction results.
- [`tests/test_information_engine.py`](./tests/test_information_engine.py) —
  Checks entropy and information-gain calculations.
- [`tests/test_agent.py`](./tests/test_agent.py) — Checks the separate agent's
  safety, follow-up, and correction behavior.
- [`tests/test_reporting.py`](./tests/test_reporting.py) — Checks the separate
  plain-text report output.
- [`requirements.txt`](./requirements.txt) — Lists the Python packages needed
  by the project and their allowed version ranges.
- [`README.md`](./README.md) — The shorter project and setup overview.
- [`INSTRUCTOR_DEMO.md`](./INSTRUCTOR_DEMO.md) — A short instructor-facing
  demonstration script.
- [`.gitignore`](./.gitignore) — Lists generated or local files that should
  normally be left out of version control.

## 8. What MEDNEX does not do

- General-information chat does not run the reasoning engines; Reasoning
  Agent mode does.
- Neither mode diagnoses a condition, estimates real-world medical risk, or
  prescribes or recommends medicines or dosages.
- It does not understand arbitrary natural language like a general-purpose
  language model. It only recognizes its programmed phrases and patterns.
- It does not have a general spelling corrector. It recognizes only listed
  variants.
- Its safety phrase checks can miss unfamiliar wording or react to wording
  without understanding all context.
- It does not provide comprehensive medical guidance. Ten topic records do not
  cover every everyday concern, age group, or circumstance.

## 9. Running MEDNEX

In PowerShell, open the project folder and start the app:

```powershell
cd C:\Users\Ronak\Desktop\MEDNEX
.\.venv\Scripts\python.exe app.py
```

Then open `http://127.0.0.1:8000/`. Keep the terminal open while using the
app. Press **Ctrl+C** in that terminal to stop it.

If the virtual environment has not been created or installed yet:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The server listens on this computer's loopback address, not as a public
internet service. The default port is 8000; `MEDNEX_PORT` can select a
different local port.

## 10. Running the tests

From the project directory:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The tests check data validation, parser behavior, negation, contradictions,
the graph and reasoning engines, calculations, agent follow-ups, reports,
guidance selection, and local web endpoints. Passing software tests show that
tested code paths behaved as expected; they do not prove medical correctness,
coverage of every phrase, or safety for real-world use.

## 11. A short demonstration plan

1. Start the server and point out the educational warning and the phrase
   “Deterministic guidance · No LLM.”
2. Try “I have a cold,” “I have stomach pain,” and “I have a feaver.” Explain
   that each reply is selected from programmed patterns and local prepared
   text.
3. Try “I do not have a cough.” Show that a recognized negated symptom is not
   presented as a present symptom.
4. Try an unsupported phrase. Show that MEDNEX asks for clarification rather
   than claiming it understands.
5. Select **Reasoning Agent** and enter:

   ```text
   A fictional 45-year-old male has a dry cough and fever for three days.
   His temperature is 38.5 degrees Celsius and oxygen saturation is 96 percent.
   ```

6. Show the single plain-English answer and planner-selected follow-up. Answer
   it, then point out that the session returns an updated simple response.
7. Enter a contradictory fictional case to show a plain-language explanation
   that avoids making a conclusion. Open `mednex/agent.py` and the relevant
   engine/data files to explain how results are produced internally.
8. End by explaining the limitations: narrow phrase coverage, synthetic
   reasoning examples, no clinical validation, and no use for real patient
   decisions.

## 12. Suggested one-line quote

> “Not every intelligent system needs an LLM—some can reason through explicit
> rules, evidence, and mathematics.”

This describes the project's software approach, not the accuracy of its
medical content.
