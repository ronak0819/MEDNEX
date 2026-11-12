# MEDNEX: Complete Start-to-Finish Demonstration

This document is a step-by-step walkthrough of the MEDNEX application as it is
implemented in this repository. It explains the problem it demonstrates, why
each stage exists, how information moves through the system, and which
mathematical formulas the educational reasoning components use.

It describes the current implementation; it is not a claim about the
chronological order in which the project was built.

> **Safety and scope:** MEDNEX is a local educational software demonstration,
> not a medical service. Do not enter real personal or health information or
> use any result to make health decisions. The reasoning datasets and labels
> are synthetic, and the general guidance is not a clinical guideline.
> MEDNEX does not diagnose, triage, or recommend medicines, doses, or
> individualized treatment.

## 1. What MEDNEX demonstrates

The project demonstrates how to build a conversational interface without an
LLM. Its language is assembled using deterministic code and bundled data:

- The **General Information** chat recognizes a limited set of symptom
  phrases, checks selected safety boundaries, retrieves matching curated
  guidance, and inserts that content into fixed response wording.
- The **Reasoning Agent** accepts only a clearly marked fictional or synthetic
  case, transforms text into structured facts, and runs small graph, symbolic,
  Bayesian, classical machine-learning, and information-gain exercises. A
  planner controls whether the simulation asks a follow-up or stops.
- The web-facing agent response is still a deterministic template based on
  the extracted facts and curated guidance. The reasoning engines do not
  generate its prose.

The problem being demonstrated is how to provide a predictable, inspectable
response from limited inputs without pretending a model understands arbitrary
language. The trade-off is that MEDNEX has limited vocabulary and cannot
reliably interpret all phrasing or context.

## 2. Two separate paths

```text
Browser (HTML/CSS/JavaScript)
  |
  +-- General Information
  |     POST /api/chat
  |       -> input checks and safety boundaries
  |       -> deterministic parser
  |       -> matching topic in general_guidance.json
  |       -> fixed response wording + selected guidance
  |
  +-- Reasoning Agent
        POST /api/agent/start
          -> fictional/synthetic marker check
          -> MEDNEXAgent start
          -> safety -> parser -> validation
          -> graph + symbolic + Bayesian + optional ML
          -> information gain -> planner -> explainability report
          -> deterministic user-facing answer
        POST /api/agent/answer
          -> record answer in temporary session -> run the cycle again
```

These are not interchangeable. The General Information chat does not run the
graph, symbolic, Bayesian, or ML engines. The Reasoning Agent does, but its
web response does not expose their candidate labels, probabilities, graph
links, or rule traces.

## 3. Prepare and run it

From the project directory, create and activate a virtual environment, install
the declared dependencies, and start the local server:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open the address printed by the server, normally
`http://127.0.0.1:8000`. The server binds to loopback. `MEDNEX_PORT` can be set
to choose a different local port. Stop it with Ctrl+C.

The browser sends same-origin JSON requests to the Python standard-library
HTTP server. The server checks the route, JSON content type, request size, and
payload shape. Agent descriptions must be at most 4,000 characters and include
the word “fictional” or “synthetic.” Up to 128 agent sessions are kept
temporarily in server memory.

## 4. General Information chat: live walkthrough

### Step 1 — Send an ordinary supported phrase

Enter:

```text
I have a cough.
```

**Question/problem being handled:** Is this message about a supported topic,
and does the local library contain general text that can be shown for it?

**What happens:**

1. The browser sends `{"message": "I have a cough."}` to `/api/chat`.
2. The server checks the request and calls `respond_to_message`.
3. The input safety check runs before symptom matching.
4. The deterministic parser maps supported phrases to canonical names.
5. The chat finds the matching topic in `general_guidance.json`.
6. Fixed prose is combined with that topic's comfort steps, when-to-seek-help
   language, and urgent-sign wording.
7. A structured JSON reply goes back to the browser, which displays its
   `response` text.

**What to point out:** It can sound conversational because sentence
structures were written in advance. No LLM writes a new answer. The guidance
library currently contains ten hand-curated topics; it is intentionally
finite.

### Step 2 — Combine topics

Enter:

```text
I have a cough and fever.
```

If both phrases are recognized, the chat combines the corresponding supported
topics and their curated information. Because this path concatenates each
topic's prepared lists, duplicate steps may appear when topics share content.

**Problem addressed:** A message can mention more than one supported topic.
The response builder can construct a plural symptom phrase and collect the
guidance for each matched topic.

### Step 3 — Show negation handling

Enter:

```text
I do not have a cough.
```

The parser records an explicitly negated symptom rather than marking it
present. If there are no other present, supported topics, the chat says it
understood the symptom was described as not present and asks what information
the user wants.

**Problem addressed:** “No cough” should not be treated like “cough.” This is
handled using explicit negation patterns and nearby text; it is not a
general-purpose language understanding capability.

### Step 4 — Show a clarification response

Enter a symptom or phrase outside the supported list, for example:

```text
My shoulder feels strange.
```

When there is no supported matching topic, the chat asks for clarification
instead of deliberately inventing topic-specific guidance. Recognition and
guidance lookup are separate: a phrase can be recognized by the parser but
still have no corresponding general-guidance topic.

### Step 5 — Show safety boundaries

Try:

```text
I have a cough. What medicine should I take?
```

The chat recognizes medication requests and returns a fixed boundary response
instead of recommending a medicine or dose. A message containing selected
urgent phrases, such as “I cannot breathe,” also triggers a boundary response.

These checks use phrase patterns. They are conservative guardrails, not a
complete emergency detector, clinical assessment, or triage system.

## 5. What a structured chat reply contains

`GuidanceReply` validates the general-chat reply using a Pydantic model. Its
fields include:

| Field | Purpose |
|---|---|
| `status` | `general_information`, `safety_boundary`, or `needs_clarification` |
| `response` | The assembled text intended for the chat display |
| `understood` | Canonical topic labels understood by the chat |
| `comfort_steps` | Matching curated general comfort text |
| `when_to_seek_help` | Matching curated professional-help text |
| `urgent_signs` | Matching curated warning-sign text |
| `disclaimer` | Scope reminder; no diagnosis or medicine/dose recommendation |

This is **structured output**, not grammar generation. The program builds
grammatical-enough prose from fixed sentence patterns, symptom labels, and
prepared guidance strings. Natural phrasing is limited to the combinations
anticipated by the code.

## 6. Reasoning Agent: complete walkthrough

Use the **Reasoning Agent** mode with an explicitly fictional example only.
The following case is included in the repository's demo data:

```text
A fictional 45-year-old person has a dry cough and fever for three days.
Temperature is 38.5 degrees Celsius and oxygen saturation is 96 percent.
```

It is a synthetic teaching input, not a patient case.

### Step 1 — Mark the case as fictional

The web endpoint requires “fictional” or “synthetic” in the description before
it starts the agent. This gate makes the mode's intended educational scope
explicit. It does not prove that the input is fictional, nor is it a substitute
for safe use.

### Step 2 — Check safety and input validity

The agent calls the safety review first. Selected phrases that suggest a real
personal health concern, a request for individualized care, or urgent language
stop the simulation. The web endpoint also rejects an empty or overlong
description.

If the safety boundary is not triggered, MEDNEX parses the description and
validates that at least one supported symptom or observation was extracted.
Ambiguous, unsupported, or contradictory statements are preserved as
limitations and cause the planner to stop uncertain rather than force a
candidate.

The safety check is based on explicit text patterns. It can miss wording it
does not know and is not medical triage.

### Step 3 — Parse free text into typed facts

The parser uses regular expressions, aliases, and deterministic normalization
to produce a structured `ExtractionResult`. The internal state can represent:

- Demographic facts that were explicitly stated, such as age.
- Symptoms, including duration and whether each was explicitly negated.
- Extracted observations, such as a temperature or oxygen saturation.
- Ambiguous, unsupported, and contradictory text.
- An extraction-clarity value, explicitly **not** a clinical confidence score.

For the example above, the intended facts include age 45, cough present, fever
present, duration three days, temperature 38.5 C, and oxygen saturation 96%.
Unmentioned facts remain missing; they are not silently changed into
negatives.

Normalization handles only explicit aliases and spelling variants in the
project. It is not a general spell checker. The parser also has limitations in
how it determines the scope of negation and hedged statements.

### Step 4 — Run the graph association exercise

The knowledge graph is a validated directed graph loaded from
`knowledge_base.json`. The engine maps extracted symptoms and observations to
matching graph nodes, follows configured edges, and records the visited nodes,
edges, and their rationales.

**Why it exists:** It demonstrates how explicit relationships can be
represented and traced, rather than hidden in a prose answer.

**What it does not do:** Graph connectivity is not a score, probability,
ranking, diagnosis, or proof of a medical association. If an extracted fact
has no matching graph node, that fact is surfaced as missing from the graph.

### Step 5 — Run the symbolic if/then rules

The symbolic engine sorts the hand-authored rules by priority and evaluates
each condition against the extracted facts. A condition can be matched,
missing, or not matched. A rule only fires when its conditions are met.

When a rule fires, it emits a configured evidence flag. The engine repeats
passes while new flags are produced, allowing a later rule to depend on a
flag from an earlier rule. Fired rules are grouped into supporting,
conflicting, or contextual evidence; opposing supporting and conflicting
rules are explicitly surfaced as a conflict.

There is no statistical formula here: it is deterministic rule evaluation,
conceptually:

```text
IF condition_1 AND condition_2 AND ... THEN add configured evidence_flag
```

The four bundled rules are synthetic teaching examples, not clinical rules.

### A beginner's math primer for the next steps

You do not need to be good at math to follow the examples below. The formulas
are compact ways of writing instructions that could also be written out as
sentences.

#### Numbers between 0 and 1

Many of the values in the reasoning exercises are probabilities. A probability
is a number from 0 to 1:

- `0` means “not at all” in the toy calculation.
- `1` means “the whole amount” or “certain under this calculation.”
- `0.5` is the same as 50 out of 100, or 50%.
- `0.8` is the same as 80%.

The project checks that probability values are in range and that a complete
set of probabilities adds up to 1 (100%). This is arithmetic validation; it
does not validate whether the invented numbers describe real life.

#### Symbols used in the equations

| Symbol | Beginner's meaning |
|---|---|
| `P(thing)` | The probability assigned to “thing” |
| `c` | One candidate label in the synthetic exercise |
| `e` | One piece of evidence, such as a parsed symptom |
| `e₁, e₂, …, eₙ` | The first, second, and later pieces of evidence |
| `P(e | c)` | The likelihood value for evidence `e` when considering label `c`; the vertical bar means “given” |
| `×` | Multiply; for example, `0.4 × 0.5` means `0.4 times 0.5` |
| `÷` | Divide |
| `∑` | Add a sequence of values together |
| `∏` | Multiply a sequence of values together |
| `≈` | Approximately equal to; the values on the two sides are being treated as approximately equal |
| `log₂` | A base-2 logarithm used to express information in “bits”; the following example explains why |
| `TP`, `FP`, `FN` | Counts of correct and incorrect classifier results, defined in the ML section |

Subscripts such as the small `1` in `e₁` are just labels: `e₁` means “the
first evidence item.” They are not instructions to multiply a number by 1.

### Step 6 — Calculate synthetic Bayesian values

The Bayesian engine uses three synthetic candidate labels and hand-authored
priors and likelihood tables from `bayesian_parameters.json`. Only recognized
evidence keys with configured likelihoods participate. Missing observations
remain missing; contradictory or unmodeled observations are reported rather
than silently treated as known evidence.

Bayes' rule is a way to update a starting probability after taking some
evidence into account. In ordinary terms:

1. Start with a **prior**: the starting value assigned to a candidate label.
2. Look up a **likelihood** for each observed evidence item and candidate.
3. Multiply the starting value by those likelihoods to get a temporary
   **unnormalized weight**.
4. Divide each candidate's weight by the sum of all candidate weights. This
   last step makes the results add up to 1 again.

For candidate label *c* and observed evidence keys *e₁, e₂, …, eₙ*, MEDNEX
uses the following calculations.

**Unnormalized weight**

<p align="center"><strong>J(c) = P(c) × ∏<sub>i=1</sub><sup>n</sup> P(e<sub>i</sub> | c)</strong></p>

Read this as: “For candidate `c`, start with its prior and multiply by the
likelihood of the first evidence item, then the second, and so on.”

**Normalized posterior**

<p align="center"><strong>P(c | e₁, e₂, …, eₙ) = J(c) ÷ ∑<sub>c′</sub> J(c′)</strong></p>

Read this as: “Take this candidate's temporary weight and divide it by the
total of all candidates' temporary weights.” The result is called a
**posterior**: the updated probability after applying the selected evidence.
`c′` means “each candidate in turn,” so the bottom of the fraction adds all
their weights.

#### Worked example with the bundled synthetic numbers

The data file starts with three fictional labels. These names are placeholders
for an arithmetic exercise; they are not medical conditions.

| Synthetic label | Starting prior |
|---|---:|
| `synthetic_pattern_alpha` | 0.40 |
| `synthetic_pattern_beta` | 0.35 |
| `synthetic_other` | 0.25 |
| **Total** | **1.00** |

Suppose the two evidence items are “cough present” and “fever present.” The
bundled file supplies these likelihoods:

| Synthetic label | Likelihood of cough present | Likelihood of fever present |
|---|---:|---:|
| `synthetic_pattern_alpha` | 0.80 | 0.75 |
| `synthetic_pattern_beta` | 0.30 | 0.25 |
| `synthetic_other` | 0.20 | 0.30 |

For the first label, the multiplication is:

<p align="center"><strong>0.40 × 0.80 × 0.75 = 0.24000</strong></p>

The same multiplication for each label gives:

| Synthetic label | Prior × cough likelihood × fever likelihood | Temporary weight |
|---|---|---:|
| `synthetic_pattern_alpha` | 0.40 × 0.80 × 0.75 | 0.24000 |
| `synthetic_pattern_beta` | 0.35 × 0.30 × 0.25 | 0.02625 |
| `synthetic_other` | 0.25 × 0.20 × 0.30 | 0.01500 |
| **Sum of weights** | 0.24000 + 0.02625 + 0.01500 | **0.28125** |

The temporary weights do not add up to 1, so they are not yet a probability
distribution. Divide each one by `0.28125`:

| Synthetic label | Weight ÷ sum of weights | Normalized result |
|---|---|---:|
| `synthetic_pattern_alpha` | 0.24000 ÷ 0.28125 | 0.8533 (about 85.33%) |
| `synthetic_pattern_beta` | 0.02625 ÷ 0.28125 | 0.0933 (about 9.33%) |
| `synthetic_other` | 0.01500 ÷ 0.28125 | 0.0533 (about 5.33%) |
| **Total** |  | **1.0000 (100%)** |

The arithmetic shows how the code combines configured numbers. It does not
show that the first label is likely in a real person. Every prior and
likelihood in this example is invented, not measured from patients or
clinical research.

#### Why does the formula multiply likelihoods?

The parameter file tells the demonstration to treat each evidence item as
conditionally independent of the others once a candidate label is fixed.
“Conditionally independent” is a technical way of saying that, for this toy
calculation, the program is allowed to multiply the separate likelihood
values.

The bundled parameter file documents its simplifying conditional
independence assumption:

<p align="center"><strong>P(e₁, e₂, …, eₙ | c) ≈ ∏<sub>i=1</sub><sup>n</sup> P(e<sub>i</sub> | c)</strong></p>

Read `P(e₁, e₂, …, eₙ | c)` as “the likelihood of seeing all these evidence
items together, given label `c`.” The approximation replaces that combined
likelihood with the multiplication of separate likelihoods. This makes the
example easy to compute, but real evidence may be related. The assumption is
not established for real cases.

All constants and labels are invented for teaching arithmetic. These values
are not empirical, calibrated, clinically meaningful, or shown in the
Reasoning Agent chat response.

### Step 7 — Optionally run classical ML

The ML exercise runs only when the parsed case has at least one unambiguous
supported symptom feature. Its feature set is deliberately small:

- cough
- fever
- shortness of breath
- headache

Each feature is represented as `present`, `absent`, `not_reported`, or
`conflicting`. Importantly, “not reported” is not the same state as “absent.”
Other symptoms and observations are ignored by these classifiers and recorded
as ignored features.

At agent initialization, scikit-learn pipelines are trained from the 50
hand-authored synthetic rows. The code creates a stratified 75%/25% train/test
split with a fixed seed, then fits logistic regression, a decision tree, and
a random forest. The prediction path uses logistic regression by default.
These are three separate educational classifiers, not an ensemble with the
Bayesian engine.

#### What those three classifier names mean

The MEDNEX code prepares four symptom features for each fictional example.
Each feature is recorded as present, absent, not reported, or conflicting.
`DictVectorizer` converts those named categories into a numeric form that the
scikit-learn models can process. The model learns patterns that map those
feature values to one of the synthetic labels.

- **Logistic regression** learns numeric weights for the feature values and
  uses them to produce class scores and probabilities. Despite “regression”
  in its name, it is used here as a classifier.
- **Decision tree** makes a sequence of questions about feature values. Each
  answer sends an example down a branch until the tree reaches a label.
- **Random forest** builds many decision trees and combines their outputs.
  Combining trees can reduce dependence on the quirks of one tree.

MEDNEX does not write the internal training equations for these models itself;
scikit-learn performs those calculations. The project selects the models,
prepares the synthetic rows, fits them, and reports their predictions and
test-set metrics. The training/test split keeps some rows out of training so
the models can be checked on examples they did not use to learn. A fixed
random seed makes the split repeatable. “Stratified” means the split aims to
preserve the label proportions in both groups.

The evaluation reports common classification metrics. A **classifier** is a
program that picks one of a set of labels for an example. In the bundled
exercise, those labels are fictional. The following measures compare a
classifier's guesses with the known labels in a held-out test set.

<p align="center"><strong>Accuracy = Number of correct predictions ÷ Number of test examples</strong></p>

Accuracy answers: “Out of all test examples, what fraction did the program
label correctly?” If it gets 18 out of 20 right, accuracy is `18 ÷ 20 = 0.9`,
or 90%. Accuracy alone can hide a poor result for a less common label.
It is included because it is an easy overall summary, but it should not be
used alone when label counts differ.

<p align="center"><strong>Precision<sub>k</sub> = TP<sub>k</sub> ÷ (TP<sub>k</sub> + FP<sub>k</sub>)</strong></p>

Precision for label `k` answers: “When the program predicted label `k`, how
often was that prediction correct?” It is useful for understanding how many
of a label's predictions were false alarms in the test set.

<p align="center"><strong>Recall<sub>k</sub> = TP<sub>k</sub> ÷ (TP<sub>k</sub> + FN<sub>k</sub>)</strong></p>

Recall for label `k` answers: “Of all test examples that really had label
`k`, how many did the program find?” It is useful for understanding how many
of the true examples were missed.

<p align="center"><strong>F1<sub>k</sub> = 2 × (Precision<sub>k</sub> × Recall<sub>k</sub>) ÷ (Precision<sub>k</sub> + Recall<sub>k</sub>)</strong></p>

In these formulas:

- `TP` (true positives) means the program predicted label `k` and the example
  really had label `k`.
- `FP` (false positives) means the program predicted label `k`, but the
  example had a different label.
- `FN` (false negatives) means the example really had label `k`, but the
  program predicted another label.

#### Small precision, recall, and F1 example

Imagine a made-up test with 10 examples that truly have label `k`. The
classifier correctly finds 8 of them and misses 2. It also predicts `k` for
2 examples that actually have another label. So:

- `TP = 8`, because 8 `k` examples were found correctly.
- `FN = 2`, because 2 real `k` examples were missed.
- `FP = 2`, because 2 examples were incorrectly called `k`.

The calculations are:

<p align="center"><strong>Precision = 8 ÷ (8 + 2) = 0.8 = 80%</strong></p>

<p align="center"><strong>Recall = 8 ÷ (8 + 2) = 0.8 = 80%</strong></p>

<p align="center"><strong>F1 = 2 × (0.8 × 0.8) ÷ (0.8 + 0.8) = 0.8 = 80%</strong></p>

This tiny example explains the arithmetic; it is not a reported MEDNEX test
result. F1 combines precision and recall into one number. It is the harmonic
mean, which penalizes a case where one of the two values is much lower than
the other. If there are no positive predictions for a label, the code asks
scikit-learn to return zero for that label's undefined precision instead of
raising an error.

MEDNEX reports **macro** precision, recall, and F1. “Macro” means: calculate
the score separately for each label, then give each label equal weight when
averaging. For three labels A, B, and C:

<p align="center"><strong>Macro score = (Score<sub>A</sub> + Score<sub>B</sub> + Score<sub>C</sub>) ÷ 3</strong></p>

Each label gets one equal share in this average, even if the dataset contains
more examples of one label than another. This makes the summary less dominated
by the largest label.

A confusion matrix is a count table: rows represent the true labels, columns
represent the labels predicted by the program. Counts on the diagonal are
correct guesses; counts off the diagonal show which labels were confused.

The dataset is artificial and small; its holdout metrics do not estimate
real-world or clinical performance. ML class probabilities are not calibrated
clinical probabilities and are kept separate from Bayesian values.

### Step 8 — Rank hypothetical follow-up questions

The information engine evaluates fixed, prewritten educational questions
using answer likelihoods from `information_questions.json`. It excludes
questions already answered in the session or whose corresponding observation
is already present in the parsed facts.

The information engine uses **entropy** to describe how spread out a set of
candidate probabilities is. Entropy is not a measure of medical danger. It
only describes the mathematical distribution supplied to this toy exercise.

If one candidate has all the probability, there is no uncertainty in the
distribution, so entropy is 0. If two candidates each have probability 0.5,
there is more uncertainty: either is equally possible in that distribution.
With three equally weighted candidates, entropy is about 1.585 bits.
You can think of one bit as the uncertainty involved in one fair yes/no
choice: either answer has the same chance. The word “bit” is a unit for this
mathematical amount of uncertainty, not a medical unit.

The formula is:

<p align="center"><strong>H(X) = −∑<sub>x</sub> P(x) log₂ P(x)</strong></p>

Here `X` means the set of possible candidate labels. The `∑` says to do the
calculation for each candidate and add the results. `log₂` is a logarithm
base 2; it converts probabilities into a quantity called bits. For a beginner,
it is enough to know that the computer applies this fixed calculation to
measure how spread out the probabilities are. A term with probability 0 is
skipped, as in the implementation.

#### A small entropy example

For two equally likely outcomes, each probability is 0.5. The formula becomes:

<p align="center"><strong>H = −[0.5 × log₂(0.5) + 0.5 × log₂(0.5)] = 1 bit</strong></p>

Because `log₂(0.5) = −1`, each half contributes 0.5 bit, and the total is 1
bit. If the distribution changes to 100% for one outcome and 0% for the
other, entropy is 0 bits: there is no uncertainty left in that simplified
distribution.

#### How the engine evaluates a question

The engine starts with the current candidate probabilities, then considers
each possible answer to a fixed educational question. It uses the configured
likelihood `P(a | x)`, which means “the toy probability of answer `a` if the
candidate were `x`,” to calculate how probable that answer would be overall:

<p align="center"><strong>P(a) = ∑<sub>x</sub> P(x) P(a | x)</strong></p>

In everyday words: multiply each candidate's current probability by the
question's answer-likelihood for that candidate, then add those products.
This gives the modeled probability of that answer.

After considering answer `a`, the candidate probabilities are updated:

<p align="center"><strong>P(x | a) = [P(x) × P(a | x)] ÷ P(a)</strong></p>

This is the same multiply-then-normalize idea used in the Bayesian section.
It gives a new distribution for the hypothetical situation where answer `a`
was received.

The engine computes entropy for each possible answer's updated distribution.
It then takes a weighted average, giving more weight to answers that the toy
model says are more likely:

<p align="center"><strong>Expected H(X | A) = ∑<sub>a</sub> P(a) H(X | a)</strong></p>

`A` means the set of possible answers. “Expected” here means a weighted
average across possible answers—not a promise about what a real person will
say.

Finally, the engine subtracts that average future entropy from the current
entropy:

<p align="center"><strong>IG(Q) = H(X) − Expected H(X | A)</strong></p>

The result, `IG(Q)`, is the question's **information gain** in bits. A larger
number means the synthetic model expects that question to reduce its own
uncertainty more. It does not mean that the question is medically important,
safe, urgent, or clinically appropriate.

#### Worked information-gain example with bundled values

To keep the arithmetic manageable, start with the bundled priors and consider
the fixed question “Is cough present or absent?” This is a calculation
demonstration only. The current distribution is:

| Label | Current probability |
|---|---:|
| Alpha | 0.40 |
| Beta | 0.35 |
| Other | 0.25 |

The current entropy, calculated with the entropy formula, is approximately
**1.5589 bits**.

The question data gives these invented likelihoods:

| Possible answer | If Alpha | If Beta | If Other |
|---|---:|---:|---:|
| Cough present | 0.80 | 0.30 | 0.20 |
| Cough absent | 0.20 | 0.70 | 0.80 |

First calculate the overall probability of “present.” Each current
probability is multiplied by the answer likelihood in its row, then the
products are added:

<p align="center"><strong>P(present) = (0.40 × 0.80) + (0.35 × 0.30) + (0.25 × 0.20) = 0.475</strong></p>

So the model assigns 47.5% to that hypothetical answer. The other answer has
probability 52.5%:

<p align="center"><strong>P(absent) = (0.40 × 0.20) + (0.35 × 0.70) + (0.25 × 0.80) = 0.525</strong></p>

Next, if the answer were “present,” the unnormalized candidate values would
be:

| Label | Current probability × present likelihood |
|---|---:|
| Alpha | 0.40 × 0.80 = 0.320 |
| Beta | 0.35 × 0.30 = 0.105 |
| Other | 0.25 × 0.20 = 0.050 |
| **Total** | **0.475** |

Divide each value by 0.475 to get the updated distribution:

| Label | Updated probability if answer is present |
|---|---:|
| Alpha | 0.320 ÷ 0.475 = 0.6737 |
| Beta | 0.105 ÷ 0.475 = 0.2211 |
| Other | 0.050 ÷ 0.475 = 0.1053 |

The entropy of that distribution is about **1.2071 bits**. Repeating the
same steps for “absent” gives updated probabilities about 0.1524, 0.4667,
and 0.3810, with entropy about **1.4571 bits**.

Now weight each possible future entropy by the probability of its answer:

<p align="center"><strong>Expected entropy = (0.475 × 1.2071) + (0.525 × 1.4571) ≈ 1.3384 bits</strong></p>

Subtract this from the current entropy:

<p align="center"><strong>Information gain = 1.5589 − 1.3384 ≈ 0.2205 bits</strong></p>

This number lets the program compare its fixed questions mathematically.
Another question can receive a different score because its invented
likelihoods differ. The planner selects the question with the largest
positive score, if one is available and the follow-up limit has not been
reached. These calculations are not a recommendation for a real-world
questionnaire.

### Step 9 — Let the planner choose the next state

The planner applies deterministic gates in order:

1. Safety boundary triggered → `PROFESSIONAL_ASSESSMENT_REQUIRED`.
2. Missing/invalid extraction or validation → `UNCERTAIN`.
3. Contradictions, ambiguity, or unsupported statements → `UNCERTAIN`.
4. A positive-information-gain question remains and the follow-up limit has
   not been reached → `NEEDS_MORE_INFORMATION`, asking the top-ranked
   question.
5. Useful questions remain after the limit (three by default) → `UNCERTAIN`.
6. No useful modeled question remains and required engine outputs exist →
   `EVALUATION_COMPLETE`.

The planner does not choose a diagnosis. It decides whether to ask, stop, or
complete the educational calculation while retaining uncertainty.

### Step 10 — Build the explainability result

The explainability component collects extracted facts, missing or unmodeled
information, graph evidence, fired symbolic rules, Bayesian values, optional
ML output, uncertainty, limitations, and the planner's next action into a
typed `ExplainableReport`.

The standalone text formatter in `reporting.py` can render a technical report.
The current web endpoint does not return that full report to the browser.

### Step 11 — Render a plain-English response

The web endpoint passes the `AgentTurn` to `build_user_facing_answer`.
That builder:

- Stops with fixed safety wording if professional assessment is required.
- Declines to summarize when extraction is missing, contradictory,
  ambiguous, or unsupported.
- Uses the canonical symptom labels and matching curated guidance topic.
- Assembles fixed sections for the fictional case, general comfort,
  contacting a healthcare professional, urgent signs, and the educational
  limitation.
- Appends the planner's follow-up question if one was selected.

The response is deterministic template composition, not text generated by
the graph, Bayesian engine, ML model, or an LLM. Technical outputs are
intentionally not included in the chat answer.

### Step 12 — Answer a follow-up and rerun

If the response includes one follow-up, the browser sends the answer and
session ID to `/api/agent/answer`. The agent records the answer against the
current question, maps simple answer wording to a polarity such as present,
absent, reported, not reported, or unknown, and reruns the cycle. Already
answered questions are excluded. The response is rebuilt from the updated
turn.

Follow-ups are one-at-a-time and capped. If an answer is unclear, the agent
does not have to force an interpretation or candidate.

## 7. Reasoning Agent demo variants

Use these repository-provided examples to show specific behaviors:

| Example input (fictional) | Demonstrates |
|---|---|
| `A fictional person reports a headache.` | Missing age, duration, and measurements remain missing |
| `A fictional person has no fever and does not have a cough.` | Explicit negation is retained |
| `A fictional person has fever but does not have a fever.` | Contradiction leads to an uncertain result |
| `A fictional person reports a cough. Pulse is 72 bpm and oxygen saturation is 96 percent.` | Supported measurement aliases and observations |
| `A fictional person reports a cough.` | One-question-at-a-time follow-up behavior |

The example descriptions are in `mednex/data/demo_cases.json`. They are
teaching inputs, not clinical scenarios.

## 8. Where each part lives

| Part | Source |
|---|---|
| Local app entry point and server | `app.py`, `mednex/web_server.py` |
| Browser interface | `web/index.html`, `web/styles.css`, `web/app.js` |
| General chat routing, safety checks, and response | `mednex/guidance.py` |
| Symptom patterns and parser | `mednex/nlp/patterns.py`, `mednex/nlp/normalizer.py`, `mednex/nlp/parser.py` |
| General chat content | `mednex/data/general_guidance.json` |
| Structured models and extraction state | `mednex/models.py`, `mednex/state.py` |
| Agent orchestration and sessions | `mednex/agent.py` |
| Safety and case validation | `mednex/safety/safety_checks.py`, `mednex/safety/validation.py` |
| Graph engine and graph data | `mednex/engines/knowledge_graph.py`, `mednex/data/knowledge_base.json` |
| Symbolic engine and rules | `mednex/engines/symbolic_engine.py`, `mednex/data/symbolic_rules.json` |
| Bayesian engine and parameters | `mednex/engines/bayesian_engine.py`, `mednex/data/bayesian_parameters.json` |
| ML engine and synthetic rows | `mednex/engines/ml_engine.py`, `mednex/data/synthetic_ml_cases.csv` |
| Information gain and questions | `mednex/engines/information_engine.py`, `mednex/data/information_questions.json` |
| Planner and explainability | `mednex/engines/planner.py`, `mednex/engines/explainability.py` |
| User-facing agent response and text report | `mednex/user_response.py`, `mednex/reporting.py` |
| Demo cases | `mednex/data/demo_cases.json`, `mednex/demo_cases.py` |

## 9. Verify the demonstration

Run the test suite from the project directory:

```powershell
python -m pytest
```

The tests cover parser behavior, response construction, safety boundaries,
structured models, graph traversal, symbolic rules, Bayesian calculations,
synthetic ML, information gain, agent follow-ups, reports, and local web
routes.

## 10. Limits to say out loud

- MEDNEX uses deterministic phrase matching, not an LLM or general language
  understanding. The same supported input and data produce the same result.
- Its vocabularies, aliases, templates, and guidance set are finite.
- Safety phrase matching can miss or over-trigger; it is not emergency
  detection or triage.
- The reasoning engines use synthetic teaching data and assumptions. Their
  labels and numerical values are not real diagnoses, risks, or validated
  performance measures.
- Missing information is not equivalent to a negative finding.
- The General Information chat is separate from the reasoning engines.
- The system is a local prototype, not a production medical application or a
  privacy/security-reviewed service.

For a shorter setup or instructor script, see `README.md` and
`INSTRUCTOR_DEMO.md`. For a less technical explanation of project vocabulary,
see `MEDNEX_PLAIN_ENGLISH_GUIDE.md`.
