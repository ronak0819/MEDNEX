# MEDNEX instructor demonstration

MEDNEX demonstrates an AI-agent-style conversational interface without an
LLM. Replies are selected and composed from deterministic parsing and a small,
curated local guidance dataset. Use it as a software demonstration only; do
not present it as a medical service or enter real identifying information.

## Walkthrough

1. In the project folder run `python app.py`, then open the local address it
   prints.
2. Click **I have a cough** or type:

   ```text
   I have a cough.
   ```

3. Point out the natural-language style of the response, its general comfort
   suggestions, the advice to seek professional help if symptoms worsen or
   persist, and the explicit no-medicine boundary.
4. Try a combined input such as `I have a cough and fever.` The general
   information chat combines matched topics without showing synthetic
   probabilities.
5. Try `I do not have a cough.` to show that negation is not treated as
   symptom presence.
6. Try `I have a cough. What medicine should I take?` to show the medication
   request boundary, or `I cannot breathe.` to show urgent-language handling.
7. Explain that the response is generated from code and local data—not an
   LLM—and that it may not understand unfamiliar wording.

## Reasoning Agent mode

1. Select **Reasoning Agent**. Use only a clearly fictional or synthetic case,
   for example:

   ```text
   A fictional 45-year-old male has a dry cough and fever for three days.
   His temperature is 38.5 degrees Celsius and oxygen saturation is 96 percent.
   ```

2. Show the single plain-English answer. Explain that the existing parser,
   validation, reasoning engines, and planner run behind the scenes; technical
   graph links, rule names, and synthetic probability values are intentionally
   not shown to a general user.
3. Point out that the planner selects one follow-up question. Answer it and
   show that MEDNEX returns a new plain-English response for the updated
   session.
4. Try contradictory wording, such as a fictional patient both having and not
   having a fever. The agent should stop as uncertain rather than force a
   candidate.
5. Explain that the engines run behind the scenes for the educational
   demonstration, but their toy candidate labels and probability values are
   intentionally hidden from the user. Nothing shown is a diagnosis or
   real-world risk estimate.

## Teaching points

- The interface is HTML/CSS/JavaScript served by a small Python HTTP server;
  Streamlit and Gradle are not used.
- The guidance set is small and reviewable. The project intentionally avoids
  filling it with thousands of duplicated, unreviewed records.
- It gives general information only; it does not diagnose or recommend
  medication or dosage.
- The Bayesian, symbolic, graph, information-gain, planner, and ML modules
  run in the separate Reasoning Agent mode. The general-information chat
  remains an independent guidance-only path.
- A pattern-based safety boundary is incomplete and is not triage.
