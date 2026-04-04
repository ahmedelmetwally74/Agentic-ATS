# Applicant Mode Technique

## Overview

For applicant mode, I do not want to use the same approach as company mode.

In company mode, the goal is to go through many CVs fast, narrow them down, and send only the best few to the reasoning step. That makes sense for HR workflows.

Applicant mode is different. Here I am working with one CV only, and the goal is not ranking. The goal is to understand how well this CV matches one job description, show what is already supported by the CV, show what is unclear, and suggest honest improvements without making things up.

Because of that, I decided to use a **requirement-centric** approach instead of a pure **section-vs-section** comparison.

---

## Why I chose this approach

A direct section-vs-section comparison looks simple, but it causes a lot of problems:

- A requirement may exist in the CV, but not in the same section name I expected.
- The model may pull text from the wrong section and build the comparison on weak evidence.
- Some requirements, like education, languages, and years of experience, need stricter handling than normal semantic matching.
- I want the output to stay grounded in the CV and avoid random assumptions.

So instead of asking:

- JD Skills vs CV Skills
- JD Experience vs CV Experience
- JD Education vs CV Education

I want to break the JD into small requirements, then check each requirement against the CV in a controlled way.

---

## Main idea

The pipeline works like this:

1. Decompose the job description into small requirement objects.
2. Save those requirements into a JSON file for debugging and traceability.
3. Parse the CV into structured items.
4. For each requirement, search for the best evidence inside the CV.
5. Apply matching logic based on the requirement type.
6. Assign a final status for each requirement.
7. Build the final applicant report from those requirement-level results.

This gives me a more stable and explainable result.

---

## Step 1: JD decomposition

First, I split the job description into atomic requirements.

Each requirement should be stored as a small JSON object like this:

```json
{
  "id": "req_01",
  "text": "Experience with SQL",
  "category": "tool",
  "importance": "required",
  "section_group": "skills",
  "matching_strategy": "exact_plus_semantic"
}
```

### Why I save it to JSON

I want the requirements saved in a JSON file because it helps me:

- inspect how the JD was interpreted
- debug bad results more easily
- keep the pipeline transparent
- reuse the same parsed JD in later steps if needed

Example file name:

`jd_requirements.json`

---

## Step 2: CV parsing

After that, I parse the CV into structured pieces instead of treating it as one long block.

Each parsed item should keep some metadata, for example:

```json
{
  "section_name": "Experience",
  "entry_index": 0,
  "bullet_index": 1,
  "date_range": "Feb 2025 – Present",
  "text": "Designed and built an NDA Reviewer using FastAPI."
}
```

### Why metadata matters

This helps me know:

- which section the text came from
- which job entry it belongs to
- the original order of bullets
- where the evidence came from when I explain the result

At this stage, I only want to rely on the **Experience** section for work-duration calculation. I may expand this later, but the initial version should stay simple and controlled.

---

## Step 3: Evidence retrieval

For each requirement from the JD, I search the whole CV for the best supporting evidence.

Here, **evidence retrieval** means finding the most relevant CV snippets, bullets, or chunks that support or fail to support a specific requirement.

### Example

If the requirement is:

- `Experience with SQL`

Then I search across the CV and retrieve the top matching pieces, such as:

1. `Programming Languages Python | C | C++ | SQL`
2. `Worked on SAP invoice prediction and dashboarding`
3. `Built an NDA Reviewer using FastAPI`

Then I rerank these results and keep only the strongest evidence.

### Why I do this

This is better than forcing one JD section to compare only against one CV section, because the real evidence may appear in different parts of the CV.

---

## Step 4: Type-specific matching

Not every requirement should be handled the same way.

That is why I want the matching strategy to depend on the requirement type.

### 1. Tools and technologies

Examples:

- SQL
- PyTorch
- TensorFlow
- FastAPI

These should use a mix of:

- exact or alias matching
- semantic retrieval
- reranking

### 2. Education

This should be handled with stricter rules.

For example, if the JD asks for:

- Bachelor's degree in Computer Science, Engineering, or Statistics

Then I should extract the education entries and compare them directly, instead of leaving it fully to semantic matching.

### 3. Languages

Languages should be treated carefully.

If the CV does not explicitly mention English or Arabic, I should not rush to mark them as missing. In many cases, the safer label is:

- `not_explicitly_stated`

### 4. Years of experience

For the first version, I want years of experience to be calculated only from the **Experience** section.

I do not want to include student activities, extracurricular work, or other sections in this calculation for now.

The plan is:

- extract date ranges only from Experience entries
- treat `Present` as today's date at runtime
- handle overlaps between jobs
- calculate the actual covered time range instead of simply summing everything

This keeps the calculation more honest.

### 5. Soft skills

Soft skills should not rely only on exact keyword matches.

For example, leadership may be supported indirectly through teaching, leading a team, or organizing work. In these cases, I can use semantic evidence and careful reasoning.

Still, I want this part to stay conservative. If the evidence is indirect, the result should be partial, not overstated.

---

## Step 5: Final status labels

For each requirement, I want one of these final labels:

- `matched_explicitly`
- `partially_matched`
- `not_explicitly_stated`
- `missing_or_insufficient`

### Why these labels

I do not want everything to be forced into a simple matched vs missing result.

Some things are truly missing.
Some things are there, but weak.
Some things may be true, but are not written clearly in the CV.

These labels make the output more honest and more useful.

---

## Step 6: Final output per requirement

Each requirement should produce a result object like this:

```json
{
  "requirement_id": "req_03",
  "requirement_text": "4+ years of experience in AI, machine learning, and deep learning",
  "status": "missing_or_insufficient",
  "top_evidence": [
    "AI and Software Engineer at Siemens (Feb 2025 – Present)",
    "Machine Learning Engineer at Giza Systems (Oct 2023 – July 2024)"
  ],
  "notes": "The CV shows relevant experience, but it does not clearly support 4+ years of directly relevant experience from the Experience section alone.",
  "suggestion": "Strengthen the Experience section with clearer dates, role scope, and directly relevant AI/ML work."
}
```

This gives me a structured result before I generate the final report.

---

## Step 7: Final report generation

Once all requirement-level results are ready, I can build the final applicant report.

The report should focus on:

- what the CV already supports well
- what is only partially supported
- what is unclear because it is not explicitly stated
- what is genuinely missing or still too weak

The suggestions should stay practical and honest.

I do not want the system to suggest fake seniority, fake years of experience, or rewritten claims that are not supported by the CV.

---

## Why this technique fits applicant mode better

This approach fits applicant mode better because:

- it is centered around one CV, not ranking many candidates
- it gives more control over evidence
- it reduces wrong section matching
- it is easier to debug
- it keeps the output closer to the real content of the CV
- it lets me treat tricky cases, such as education and years of experience, in a cleaner way

---

## Initial scope

To keep the first version stable, I want to start with this scope:

- Save JD requirements to JSON
- Parse the CV into structured items
- Retrieve evidence for each requirement
- Use requirement-specific matching
- Calculate years of experience from the **Experience** section only
- Handle `Present` using today's date
- Handle overlapping date ranges
- Return requirement-level statuses
- Generate the final applicant report from those statuses

---

## Future improvements

Later, I may extend this by:

- expanding experience logic beyond one section if needed
- improving soft-skill detection
- improving relevance scoring
- adding better handling for aliases and synonyms
- storing intermediate debug files for each stage

But for now, I want the first version to stay focused, clear, and easy to validate.

---

## Summary

In short, my applicant mode technique is:

- break the JD into clear requirement objects
- save them in JSON
- search for evidence in the CV for each requirement
- use different matching logic depending on the requirement type
- calculate years of experience from the Experience section only
- use careful status labels instead of treating everything as simply missing
- generate a grounded, honest applicant report

This should give me a much better base than a simple section-vs-section comparison.
