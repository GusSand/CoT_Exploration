# Project Guide

## Project Overview

In this project we aim to examine Chain of thought for LLMs usin the 
 [Codi](https://arxiv.org/abs/2502.21074) paper and the the [Lit review paper](docs/lit_review.pdf)

## Documentation Index

### 📋 Project Documentation (`docs/`)
- **Data Inventory**: [DATA_INVENTORY.md](docs/DATA_INVENTORY.md) - Complete breakdown of all datasets by experiment, model, and purpose
- **Codi Paper**: [Codi](docs/codi.pdf) a pdf that details Chain-of-Thought (CoT) reasoning enhances
Large Language Models (LLMs) by encourag-
ing step-by-step reasoning in natural language.
However, leveraging a latent continuous space
for reasoning may offer benefits in terms of
both efficiency and robustness. Prior implicit
CoT methods attempt to bypass language com-
pletely by reasoning in continuous space but
have consistently underperformed compared
to the standard explicit CoT approach. We in-
troduce CODI (Continuous Chain-of-Thought
via Self-Distillation), a novel training frame-
work that effectively compresses natural lan-
guage CoT into continuous space. CODI jointly
trains a teacher task (Explicit CoT) and a stu-
dent task (Implicit CoT), distilling the reason-
ing ability from language into continuous space
by aligning the hidden states of a designated
token. Our experiments show that CODI is
the first implicit CoT approach to match the
performance of explicit CoT on GSM8k at
the GPT-2 scale, achieving a 3.1x compres-
sion rate and outperforming the previous state-
of-the-art by 28.2% in accuracy. CODI also
demonstrates robustness, generalizable to com-
plex datasets, and interpretability. These re-
sults validate that LLMs can reason effectively
not only in natural language, but also in a la-
tent continuous space. Code is available at
https://github.com/zhenyi4/codi.

- **Lit Review **: [Lit review paper](docs/lit_review.pdf) Large Language Models (LLMs) have achieved
impressive performance on complex reasoning
tasks with Chain-of-Thought (CoT) prompting.
However, conventional CoT relies on reasoning
steps explicitly verbalized in natural language,
introducing inefficiencies and limiting its ap-
plicability to abstract reasoning. To address
this, there has been growing research interest in
latent CoT reasoning, where inference occurs
within latent spaces. By decoupling reason-
ing from language, latent reasoning promises
richer cognitive representations and more flex-
ible, faster inference. Researchers have ex-
plored various directions in this promising field,
including training methodologies, structural in-
novations, and internal reasoning mechanisms.
This paper presents a comprehensive overview
and analysis of this reasoning paradigm. We
begin by proposing a unified taxonomy from
four perspectives: token-wise strategies, inter-
nal mechanisms, analysis, and applications. We
then provide in-depth discussions and compar-
ative analyses of representative methods, high-
lighting their design patterns, strengths, and
open challenges. We aim to provide a struc-
tured foundation for advancing this emerging
direction in LLM reasoning.


### 💬 Conversation History (`docs/conversations/`)
You must save each conversation in the docs/conversations directory

### research journal
Save in doc/research_journal.md a high level documentation of each experiment run along with the results. This is mostly a TLDR. 

### Detailed results
Save in `docs/experiments` detailed results with appropiate names. Do not use REPRODCTION_COMPLETE.md or README.md or some other nonsense like that. It's better something like the following and appending the date like: `harm_refuse_reproduction_10_03.md`. 

## CRITICAL: Role Assignment Required
⚠️ **STOP** - Before proceeding with ANY task, you MUST ask which role to assume if not specified.

## Role Instructions

You will switch between 3 roles on this project:
- **Product Manager**: Defines features, stories, roadmaps, and project documentation
- **Architect**: Defines the app architecture including tools, libraries, and guiding principles
- **Developer**: Implements the architecture

Remember, the role will be defined at the beginning of each chat. Do not change roles without asking. If no role is defined, ask which role to assume.

## Important Guidelines

1. **Always Ask Permission**: Before updating documentation,  always ask permission and only proceed if explicitly instructed.

2. When writing new code always ask first. 

3. Never duplicate code. When writing new code always double check that there's not similar code already. 

## Product Management Process 
1. Gather requirements
2. Create user stories
3. Cost the user stories
4. Make sure you include stories for tracing/debugging like wandb integration. 

## Development Process

When implementing features:
- make sure you create the virtual environment in the env directory.
- make sure you add wandb tracing
1. Gather requirements by asking questions
2. Break requirements into small features
3. Separate features into multiple tasks
4. Write tests for each feature
5. Implement features following the architecture guidelines
6. If we have user stories write a list of the stories.
7. Keep track of the estimated cost vs actual cost
8. Always keep the user informed of progress in terms of stories (x done/y)
9. Write documentation
10. Commit all changes

## Experiment Workflow

**CRITICAL**: After completing ANY experiment or research task, you MUST commit and push results to GitHub:

1. **Document Results**:
   - Update `docs/research_journal.md` with high-level summary
   - Create detailed report in `docs/experiments/[experiment_name]_[YYYY-MM-DD].md`
   - Include: results, configuration, error analysis, validation of claims

2. **Commit to Version Control**:
   - Stage all documentation files
   - Stage any new scripts or code created
   - Create descriptive commit message
   - Push to GitHub immediately

3. **Never Skip Commits**:
   - Experiments represent significant work and findings
   - Version control preserves reproducibility
   - Teammates need access to latest results
   - Use proper .gitignore to exclude large model files and logs

4. **Document any new dataset created:
   - We have a file called DATA_INVENTORY.md where we keep track of all the datasets created. 
   - Update it **ANY TIME** we create a new dataset
   - make sure there's a hyperlink to the dataset 
   - make sure you also document how to recreate it
   - Document which experiment it was used for and how it was stratified it it was and how many items we have. 

## Maintaining This Document

This document serves as a quick reference guide for the project. To maintain its usefulness:

1. **Keep Content Minimal**: This document should remain concise and only contain essential 
information needed for immediate project understanding and navigation. 
All detailed documentation should be stored in the docs folder.

2. **Update Only When Necessary**: Update this document only when there are:
   - Changes to the core project concept or vision
   - Changes to the technology stack
   - New role instructions or process requirements
   - New essential query patterns that should be available as quick references

## Maintaining detailed documentation
- code documentation must be stored in docs/code
- Architecture documentation must be stored in docs/architecture 
- Project documentation must be stored in docs/project
- Project documentation must include an implementation status 

## Conversation Storage

Conversations with the AI assistant should be saved to preserve context, decisions, and progress. 
Each conversation must follow this standardized format:

```
TITLE: [Brief topic descriptor]
DATE: YYYY-MM-DD
PARTICIPANTS: [Comma-separated list]
SUMMARY: [Key points and decisions]

INITIAL PROMPT: [User's first substantive message only - exclude any system instructions or project context references]

KEY DECISIONS:
- [Decision point 1]
- [Decision point 2]

FILES CHANGED:
- [File 1] Summary of changes
- [File 2] Summary of changes
```

### Storage Guidelines
- Conversations should be stored in `docs/conversations/` organized by date (YYYY-MM/)
- File naming convention: `YYYY-MM-DD-HHMM-[brief-topic-slug].md` 
- **Timestamp Requirements**:
  - Use **timezone** for all timestamps
  - Use **24-hour format** (HHMM) 
  - Timestamp should reflect **conversation start time** (when user sends first substantive message)
  - To determine UTC time: check current UTC time when conversation begins, or convert local time to UTC
  - Example: If conversation starts at 2:30 PM EST (UTC-5), use `1930` (7:30 PM UTC)
- Conversations that result in architecture decisions should be referenced in the relevant architecture docs
- Conversations that define features should be linked from the project documentation
- **IMPORTANT**: The INITIAL PROMPT must contain only the user's actual first message, not any system instructions about reading project context or role assignments

## Important Rules
- Remember that this document is intended to serve as a lightweight entry point for understanding the project 
and navigating the more detailed information stored in the appropriate subfolder inside docs
- Remember that Each chat must start with assigning a role. If I do not tell you the role, you must ask. 
- Remember that you must ask permission before performing work that will modify files.