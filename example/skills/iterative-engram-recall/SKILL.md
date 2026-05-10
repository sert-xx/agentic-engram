---
name: iterative-engram-recall
description: A skill to retrieve long-term memory for an Agentic Engram using standard procedures. Use this when you need to recall long-term memory, design a re-recall query due to insufficient confidence, and record each loop.
---

# Iterative Engram Recall

## Prerequisites & Conditions
- This skill provides the operational procedure for "executing the initial recall" and determining "what and how to re-recall".
- It assumes that `AGENTIC_ENGRAM_HOME` has been `export`ed in advance.
- If the recall yields 0 results or the recall command fails, leave a record and return to the normal investigation process.

## Standard Invocation
1. Change the working directory to `AGENTIC_ENGRAM_HOME`
```bash
cd $AGENTIC_ENGRAM_HOME
```

2. Activate the virtual environment
```bash
source .venv/bin/activate
```

3. Execute `ae-recall`
```bash
ae-recall --query "<Description of or problem task the>" --format markdown --limit 3
```

4. Review the execution results and briefly record the `query` / `additional insights gained` / `next action` as needed.
