# AirTrace Guide for Gemini

👋 Hello, Gemini! Welcome to the AirTrace project.

## Quick Start

This project has comprehensive documentation for AI agents working on the codebase. **Please read `CLAUDE.md` first** - it contains all the essential information you need:

- Project structure and navigation
- Core framework principles and rules
- Common tasks and how to do them
- Pitfalls to avoid
- Development workflow

## Essential Files for AI Agents

1. **`CLAUDE.md`** ← **START HERE**
   - Complete guide to working on AirTrace
   - Framework principles and rules
   - Navigation cheatsheet
   - Common tasks and patterns

2. **`MEMORY.md`**
   - Learnings from previous AI agents
   - Surprising discoveries and gotchas
   - **Read this before making significant changes**

3. **`README.md`**
   - Project overview for humans
   - Installation and quick start
   - General project information

4. **`docs/architecture.md`**
   - Detailed design philosophy
   - Component interactions
   - In-depth technical explanations

## Why CLAUDE.md?

While this file is named for Claude Code, the guidance is **universal for all AI agents**. The principles of config-code synchronization, modular architecture, and framework conventions apply regardless of which AI system is working on the code.

Think of CLAUDE.md as "Agent Guide" - it just happens to be named after the tool that created it.

## Quick Rules

⚠️ **Critical things to know**:

- **Config ↔ Code**: Every model/transform/task needs BOTH a Python file AND a config file
- **Don't commit to `data/`**: It's gitignored and ephemeral
- **Use base classes**: Inherit from `ARBaseModel`, `Transform`, or `Task`
- **Register components**: Use `@register("name")` decorator
- **Add type hints**: All functions need type annotations
- **Update MEMORY.md**: When you learn something surprising

## Project Type

AirTrace is a **config-driven ML framework** built on:
- Hydra (hierarchical configuration)
- PyTorch (models and training)
- Registry pattern (pluggable components)

It's NOT a monolithic training script - it's a framework for composing experiments from modular pieces.

## Getting Started

```bash
# 1. Read the guide
cat CLAUDE.md

# 2. Read the learnings
cat MEMORY.md

# 3. Understand the project
cat README.md

# 4. Start working!
# Always follow the patterns in CLAUDE.md
```

## Working Together

If you discover something surprising or non-obvious while working on AirTrace:

1. Add it to `MEMORY.md` using the template provided
2. This helps future agents (including other instances of Gemini, Claude, or any AI) avoid the same issues

AI agents are a team working across time. Good documentation helps everyone.

---

**TL;DR**: Read `CLAUDE.md` first, then `MEMORY.md`. They contain everything you need to work effectively on AirTrace.
