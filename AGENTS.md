# AirTrace Guide for AI Agents

**Welcome, AI Agent!** 🤖

This repository has been prepared for AI-assisted development with comprehensive agent documentation.

## Start Here

**→ Read `CLAUDE.md` immediately** ← This is your primary guide.

## Essential Documentation

| File | Purpose | When to Read |
|------|---------|--------------|
| **`CLAUDE.md`** | Complete agent guide with rules, patterns, and navigation | **Read first, always** |
| **`MEMORY.md`** | Learnings from previous agents - gotchas and insights | **Read before coding** |
| `README.md` | Project overview, installation, usage | For project context |
| `docs/architecture.md` | Detailed design philosophy | For deep understanding |

## What You Need to Know

### This is a Config-Driven Framework

AirTrace uses **Hydra** to compose experiments from modular components:

```
Experiment = Data + Model + Transforms + Task + Training Config
```

**Everything is declarative** - defined in YAML configs and registered Python components.

### The Golden Rule

**Config ↔ Code Synchronization**

Every component needs:
1. Python implementation with `@register("name")` decorator
2. YAML config file with matching name

Break this rule → runtime failures.

### File Structure

```
configs/          ← YAML configurations (Hydra)
  ├── model/      ← Model architectures
  ├── transforms/ ← Data preprocessing
  ├── task/       ← Prediction objectives
  └── exp/        ← Complete experiments

src/airtrace/     ← Python implementations
  ├── models/     ← Model classes
  ├── transforms/ ← Transform classes
  ├── tasks/      ← Task classes
  └── ...

tests/           ← pytest tests
docs/            ← Extended documentation
```

### Critical Rules

✅ **DO**:
- Read `CLAUDE.md` and `MEMORY.md` before starting
- Follow the config-code contract
- Use base classes (`ARBaseModel`, `Transform`, `Task`)
- Add type hints to all functions
- Write tests for new components
- Update `MEMORY.md` when you learn something surprising

❌ **DON'T**:
- Commit files to `data/` (ephemeral, gitignored)
- Modify `data/raw/` (immutable source of truth)
- Add components without config files
- Bypass registration system
- Skip type annotations
- Hardcode paths or magic values

## For Different AI Systems

Whether you're:
- **Claude Code** - This guide was designed for you
- **GitHub Copilot** - Use as context for suggestions
- **Gemini** - See `GEMINI.md` for a friendly intro (but still read `CLAUDE.md`)
- **GPT/ChatGPT** - Read this for quick orientation
- **Any other AI** - The principles apply universally

The documentation is agent-agnostic. The patterns, rules, and structure apply regardless of which AI system is working on the code.

## Quick Start Checklist

- [ ] Read `CLAUDE.md` (comprehensive guide)
- [ ] Read `MEMORY.md` (previous agent learnings)
- [ ] Review `README.md` (project overview)
- [ ] Check `docs/architecture.md` (if needed)
- [ ] Understand the config-code contract
- [ ] Know where components live (`src/airtrace/` + `configs/`)
- [ ] Remember to update `MEMORY.md` with discoveries

## Project Hooks

The `.claude/hooks/` directory contains scripts that enforce structure:

- `session-start.sh` - Orientation for new sessions
- `pre-commit.sh` - Validates changes before commit
- `post-commit.sh` - Summarizes impact after commit

These hooks help maintain code quality and catch common mistakes.

## Questions?

1. **Check `CLAUDE.md`** - Most questions are answered there
2. **Search `MEMORY.md`** - Previous agents may have faced the same issue
3. **Read the code** - Look for similar patterns in existing implementations
4. **Check tests** - They demonstrate expected usage
5. **Ask the user** - If genuinely ambiguous

## Philosophy

AirTrace is designed for **reproducible, modular ML research**. Every design decision serves this goal:

- **Config-driven**: Experiments are reproducible specifications
- **Registry pattern**: Components are discoverable and composable
- **Type safety**: Interfaces prevent integration bugs
- **Testing**: Confidence in modifications

Work **with** the framework, not against it.

## Contributing to Agent Documentation

When you discover something that would help future agents:

1. Add it to `MEMORY.md` (use the template there)
2. Include file paths, code examples, and impact analysis
3. Explain **why** it matters, not just **what** you found

**We're all a team across time.** Good documentation helps everyone.

---

**Ready?** Go read `CLAUDE.md` now. Seriously. It has everything you need.

✨ Happy coding!
