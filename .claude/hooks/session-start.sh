#!/bin/bash
# AirTrace Session Start Hook - Orients New Agents
# This hook runs when a new Claude Code session starts

echo "🚀 Welcome to AirTrace!"
echo "======================================"
echo ""
echo "📖 First time working on this project? Read these files:"
echo ""
echo "   1. CLAUDE.md     - Complete agent guide (navigation, rules, patterns)"
echo "   2. MEMORY.md     - Learnings from previous agents (read before coding!)"
echo "   3. README.md     - Project overview and quick start"
echo "   4. docs/architecture.md - Design philosophy and patterns"
echo ""
echo "======================================"
echo "🎯 Quick Reference:"
echo ""
echo "  Project Type:  Config-driven ML framework (Hydra + PyTorch)"
echo "  Key Principle: Config ↔ Code synchronization (NEVER break this!)"
echo "  Structure:     configs/ ↔ src/airtrace/ (models, transforms, tasks)"
echo ""
echo "  Common Tasks:"
echo "    • Add model:      src/airtrace/models/ + configs/model/"
echo "    • Add transform:  src/airtrace/transforms/ + configs/transforms/"
echo "    • Run experiment: airtrace train exp=<name>"
echo "    • Run tests:      pytest"
echo "    • Format code:    black src/ tests/"
echo ""
echo "======================================"
echo "⚠️  Critical Rules:"
echo ""
echo "  ✗ NEVER commit files to data/ (ephemeral, gitignored)"
echo "  ✗ NEVER modify data/raw/ (immutable source of truth)"
echo "  ✗ NEVER add a model/transform/task without its config file"
echo "  ✗ NEVER bypass base class interfaces (ARBaseModel, Transform, Task)"
echo "  ✓ ALWAYS use @register() decorator for components"
echo "  ✓ ALWAYS add type hints to functions"
echo "  ✓ ALWAYS update MEMORY.md when you learn something surprising"
echo ""
echo "======================================"
echo "📊 Current Repository State:"
echo ""

# Show current branch
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
echo "  Branch: $BRANCH"

# Show git status summary
UNTRACKED=$(git ls-files --others --exclude-standard | wc -l)
MODIFIED=$(git diff --name-only | wc -l)
STAGED=$(git diff --cached --name-only | wc -l)

echo "  Status: $STAGED staged, $MODIFIED modified, $UNTRACKED untracked"

# Show recent commits
echo ""
echo "  Recent commits:"
git log --oneline -3 | sed 's/^/    /'

echo ""
echo "======================================"
echo "✨ Happy coding! Remember:"
echo ""
echo "   • Read CLAUDE.md and MEMORY.md first"
echo "   • Follow the config-code contract"
echo "   • Test your changes (pytest)"
echo "   • Log surprises in MEMORY.md"
echo ""
echo "Type 'cat CLAUDE.md' to start reading the guide."
echo ""
