#!/bin/bash
# AirTrace Post-Commit Hook - Summarizes Changes
# This hook runs after each successful commit to provide a summary and reminders

set -e

echo ""
echo "📦 AirTrace Post-Commit Summary"
echo "======================================"

# Get the latest commit info
COMMIT_HASH=$(git rev-parse --short HEAD)
COMMIT_MSG=$(git log -1 --pretty=%B)
COMMIT_FILES=$(git diff-tree --no-commit-id --name-only -r HEAD)

echo "Commit: $COMMIT_HASH"
echo "Message: $COMMIT_MSG"
echo ""

# Categorize changed files
MODELS_CHANGED=false
TRANSFORMS_CHANGED=false
TASKS_CHANGED=false
CONFIGS_CHANGED=false
TESTS_CHANGED=false
DOCS_CHANGED=false
DATA_CHANGED=false

for file in $COMMIT_FILES; do
    if [[ "$file" =~ ^src/airtrace/models/ ]]; then
        MODELS_CHANGED=true
    elif [[ "$file" =~ ^src/airtrace/transforms/ ]]; then
        TRANSFORMS_CHANGED=true
    elif [[ "$file" =~ ^src/airtrace/tasks/ ]]; then
        TASKS_CHANGED=true
    elif [[ "$file" =~ ^configs/ ]]; then
        CONFIGS_CHANGED=true
    elif [[ "$file" =~ ^tests/ ]]; then
        TESTS_CHANGED=true
    elif [[ "$file" =~ ^docs/ ]] || [[ "$file" =~ \.md$ ]]; then
        DOCS_CHANGED=true
    elif [[ "$file" =~ ^src/airtrace/data/ ]]; then
        DATA_CHANGED=true
    fi
done

echo "📊 Impact Analysis:"
echo ""

# Provide context-aware reminders
if [ "$MODELS_CHANGED" = true ]; then
    echo "  🧠 Models modified"
    echo "     → Verify configs/model/*.yaml are in sync"
    echo "     → Run: pytest tests/models/"
    echo "     → Consider updating docs/architecture.md"
    echo ""
fi

if [ "$TRANSFORMS_CHANGED" = true ]; then
    echo "  🔄 Transforms modified"
    echo "     → Verify configs/transforms/*.yaml are in sync"
    echo "     → Run: pytest tests/transforms/"
    echo "     → Test inverse_transform() if applicable"
    echo ""
fi

if [ "$TASKS_CHANGED" = true ]; then
    echo "  🎯 Tasks modified"
    echo "     → Verify configs/task/*.yaml are in sync"
    echo "     → Run: pytest tests/tasks/"
    echo "     → Validate loss computation and metrics"
    echo ""
fi

if [ "$CONFIGS_CHANGED" = true ]; then
    echo "  ⚙️  Configs modified"
    echo "     → Test with: airtrace train exp=<your_exp> --dry-run"
    echo "     → Verify Hydra composition works"
    echo ""
fi

if [ "$DATA_CHANGED" = true ]; then
    echo "  💾 Data pipeline modified"
    echo "     → Consider deleting data/interim/ and data/processed/ caches"
    echo "     → Verify windowing logic with small dataset"
    echo "     → Run: pytest tests/data/"
    echo ""
fi

if [ "$TESTS_CHANGED" = true ]; then
    echo "  ✅ Tests modified"
    echo "     → Run full test suite: pytest"
    echo "     → Check coverage: pytest --cov=airtrace"
    echo ""
fi

if [ "$DOCS_CHANGED" = true ]; then
    echo "  📚 Documentation modified"
    echo "     → Review for clarity and accuracy"
    echo "     → Update MEMORY.md if you learned something surprising"
    echo ""
fi

# General reminders
echo "======================================"
echo "📝 Reminders:"
echo ""
echo "  • Did you discover something surprising? → Update MEMORY.md"
echo "  • Ready to push? → git push -u origin $(git branch --show-current)"
echo "  • Need to test? → Run pytest before pushing"
echo "  • Major change? → Update docs/architecture.md"
echo ""
echo "See CLAUDE.md for development workflow details."
echo ""
