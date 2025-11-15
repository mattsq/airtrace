#!/bin/bash
# AirTrace Pre-Commit Hook - Enforces Project Structure
# This hook runs before each git commit to validate changes comply with AirTrace framework rules

set -e

echo "🔍 AirTrace Structure Validation Hook"
echo "======================================"

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    echo "✅ No files to validate"
    exit 0
fi

ERRORS=0
WARNINGS=0

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Rule 1: Check for files created in disallowed locations
echo ""
echo "📋 Rule 1: Validating file locations..."
for file in $STAGED_FILES; do
    # Skip deleted files
    if [ ! -f "$file" ]; then
        continue
    fi

    # Allowed patterns
    if [[ "$file" =~ ^src/airtrace/ ]] || \
       [[ "$file" =~ ^configs/ ]] || \
       [[ "$file" =~ ^tests/ ]] || \
       [[ "$file" =~ ^notebooks/.*\.ipynb$ ]] || \
       [[ "$file" =~ ^docs/.*\.md$ ]] || \
       [[ "$file" =~ ^\.claude/ ]] || \
       [[ "$file" =~ ^(README\.md|CLAUDE\.md|MEMORY\.md|AGENTS\.md|GEMINI\.md|LICENSE|\.gitignore|pyproject\.toml|setup\.py|requirements\.txt)$ ]]; then
        echo "  ✓ $file (valid location)"
    elif [[ "$file" =~ ^data/ ]]; then
        echo -e "  ${RED}✗ $file${NC} - NEVER commit files in data/ (should be gitignored)"
        ERRORS=$((ERRORS + 1))
    else
        echo -e "  ${YELLOW}⚠ $file${NC} - Unusual location, verify this is intentional"
        WARNINGS=$((WARNINGS + 1))
    fi
done

# Rule 2: Check config-code synchronization for models
echo ""
echo "📋 Rule 2: Validating model config-code sync..."
for file in $STAGED_FILES; do
    if [[ "$file" =~ ^src/airtrace/models/(.+)\.py$ ]]; then
        model_name="${BASH_REMATCH[1]}"
        if [ "$model_name" != "base" ] && [ "$model_name" != "__init__" ]; then
            # Check if model has @register decorator
            if grep -q "@register" "$file"; then
                # Extract registered name(s)
                registered_names=$(grep -oP "@register\(\s*['\"]\\K[^'\"]+(?=['\"])" "$file" || echo "")
                for reg_name in $registered_names; do
                    config_file="configs/model/${reg_name}.yaml"
                    if [ -f "$config_file" ]; then
                        echo "  ✓ Model $reg_name: $file ↔ $config_file"
                    else
                        echo -e "  ${YELLOW}⚠ Model $reg_name${NC}: $file exists but $config_file missing"
                        echo "    Create $config_file or this model won't be usable in experiments"
                        WARNINGS=$((WARNINGS + 1))
                    fi
                done
            fi
        fi
    fi
done

# Rule 3: Check config-code synchronization for transforms
echo ""
echo "📋 Rule 3: Validating transform config-code sync..."
for file in $STAGED_FILES; do
    if [[ "$file" =~ ^src/airtrace/transforms/(.+)\.py$ ]]; then
        transform_name="${BASH_REMATCH[1]}"
        if [ "$transform_name" != "base" ] && [ "$transform_name" != "__init__" ]; then
            if grep -q "@register" "$file"; then
                registered_names=$(grep -oP "@register\(\s*['\"]\\K[^'\"]+(?=['\"])" "$file" || echo "")
                for reg_name in $registered_names; do
                    config_file="configs/transforms/${reg_name}.yaml"
                    if [ -f "$config_file" ]; then
                        echo "  ✓ Transform $reg_name: $file ↔ $config_file"
                    else
                        echo -e "  ${YELLOW}⚠ Transform $reg_name${NC}: $file exists but $config_file missing"
                        echo "    Create $config_file or include in a transform pipeline config"
                        WARNINGS=$((WARNINGS + 1))
                    fi
                done
            fi
        fi
    fi
done

# Rule 4: Check config-code synchronization for tasks
echo ""
echo "📋 Rule 4: Validating task config-code sync..."
for file in $STAGED_FILES; do
    if [[ "$file" =~ ^src/airtrace/tasks/(.+)\.py$ ]]; then
        task_name="${BASH_REMATCH[1]}"
        if [ "$task_name" != "base" ] && [ "$task_name" != "__init__" ]; then
            if grep -q "@register" "$file"; then
                registered_names=$(grep -oP "@register\(\s*['\"]\\K[^'\"]+(?=['\"])" "$file" || echo "")
                for reg_name in $registered_names; do
                    config_file="configs/task/${reg_name}.yaml"
                    if [ -f "$config_file" ]; then
                        echo "  ✓ Task $reg_name: $file ↔ $config_file"
                    else
                        echo -e "  ${YELLOW}⚠ Task $reg_name${NC}: $file exists but $config_file missing"
                        echo "    Create $config_file or this task won't be usable in experiments"
                        WARNINGS=$((WARNINGS + 1))
                    fi
                done
            fi
        fi
    fi
done

# Rule 5: Check for __init__.py updates when adding new modules
echo ""
echo "📋 Rule 5: Validating __init__.py updates..."
for file in $STAGED_FILES; do
    if [[ "$file" =~ ^src/airtrace/(models|transforms|tasks|data)/(.+)\.py$ ]]; then
        component_dir="${BASH_REMATCH[1]}"
        module_name="${BASH_REMATCH[2]}"
        if [ "$module_name" != "__init__" ] && [ "$module_name" != "base" ]; then
            init_file="src/airtrace/${component_dir}/__init__.py"
            if [ -f "$init_file" ]; then
                # Check if new module is imported in __init__.py
                if git diff --cached "$init_file" | grep -q "from.*${module_name}"; then
                    echo "  ✓ $module_name added to $init_file"
                else
                    echo -e "  ${YELLOW}⚠ New module $file${NC} - consider adding to $init_file"
                    WARNINGS=$((WARNINGS + 1))
                fi
            fi
        fi
    fi
done

# Rule 6: Check for tests when adding new implementations
echo ""
echo "📋 Rule 6: Validating test coverage..."
for file in $STAGED_FILES; do
    if [[ "$file" =~ ^src/airtrace/(models|transforms|tasks)/(.+)\.py$ ]]; then
        component_type="${BASH_REMATCH[1]}"
        module_name="${BASH_REMATCH[2]}"
        if [ "$module_name" != "__init__" ] && [ "$module_name" != "base" ]; then
            test_file="tests/${component_type}/test_${module_name}.py"
            if echo "$STAGED_FILES" | grep -q "$test_file"; then
                echo "  ✓ $file has corresponding $test_file"
            elif [ -f "$test_file" ]; then
                echo "  ✓ $file (existing test: $test_file)"
            else
                echo -e "  ${YELLOW}⚠ $file${NC} - no test file found at $test_file"
                echo "    Consider adding tests for new implementations"
                WARNINGS=$((WARNINGS + 1))
            fi
        fi
    fi
done

# Rule 7: Validate Python files have type hints (sample check)
echo ""
echo "📋 Rule 7: Checking for type hints in new Python files..."
for file in $STAGED_FILES; do
    if [[ "$file" =~ \.py$ ]] && [ -f "$file" ]; then
        # Skip __init__.py and check if file has functions without type hints
        if [ "$(basename $file)" != "__init__.py" ]; then
            # Simple heuristic: check if 'def ' exists without '->' in the same line
            untyped_funcs=$(grep -n "def " "$file" | grep -v "__init__" | grep -v -- "->") || true
            if [ ! -z "$untyped_funcs" ]; then
                echo -e "  ${YELLOW}⚠ $file${NC} - some functions may be missing return type hints"
                WARNINGS=$((WARNINGS + 1))
            else
                echo "  ✓ $file (type hints look good)"
            fi
        fi
    fi
done

# Rule 8: Check README.md is updated when models are added/modified
echo ""
echo "📋 Rule 8: Validating README.md updates for new/modified models..."
MODEL_FILES_CHANGED=false
README_CHANGED=false

for file in $STAGED_FILES; do
    if [[ "$file" =~ ^src/airtrace/models/(.+)\.py$ ]]; then
        model_name="${BASH_REMATCH[1]}"
        # Check if this is a new model file or a modified one with new @register
        if [ "$model_name" != "base" ] && [ "$model_name" != "__init__" ] && [ "$model_name" != "registry" ]; then
            if grep -q "@register" "$file" 2>/dev/null; then
                MODEL_FILES_CHANGED=true
                break
            fi
        fi
    fi
done

if echo "$STAGED_FILES" | grep -q "README.md"; then
    README_CHANGED=true
fi

if [ "$MODEL_FILES_CHANGED" = true ]; then
    if [ "$README_CHANGED" = true ]; then
        # Check if the README Model Registry section was actually updated
        if git diff --cached README.md | grep -q "Model Registry"; then
            echo "  ✓ Model files changed and README.md Model Registry section updated"
        else
            echo -e "  ${YELLOW}⚠ Model files changed${NC} - ensure Model Registry section in README.md is updated"
            echo "    Add model details (name, class, description) to the appropriate table"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo -e "  ${YELLOW}⚠ Model files changed${NC} but README.md is not being updated"
        echo "    Update the Model Registry section in README.md with your new model's details"
        echo "    See CLAUDE.md section 'Add a new model' for instructions"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✓ No model files changed"
fi

# Rule 9: Run pytest to ensure all tests pass
echo ""
echo "📋 Rule 9: Running pytest to validate code changes..."

# Check if there are any Python files being committed
PYTHON_FILES_CHANGED=false
for file in $STAGED_FILES; do
    if [[ "$file" =~ \.py$ ]]; then
        PYTHON_FILES_CHANGED=true
        break
    fi
done

if [ "$PYTHON_FILES_CHANGED" = true ]; then
    echo "  Running pytest..."

    # Run pytest with minimal output unless there are failures
    if pytest --tb=short -q 2>&1 | tee /tmp/pytest_output.txt; then
        TEST_COUNT=$(grep -oP '\d+(?= passed)' /tmp/pytest_output.txt | head -1)
        if [ -n "$TEST_COUNT" ]; then
            echo -e "  ${GREEN}✓ All $TEST_COUNT tests passed${NC}"
        else
            echo -e "  ${GREEN}✓ All tests passed${NC}"
        fi
    else
        echo -e "  ${RED}✗ pytest failed${NC}"
        echo ""
        echo "Test failures detected:"
        cat /tmp/pytest_output.txt
        echo ""
        echo "Please fix failing tests before committing."
        ERRORS=$((ERRORS + 1))
    fi

    # Clean up temp file
    rm -f /tmp/pytest_output.txt
else
    echo "  ✓ No Python files changed, skipping pytest"
fi

# Summary
echo ""
echo "======================================"
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}❌ Validation FAILED with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before committing."
    echo "See CLAUDE.md for project structure guidelines."
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Validation passed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "Review warnings above. They may indicate missing configs, tests, or docs."
    echo "See CLAUDE.md for best practices."
    echo ""
    echo "Proceeding with commit..."
    exit 0
else
    echo -e "${GREEN}✅ All validations passed!${NC}"
    exit 0
fi
