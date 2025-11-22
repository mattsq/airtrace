#!/usr/bin/env bash
# Test Integrity Audit Script
# Usage: ./scripts/audit_tests.sh [--detailed]

set -euo pipefail

TESTS_DIR="tests"
DETAILED=${1:-""}

echo "========================================="
echo "Test Integrity Audit"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# === METRIC 1: Weak Assertions ===
echo "📊 METRIC 1: Weak Assertions"
echo "----------------------------"

is_not_none_count=$(grep -rn "assert.*is not None" "$TESTS_DIR/" 2>/dev/null | wc -l)
echo "  • 'assert ... is not None': $is_not_none_count instances"

if [ "$is_not_none_count" -gt 50 ]; then
    echo -e "  ${RED}❌ CRITICAL: >50 weak assertions${NC}"
elif [ "$is_not_none_count" -gt 20 ]; then
    echo -e "  ${YELLOW}⚠️  WARNING: >20 weak assertions${NC}"
else
    echo -e "  ${GREEN}✓ OK${NC}"
fi

if [ "$DETAILED" = "--detailed" ]; then
    echo ""
    echo "  Top files with weak assertions:"
    grep -rn "assert.*is not None" "$TESTS_DIR/" 2>/dev/null | cut -d: -f1 | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'
fi

echo ""

# === METRIC 2: Excessive Mocking ===
echo "📊 METRIC 2: Excessive Mocking"
echo "----------------------------"

mock_count=$(grep -rn "monkeypatch\.setattr\|@patch\|@mock.patch" "$TESTS_DIR/" 2>/dev/null | wc -l)
echo "  • Total mocking calls: $mock_count"

if [ "$DETAILED" = "--detailed" ]; then
    echo ""
    echo "  Files with most mocking:"
    grep -rn "monkeypatch\.setattr\|@patch\|@mock.patch" "$TESTS_DIR/" 2>/dev/null | cut -d: -f1 | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'
    echo ""
    echo "  Tests with >5 mocks (potential over-mocking):"
    for file in $(find "$TESTS_DIR" -name "test_*.py"); do
        if [ -f "$file" ]; then
            count=$(grep "monkeypatch\.setattr\|@patch\|@mock.patch" "$file" 2>/dev/null | wc -l)
            if [ "$count" -gt 5 ]; then
                echo "    ⚠️  $file: $count mocks"
            fi
        fi
    done
fi

echo ""

# === METRIC 3: Stub Classes ===
echo "📊 METRIC 3: Stub/Fake Classes"
echo "----------------------------"

stub_count=$(grep -rn "^class _[A-Z]\|^class Fake\|^class Stub\|^class Dummy[A-Z]" "$TESTS_DIR/" 2>/dev/null | wc -l)
echo "  • Stub classes defined: $stub_count"

if [ "$DETAILED" = "--detailed" ]; then
    echo ""
    echo "  Files with most stubs:"
    grep -rn "^class _[A-Z]\|^class Fake\|^class Stub\|^class Dummy[A-Z]" "$TESTS_DIR/" 2>/dev/null | cut -d: -f1 | sort | uniq -c | sort -rn | head -5 | sed 's/^/    /'
fi

echo ""

# === METRIC 4: Test Count ===
echo "📊 METRIC 4: Test Coverage"
echo "----------------------------"

test_file_count=$(find "$TESTS_DIR" -name "test_*.py" | wc -l)
test_func_count=$(grep -rn "^def test_\|^    def test_" "$TESTS_DIR/" 2>/dev/null | wc -l)

echo "  • Test files: $test_file_count"
echo "  • Test functions: ~$test_func_count"

echo ""

# === METRIC 5: Problematic Patterns ===
echo "📊 METRIC 5: Problematic Patterns"
echo "----------------------------"

# Check for tests with no assertions
echo -n "  • Scanning for tests with suspicious patterns... "

empty_asserts=0
hardcoded_returns=$(grep -rn "return.*\[\]$\|return None$\|return.*0\.0\)" "$TESTS_DIR/" 2>/dev/null | grep -v "pragma: no cover" | wc -l)

echo "Done"
echo "  • Hardcoded returns in stubs: $hardcoded_returns instances"

if [ "$DETAILED" = "--detailed" ]; then
    echo ""
    echo "  Hardcoded return examples:"
    grep -rn "return.*\[\]$\|return None$\|return.*0\.0\)" "$TESTS_DIR/" 2>/dev/null | grep -v "pragma: no cover" | head -5 | sed 's/^/    /'
fi

echo ""

# === SUMMARY ===
echo "========================================="
echo "SUMMARY"
echo "========================================="

issues=0

if [ "$is_not_none_count" -gt 50 ]; then
    echo -e "${RED}❌ CRITICAL: Too many weak assertions ($is_not_none_count)${NC}"
    issues=$((issues + 1))
elif [ "$is_not_none_count" -gt 20 ]; then
    echo -e "${YELLOW}⚠️  WARNING: High number of weak assertions ($is_not_none_count)${NC}"
fi

if [ "$mock_count" -gt 30 ]; then
    echo -e "${YELLOW}⚠️  WARNING: High number of mocking calls ($mock_count)${NC}"
fi

if [ "$stub_count" -gt 40 ]; then
    echo -e "${YELLOW}⚠️  WARNING: Many stub classes ($stub_count)${NC}"
fi

if [ "$issues" -eq 0 ]; then
    echo -e "${GREEN}✓ No critical issues detected${NC}"
    echo ""
    echo "Note: This is an automated scan. Manual review may still find issues."
else
    echo ""
    echo "Run with --detailed flag for more information:"
    echo "  ./scripts/audit_tests.sh --detailed"
fi

echo ""
echo "For full audit report, see: docs/audit/test_integrity_audit.md"
echo "For remediation examples, see: docs/audit/remediation_examples.md"
echo ""

exit $issues
