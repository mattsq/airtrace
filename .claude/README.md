# Claude Code Configuration

This directory contains Claude Code specific configuration and hooks for the AirTrace project.

## Hooks

The `hooks/` directory contains shell scripts that enforce AirTrace's project structure and provide helpful guidance to AI agents.

### Available Hooks

#### `session-start.sh`
Runs when a new Claude Code session starts.

**Purpose**: Orient new agents to the project
- Displays welcome message with key files to read
- Shows quick reference for common tasks
- Lists critical rules and principles
- Displays current repository state

**When it runs**: Automatically at session start (if configured in Claude Code)

#### `pre-commit.sh`
Runs before each git commit.

**Purpose**: Validate changes comply with AirTrace framework rules
- **Rule 1**: Validates file locations (prevents commits to `data/`, etc.)
- **Rule 2-4**: Checks config-code synchronization for models, transforms, tasks
- **Rule 5**: Validates `__init__.py` updates when adding modules
- **Rule 6**: Checks for test coverage
- **Rule 7**: Verifies type hints in Python files

**Exit codes**:
- `0`: All checks passed (or warnings only)
- `1`: Errors found, commit blocked

**When it runs**: Automatically before `git commit` (if configured as a git hook)

#### `post-commit.sh`
Runs after a successful git commit.

**Purpose**: Provide impact analysis and reminders
- Categorizes changed files (models, configs, tests, etc.)
- Provides context-aware reminders for testing and validation
- Suggests next steps based on what was changed

**When it runs**: Automatically after `git commit` (if configured as a git hook)

## Setting Up Hooks

### As Git Hooks (Optional)

To use these as standard git hooks, symlink them:

```bash
# From repository root
ln -sf ../../.claude/hooks/pre-commit.sh .git/hooks/pre-commit
ln -sf ../../.claude/hooks/post-commit.sh .git/hooks/post-commit
```

### As Claude Code Hooks

Claude Code can automatically run these hooks. Configuration depends on your Claude Code setup.

Typical configuration in `.claude/config.json`:

```json
{
  "hooks": {
    "session-start": ".claude/hooks/session-start.sh",
    "pre-commit": ".claude/hooks/pre-commit.sh",
    "post-commit": ".claude/hooks/post-commit.sh"
  }
}
```

## Customizing Hooks

Feel free to modify hooks to match your workflow:

1. **Add new validation rules**: Edit `pre-commit.sh`
2. **Change reminder messages**: Edit `post-commit.sh`
3. **Add new hooks**: Create new `.sh` files and make executable

### Hook Best Practices

- Keep hooks fast (< 5 seconds)
- Provide clear, actionable error messages
- Use warnings for suggestions, errors for blockers
- Test hooks manually: `./.claude/hooks/pre-commit.sh`

## Disabling Hooks Temporarily

If you need to bypass hooks:

```bash
# For git hooks
git commit --no-verify

# For Claude Code hooks
# (depends on your Claude Code configuration)
```

**Warning**: Only bypass hooks if you know what you're doing. They exist to prevent common mistakes.

## Troubleshooting

### Hook not running
- Check if file is executable: `ls -l .claude/hooks/*.sh`
- Make executable: `chmod +x .claude/hooks/*.sh`
- Verify hook configuration in Claude Code settings

### Hook failing incorrectly
- Run hook manually to see full output: `./.claude/hooks/pre-commit.sh`
- Check if you're following AirTrace conventions (see `CLAUDE.md`)
- File an issue if the hook has a bug

### Hook too strict
- Hooks are opinionated to enforce best practices
- Review the specific rule in the hook script
- Modify the hook if your use case is legitimate
- Consider if there's a better way that follows conventions

## Philosophy

These hooks embody AirTrace's core principles:

1. **Config-Code Synchronization**: Every component needs both implementation and config
2. **Structural Discipline**: Files belong in specific places
3. **Testing Culture**: New code should have tests
4. **Documentation**: Surprising findings go in `MEMORY.md`

The hooks are guardrails, not restrictions. They help maintain consistency and catch common mistakes early.

---

*For more information, see `CLAUDE.md` (agent guide) and `MEMORY.md` (agent learnings).*
