# Claude Configuration

Configuration files for Claude Code.

## Statusline Preview

```
📂 my-app | ★ Claude Haiku | Context: ▓▓▓░░░░░░░ 35% | Session: ▓▓░░░░░░░░ 22% | Weekly: ▓░░░░░░░░░ 9%
```

Custom statusline showing folder, model, context usage, and plan limits with color-coded progress bars.

## Files

- **CLAUDE.md** (repo root) — Global instructions and coding guidelines for Claude Code sessions
- **.claude/settings.json** — Claude Code settings including permissions, statusline configuration, and preferences

## Statusline

The `settings.json` includes a custom statusline that displays:
- Current folder
- Active model
- Context window usage
- Session and weekly plan usage limits

All with color-coded progress bars for easy monitoring.

**Note:** The statusline command points to `~/.claude/statusline.sh`. Install that script in your home `.claude` directory for the statusline to work.
