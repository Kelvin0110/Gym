# CLI Command Reference

This page documents all available NeMo Gym CLI commands.

:::{note}
The NeMo Gym CLI uses a unified entry point with organized subcommands. All commands are accessed through `ng` or `nemo_gym`.
:::

## Quick Reference

```bash
# Display help
ng --help

# Get help for a command group
ng server --help

# Get detailed help for a specific command
ng server run --help
```

---

## Command Structure

```bash
ng <command-group> <command> [options]
```

**Command Groups:**
- `server`: Server management
- `data`: Data collection and preparation
- `dataset`: Dataset registry operations
- `config`: Configuration utilities
- `test`: Test commands

**Utility Commands:**
- `ng version [--json]`: Show version and system information
- `ng --help`: Show help message

---

## Main CLI

```{eval-rst}
.. click:: nemo_gym.cli_new:cli
   :prog: ng
   :nested: full
```

---

## Getting Help

For detailed help on any command, use the `--help` flag:

```bash
ng --help                    # Show all command groups
ng server --help             # Show server commands
ng server run --help         # Show detailed help for run command
ng test --help               # Show test commands
ng data --help               # Show data commands
ng dataset --help            # Show dataset commands
ng config --help             # Show config commands
```

---

## Tab Completion

The NeMo Gym CLI supports tab completion for bash, zsh, and fish. To enable it:

**Bash:**
```bash
_NG_COMPLETE=bash_source ng > ~/.ng-complete.bash
echo ". ~/.ng-complete.bash" >> ~/.bashrc
source ~/.bashrc
```

**Zsh:**
```bash
_NG_COMPLETE=zsh_source ng > ~/.ng-complete.zsh
echo ". ~/.ng-complete.zsh" >> ~/.zshrc
source ~/.zshrc
```

**Fish:**
```bash
_NG_COMPLETE=fish_source ng > ~/.config/fish/completions/ng.fish
```

After enabling, you can use tab completion:
```bash
ng <TAB>              # Shows: server, data, dataset, config, test, version
ng server <TAB>       # Shows: run, init
ng test <TAB>         # Shows: server, all
ng data <TAB>         # Shows: collect, prepare, view
ng dataset <TAB>      # Shows: upload, download, delete, migrate
ng config <TAB>       # Shows: dump, validate
```
