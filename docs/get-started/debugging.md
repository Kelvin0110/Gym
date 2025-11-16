# Debugging Servers

This guide shows you how to debug NeMo Gym servers using VS Code's debugger.

## Prerequisites

Make sure you have the dev dependencies installed:

```bash
pip install -e ".[dev]"
```

This includes `debugpy`, which is required for remote debugging.

## VS Code Configuration

Add these configurations to your `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "📎 Attach: Server Port 5678",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "${workspaceFolder}"
        }
      ],
      "justMyCode": false
    },
    {
      "name": "📎 Attach: Server Port 5679",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5679
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "${workspaceFolder}"
        }
      ],
      "justMyCode": false
    },
    {
      "name": "📎 Attach: Server Port 5680",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5680
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "${workspaceFolder}"
        }
      ],
      "justMyCode": false
    },
    {
      "name": "📎 Attach: Server Port 5681",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5681
      },
      "pathMappings": [
        {
          "localRoot": "${workspaceFolder}",
          "remoteRoot": "${workspaceFolder}"
        }
      ],
      "justMyCode": false
    }
  ]
}
```

:::{note}
If you already have a `launch.json`, just add these configurations to your existing `"configurations"` array.
:::

## Debugging Workflow

### Step 1: Start Servers with Debug Mode

In your terminal, enable debug mode and start your servers:

```bash
# Enable subprocess debugging
export NEMO_GYM_DEBUG_SUBPROCESS=1
export NEMO_GYM_DEBUG_PORT=5678

# Start your servers (adjust config paths as needed)
ng_run "+config_paths=[resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

You'll see output like:
```
🐛 Subprocess debug mode enabled - servers will use ports starting from 5678
   [openai_model] listening for debugger on port 5678
   [weather_server] listening for debugger on port 5679
```

The servers are now running and waiting for debugger attachment.

### Step 2: Attach VS Code Debugger

1. **Set your breakpoints** in the server code (e.g., `responses_api_models/openai_model/app.py`)

2. **Attach the debugger:**
   - Press `F5` or `Ctrl+Shift+D` (Cmd+Shift+D on Mac)
   - Select **"📎 Attach: Server Port 5678"** (for the first server)
   - Or select the port corresponding to the server you want to debug

3. You'll see the debugger attach successfully in VS Code's debug panel.

### Step 3: Trigger Your Code

Run your client or test to trigger requests to the servers:

```bash
python responses_api_agents/simple_agent/client.py
```

Your breakpoints will now hit! 🎉

## Quick Reference

### Environment Variables

| Variable | Description |
|----------|-------------|
| `NEMO_GYM_DEBUG_SUBPROCESS` | Set to `1` to enable debug mode for subprocesses |
| `NEMO_GYM_DEBUG_PORT` | Base port number (default: 5678). Each server gets sequential ports |

### Port Assignment

When debug mode is enabled, servers are assigned sequential ports:
- First server: 5678 (base port)
- Second server: 5679 (base + 1)
- Third server: 5680 (base + 2)
- And so on...

### Debugging Multiple Servers

You can attach to multiple servers simultaneously:

1. Attach to port 5678 (press F5 → select port 5678)
2. Attach to port 5679 (press F5 again → select port 5679)
3. Both servers are now being debugged!

VS Code will show multiple debug sessions in the Call Stack panel.

## Troubleshooting

### "Failed to attach"

- Make sure servers are running with `ng_run`
- Check the port numbers in the terminal output
- Verify `NEMO_GYM_DEBUG_SUBPROCESS=1` is set

### Breakpoints not hitting

- Verify you attached to the correct port
- Make sure you're triggering the right endpoint
- Check that the request is actually reaching the server
- Ensure breakpoints are set before the code executes

### "debugpy not found"

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

## How It Works

When `NEMO_GYM_DEBUG_SUBPROCESS=1` is set, the CLI automatically starts each server subprocess with:

```bash
python -m debugpy --listen 0.0.0.0:PORT app.py
```

This enables remote debugging without modifying your application code. VS Code connects to these debug ports using the attach configurations.

