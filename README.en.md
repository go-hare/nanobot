# nanobot (English)

nanobot is an ultra-lightweight personal AI assistant with a **fusion pipeline architecture (IQ + EQ)**.

## Install

Install from source (recommended for contributors):

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

Install from PyPI:

```bash
pip install nanobot-ai
```

## Quick Start

1) Initialize local workspace:

```bash
nanobot onboard
```

2) Configure `~/.nanobot/config.json` (minimum):

```json
{
  "providers": {
    "openrouter": {
      "apiKey": "sk-or-v1-xxx"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-opus-4-5",
      "provider": "openrouter"
    }
  }
}
```

3) Chat in CLI:

```bash
nanobot agent
```

4) Run gateway for chat channels:

```bash
nanobot gateway
```

## Architecture

Current runtime is fusion-first:

1. `SignalExtractor` -> turn-level signal extraction
2. `PolicyEngine` -> continuous IQ/EQ mix generation
3. `IQEngine` -> factual/tool execution
4. `EQEngine` -> empathy + style rendering
5. `Composer` -> final response composition
6. `ReflectionEngine` -> structured policy adjustment write-back

## Memory Layer

- `SemanticStore`: factual/semantic memory
- `RelationalStore`: preference/relationship memory
- `AffectiveStore`: PAD emotional traces
- `PolicyStateStore`: runtime policy adjustment state
- `MemoryFacade`: unified read/write API

## Channels

Supported channels include:

- Telegram
- Discord
- WhatsApp
- Feishu
- DingTalk
- Slack
- Email
- QQ
- Matrix
- Mochat

Channel config is in `~/.nanobot/config.json` under `channels`.

## MCP (Model Context Protocol)

nanobot supports MCP servers via config:

```json
{
  "tools": {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"]
      },
      "remote-mcp": {
        "url": "https://example.com/mcp/",
        "headers": {
          "Authorization": "Bearer xxx"
        }
      }
    }
  }
}
```

## Security

For production, enable workspace sandbox:

```json
{
  "tools": {
    "restrictToWorkspace": true
  }
}
```

## Docker

```bash
docker build -t nanobot .
docker run -v ~/.nanobot:/root/.nanobot --rm nanobot onboard
docker run -v ~/.nanobot:/root/.nanobot -p 18790:18790 nanobot gateway
```

## CLI Commands

```bash
nanobot onboard
nanobot agent
nanobot gateway
nanobot status
nanobot cron list
nanobot channels status
```

## Project Structure

```text
nanobot/
├── core/                 # fusion orchestration
├── engines/              # IQ / EQ execution
├── memory/               # layered memory stores
├── meta/                 # reflection and policy adjustment
├── runtime/              # daemon/scheduler runtime
├── agent/                # loop/context/tools/subagent
├── channels/             # channel integrations
├── providers/            # LLM providers
├── session/              # session management
└── cli/                  # CLI entrypoints
```

## Community

See `COMMUNICATION.md`.

## License

MIT.

