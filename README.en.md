# emoticorebot (English)

emoticorebot is an ultra-lightweight personal AI assistant with a **fusion pipeline architecture (IQ + EQ)**, derived from the original Nanobot project.

## Install

Install from source (recommended for contributors):

```bash
git clone https://github.com/HKUDS/emoticorebot.git
cd emoticorebot
pip install -e .
```

Install from PyPI:

```bash
pip install emoticorebot-ai
```

## Quick Start

1) Initialize local workspace:

```bash
emoticorebot onboard
```

2) Configure `~/.emoticorebot/config.json` (minimum):

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
emoticorebot agent
```

4) Run gateway for chat channels:

```bash
emoticorebot gateway
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

Channel config is in `~/.emoticorebot/config.json` under `channels`.

## MCP (Model Context Protocol)

emoticorebot supports MCP servers via config:

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
docker build -t emoticorebot .
docker run -v ~/.emoticorebot:/root/.emoticorebot --rm emoticorebot onboard
docker run -v ~/.emoticorebot:/root/.emoticorebot -p 18790:18790 emoticorebot gateway
```

## CLI Commands

```bash
emoticorebot onboard
emoticorebot agent
emoticorebot gateway
emoticorebot status
emoticorebot cron list
emoticorebot channels status
```

## Project Structure

```text
emoticorebot/
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

