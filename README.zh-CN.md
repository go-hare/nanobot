# nanobot（中文）

nanobot 是一个超轻量级个人 AI 助手，采用 **融合管线架构（IQ + EQ）**。

## 安装

从源码安装（推荐开发场景）：

```bash
git clone https://github.com/HKUDS/nanobot.git
cd nanobot
pip install -e .
```

从 PyPI 安装：

```bash
pip install nanobot-ai
```

## 快速开始

1）初始化本地工作区：

```bash
nanobot onboard
```

2）编辑 `~/.nanobot/config.json`（最小配置）：

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

3）CLI 对话：

```bash
nanobot agent
```

4）启动网关（对接聊天渠道）：

```bash
nanobot gateway
```

## 架构

当前运行时采用融合优先主流程：

1. `SignalExtractor`：回合信号提取
2. `PolicyEngine`：连续 IQ/EQ 配方生成
3. `IQEngine`：事实/工具执行
4. `EQEngine`：共情与表达渲染
5. `Composer`：最终回复合成
6. `ReflectionEngine`：结构化策略调整写回

## 记忆层

- `SemanticStore`：语义/事实记忆
- `RelationalStore`：关系/偏好记忆
- `AffectiveStore`：PAD 情绪轨迹
- `PolicyStateStore`：运行时策略状态
- `MemoryFacade`：统一读写入口

## 渠道支持

当前支持：

- Telegram
- Discord
- WhatsApp
- 飞书（Feishu）
- 钉钉（DingTalk）
- Slack
- Email
- QQ
- Matrix
- Mochat

渠道配置位于 `~/.nanobot/config.json` 的 `channels` 字段下。

## MCP（Model Context Protocol）

可通过配置接入 MCP 服务：

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

## 安全建议

生产环境建议开启工作区沙箱：

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

## 常用命令

```bash
nanobot onboard
nanobot agent
nanobot gateway
nanobot status
nanobot cron list
nanobot channels status
```

## 项目结构

```text
nanobot/
├── core/                 # 融合流程编排
├── engines/              # IQ / EQ 执行层
├── memory/               # 分层记忆实现
├── meta/                 # 反思与策略调整
├── runtime/              # 守护进程与运行时调度
├── agent/                # loop/context/tools/subagent
├── channels/             # 聊天渠道集成
├── providers/            # LLM provider 适配
├── session/              # 会话管理
└── cli/                  # 命令行入口
```

## 社区

见 `COMMUNICATION.md`。

## 许可证

MIT。

