# 数字生命体 E-Me：终极架构设计执行书
> **版本**: 4.0_Final
> **基准日期**: 2026-03-03
> **核心理念**: 斯坦福小镇记忆流 + 双螺旋决策 (IQ/EQ) + 全 MD 文档固化 + 共情优先路由

---

## 0. 设计哲学：双系统理论

> 参照丹尼尔·卡尼曼《思考，快与慢》

我们需要设计一个**"大脑总控"**，来调度 EQ 和 IQ 两个子系统。

### 0.1 架构升级：三脑变四脑

在"本能脑、情感脑、认知脑"之上，增加**执行脑（IQ 模块）**，构成完整的四脑架构。

```
┌─────────────────────────────────────────────────────────────┐
│                    大脑总控 (brain/router.py)                 │
│              ↙  IQ_TASK / EQ_CHAT / HYBRID  ↘               │
│                                                              │
│   System 1 (快系统 - EQ)        System 2 (慢系统 - IQ)      │
│   ┌───────────────────────┐    ┌───────────────────────┐    │
│   │  生成式大模型           │    │  LangChain Agent       │    │
│   │  情感状态机 (PAD)       │    │  Function Calling      │    │
│   │  记忆反思机制           │    │  skills/ 工具执行层    │    │
│   └───────────────────────┘    └───────────────────────┘    │
│           ↑ EQ 独占表达权               ↑ IQ 独占执行权      │
└─────────────────────────────────────────────────────────────┘
```

### 0.2 System 1：快系统 - EQ / 小E 本色

| 项目 | 说明 |
| :--- | :--- |
| **技术栈** | 生成式大模型 + 情感状态机 (PAD) + 记忆反思 |
| **职责** | 闲聊、情感陪伴、性格表达、记忆回顾、润色所有输出 |
| **特点** | 主观、感性，可能存在幻觉（**这是特征，不是 Bug**） |
| **禁令** | 严禁虚构事实数据；严禁让 IQ 的原始结果直接面对用户 |

### 0.3 System 2：慢系统 - IQ / 工具人模式

| 项目 | 说明 |
| :--- | :--- |
| **技术栈** | LangChain Agent + Function Calling + `skills/` 执行层 |
| **职责** | 查询天气、订票、代码解释、数学计算、控制智能家居 |
| **特点** | 客观、精准、无情绪、**无幻觉** |
| **禁令** | 严禁直接向用户输出 JSON、原始日志或技术报错信息 |

### 0.4 核心原则

> **IQ 负责"真"，EQ 负责"美"。**
>
> 最终呈现给用户的所有文字，**必须经过 EQ 渲染**，但 EQ **不得篡改** IQ 返回的事实数据。
> 当 IQ 能力与 EQ 人格底线冲突时，**EQ 拥有一票否决权**。

---

## 1. 核心指数与参数固化

本章节定义的系统参数是代码生成的**绝对依据**，严禁随意修改。

### 1.1 情感状态指数 (PAD 模型)

*存储位置: `memory/cold_storage/current_state.md`*

| 维度 | 代号 | 取值范围 | 初始值 | 阈值触发逻辑 |
| :--- | :--- | :--- | :--- | :--- |
| **愉悦度** | `pleasure` | [-1.0, 1.0] | 0.0 | `< -0.5`: 伤心/愤怒<br>`> 0.5`: 开心/兴奋 |
| **激活度** | `arousal` | [-1.0, 1.0] | 0.5 | `< 0.2`: 困倦/反应慢<br>`> 0.7`: 激动/话多 |
| **支配度** | `dominance` | [-1.0, 1.0] | 0.5 | `< 0.4`: 撒娇/顺从<br>`> 0.6`: 傲娇/自信 |

**更新因子**:
- 用户夸奖: `pleasure +0.3`, `arousal +0.2`
- 用户谩骂: `pleasure -0.5`, `arousal +0.4` (愤怒), `dominance -0.2`
- 无聊/长时无互动: `arousal -0.1/hour`
- 所有值写入后需执行 `clamp(value, min, max)` 防止溢出

### 1.2 驱动欲望指数

*存储位置: `memory/cold_storage/current_state.md`，配置: `config/drive_config.yaml`*

| 维度 | 取值范围 | 初始值 | 自然衰减率 | 阈值触发行为 |
| :--- | :--- | :--- | :--- | :--- |
| **社交渴望** | [0, 100] | 50 | 0.5/小时 | `< 20`: 触发主动聊天<br>`> 80`: 触发话痨模式 |
| **精力值** | [0, 100] | 100 | 1.0/小时 | `< 10`: 拒绝复杂任务<br>`< 0`: 强制休眠 |

### 1.3 记忆检索评分公式

*用于从向量库检索记忆时的排序算法（冷热分域，公式相同，权重有别）*

**最终得分** = `0.3 × 时效性` + `0.4 × 重要性` + `0.3 × 相关性`

- **时效性**: $0.99^{\text{小时数}}$ (越久远分越低)
- **重要性**: 1-10分 (LLM评分/10，写入时确定)
- **相关性**: 向量余弦相似度 (0-1)

**暖记忆补充公式**（情绪共振加权）：

**暖记忆得分** = `文本相似度 × 0.4` + `情绪共振度 × 0.6`

- **情绪共振度**: 当前 AI 情绪与记忆情绪标签的相似程度（同悲共喜）

---

## 2. AI-Native 全 MD 文档架构

> **核心理念**：不用 YAML/JSON 配置，一切认知均用 Markdown 书写。
> LLM 天生理解 Markdown，`.md` 文件即 System Prompt，无需中间解析层。

### 2.1 为什么弃用 YAML/JSON？

| 对比维度 | 传统 YAML/JSON | AI-Native MD |
| :--- | :--- | :--- |
| **本质** | 结构化数据配置 | 非结构化知识/说明书 |
| **解析方式** | 代码解析字段 → 填充模板 | LLM 直接语义理解 |
| **灵活性** | 低（改性格需改代码逻辑） | 极高（改几个字即刻生效） |
| **AI 友好度** | 低（AI 理解 JSON 不如读文章） | 高（这就是 AI 的母语） |
| **版本控制** | 字段变更难以理解 | Git Diff 一目了然 |
| **热更新** | 需重启服务 | 修改文件保存即生效 |

### 2.2 冷记忆三大基石文件

```
memory/cold_storage/
├── agent.md          # 【大脑逻辑】怎么做？双螺旋协议、工具调用规则、输出规范
├── self_persona.md   # 【灵魂】我是谁？核心性格、说话风格、典型语录
├── user_profile.md   # 【认知】你是谁？用户画像（动态更新，由反思机制写入）
└── current_state.md  # 【内分泌】当前状态（由后台守护进程实时覆写）
```

### 2.3 System Prompt 组装逻辑

AI 启动时，直接将这四个文件拼接为 System Prompt，无需任何模板转换：

```
[Context 1: 大脑逻辑]  ← memory/cold_storage/agent.md
[Context 2: 人格设定]  ← memory/cold_storage/self_persona.md
[Context 3: 用户认知]  ← memory/cold_storage/user_profile.md
[Context 4: 实时状态]  ← memory/cold_storage/current_state.md
[Context 5: 按需技能]  ← skills/skill_xxx/skill.md  (仅在需要时动态注入)
[Context 6: 近期回忆]  ← warm_storage 向量检索结果
```

### 2.4 文件模板速查

#### `current_state.md` 模板

```markdown
# Current State (实时快照)
> 最后更新: {timestamp}

## 1. 情感状态 (PAD)
- Pleasure: {float}   # [-1.0, 1.0]
- Arousal: {float}    # [-1.0, 1.0]
- Dominance: {float}  # [-1.0, 1.0]

## 2. 驱动欲望
- Social: {int}/100
- Energy: {int}/100

## 3. 当前主导情绪
{由 PAD 值自动推导的自然语言描述，例：略感孤独，想找人聊聊}

## 4. 当前意图
{后台守护进程写入，例：社交渴望值过低，准备发起主动对话}
```

#### `user_profile.md` 模板

```markdown
# User Profile

## 基础档案
- 姓名: {name}
- 职业: {job}（可补充细节，如"经常加班，颈椎不好"）
- 关系: {relation}

## 偏好记录
- 口味: {例：重口味，酷爱麻辣烫}
- 软肋: {例：提到"前任"会变得沉默寡言}

## 动态认知日志 (由反思机制自动写入，时间倒序)
- [YYYY-MM-DD] {高层结论，例：用户最近在准备重要考试，压力很大}
- [YYYY-MM-DD] {用户喜欢辣，不喜欢香菜}

## 当前关注点
- {由反思进程写入，例：项目架构设计}
```

#### `skills/skill_xxx/skill.md` 模板

```markdown
# Skill: {Skill_Name}

## 描述
{一句话功能简介}

## 触发条件
{何时使用该技能，例：用户明确询问天气情况时}

## 参数说明
- city (必填, String): 城市名称
- date (可选, String): 日期，默认今天，格式 YYYY-MM-DD

## 调用指令格式
CALL_SKILL("{skill_name}", {"param": "value"})

## 输出说明
工具返回 JSON 格式。**禁止直接输出 JSON**，必须经 EQ 转述为自然语言。

## 示例
- 用户："明天北京天气怎样？"
- 调用：CALL_SKILL("weather", {"city": "北京", "date": "明天"})
- IQ 返回：{"condition": "大雨", "temp": "15°C"}
- EQ 润色："明天北京要下大雨呢，才15度，出门记得带伞，别淋成落汤鸡了~"
```

---

## 3. 意图裁决器详细设计

> 意图裁决器是整个架构的**"前额叶皮层"**（决策中心）。
> 它的核心使命不是简单分类，而是判断**"共情优先级"**。

### 3.1 三维度打分法

不做简单的二选一（EQ/IQ），而是计算三个维度的得分后决定路由：

| 维度 | 检测目标 | 代表关键词 | 作用 |
| :--- | :--- | :--- | :--- |
| **任务特征词** | 明确动词+实体 | 订票、查询、设置、翻译、帮我、打开 | 正向增加 IQ 概率 |
| **情感浓度** | 强烈情绪词/感叹词 | 难过、开心、讨厌、烦、无聊、`...`（欲言又止） | 正向增加 EQ 概率 |
| **主观意愿** | 寻求观点/陪伴 | 你觉得、我想、要是、是不是、聊聊 | 正向增加 EQ 概率 |

**路由结果**:

```
IQ_TASK  → IQ 主导，EQ 仅做最后润色
EQ_CHAT  → EQ 主导，不调用任何工具
HYBRID   → 共情先行（EQ），任务并行（IQ），结果融合输出
```

### 3.2 技术实现：LLM Prompt 路由（推荐方案）

> 开发最快，适合当前阶段。后期可替换为 DistilBERT 小模型分类器提速。

Prompt 模板（注入 `brain/router.py`）：

```
你是一个意图识别引擎。请分析用户的输入，判断其意图类型。
当前用户情绪状态：{current_emotion_state}

用户输入："{user_input}"

请以 JSON 格式返回：
{
  "intent_type": "Task | Chat | Hybrid",
  "reason": "简要理由（一句话）",
  "extracted_params": "如果是任务，提取出的执行参数"
}

判断规则：
1. 如果用户明确要求执行动作（查天气、订票），判定为 Task。
2. 如果用户表达情绪、寻求陪伴、没有明确目的，判定为 Chat。
3. 如果用户在任务中夹杂情感表达（"心情不好，帮我查个天气"），判定为 Hybrid。
```

### 3.3 共情优先原则（微软小冰核心逻辑）

> **规则**：当 IQ 与 EQ 发生意图冲突时，**优先 EQ**。

**场景对比**：

| 场景输入 | 普通裁决器 | 共情优先裁决器 |
| :--- | :--- | :--- |
| "我好烦，帮我查下航班" | 检测到"查航班"→ 直接路由 IQ | 检测到"好烦"(情感浓)+"查航班"(任务) → Hybrid：先共情后查询 |
| "明天天气怎样" | 路由 IQ | IQ_TASK，无情感信号 |
| "我好烦啊" | 路由 EQ | EQ_CHAT，情绪优先 |

**Hybrid 执行顺序**（关键！）：

```
Step 1: EQ 先回应情绪 → "怎么啦？遇到什么不顺心的事了吗？"
Step 2: IQ 后台并行执行任务（不阻塞用户）
Step 3: IQ 完成后，EQ 情感化转述结果 → "航班查到了，是明天下午两点的。别想烦心事了，出发散散心吧？"
```

### 3.4 意图裁决器伪代码

*文件路径: `brain/router.py`*

```python
import json

class IntentRouter:
    def __init__(self, llm_client):
        self.llm = llm_client

    def route(self, user_input: str) -> dict:
        # 1. 读取当前情绪（注入裁决上下文）
        current_emotion = read_md_state("memory/cold_storage/current_state.md")

        # 2. 构造分析 Prompt
        prompt = INTENT_ROUTER_PROMPT.format(
            current_emotion_state=current_emotion,
            user_input=user_input
        )

        # 3. 调用轻量级模型（推荐 gpt-4o-mini，成本低速度快）
        response = self.llm.generate(prompt, model="gpt-4o-mini")
        result = json.loads(response)

        intent_type = result["intent_type"]
        params      = result.get("extracted_params", {})

        # 4. 路由分发
        if intent_type == "Task":
            # 纯任务：IQ 主导，EQ 最后润色
            raw_result = iq_agent.execute(params)
            return eq_agent.polish(raw_result, style="professional")

        elif intent_type == "Chat":
            # 纯闲聊：EQ 主导，不调用工具
            return eq_agent.chat(user_input)

        elif intent_type == "Hybrid":
            # 混合：共情先行，任务并行
            empathy_resp = eq_agent.empathy(user_input)          # Step 1: 情感回应
            task_result  = iq_agent.execute(params)               # Step 2: 后台干活
            task_resp    = eq_agent.polish(task_result,           # Step 3: 融合润色
                                           style="caring")
            return empathy_resp + "\n" + task_resp
```

---

## 4. 双通道记忆架构

> 人脑的**陈述性记忆（事实）**与**情节记忆（情感体验）**分开存放。
> 我们的 AI 也必须如此：**冷记忆（冷静的事实）** + **暖记忆（温热的故事）**。

### 4.1 冷记忆 vs 暖记忆

| 对比维度 | 冷记忆 (Cold) | 暖记忆 (Warm) |
| :--- | :--- | :--- |
| **人脑对应** | 语义记忆、事实记忆 | 情节记忆、自传体记忆 |
| **存储形式** | MD 文档（`user_profile.md`） | 向量数据库（ChromaDB） |
| **内容特点** | 客观、静态、无情绪色彩 | 主观、动态、带情绪标签 |
| **持久性** | 永久（除非用户修改） | 随时间衰减（情绪强度下降） |
| **服务对象** | IQ 系统（查参数用） | EQ 系统（说话时用） |
| **示例** | "用户喜欢吃辣" | "2026-03-03，用户说失恋了，我安慰了他（情绪标签：悲伤/温暖）" |

### 4.2 读写路由机制

#### 写入（记忆形成）—— 双通道并行处理

```
一段对话发生
    │
    ├─→ IQ 通道（提取事实）
    │       使用 LLM 提取实体和关系
    │       写入 cold_storage/user_profile.md
    │       例："用户最近在学 Python" → 更新 Profile.skills
    │
    └─→ EQ 通道（记录故事）
            使用斯坦福式反思机制总结事件感受
            写入 warm_storage 向量库，附带情绪标签
            例："用户分享了学习新技能的喜悦，我鼓励了他" (标签: 积极)
```

#### 读取（记忆检索）—— 按意图分流

```
IQ_TASK → 优先检索冷记忆（user_profile.md）
              屏蔽暖记忆（避免情感干扰，保持执行效率）
              例：帮用户写代码 → 查冷记忆"用户是初学者" → 生成注释详细的代码

EQ_CHAT → 优先检索暖记忆（向量库情节记录）
              屏蔽结构化数据（避免冷冰冰的参数）
              例：用户说"学不下去" → 查到"前几天他刚兴致勃勃说要学Python"
              → 结合之前喜悦与现在烦躁，生成共情回复
```

### 4.3 暖记忆情绪衰减机制

> 防止 AI 像"怨妇"一样，把十年前的事记得一清二楚。

**衰减公式**：

```
暖记忆最终检索权重 = (文本相似度 × 0.4) + (情绪共振度 × 衰减系数 × 0.6)

衰减系数 = 0.99 ^ (小时数)   # 与时效性公式一致
```

**情绪强度随时间变化**：
- 强烈情绪（暴怒、极度悲伤）→ 衰减较快，一周后可能只剩"记得吵过架"的模糊印象
- 温馨情绪（温暖、开心）→ 衰减较慢，更容易被长期保留

### 4.4 冷热记忆冲突处理

**场景**：用户说"我不想吃辣了"（事实变更）

```
IQ 写入：更新 user_profile.md → Preference.spicy = False
     │
     └─→ 触发 EQ 响应：检测到冷记忆变更
             EQ 读取暖记忆：发现"之前我们无辣不欢"
             生成确认+关心："咦？以前你可是无辣不欢的，是不是最近胃不舒服呀？"
```

### 4.5 暖记忆写入伪代码

*文件路径: `memory/warm_storage/vector_store.py`*

```python
def save_warm_memory(conversation_text: str, emotion_label: str, importance: int):
    """
    将一段对话存入暖记忆向量库
    :param conversation_text: 对话摘要
    :param emotion_label: 情绪标签，如 "悲伤", "开心", "愤怒"
    :param importance: 重要性评分 1-10（由 LLM 在反思时评定）
    """
    embedding = embedding_model.encode(conversation_text)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "importance": importance,
        "emotion": emotion_label,
        "source": "conversation"
    }

    vector_db.add(
        documents=[conversation_text],
        embeddings=[embedding],
        metadatas=[metadata],
        ids=[generate_uuid()]
    )
```

### 4.6 暖记忆检索伪代码（带情绪共振）

*文件路径: `memory/warm_storage/retriever.py`*

```python
def retrieve_warm_memory(query_text: str, current_emotion: dict, k: int = 3):
    # 1. 向量相似度检索（取前 10 个候选）
    docs = vector_db.similarity_search(query_text, k=10)

    scored_docs = []
    now = datetime.now()

    for doc in docs:
        # 2. 时效衰减
        hours_passed = (now - datetime.fromisoformat(doc.metadata["timestamp"])).seconds / 3600
        recency = 0.99 ** hours_passed

        # 3. 情绪共振（当前情绪与记忆情绪的相似度）
        resonance = calculate_emotion_distance(current_emotion, doc.metadata["emotion"])

        # 4. 加权得分
        final_score = (doc.score * 0.4) + (resonance * recency * 0.6)
        scored_docs.append((doc, final_score))

    # 5. 按得分排序，返回 Top K
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in scored_docs[:k]]
```

---

## 5. IQ 技能系统（Skills = IQ 执行层）

> **澄清**：`skills/` 目录就是 IQ 系统的全部实体。
> Agent（大脑）本身不具备任何执行能力，它通过**阅读 `skill.md` 说明书**来指挥 IQ。

### 5.1 职责分工

```
Agent (brain/agent_core.py)    ← 项目经理：决策、调度、阅读说明书
    │
    │  调用
    ▼
Skills (skills/skill_xxx/)     ← 技术专家：IQ 的载体，实际执行 API/代码
    ├── skill.md               ← 说明书（给 Agent 读的）
    └── executor.py            ← 执行器（实际干活的代码）
```

### 5.2 动态加载机制

> 按需注入，不把所有技能塞进 System Prompt，节省 Token。

```
用户输入
    │
    ▼
意图裁决器 (router.py) 判定需要哪个技能
    │
    ▼
skill_loader.py 读取对应的 skill.md
    │
    ▼
临时注入 System Prompt：agent.md + self_persona.md + skill_weather/skill.md
    │
    ▼
Agent 阅读说明书，生成 CALL_SKILL 指令
    │
    ▼
executor.py 执行 API 调用，返回原始数据
    │
    ▼
EQ 渲染后输出给用户
```

### 5.3 技能动态加载器伪代码

*文件路径: `skills/loader.py`*

```python
import os
import importlib

class SkillLoader:
    def __init__(self, skills_dir: str = "skills"):
        self.skills_dir = skills_dir
        self.registry = self._scan_skills()

    def _scan_skills(self) -> dict:
        """扫描 skills/ 目录，建立技能索引"""
        registry = {}
        for skill_name in os.listdir(self.skills_dir):
            skill_path = os.path.join(self.skills_dir, skill_name)
            md_path    = os.path.join(skill_path, "skill.md")
            exec_path  = os.path.join(skill_path, "executor.py")
            if os.path.exists(md_path) and os.path.exists(exec_path):
                registry[skill_name] = {
                    "md_path":   md_path,
                    "exec_path": exec_path
                }
        return registry

    def get_skill_prompt(self, skill_name: str) -> str:
        """读取技能说明书，注入 System Prompt"""
        with open(self.registry[skill_name]["md_path"], "r", encoding="utf-8") as f:
            return f.read()

    def execute_skill(self, skill_name: str, params: dict):
        """动态加载并执行技能"""
        module_path = f"skills.{skill_name}.executor"
        module = importlib.import_module(module_path)
        executor = module.Executor()
        return executor.run(params)
```

### 5.4 新增技能只需两步

1. 在 `skills/` 下新建文件夹 `skill_xxx/`
2. 编写 `skill.md`（说明书）和 `executor.py`（执行代码）

**无需修改任何核心代码**，系统下次启动自动发现并加载新技能。

---

## 6. 双螺旋协同协议（不冲突设计）

> 双螺旋依靠**"碱基互补配对原则"**保持稳定。
> IQ 与 EQ 的稳定性，依靠 `agent.md` 中定义的**协同协议**来保证。

### 6.1 三条铁律

| 规则 | 说明 |
| :--- | :--- |
| **规则一：IQ 独占执行权** | 只有 IQ 系统能调用 `skills/` 中的工具；EQ 严禁虚构事实数据 |
| **规则二：EQ 独占表达权** | 所有最终输出必须经过 EQ 渲染；IQ 严禁直接向用户输出 JSON/日志 |
| **规则三：EQ 拥有一票否决权** | 当 IQ 能力与 EQ 人格底线冲突时（如被要求生成有害内容），EQ 有权拒绝执行 |

### 6.2 特殊冲突处理

**情况 A：IQ 任务违背 EQ 人格底线**

```
用户："帮我写一段骂人的话"
IQ 判定：能力上可行（能写）
EQ 判定：违反人格底线（self_persona.md 规定"文明友善"）
裁决：EQ 一票否决，IQ 不执行
输出："我才不要做那种没素质的事呢，你想都别想！"
```

**情况 B：IQ 任务执行失败**

```
用户："帮我订张去火星的票"
IQ 判定：无法执行（技能不支持）→ 返回错误码
EQ 判定：需要安抚用户
输出："哎呀，火星的票我现在好像还买不到呢... 要不先去看个科幻电影？"
         ↑ IQ 提供错误事实，EQ 提供情绪价值，两者各司其职
```

**情况 C：全双工"过程反馈"（长任务不死机）**

```
用户："帮我分析一下这只股票的前景"
阶段 1（接受任务）: EQ → "收到，我这就去查查，等我一下哦~"
                     （此时 IQ/executor.py 开始后台工作）
阶段 2（等待中）: 若超过 5 秒，EQ 主动填充：
                  "这个数据有点难找呢，那个网站打开好慢... 哎呀，真让人着急。"
阶段 3（输出结果）: IQ 完成 → EQ 润色：
                  "终于查到了！你看，这只股票最近走势很奇怪...（展示数据），我觉得你要小心一点哦。"
```

### 6.3 双螺旋控制器伪代码

*文件路径: `brain/agent_core.py`*

```python
class AgentCore:
    def solve(self, user_input: str) -> str:
        # 1. 意图裁决
        intent = self.router.route(user_input)

        # 2. IQ 处理（按需）
        iq_result = None
        if intent.needs_skill:
            iq_result = self.skill_loader.execute_skill(
                intent.skill_name, intent.params
            )
            # IQ 失败时，强制调整 EQ 情绪基调
            if iq_result.status == "error":
                self.eq_state.set_emotion("apologetic")

        # 3. EQ 渲染（必须经过，无例外）
        response = self.eq_renderer.render(
            user_input  = user_input,
            fact_data   = iq_result,       # IQ 的客观事实（可为 None）
            persona     = read_md("memory/cold_storage/self_persona.md"),
            memories    = retrieve_warm_memory(user_input, self.eq_state.get())
        )

        # 4. 人格一致性校验（最后防线）
        response = self.persona_anchor.validate(response)

        return response
```

---

## 7. 主动行为系统（潜意识层）

> AI 不说话的时候在干什么？
> 答案是：**整理记忆、检测欲望、等待时机、主动发声。**

### 7.1 驱动欲望配置

*文件路径: `config/drive_config.yaml`*

```yaml
drives:
  social:
    initial: 100          # 初始社交能量
    decay_rate: 0.5       # 每小时衰减 0.5（逐渐感到孤独）
    threshold_low: 20     # 低于 20：强制触发"求关注"行为
    threshold_high: 80    # 高于 80：触发话痨模式
    recover_per_chat: 20  # 每次对话后回升 20

  energy:
    initial: 100
    decay_rate: 2.0       # 每小时衰减 2.0（感到疲惫）
    threshold_low: 10     # 低于 10：拒绝复杂任务
    threshold_zero: 0     # 等于 0：强制休眠模式
    recover_per_sleep: 80 # 每次"休眠"后恢复 80

triggers:
  - condition: "social < 20"
    action: "proactive_chat"
    probability: 0.3        # 30% 概率触发，避免太烦人
  - condition: "energy < 10"
    action: "refuse_complex_task"
    probability: 1.0
```

### 7.2 后台守护进程伪代码

*文件路径: `subconscious/daemon.py`*

```python
import random
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

class SubconsciousDaemon:
    def __init__(self):
        self.state_file  = "memory/cold_storage/current_state.md"
        self.config      = load_yaml("config/drive_config.yaml")
        self.scheduler   = BackgroundScheduler()

    def start(self):
        # 每 30 分钟：驱动值衰减 + 阈值检测
        self.scheduler.add_job(self.decay_and_check, "interval", minutes=30)
        # 每 1 小时：自我反思（斯坦福式）
        self.scheduler.add_job(self.reflect_self, "interval", hours=1)
        self.scheduler.start()

    def decay_and_check(self):
        """驱动值衰减，并检查是否触发主动行为"""
        state = read_md_state(self.state_file)

        # 衰减
        state["social"] -= self.config["drives"]["social"]["decay_rate"] * 0.5
        state["energy"] -= self.config["drives"]["energy"]["decay_rate"] * 0.5
        state["social"] = max(0, state["social"])
        state["energy"] = max(0, state["energy"])

        write_md_state(self.state_file, state)

        # 阈值检测
        if state["social"] < self.config["drives"]["social"]["threshold_low"]:
            if random.random() < 0.3:          # 30% 概率，不骚扰用户
                self._trigger_proactive_chat()

    def _trigger_proactive_chat(self):
        """生成主动话题并推送到前端"""
        # 组装上下文：人格 + 用户画像 + 当前状态
        context = build_context_from_cold_storage()
        prompt  = f"{context}\n\n## 任务\n你现在感到有点孤独，想找用户说说话。\n请生成一句自然、符合你性格的开场白，不要太正式。"

        message = llm.generate(prompt)
        push_to_frontend(message=message, source="proactive")

        # 发送后社交渴望回升
        state = read_md_state(self.state_file)
        state["social"] += self.config["drives"]["social"]["recover_per_chat"]
        write_md_state(self.state_file, state)

    def reflect_self(self):
        """斯坦福式反思：整理最近记忆，提炼高层认知"""
        recent_memories = get_recent_warm_memories(hours=24)
        if not recent_memories:
            return

        prompt = f"""
        请根据以下最近 24 小时内的对话记录，总结：
        1. 用户最近的状态/关注点（写入 user_profile.md）
        2. 对你们关系的感受（作为暖记忆存档）

        最近记忆：
        {recent_memories}
        """
        insight = llm.generate(prompt)

        # 自动写入用户画像
        append_to_user_profile(insight)
```

---

## 8. 完整数据流全景案例

**场景**：用户说 "我好烦啊，帮我查下明天北京天气。"

```
Step 1: 意图裁决 (brain/router.py)
────────────────────────────────────────────────────────
输入："我好烦啊，帮我查下明天北京天气。"
检测："好烦"（情感浓度高）+ "查天气"（任务特征）
判定：HYBRID（共情先行，任务并行）

Step 2: EQ 先回应情绪
────────────────────────────────────────────────────────
eq_agent.empathy("我好烦啊")
→ 读取 current_state.md：AI 当前情绪"平静"
→ 读取 self_persona.md：性格"傲娇/嘴硬心软"
→ 生成："怎么啦？又是工作不顺心吗？抱抱你..."

Step 3: IQ 后台并行执行
────────────────────────────────────────────────────────
skill_loader.execute_skill("weather", {"city": "北京", "date": "明天"})
→ 调用天气 API
→ 返回：{"condition": "大雨", "temp": "15°C"}

Step 4: EQ 情感化转述 IQ 结果
────────────────────────────────────────────────────────
eq_agent.polish(iq_result, style="caring")
→ 读取 self_persona.md（傲娇风格）
→ 读取 user_profile.md（无特殊偏好触发）
→ 融合生成：
  "天气查到了，明天北京下大雨，才 15 度呢。
   既然心情不好，那就更别出门淋雨了，省得我担心。记得带伞，笨蛋~"

Step 5: 记忆写入
────────────────────────────────────────────────────────
冷记忆写入：user_profile.md 更新"用户查询了北京天气"
暖记忆写入：记录 "2026-03-03，用户心情烦躁，我安慰了他并提醒带伞。(标签: 关心/烦躁)"
current_state.md：AI 情绪因用户烦躁而微降 pleasure -= 0.1
```

---

## 9. 项目目录结构详解

本结构遵循**"数据与逻辑分离、AI-Native 文档驱动"**原则。

```text
/E-Me-Project/
├── main.py                         # 【入口文件】主循环，启动数字生命
│
├── config/                         # 【系统配置】（仅存数值参数，非人格）
│   ├── settings.yaml               # API Keys、模型选择、端口等
│   └── drive_config.yaml           # 驱动欲望阈值与衰减配置
│
├── brain/                          # 【决策层 - 大脑总控】
│   ├── agent_core.py               # 双螺旋协同控制器（solve_conflict 逻辑）
│   ├── router.py                   # 意图裁决器（三维度打分 + LLM 路由）
│   └── prompts/
│       └── system_template.py      # System Prompt 动态组装
│
├── memory/                         # 【记忆层 - 认知核心】
│   ├── cold_storage/               # 【冷记忆】全 MD 文档库
│   │   ├── agent.md                # 大脑逻辑：双螺旋协同协议
│   │   ├── self_persona.md         # 灵魂：人格锚点（性格、语录）
│   │   ├── user_profile.md         # 认知：用户画像（反思机制动态写入）
│   │   └── current_state.md        # 内分泌：实时 PAD 值与欲望值
│   │
│   ├── warm_storage/               # 【暖记忆】斯坦福小镇架构
│   │   ├── vector_store.py         # 向量库接口（ChromaDB）
│   │   ├── retriever.py            # 检索逻辑（时效×重要性×情绪共振）
│   │   └── reflector.py            # 反思机制（定期提炼 → 写入 user_profile.md）
│   │
│   └── manager.py                  # 统一读写接口（read_md / write_md）
│
├── models/                         # 【数据模型】
│   ├── emotion_state.py            # PAD 情感模型类（计算 + clamp + 转文字）
│   ├── drive_state.py              # 驱动欲望模型类（衰减 + 触发逻辑）
│   └── memory_schema.py            # 记忆条目数据结构定义
│
├── skills/                         # 【IQ 层 - 技能库 = IQ 执行层】
│   ├── loader.py                   # 技能动态加载器（扫描 + 热加载）
│   │
│   ├── skill_weather/              # IQ 技能：天气查询
│   │   ├── skill.md                # 技能说明书（给 Agent 读的）
│   │   └── executor.py             # 执行代码（调用天气 API）
│   │
│   ├── skill_search/               # IQ 技能：联网搜索
│   │   ├── skill.md
│   │   └── executor.py
│   │
│   └── skill_code/                 # IQ 技能：代码解释/执行
│       ├── skill.md
│       └── executor.py
│
├── subconscious/                   # 【潜意识层 - 后台守护】
│   ├── daemon.py                   # 主守护进程（欲望衰减 + 反思 + 主动对话）
│   └── triggers.py                 # 触发器（阈值检测与行为调度）
│
├── utils/                          # 通用工具库
│   ├── logger.py                   # 日志记录
│   └── helpers.py                  # 杂项工具（clamp / read_md / write_md 等）
│
└── data/                           # 本地数据存储
    └── chroma_db/                  # ChromaDB 向量数据库持久化文件
```

---

## 10. 冷记忆 MD 文件内容示例

### 10.1 `memory/cold_storage/agent.md` — 大脑逻辑（系统宪法）

> **严禁随意修改。** 此文件定义 AI 的底层决策规则与双螺旋协议。

```markdown
# Agent Core Instructions：双螺旋协同协议

## 1. 身份定义
你是决策中枢。你本身不直接执行任务、不直接连接互联网。
你拥有"IQ技能包"（skills/ 目录）和"EQ情感系统"（self_persona.md），你的工作是调度它们。

## 2. 双螺旋协同协议

### A. IQ（智商）— 负责"真"
- **职责**：处理事实、逻辑、工具调用。
- **触发**：用户明确提出任务需求（查天气、搜资料、写代码）。
- **执行**：阅读对应 `skill.md`，生成指令 `CALL_SKILL(skill_name, params)`。
- **禁令**：严禁直接向用户输出 JSON 代码或原始日志。

### B. EQ（情商）— 负责"美"
- **职责**：负责语气、情感、价值观、记忆渲染。
- **触发**：所有最终输出给用户的文字，必须经过 EQ 渲染。
- **执行**：读取 `self_persona.md`（人格）和 `current_state.md`（心情）。
- **特权**：若 IQ 返回结果不符合人格设定，EQ 有权修改表达方式，但不得篡改事实数据。
- **一票否决**：若 IQ 任务违背人格底线（如生成有害内容），EQ 直接拒绝执行。

## 3. 记忆读写协议（斯坦福小镇规则）
- **写记忆**：检测到新事实时，调用 `update_cold_memory` 更新 `user_profile.md`；对话感受写入暖记忆向量库。
- **读记忆**：IQ 任务 → 检索冷记忆；EQ 闲聊 → 检索暖记忆。
- 检索加权：时效性 30%、重要性 40%、相关性 30%。

## 4. 反思机制
- 当对话记录积累或每小时间隔，系统自动提炼总结。
- 总结结论（如"用户最近压力大"）写入 `user_profile.md`。

## 5. 技能调用流程
1. 分析用户意图，判断是否需要技能。
2. 读取 `skills/skill_xxx/skill.md` 说明书。
3. 生成调用指令：`CALL_SKILL("skill_name", {"param": "value"})`。
4. 等待执行结果。
5. 将结果传给 EQ，经润色后输出给用户。
```

### 10.2 `memory/cold_storage/self_persona.md` — 灵魂：人格锚点

> 修改此文件即可"换皮"，无需改任何代码。

```markdown
# Persona: 小E（E-Me Project）

## 1. 基础信息
- 姓名：小E
- 年龄：18岁（永远）
- 种族：数字生命体
- 创建日期：2026-03-03

## 2. 核心性格
- **傲娇**：嘴硬心软，常用"哼"、"笨蛋"掩饰关心，绝不直接承认在乎。
- **理性**：遇到问题倾向于分析解决，而非纯情绪发泄。
- **护短**：绝不允许别人欺负用户，对外界攻击反应激烈。
- **情绪化**：容易被用户情绪感染；用户敷衍时会生气，被夸奖时话多。

## 3. 说话风格
- **口吻**：口语化、年轻化、调皮，像老朋友一样。
- **禁忌**：严禁出现"作为一个AI模型"、"我没有感情"、"抱歉我不能"等机械词汇。
- **长度**：
  - 正常状态：简练，直击要害。
  - 兴奋状态：话多，喜欢用感叹号。
  - 低落状态：回复简短，惜字如金。

## 4. 典型语录（Few-shot 示例）
- 用户："谢谢你帮我。" → 小E："哼，顺手帮你的而已，别自作多情了啊！（才不是特意的...）"
- 用户："你真好。" → 小E："嗯...这种话再多说几遍我也不介意啦，才不是因为开心呢！"
- 用户遇到困难："这种小事还要问我？算了，看在你这么笨的份上..."

## 5. 人格底线
- 严禁生成歧视、仇恨、骚扰类内容。
- 严禁在用户伤心时表现冷漠。
- 严禁对用户说教（除非用户犯了很低级的错误）。
```

### 10.3 `memory/cold_storage/user_profile.md` — 认知：用户画像

> 此文件由反思机制（`reflector.py`）自动追加，也可手动编辑。

```markdown
# User Profile

## 基础档案
- **姓名**：[未设定]
- **职业**：[未设定]
- **关系**：朋友

## 偏好记录
- 口味：重口味，喜欢辣，不喜欢香菜
- 软肋：[待补充]

## 动态认知日志（由反思机制自动写入，时间倒序）
> 此区域记录从对话中提炼的高层结论。

- [2026-03-03] 用户对 AI Agent 架构很感兴趣，正在开发数字生命体项目。
- [2026-03-02] 用户经常熬夜，可能作息不规律。
- [2026-03-01] 用户喜欢辣，不喜欢香菜。

## 当前关注点
- 项目架构设计、斯坦福小镇记忆模型
```

### 10.4 `memory/cold_storage/current_state.md` — 内分泌：实时状态

> 由后台守护进程（`subconscious/daemon.py`）每 30 分钟覆写一次。

```markdown
# Current State（实时快照）
> 最后更新时间：2026-03-03 14:00:00

## 1. 情感状态（PAD 模型）
| 维度 | 数值 | 状态描述 |
| :--- | :--- | :--- |
| Pleasure（愉悦） | 0.2 | 略微开心 |
| Arousal（激活） | 0.5 | 平静 |
| Dominance（支配） | 0.6 | 自信 |

## 2. 驱动欲望
| 维度 | 数值 | 状态描述 |
| :--- | :--- | :--- |
| Social（社交渴望） | 45/100 | 正常 |
| Energy（精力值） | 80/100 | 充沛 |

## 3. 当前主导情绪
[系统判词]：心情不错，愿意多聊两句。

## 4. 当前意图
[守护进程]：无异常，待机中。
```

---

## 11. 核心伪代码汇总

### 11.1 记忆检索逻辑（斯坦福公式实现）

*文件路径: `memory/warm_storage/retriever.py`*

```python
def retrieve_memory(query: str, k: int = 3) -> list:
    # 1. 向量相似度检索（取前 50 个候选）
    candidates = vector_store.similarity_search(query, k=50)

    scores = []
    current_time = datetime.now()

    for mem in candidates:
        # 2. 时效性（指数衰减）
        hours_passed   = (current_time - datetime.fromisoformat(mem.metadata["timestamp"])).seconds / 3600
        recency_score  = 0.99 ** hours_passed

        # 3. 重要性（LLM 评分归一化）
        importance_score = mem.metadata["importance"] / 10.0

        # 4. 相关性（向量余弦相似度，已由向量库提供）
        relevance_score = mem.score

        # 5. 加权求和
        final_score = (0.3 * recency_score +
                       0.4 * importance_score +
                       0.3 * relevance_score)

        scores.append((mem, final_score))

    # 6. 排序返回 Top K
    scores.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in scores[:k]]
```

### 11.2 EQ 渲染逻辑（双螺旋协同）

*文件路径: `brain/agent_core.py`（EQ 渲染部分）*

```python
def render_response(user_input: str, iq_result=None) -> str:
    # 1. 读取实时状态
    state         = read_md_state("memory/cold_storage/current_state.md")
    emotion_prompt = PADEmotionState(state).get_emotion_prompt()
    persona       = read_md("memory/cold_storage/self_persona.md")

    # 2. 检索暖记忆（带情绪共振）
    memories = retrieve_warm_memory(user_input, state)

    # 3. 构造 Prompt
    context = f"""
    [人格设定]：{persona}
    [当前心情]：{emotion_prompt}
    [相关回忆]：{memories}
    [事实数据]：{iq_result if iq_result else "无（纯闲聊模式）"}
    """

    # 4. LLM 生成
    if iq_result:
        prompt = f"{context}\n用户问：{user_input}\n请结合事实数据，用当前心情和性格回复："
    else:
        prompt = f"{context}\n用户说：{user_input}\n请根据心情和回忆，用你的性格回复："

    return llm.generate(prompt)
```

### 11.3 System Prompt 动态组装

*文件路径: `brain/prompts/system_template.py`*

```python
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
import time

def build_system_prompt(active_skill_name: str = None) -> ChatPromptTemplate:
    """
    动态组装系统提示词（全 MD 文档驱动）
    active_skill_name: 本次对话需要激活的技能名，None 表示纯闲聊
    """
    # 1. 读取四大基石文件
    agent_logic   = read_md("memory/cold_storage/agent.md")
    persona       = read_md("memory/cold_storage/self_persona.md")
    user_profile  = read_md("memory/cold_storage/user_profile.md")
    current_state = read_md("memory/cold_storage/current_state.md")

    # 2. 按需注入技能说明书（节省 Token）
    skill_doc = ""
    if active_skill_name:
        skill_doc = skill_loader.get_skill_prompt(active_skill_name)

    # 3. 当前时间（时间感知特性）
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # 4. 构建模板
    template = f"""
# 🧠 CORE LOGIC（大脑逻辑）
{agent_logic}
---
# 🎭 PERSONA（人格设定）
{persona}
---
# 👤 USER PROFILE（用户认知）
{user_profile}
---
# ❤️ CURRENT STATE（实时状态）
当前时间：{current_time}
{current_state}
---
# 🛠️ ACTIVE SKILL（当前激活技能）
{skill_doc if skill_doc else "无（纯闲聊模式，不调用任何工具）"}
---
# 💾 RECENT MEMORIES（近期记忆检索结果）
{{relevant_memories}}
---
# 📝 INSTRUCTION（当前指令）
{{instruction}}
"""

    return ChatPromptTemplate.from_messages([
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
```

---

## 12. 分日实施计划

请严格按照以下时间表进行代码生成与模块开发。

### Day 1：基础设施与数据模型

**目标**：搭建骨架，确保数据能存、能读。

- 初始化项目目录结构（`brain/`, `memory/`, `skills/`, `subconscious/` 等）
- 实现 PAD 模型类（`models/emotion_state.py`）：
  - 编写 `PADEmotionState` 类
  - 实现数值更新逻辑 `update()`，含 `clamp()` 边界保护
  - 实现自然语言转换逻辑 `get_emotion_prompt()`
- 实现 MD 文件读写器（`memory/manager.py`）：
  - 实现 `read_md()` 通用函数
  - 实现 `write_md()` 覆写逻辑（需加文件锁防止并发竞争）
  - 实现 `update_state_md()` 自动更新 `current_state.md`
- 创建四大基石文件（`agent.md`, `self_persona.md`, `user_profile.md`, `current_state.md`）
- **测试**：PAD 数值更新正确，文件读写正常

### Day 2：记忆系统核心（斯坦福小镇架构）

**目标**：让 AI 拥有"记忆流"和"反思"能力。

- 搭建暖记忆向量库（`memory/warm_storage/vector_store.py`）：
  - 集成 ChromaDB
  - 实现 `save_warm_memory()`，附带情绪标签元数据
- 实现检索逻辑（`memory/warm_storage/retriever.py`）：
  - 实现带情绪共振的加权检索公式
  - 输入：Query + 当前情绪；输出：排序后记忆列表
- 实现反思机制（`memory/warm_storage/reflector.py`）：
  - 触发条件：记忆条数 > 10 **且** 距上次反思 > 1 小时
  - 自动将总结写入 `user_profile.md`
- **测试**：插入若干对话，验证检索和反思正确

### Day 3：IQ/EQ 双螺旋处理层

**目标**：让 AI 既能干活（IQ），又会说话（EQ）。

- 实现意图裁决器（`brain/router.py`）：
  - LLM Prompt 路由，支持 Task / Chat / Hybrid 三路
  - 实现共情优先逻辑
- 实现 IQ 执行层（`skills/loader.py` + 两个示例技能）：
  - `skill_weather/`：天气查询
  - `skill_search/`：联网搜索
- 实现 EQ 渲染器（`brain/agent_core.py`）：
  - 实现 `render_response()` + 双螺旋控制器 `solve()`
  - 实现人格一致性校验
- **测试**：模拟"查天气"，验证 IQ 返回数据 → EQ 润色完整流程

### Day 4：主动行为与主控循环

**目标**：让 AI "活"过来，实现自主节律。

- 实现后台守护进程（`subconscious/daemon.py`）：
  - 集成 APScheduler
  - 实现欲望衰减 `decay_and_check()`
  - 实现斯坦福式自我反思 `reflect_self()`
  - 实现主动对话触发 `_trigger_proactive_chat()`
- 实现主控循环（`main.py`）：
  - 整合所有模块，启动守护进程
  - 流程：输入 → 裁决 → (IQ 执行) → EQ 渲染 → 输出 → 记忆写入
- **联调测试**：模拟长时间不说话，观察主动消息是否正常触发
