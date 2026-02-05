"""Agent loop: the core processing engine.

集成 Stanford Generative Agents 记忆系统的流程：

1. 收消息 
2. → [新增] 存入记忆(observation) + 评估重要性
3. → [新增] 检索相关记忆(3D retrieval)
4. → 构建context（包含检索到的记忆）
5. → 调LLM 
6. → 执行tools 
7. → [新增] 检查是否触发反思
8. → [新增] 存入记忆(assistant response)
9. → 响应
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_manager import MemoryManager
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebSearchTool, WebFetchTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.subagent import SubagentManager
from nanobot.session.manager import SessionManager


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    集成 Stanford Generative Agents 记忆系统：
    1. Receives messages from the bus
    2. [NEW] Stores observation + evaluates importance
    3. [NEW] Retrieves relevant memories (3D retrieval)
    4. Builds context with history, memory, skills
    5. Calls the LLM
    6. Executes tool calls
    7. [NEW] Checks reflection trigger
    8. [NEW] Stores assistant response
    9. Sends responses back
    """
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        # 记忆系统配置
        memory_enabled: bool = True,
        embedding_config: Optional[Dict[str, Any]] = None,
        reflection_threshold: int = 150,
        reflection_enabled: bool = True,
    ):
        from nanobot.config.schema import ExecToolConfig
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        
        # 记忆系统配置
        self.memory_enabled = memory_enabled
        self.embedding_config = embedding_config
        self.reflection_threshold = reflection_threshold
        self.reflection_enabled = reflection_enabled
        
        self.context = ContextBuilder(workspace)
        self.sessions = SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
        )
        
        # 记忆管理器缓存（每个session一个）
        self._memory_managers: Dict[str, MemoryManager] = {}
        
        self._running = False
        self._register_default_tools()
    
    def _get_memory_manager(self, session_key: str) -> Optional[MemoryManager]:
        """
        获取或创建会话的记忆管理器
        
        Args:
            session_key: 会话标识
        
        Returns:
            MemoryManager实例，如果记忆系统禁用则返回None
        """
        if not self.memory_enabled:
            return None
        
        if session_key not in self._memory_managers:
            self._memory_managers[session_key] = MemoryManager(
                role_id=session_key.replace(":", "_"),
                workspace=self.workspace,
                embedding_config=self.embedding_config,
                reflection_threshold=self.reflection_threshold,
                reflection_enabled=self.reflection_enabled,
            )
        
        return self._memory_managers[session_key]
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools
        self.tools.register(ReadFileTool())
        self.tools.register(WriteFileTool())
        self.tools.register(EditFileTool())
        self.tools.register(ListDirTool())
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.exec_config.restrict_to_workspace,
        ))
        
        # Web tools
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")
    
    async def _process_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        实现 Stanford GA 记忆增强流程：
        1. 收消息 
        2. [新增] 存入记忆(observation) + 评估重要性
        3. [新增] 检索相关记忆(3D retrieval)
        4. 构建context（包含检索到的记忆）
        5. 调LLM 
        6. 执行tools 
        7. [新增] 检查是否触发反思
        8. [新增] 存入记忆(assistant response)
        9. 响应
        
        Args:
            msg: The inbound message to process.
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}")
        
        # Get or create session
        session = self.sessions.get_or_create(msg.session_key)
        
        # 获取记忆管理器
        memory_manager = self._get_memory_manager(msg.session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        # ========== 步骤2：存入用户消息到记忆 + 评估重要性 ==========
        memory_context = ""
        if memory_manager:
            try:
                await memory_manager.add_observation(
                    content=msg.content,
                    role="user",
                    metadata={
                        "channel": msg.channel,
                        "sender_id": msg.sender_id,
                    }
                )
                logger.debug(f"Added user message to memory: {msg.content[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to add user message to memory: {e}")
            
            # ========== 步骤3：检索相关记忆 (3D retrieval) ==========
            try:
                relevant_memories = await memory_manager.retrieve_relevant_memories(
                    query=msg.content,
                    k=5,
                )
                memory_context = memory_manager.format_memories_for_context(relevant_memories)
                logger.debug(f"Retrieved {len(relevant_memories)} relevant memories")
            except Exception as e:
                logger.warning(f"Failed to retrieve memories: {e}")
        
        # ========== 步骤4：构建context（包含检索到的记忆）==========
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            memory_context=memory_context,  # 新增：记忆上下文
        )
        
        # ========== 步骤5-6：调LLM + 执行tools ==========
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            logger.bind(tag="agents").info(f"Response: {response}")
            # Handle tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                
                # Execute tools
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # ========== 步骤7：检查是否触发反思 ==========
        if memory_manager:
            try:
                reflection_nodes = await memory_manager.check_and_reflect(
                    llm=self.provider,
                    agent_name="assistant",
                )
                if reflection_nodes:
                    logger.info(f"Reflection generated {len(reflection_nodes)} insights")
            except Exception as e:
                logger.warning(f"Reflection failed: {e}")
            
            # ========== 步骤8：存入助手回复到记忆 ==========
            try:
                await memory_manager.add_observation(
                    content=final_content,
                    role="assistant",
                    metadata={
                        "channel": msg.channel,
                    }
                )
                logger.debug(f"Added assistant response to memory: {final_content[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to add assistant response to memory: {e}")
        
        # Save to session (保持原有的session机制)
        session.add_message("user", msg.content)
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        # ========== 步骤9：响应 ==========
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with the announce content
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments)
                    logger.debug(f"Executing tool: {tool_call.name} with arguments: {args_str}")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    async def process_direct(self, content: str, session_key: str = "cli:direct") -> str:
        """
        Process a message directly (for CLI usage).
        
        Args:
            content: The message content.
            session_key: Session identifier.
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=content
        )
        
        response = await self._process_message(msg)
        return response.content if response else ""
