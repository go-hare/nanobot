"""
MemoryStream - 记忆流

管理所有记忆节点，按时间顺序存储
提供基础的增删查改功能
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import json
import os
from .concept_node import ConceptNode
from loguru import logger
TAG = __name__

class MemoryStream:
    """记忆流 - 存储和管理所有记忆节点"""
    
    def __init__(self, role_id: str, save_to_file: bool = True, memory_file: Optional[str] = None):
        """
        初始化记忆流
        
        Args:
            role_id: 用户/角色ID
            save_to_file: 是否保存到文件
            memory_file: 记忆文件路径（可选）
        """
        self.role_id = role_id
        self.save_to_file = save_to_file
        
        # 记忆存储
        self.nodes: Dict[str, ConceptNode] = {}  # {node_id: ConceptNode}
        self.event_sequence: List[str] = []  # 事件序列（按时间顺序的node_id）
        self.reflection_sequence: List[str] = []  # 反思序列
        
        # 累计重要性（用于触发反思）
        self.accumulated_importance = 0
        
        # 文件路径
        if memory_file:
            self.memory_file = memory_file
            
        # 尝试加载已有记忆
        if self.save_to_file:
            self.load_from_file()
    
    def add_node(self, node: ConceptNode, accumulate_importance: bool = True) -> str:
        """
        添加一个记忆节点
        
        Args:
            node: 记忆节点
            accumulate_importance: 是否累积重要性（默认True）。
                                   反思记忆通常设为False以避免无限反思
        
        Returns:
            节点ID
        """
        # 存储节点
        self.nodes[node.node_id] = node
        
        # 添加到对应序列
        if node.memory_type == "reflection":
            self.reflection_sequence.append(node.node_id)
        else:
            self.event_sequence.append(node.node_id)
        
        # 累积重要性（可选）
        if accumulate_importance:
            self.accumulated_importance += node.importance
        
        # 保存到文件
        if self.save_to_file:
            self.save_to_file_async()
        
        logger.bind(tag=TAG).debug(
            f"添加记忆: [{node.memory_type}] {node.content[:30]}... (importance={node.importance})"
        )
        
        return node.node_id
    
    def get_node(self, node_id: str) -> Optional[ConceptNode]:
        """获取指定记忆节点"""
        node = self.nodes.get(node_id)
        if node:
            node.record_access()  # 记录访问
        return node
    
    def get_all_nodes(self, include_expired: bool = False) -> List[ConceptNode]:
        """
        获取所有记忆节点
        
        Args:
            include_expired: 是否包含已过期的记忆
        
        Returns:
            记忆节点列表
        """
        current_time = time.time()
        
        if include_expired:
            return list(self.nodes.values())
        else:
            return [
                node for node in self.nodes.values()
                if not node.is_expired(current_time)
            ]
    
    def get_recent_nodes(self, n: int = 100, memory_type: Optional[str] = None) -> List[ConceptNode]:
        """
        获取最近的N条记忆
        
        Args:
            n: 返回数量
            memory_type: 记忆类型过滤（可选）
        
        Returns:
            记忆节点列表
        """
        # 选择序列
        if memory_type == "reflection":
            sequence = self.reflection_sequence
        elif memory_type == "observation":
            sequence = self.event_sequence
        else:
            # 合并并排序
            all_ids = self.event_sequence + self.reflection_sequence
            sequence = sorted(
                all_ids,
                key=lambda nid: self.nodes[nid].created if nid in self.nodes else 0,
                reverse=True
            )
        
        # 获取最近的N个
        recent_ids = sequence[-n:]
        recent_ids.reverse()  # 最新的在前
        
        # 返回节点
        nodes = []
        for node_id in recent_ids:
            if node_id in self.nodes:
                nodes.append(self.nodes[node_id])
        
        return nodes
    
    def get_nodes_by_type(self, memory_type: str) -> List[ConceptNode]:
        """获取指定类型的所有记忆"""
        return [
            node for node in self.nodes.values()
            if node.memory_type == memory_type
        ]
    
    def get_nodes_since(self, timestamp: float) -> List[ConceptNode]:
        """获取指定时间后的所有记忆"""
        return [
            node for node in self.nodes.values()
            if node.created >= timestamp
        ]
    
    def delete_node(self, node_id: str) -> bool:
        """删除记忆节点"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            
            # 从序列中移除
            if node.memory_type == "reflection" and node_id in self.reflection_sequence:
                self.reflection_sequence.remove(node_id)
            elif node_id in self.event_sequence:
                self.event_sequence.remove(node_id)
            
            # 删除节点
            del self.nodes[node_id]
            
            # 保存
            if self.save_to_file:
                self.save_to_file_async()
            
            logger.bind(tag=TAG).debug(f"删除记忆: {node_id}")
            return True
        
        return False
    
    def clear_expired_nodes(self) -> int:
        """清理过期的记忆节点"""
        current_time = time.time()
        expired_ids = [
            node_id for node_id, node in self.nodes.items()
            if node.is_expired(current_time)
        ]
        
        for node_id in expired_ids:
            self.delete_node(node_id)
        
        if expired_ids:
            logger.bind(tag=TAG).info(f"清理了 {len(expired_ids)} 条过期记忆")
        
        return len(expired_ids)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            'total_nodes': len(self.nodes),
            'observations': len(self.event_sequence),
            'reflections': len(self.reflection_sequence),
            'accumulated_importance': self.accumulated_importance,
            'avg_importance': sum(n.importance for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0,
            'oldest_memory_hours': min(n.get_age_hours() for n in self.nodes.values()) if self.nodes else 0,
        }
    
    def reset_accumulated_importance(self):
        """重置累计重要性（反思后调用）"""
        self.accumulated_importance = 0
        logger.bind(tag=TAG).debug("重置累计重要性计数器")
    
    def save_to_file_async(self):
        """异步保存到文件（不阻塞）"""
        if not self.save_to_file:
            return
        
        try:
            # 序列化数据
            data = {
                'role_id': self.role_id,
                'accumulated_importance': self.accumulated_importance,
                'nodes': {
                    node_id: node.to_dict()
                    for node_id, node in self.nodes.items()
                },
                'event_sequence': self.event_sequence,
                'reflection_sequence': self.reflection_sequence,
                'last_updated': time.time()
            }
            
            # 确保目录存在
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # 写入文件
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.bind(tag=TAG).debug(f"记忆已保存到文件: {self.memory_file}")
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"保存记忆到文件失败: {e}")
    
    def load_from_file(self):
        """从文件加载记忆"""
        if not os.path.exists(self.memory_file):
            logger.bind(tag=TAG).info(f"记忆文件不存在，将创建新的记忆流: {self.memory_file}")
            return
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 恢复数据
            self.role_id = data.get('role_id', self.role_id)
            self.accumulated_importance = data.get('accumulated_importance', 0)
            self.event_sequence = data.get('event_sequence', [])
            self.reflection_sequence = data.get('reflection_sequence', [])
            
            # 恢复节点
            nodes_data = data.get('nodes', {})
            for node_id, node_dict in nodes_data.items():
                try:
                    self.nodes[node_id] = ConceptNode.from_dict(node_dict)
                except Exception as e:
                    logger.bind(tag=TAG).warning(f"恢复记忆节点失败 {node_id}: {e}")
            
            logger.bind(tag=TAG).info(
                f"从文件加载了 {len(self.nodes)} 条记忆 (观察:{len(self.event_sequence)}, 反思:{len(self.reflection_sequence)})"
            )
        
        except Exception as e:
            logger.bind(tag=TAG).error(f"从文件加载记忆失败: {e}")
    
    def __len__(self) -> int:
        """返回记忆总数"""
        return len(self.nodes)
    
    def __str__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return f"MemoryStream(total={stats['total_nodes']}, obs={stats['observations']}, ref={stats['reflections']})"
