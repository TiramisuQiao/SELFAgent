from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
import numpy as np
import os
import pretty_errors

os.environ["OPENAI_API_KEY"] = "sk-K0kNVSkeaFmWL6wGi5f3Y4Ar3HOecVgP9WV6Xso4wyJuFeHV"

class HybridMemorySystem:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_texts(
            texts=["作为一名合格的辩手，只能提出三个合理的观点，过多或者过少都会显得业余"],  # 初始占位符
            embedding=self.embeddings,
            metadatas=[{"type": "init_memory"}]
        )
        self.short_term_memory = EnhancedBufferMemory(
            window_size=5,  # 保留最近5轮对话
            memory_key="short_term_context"
        )

    def promote_memories(self):
        """将短期记忆中使用次数达到阈值的条目升级到长期记忆"""
        promoted_memories = self.short_term_memory.check_promotion()
        if promoted_memories:
            texts = [memory["content"] for memory in promoted_memories]
            metadatas = [{"type": "promoted_memory"} for _ in texts]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def print_memory_state(self):
        """打印当前记忆状态"""
        print(f"\n=== Agent {self.agent_id} 记忆状态 ===")
        
        # 打印短期记忆
        print("短期记忆：")
        for i, message in enumerate(self.short_term_memory.chat_memory.messages):
            print(f"{i+1}. {message.content}")
        
        # 打印长期记忆
        print("\n长期记忆：")
        docs = self.vector_store.similarity_search("", k=10)  # 检索所有记忆
        for i, doc in enumerate(docs):
            print(f"{i+1}. {doc.page_content} (类型: {doc.metadata['type']})")


class EnhancedBufferMemory(ConversationBufferWindowMemory):
    """增强型短期记忆，带使用次数追踪"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize custom fields
        self.__dict__["usage_counter"] = {}  # 直接操作 __dict__ 来设置属性
        self.__dict__["promotion_threshold"] = 3  # Set default value for promotion_threshold

    def _track_usage(self, message_id: str):
        """更新消息使用次数"""
        self.usage_counter[message_id] = self.usage_counter.get(message_id, 0) + 1
        
    def check_promotion(self):
        """检查需要升级的短期记忆"""
        promoted_memories = []
        for msg_id, count in self.usage_counter.items():
            if count >= self.promotion_threshold:
                # 获取消息内容并转换格式
                memory_entry = next(
                    m for m in self.chat_memory.messages 
                    if m.additional_kwargs.get("message_id") == msg_id
                )
                promoted_memories.append({
                    "content": memory_entry.content,
                    "metadata": {"type": "promoted_memory"}
                })
                del self.usage_counter[msg_id]
        return promoted_memories


class DebateAgent:
    def __init__(self, agent_id: str, role: str):  
        self.memory = HybridMemorySystem(agent_id)
        self.role = role
        self.llm = ChatOpenAI(
            model="moonshot-v1-8k", 
            base_url="https://api.moonshot.cn/v1",
            temperature=0.7
        )
        
        # 构建记忆增强的对话链
        self.chain = self._build_chain()

    def _build_chain(self):
        # 记忆检索组件
        retriever = self.memory.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "filter": {"type": "promoted_memory"}}
        )
        
        # 对话模板
        prompt = ChatPromptTemplate.from_messages([ 
            ("system", "你是一个{role}辩手，当前辩论主题：{topic}"),
            ("human", "{input}")
        ])
        
        return RunnablePassthrough.assign(
            context=lambda x: retriever.get_relevant_documents(x["input"])
        ) | prompt | self.llm

    def generate_response(self, topic: str, input_text: str):
        """生成辩论响应并更新记忆"""
        response = self.chain.invoke({
            "role": self.role,
            "topic": topic,
            "input": input_text
        })
        
        # 更新短期记忆
        self.memory.short_term_memory.save_context(
            {"input": input_text},
            {"output": response.content}
        )
        return response.content


class DebateCoordinator:
    def __init__(self, agents: List[DebateAgent], rounds_before_promote: int = 3):
        self.agents = agents
        self.round_counter = 0
        self.promotion_interval = rounds_before_promote
        
    def conduct_round(self, topic: str, proposition: str, opposition: str):
        print(f"\n=== 第 {self.round_counter+1} 轮辩论 ===")
        print(f"辩题：{topic}")
        
        # 正方发言
        prop_response = self.agents[0].generate_response(
            topic, 
            f"作为正方，请回应：{proposition}"
        )
        print(f"\n正方：{prop_response}")
        
        # 反方发言
        oppo_response = self.agents[1].generate_response(
            topic, 
            f"作为反方，请回应：{opposition}。对方刚才说：{prop_response}"
        )
        print(f"\n反方：{oppo_response}")
        
        # 打印记忆状态
        for agent in self.agents:
            agent.memory.print_memory_state()
        
        # 定期升级记忆
        if (self.round_counter+1) % self.promotion_interval == 0:
            for agent in self.agents:
                agent.memory.promote_memories()
            print("\n=== 记忆升级完成 ===")
            for agent in self.agents:
                agent.memory.print_memory_state()
        
        self.round_counter += 1


if __name__ == "__main__":
    # 初始化辩论双方
    prop_agent = DebateAgent(agent_id="prop_1", role="专业级正方辩手")
    oppo_agent = DebateAgent(agent_id="oppo_1", role="资深反方辩手")
    
    coordinator = DebateCoordinator([prop_agent, oppo_agent])
    
    # 模拟辩论流程
    debate_topic = "社交媒体是否应该实施严格的内容审查制度？"
    
    coordinator.conduct_round(
        topic=debate_topic,
        proposition="社交媒体有责任维护健康的网络环境",
        opposition="过度审查会侵犯言论自由"
    )
    
    coordinator.conduct_round(
        topic=debate_topic,
        proposition="算法推荐导致信息茧房效应",
        opposition="用户应自主选择信息内容"
    )
    
    coordinator.conduct_round(
        topic=debate_topic,
        proposition="平台需要防止虚假信息传播",
        opposition="事实核查应由独立机构负责"
    )
