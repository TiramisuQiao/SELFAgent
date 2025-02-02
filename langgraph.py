from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any, Tuple
import numpy as np
import os
import pretty_errors
from collections import deque
import uuid
import time
from langchain.schema import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder

os.environ["OPENAI_API_KEY"] = "sk-K0kNVSkeaFmWL6wGi5f3Y4Ar3HOecVgP9WV6Xso4wyJuFeHV"
class HybridMemorySystem:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = FAISS.from_texts(
            texts=["作为一名合格的辩手，只能提出三个合理的观点，过多或者过少都会显得业余"],
            embedding=self.embeddings,
            metadatas=[{"type": "init_memory"}]
        )
        self.short_term_memory = EnhancedBufferMemory(
            window_size=5,
            memory_key="short_term_context"
        )
        self.feedback_history = deque(maxlen=10)  # 新增反馈历史跟踪

    def promote_memories(self):
        """增强记忆升级逻辑，加入反馈分析"""
        promoted_memories = self.short_term_memory.check_promotion()
        if promoted_memories:
            texts = [f"{m['content']} [反馈：{m['feedback']}]" for m in promoted_memories]
            metadatas = [{
                "type": "promoted_memory",
                "usage_count": m['usage_count'],
                "feedback_stats": m['feedback_stats']
            } for m in promoted_memories]
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)

    def record_feedback(self, feedback: Dict):
        """记录检查器反馈"""
        self.feedback_history.append({
            "timestamp": time.time(),
            "feedback": feedback
        })

    def get_user_preferences(self) -> Dict:
        """从长期记忆中提取用户偏好"""
        docs = self.vector_store.similarity_search("用户偏好", k=1)
        if docs:
            return docs[0].metadata.get("preferences", {})
        return {"concise_answers": True}

    def print_memory_state(self):
        """增强记忆状态输出"""
        print(f"\n=== Agent {self.agent_id} 记忆状态 ===")
        
        # 短期记忆带使用次数
        print("短期记忆（最近5轮）：")
        for msg in self.short_term_memory.chat_memory.messages[-5:]:
            count = self.short_term_memory.usage_counter.get(msg.additional_kwargs.get("message_id", ""), 0)
            print(f"[使用{count}次] {msg.content}")

        # 长期记忆带元数据
        print("\n长期记忆（前5条）：")
        docs = self.vector_store.similarity_search("", k=5)
        for doc in docs:
            print(f"{doc.page_content} | 类型：{doc.metadata['type']}")

class EnhancedBufferMemory(ConversationBufferWindowMemory):
    """增强型记忆系统，支持反馈追踪"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__dict__["usage_counter"] = {}
        self.__dict__["promotion_threshold"] = 3
        self.__dict__["feedback_log"] = []

        # self.usage_counter = {}
        # self.promotion_threshold = 3
        # self.feedback_log = []

    def _track_usage(self, message_id: str, feedback: Dict):
        """更新使用次数并记录反馈"""
        self.usage_counter[message_id] = self.usage_counter.get(message_id, 0) + 1
        self.feedback_log.append({
            "message_id": message_id,
            "feedback": feedback,
            "timestamp": time.time()
        })

    def check_promotion(self) -> List[Dict]:
        """增强升级逻辑，包含反馈分析"""
        promoted = []
        for msg_id, count in self.usage_counter.items():
            if count >= self.promotion_threshold:
                message = next(m for m in self.chat_memory.messages 
                             if m.additional_kwargs.get("message_id") == msg_id)
                
                # 分析相关反馈
                related_feedbacks = [
                    f["feedback"] for f in self.feedback_log
                    if f["message_id"] == msg_id
                ]
                
                promoted.append({
                    "content": message.content,
                    "usage_count": count,
                    "feedback_stats": {
                        "positive": sum(1 for f in related_feedbacks if f["correctness"]),
                        "negative": sum(1 for f in related_feedbacks if not f["correctness"])
                    }
                })
                del self.usage_counter[msg_id]
        return promoted

class DebateAgent:
    def __init__(self, agent_id: str, role: str):  
        self.memory = HybridMemorySystem(agent_id)
        self.role = role
        self.llm = ChatOpenAI(
            model="moonshot-v1-8k",
            base_url="https://api.moonshot.cn/v1",
            temperature=0.7
        )
        self.chain = self._build_chain()
        self.response_checker = ResponseChecker()  # 新增检查器组件

    def _build_chain(self):
        retriever = self.memory.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "filter": {"type": "promoted_memory"}}
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个{role}辩手，当前辩论主题：{topic}。用户偏好：{preferences}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        return RunnablePassthrough.assign(
            context=lambda x: retriever.get_relevant_documents(x["input"]),
            preferences=lambda x: self.memory.get_user_preferences()
        ) | prompt | self.llm

    def generate_response(self, topic: str, input_text: str) -> Tuple[str, Dict]:
        """生成带反馈机制的响应"""
        # 获取历史消息
        memory_variables = self.memory.short_term_memory.load_memory_variables({})
        history_messages = memory_variables.get(self.memory.short_term_memory.memory_key, [])
    
        # 确保 history_messages 是一个消息对象列表，并且没有空内容
        if isinstance(history_messages, str):
            # 如果 history_messages 是字符串，将其转换为消息对象列表
            history_messages = [HumanMessage(content=history_messages)] if history_messages.strip() else []
        elif isinstance(history_messages, list):
            # 如果 history_messages 是列表，确保每个元素都是消息对象，并且没有空内容
            history_messages = [
                HumanMessage(content=msg) if isinstance(msg, str) and msg.strip() else msg
                for msg in history_messages
                if (isinstance(msg, str) and msg.strip()) or (hasattr(msg, "content") and msg.content.strip())
            ]
        else:
            # 如果 history_messages 是其他类型，初始化为空列表
            history_messages = []
    
        # 确保输入文本不为空
        if not input_text.strip():
            raise ValueError("Input text cannot be empty.")
    
        # 构造 messages 列表
        messages = [
            {"role": "system", "content": f"You are a {self.role}辩手，当前辩论主题：{topic}。用户偏好：{self.memory.get_user_preferences()}"},
            *[
            {"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
                for msg in history_messages
            ],
            {"role": "user", "content": input_text}
        ]
    
        #    确保没有空消息
        messages = [msg for msg in messages if msg["content"].strip()]
    
        # 生成初始响应
        raw_response = self.llm.invoke(messages)
    
        # 检查器评估
        evaluation = self.response_checker.evaluate(
            response=raw_response.content,
            preferences=self.memory.get_user_preferences()
        )
    
        # 生成最终响应
        final_response = self._revise_response(raw_response.content, evaluation)
    
        # 更新记忆系统
        message_id = str(uuid.uuid4())
        self.memory.short_term_memory.save_context(
            {"input": input_text},  # inputs
            {"output": final_response, "message_id": message_id}  # outputs (including message_id)
        )
        self.memory.short_term_memory._track_usage(message_id, evaluation)
        self.memory.record_feedback(evaluation)
    
        return final_response, evaluation

    def _revise_response(self, response: str, feedback: Dict) -> str:
        """根据反馈修订响应"""
        if feedback.get("conciseness_violated"):
            return response.split("。")[0] + "。"  # 简单截断示例
        return response

class ResponseChecker:
    """响应检查器组件"""
    
    def evaluate(self, response: str, preferences: Dict) -> Dict:
        """执行多维度评估"""
        correctness = self._check_correctness(response)
        conciseness = len(response.split()) <= 50 if preferences.get("concise_answers") else True
        
        return {
            "correctness": correctness,
            "conciseness": conciseness,
            "feedback": self._generate_feedback(correctness, conciseness)
        }

    def _check_correctness(self, response: str) -> bool:
        """简单正确性检查（示例实现）"""
        forbidden_terms = ["不确定", "可能", "或许"]
        return not any(term in response for term in forbidden_terms)

    def _generate_feedback(self, correctness: bool, conciseness: bool) -> str:
        feedback = []
        if not correctness:
            feedback.append("响应包含不确定表述")
        if not conciseness:
            feedback.append("响应不够简洁")
        return "；".join(feedback) if feedback else "符合要求"

class DebateCoordinator:
    def __init__(self, agents: List[DebateAgent], rounds_before_promote: int = 3):
        self.agents = agents
        self.round_counter = 0
        self.promotion_interval = rounds_before_promote
        self.debate_log = []

    def conduct_round(self, topic: str, proposition: str, opposition: str):
        print(f"\n=== 第 {self.round_counter+1} 轮辩论 ===")
        
        # 正方发言带反馈
        prop_response, prop_feedback = self.agents[0].generate_response(
            topic, f"作为正方，请回应：{proposition}"
        )
        print(f"\n正方：{prop_response} [检查反馈：{prop_feedback['feedback']}]")
        
        # 反方发言带反馈
        oppo_response, oppo_feedback = self.agents[1].generate_response(
            topic, f"作为反方，请回应：{opposition}。对方刚才说：{prop_response}"
        )
        print(f"\n反方：{oppo_response} [检查反馈：{oppo_feedback['feedback']}]")
        
        # 记录辩论日志
        self.debate_log.append({
            "round": self.round_counter+1,
            "responses": {
                "proposition": (prop_response, prop_feedback),
                "opposition": (oppo_response, oppo_feedback)
            }
        })
        
        # 记忆管理
        if (self.round_counter+1) % self.promotion_interval == 0:
            for agent in self.agents:
                agent.memory.promote_memories()
            print("\n=== 记忆升级完成 ===")
        
        self.round_counter += 1

if __name__ == "__main__":
    prop_agent = DebateAgent(agent_id="prop_1", role="专业级正方辩手")
    oppo_agent = DebateAgent(agent_id="oppo_1", role="资深反方辩手")
    
    coordinator = DebateCoordinator([prop_agent, oppo_agent])
    
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
    
    # 打印最终记忆状态
    print("\n=== 最终记忆状态 ===")
    for agent in [prop_agent, oppo_agent]:
        agent.memory.print_memory_state()