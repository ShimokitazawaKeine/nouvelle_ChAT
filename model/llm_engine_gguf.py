from llama_cpp import Llama
import logging
from typing import Optional, Dict, List, Union, Callable
from functools import lru_cache
import threading
from dataclasses import dataclass
from enum import Enum
import time
import os
import gc

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义任务类型枚举
class TaskType(Enum):
    GENERAL = "general"  # 普通对话
    VOICE_INTERACTION = "voice_interaction"  # 语音助手
    HEALTH_CONSULTATION = "health_consultation"  # 健康咨询
    EMOTIONAL_SUPPORT = "emotional_support"  # 情绪支持
    EMERGENCY = "emergency"  # 紧急响应
    PERSON_IDENTIFICATION = "person_identification"  # 人物识别
    OBJECT_RECOGNITION = "object_recognition"  # 物体识别

# 定义生成配置
@dataclass
class GenerationConfig:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True

# 主模型引擎类
class QwenLLMEngine:
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8, use_cache: bool = True):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.use_cache = use_cache
        self._lock = threading.Lock()

        # 初始化统计信息
        self.stats = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "average_response_time": 0.0,
            "cache_hits": 0
        }
        self._initialize_model()  # 加载模型
        self.system_prompts = self._load_system_prompts()  # 加载系统提示
        self.post_processors = self._setup_post_processors()  # 设置后处理函数

    # 加载 GGUF 模型
    def _initialize_model(self):
        try:
            logger.info(f"[QwenLLMEngine] Loading GGUF model: {self.model_path}")
            self.model = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads)
            logger.info("[QwenLLMEngine] GGUF model loaded successfully.")
        except Exception as e:
            logger.error(f"[QwenLLMEngine] Failed to load model: {e}")
            self.model = None

    # 系统提示模板
    def _load_system_prompts(self) -> Dict[str, str]:
        return {
            TaskType.GENERAL.value: "You are a helpful AI assistant.",
            TaskType.VOICE_INTERACTION.value: "You are a friendly voice assistant.",
            TaskType.HEALTH_CONSULTATION.value: "You are a health advisory assistant.",
            TaskType.EMOTIONAL_SUPPORT.value: "You are an empathetic emotional support assistant.",
            TaskType.EMERGENCY.value: "You are an emergency response assistant.",
            TaskType.PERSON_IDENTIFICATION.value: "You help identify people based on context.",
            TaskType.OBJECT_RECOGNITION.value: "You help identify objects based on input."
        }

    # 设置每个任务的后处理函数
    def _setup_post_processors(self) -> Dict[str, Callable]:
        return {
            TaskType.VOICE_INTERACTION.value: self._voice_post_process,
            TaskType.HEALTH_CONSULTATION.value: self._health_post_process,
            TaskType.EMERGENCY.value: self._emergency_post_process,
            TaskType.PERSON_IDENTIFICATION.value: self._person_id_post_process,
        }

    # 清洗语音助手输出
    def _voice_post_process(self, response: str) -> str:
        return " ".join(response.replace("*", "").replace("_", "").split())

    # 健康咨询添加免责声明
    def _health_post_process(self, response: str) -> str:
        if "doctor" not in response.lower():
            response += "\n\nReminder: Consult a healthcare professional if symptoms persist."
        return response

    # 紧急任务突出提示
    def _emergency_post_process(self, response: str) -> str:
        if any(k in response.lower() for k in ["emergency", "urgent", "danger"]):
            response = "[URGENT] " + response
        return response

    # 人物识别未能回答时请求更多信息
    def _person_id_post_process(self, response: str) -> str:
        if "unknown" in response.lower():
            response += " Please provide more context."
        return response

    # 模型推理主函数
    def _raw_inference(self, prompt: str, config: GenerationConfig) -> str:
        try:
            result = self.model.create_completion(
                prompt=prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repetition_penalty,
            )
            return result["choices"][0]["text"].strip()
        except Exception as e:
            logger.error(f"[QwenLLMEngine] Inference error: {e}")
            return "Model failed to generate a response."

    # 构建完整 prompt
    def build_prompt(self, user_input: str, task_type: Union[str, TaskType] = TaskType.GENERAL, context: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None, system_context: Optional[str] = None) -> str:
        if isinstance(task_type, TaskType):
            task_type = task_type.value
        system_prompt = self.system_prompts.get(task_type, self.system_prompts[TaskType.GENERAL.value])
        if system_context:
            system_prompt += f"\nContext: {system_context}"
        parts = [system_prompt]
        if context:
            parts.append("\n".join(f"{k}: {v}" for k, v in context.items()))
        if conversation_history:
            formatted_history = []
            for m in conversation_history[-3:]:
                role = m.get("role", "user")  # 默认为 user
                content = m.get("content", "")  # 默认为空字符串
                prefix = "User" if role == "user" else "Assistant"
                formatted_history.append(f"{prefix}: {content}")
            parts.append("\n".join(formatted_history))

        parts.append(f"User: {user_input}\nAssistant:")
        return "\n\n".join(parts)

    # 外部调用入口
    def run_inference(self, user_input: str, task_type: Union[str, TaskType] = TaskType.GENERAL, context: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None, system_context: Optional[str] = None, config: Optional[GenerationConfig] = None) -> Dict[str, Union[str, float, int]]:
        if self.model is None:
            return {"response": "Model is not loaded.", "error": True, "response_time": 0.0}

        start = time.time()
        with self._lock:
            prompt = self.build_prompt(user_input, task_type, context, conversation_history, system_context)
            config = config or GenerationConfig()
            response = self._raw_inference(prompt, config)

            task_type_str = task_type.value if isinstance(task_type, TaskType) else task_type
            if task_type_str in self.post_processors:
                response = self.post_processors[task_type_str](response)

            elapsed = time.time() - start
            self.stats["total_requests"] += 1
            self.stats["average_response_time"] = (
                (self.stats["average_response_time"] * (self.stats["total_requests"] - 1) + elapsed)
                / self.stats["total_requests"]
            )
            tokens_generated = len(response.split()) if response else 0

            return {
                "response": response,
                "response_time": elapsed,
                "error": False,
                "task_type": task_type_str,
                "tokens_generated": tokens_generated
            }

    # 获取当前性能统计
    def get_stats(self) -> Dict:
        return self.stats.copy()

    # 析构时清理资源
    def __del__(self):
        gc.collect()

# 简化包装类
class SimpleQwenLLM:
    def __init__(self, model_path: str, max_tokens: int = 128, temperature: float = 0.7):
        self.engine = QwenLLMEngine(model_path=model_path)
        self.max_tokens = max_tokens
        self.temperature = temperature

    # 构建简化 prompt
    def build_prompt(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        return f"{system_prompt or ''}\nUser: {user_input}\nAssistant:"

    # 直接调用引擎生成文本
    def generate(self, user_input: str, system_prompt: Optional[str] = None) -> str:
        config = GenerationConfig(max_tokens=self.max_tokens, temperature=self.temperature)
        result = self.engine.run_inference(user_input=user_input, system_context=system_prompt, config=config)
        return result.get("response", "[Error]")

# 获取模型实例（工厂方法）
def get_qwen_engine(model_path: str) -> QwenLLMEngine:
    return QwenLLMEngine(model_path)

# 获取全局单例模型实例
def get_global_engine(model_path: str) -> QwenLLMEngine:
    global _global_engine
    if '_global_engine' not in globals():
        globals()['_global_engine'] = get_qwen_engine(model_path)
    return globals()['_global_engine']

# 兼容旧调用接口
def run_llm_inference(audio_path: str) -> str:
    engine = get_global_engine()
    dummy = "Hi, I feel dizzy today. Can you help me?"
    result = engine.run_inference(dummy, task_type=TaskType.VOICE_INTERACTION)
    return result.get("response", "[Error]")
















###############################################################################################################################
# from llama_cpp import Llama
#
# # 载入模型
# llm = Llama(
#     model_path="./gguf/Qwen3-0.6B-Q8_0.gguf",
#     n_ctx=2048,
#     n_threads=6,
#     verbose=True
# )
#
# def build_qwen_prompt(user_msg):
#     return (
#         "<|im_start|>system\n你是一个有帮助的AI助手。\n<|im_end|>\n"
#         f"<|im_start|>user\n{user_msg}\n<|im_end|>\n"
#         "<|im_start|>assistant\n"
#     )
#
#
# def chat(prompt):
#     formatted_prompt = build_qwen_prompt(prompt)
#     output = llm(formatted_prompt, max_tokens=256, stop=["<|im_end|>"])
#     print("模型回复：", output["choices"][0]["text"].strip())
#
#
# # 开始循环对话
# while True:
#     user_input = input("你：")
#     if user_input.lower() in ["exit", "quit"]:
#         break
#     chat(user_input)

