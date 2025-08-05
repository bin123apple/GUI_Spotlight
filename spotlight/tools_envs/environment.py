import logging
from abc import ABC, abstractmethod
from vllm import LLM, SamplingParams  # type: ignore
from transformers import PreTrainedModel
from typing import Callable, Dict, Union, Any, List, Sequence
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class Environment(ABC):

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"agenttrain.envs.{self.__class__.__name__}")
        self.tokenizer = None
        self.dataset = None
        self.eval_dataset = None
        self.eot_id = 151643
        self.message_end_id = 151645
        self.reward_funcs = []
        self.reward_weights = []

    @abstractmethod
    def get_reward_funcs(self, **kwargs: Any) -> List[RewardFunc]:
        pass
    
    @abstractmethod
    def get_reward_weights(self, **kwargs: Any) -> List[float]:
        pass
    
    @abstractmethod
    def generate(self,
                 prompts: List[List[Dict[str, Any]]],
                 llm: LLM,
                 sampling_params: SamplingParams,
                 **kwargs: Any) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        pass
