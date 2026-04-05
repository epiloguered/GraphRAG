from typing import Dict, Type

from strategies.chunk_only_strategy import ChunkOnlyStrategy
from strategies.node_relation_strategy import NodeRelationStrategy
from strategies.triple_only_strategy import TripleOnlyStrategy
from strategies.youtu_default_strategy import YoutuDefaultStrategy


class StrategyRegistry:
    def __init__(self):
        self._strategies: Dict[str, Type] = {
            YoutuDefaultStrategy.strategy_id: YoutuDefaultStrategy,
            ChunkOnlyStrategy.strategy_id: ChunkOnlyStrategy,
            TripleOnlyStrategy.strategy_id: TripleOnlyStrategy,
            NodeRelationStrategy.strategy_id: NodeRelationStrategy,
        }

    def get(self, strategy_name: str):
        if strategy_name not in self._strategies:
            raise ValueError(f"Unknown retrieval mode: {strategy_name}")
        return self._strategies[strategy_name]()

    def list_modes(self):
        modes = []
        for strategy_name, strategy_class in self._strategies.items():
            modes.append({
                "id": strategy_name,
                "name": strategy_class.display_name,
                "description": strategy_class.description,
            })
        return modes
