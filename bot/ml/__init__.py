from bot.ml.genetic    import GeneticOptimizer
from bot.ml.supervised import SupervisedStrategy
from bot.ml.rl_agent   import RLStrategy

import bot.strategy as _strat
_strat.REGISTRY.setdefault("supervised", SupervisedStrategy)
_strat.REGISTRY.setdefault("rl_ppo",     RLStrategy)

__all__ = ["GeneticOptimizer", "SupervisedStrategy", "RLStrategy"]
