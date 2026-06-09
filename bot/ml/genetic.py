"""
Genetic Algorithm optimizer for strategy hyperparameters.

Finds the best combination of strategy parameters (e.g. SMA periods,
RSI thresholds) by evolving a population of candidates over multiple
generations and evaluating each via backtesting.

Usage:
    from bot.ml.genetic import GeneticOptimizer
    from bot.strategy import SmaCrossover
    from bot.backtest.engine import BacktestEngine

    engine = BacktestEngine()
    optimizer = GeneticOptimizer(
        strategy_cls=SmaCrossover,
        param_grid={"fast": list(range(5, 21)), "slow": list(range(20, 101, 5))},
        engine=engine,
        population_size=30,
        generations=20,
        fitness_metric="sharpe",
    )
    best_params, best_result = optimizer.optimize(
        "AAPL", "1d", since="2023-01-01", until="2023-10-01"
    )
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from bot.backtest.engine import BacktestEngine, BacktestResult
    from bot.strategy.base import BaseStrategy

console = Console()


@dataclass
class Individual:
    params:  dict[str, Any]
    fitness: float = field(default=-math.inf)
    result:  Optional["BacktestResult"] = field(default=None, repr=False)


class GeneticOptimizer:
    """
    Evolves strategy parameters using:
      - Tournament selection
      - Uniform crossover
      - Per-gene random mutation
      - Elite carry-over (top N individuals survive unchanged)

    Parameters
    ----------
    strategy_cls:    Strategy class (not instance) to optimise
    param_grid:      {"param_name": [value1, value2, ...], ...}
    engine:          BacktestEngine to use for fitness evaluation
    population_size: Number of individuals per generation
    generations:     Number of evolution cycles
    elite_frac:      Fraction of top individuals to carry over unchanged
    mutation_rate:   Per-gene probability of random mutation
    fitness_metric:  Key from BacktestResult.metrics to maximise
    tournament_size: Contestants drawn for each selection
    holdout_frac:    Fraction of data reserved as an out-of-sample test set.
                     Optimisation runs ONLY on the train portion; the winner is
                     then scored on the untouched holdout to expose overfitting.
                     Set 0 to disable (not recommended).
    val_since/until: Optional explicit validation window (overrides holdout).
    """

    def __init__(
        self,
        strategy_cls: type["BaseStrategy"],
        param_grid: dict[str, list],
        engine: "BacktestEngine",
        population_size: int = 30,
        generations: int = 20,
        elite_frac: float = 0.15,
        mutation_rate: float = 0.20,
        fitness_metric: str = "sharpe",
        tournament_size: int = 4,
        holdout_frac: float = 0.30,
        val_since: Optional[str] = None,
        val_until: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.strategy_cls    = strategy_cls
        self.param_grid      = param_grid
        self.engine          = engine
        self.population_size = population_size
        self.generations     = generations
        self.elite_count     = max(1, int(population_size * elite_frac))
        self.mutation_rate   = mutation_rate
        self.fitness_metric  = fitness_metric
        self.tournament_size = tournament_size
        self.holdout_frac    = holdout_frac
        self.val_since       = val_since
        self.val_until       = val_until
        self.rng             = random.Random(seed)
        self.history: list[dict] = []

    def _random_individual(self) -> Individual:
        return Individual(params={k: self.rng.choice(v) for k, v in self.param_grid.items()})

    def _evaluate(
        self, ind: Individual,
        symbol: str, timeframe: str,
        since: str, until: Optional[str],
    ) -> Individual:
        try:
            strat      = self.strategy_cls(**ind.params)
            eval_since = self.val_since or since
            eval_until = self.val_until or until
            result     = self.engine.run(strat, symbol, timeframe, eval_since, eval_until)
            fitness    = result.metrics.get(self.fitness_metric, -math.inf)
            if not math.isfinite(fitness) or result.metrics["total_trades"] < 2:
                fitness = -math.inf
            return Individual(params=ind.params, fitness=fitness, result=result)
        except Exception:
            return Individual(params=ind.params, fitness=-math.inf)

    def _tournament_select(self, pop: list[Individual]) -> Individual:
        k = min(self.tournament_size, len(pop))
        return max(self.rng.sample(pop, k), key=lambda x: x.fitness)

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        params = {k: self.rng.choice([p1.params[k], p2.params[k]]) for k in self.param_grid}
        return Individual(params=params)

    def _mutate(self, ind: Individual) -> Individual:
        params = dict(ind.params)
        for k, choices in self.param_grid.items():
            if self.rng.random() < self.mutation_rate:
                params[k] = self.rng.choice(choices)
        return Individual(params=params)

    @staticmethod
    def _ts_to_date(ts_ms: int) -> str:
        import pandas as pd
        return pd.Timestamp(ts_ms, unit="ms").strftime("%Y-%m-%d")

    def _split_dates(
        self, symbol: str, timeframe: str, since: str, until: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Return (train_until, test_since) for the holdout split, or (None, None)."""
        if not (0 < self.holdout_frac < 1) or self.val_since:
            return None, None
        df = self.engine.store.load(symbol, timeframe, since=since, until=until)
        if len(df) < 50:
            return None, None
        split = int(len(df) * (1 - self.holdout_frac))
        if split < 1 or split >= len(df):
            return None, None
        ts = df["timestamp"]
        return self._ts_to_date(int(ts.iloc[split - 1])), self._ts_to_date(int(ts.iloc[split]))

    def _run_params(self, params, symbol, timeframe, since, until):
        try:
            return self.engine.run(self.strategy_cls(**params), symbol, timeframe, since, until)
        except Exception:
            return None

    def optimize(
        self,
        symbol: str,
        timeframe: str,
        since: str,
        until: Optional[str] = None,
    ) -> tuple[dict, "BacktestResult"]:
        """
        Run the genetic optimisation.

        Optimisation is performed on the training window only; the best
        individual is then re-scored on an untouched out-of-sample holdout
        so the train→OOS performance gap (the overfitting tell) is visible.

        Returns
        -------
        best_params : dict           — best hyperparameter values found
        best_result : BacktestResult — IN-SAMPLE backtest result for those params
        """
        train_until, test_since = self._split_dates(symbol, timeframe, since, until)
        opt_until = train_until or until   # evolve on the train window only

        population = [self._random_individual() for _ in range(self.population_size)]
        best_ever: Optional[Individual] = None
        self.history = []

        split_note = (
            f"train {since}→{train_until} | OOS {test_since}→{until or 'end'}"
            if train_until else "[yellow]no holdout (full-period, overfit-prone)[/]"
        )
        console.rule(
            f"[bold cyan]Genetic Optimiser — {self.strategy_cls.__name__} "
            f"| metric={self.fitness_metric} | {self.generations}gen × {self.population_size}pop[/]"
        )
        console.print(f"  [dim]{split_note}[/]")

        for gen in range(self.generations):
            population = [
                self._evaluate(ind, symbol, timeframe, since, opt_until)
                for ind in population
            ]
            population.sort(key=lambda x: x.fitness, reverse=True)

            best  = population[0]
            valid = [x for x in population if math.isfinite(x.fitness)]
            avg   = sum(x.fitness for x in valid) / max(1, len(valid))

            self.history.append({"gen": gen + 1, "best": best.fitness, "avg": avg})

            if best_ever is None or best.fitness > best_ever.fitness:
                best_ever = best

            console.print(
                f"  Gen {gen+1:>3}/{self.generations}  "
                f"best={best.fitness:+.4f}  avg={avg:+.4f}  "
                f"params={best.params}"
            )

            elites   = population[: self.elite_count]
            next_pop = list(elites)
            while len(next_pop) < self.population_size:
                p1    = self._tournament_select(population)
                p2    = self._tournament_select(population)
                child = self._mutate(self._crossover(p1, p2))
                next_pop.append(child)
            population = next_pop

        # ── Out-of-sample re-scoring of the winner ─────────────────────────
        oos_result = None
        if test_since:
            oos_result = self._run_params(best_ever.params, symbol, timeframe, test_since, until)

        self._print_top(best_ever, oos_result)
        return best_ever.params, best_ever.result

    def _print_top(self, best: Individual, oos_result: Optional["BacktestResult"] = None) -> None:
        console.rule("[bold green]Optimisation complete[/]")

        t = Table(title="Best Individual", box=box.SIMPLE_HEAVY, show_header=False)
        t.add_column("Key",   style="dim")
        t.add_column("Value", justify="right")
        for k, v in best.params.items():
            t.add_row(k, str(v))
        console.print(t)

        # Train vs out-of-sample comparison — the overfitting check
        metric = self.fitness_metric
        cmp = Table(title="Train → Out-of-Sample", box=box.SIMPLE_HEAVY)
        cmp.add_column("Metric")
        cmp.add_column("Train (IS)", justify="right")
        cmp.add_column("Holdout (OOS)", justify="right")

        is_m  = best.result.metrics if best.result else {}
        oos_m = oos_result.metrics if oos_result else None

        def _row(label, key, sfx=""):
            iv = is_m.get(key, float("nan"))
            ov = oos_m.get(key, float("nan")) if oos_m else None
            ostr = f"{ov:+.3f}{sfx}" if ov is not None else "[dim]—[/]"
            cmp.add_row(label, f"{iv:+.3f}{sfx}", ostr)

        _row("Sharpe",       "sharpe")
        _row("Return",       "total_return_pct", "%")
        _row("Max DD",       "max_drawdown_pct", "%")
        cmp.add_row("Trades",
                    str(is_m.get("total_trades", 0)),
                    str(oos_m.get("total_trades", 0)) if oos_m else "[dim]—[/]")
        console.print(cmp)

        # Verdict
        if oos_m is not None:
            is_fit  = is_m.get(metric, float("nan"))
            oos_fit = oos_m.get(metric, float("nan"))
            if math.isfinite(is_fit) and math.isfinite(oos_fit) and abs(is_fit) > 1e-9:
                retention = oos_fit / is_fit if is_fit != 0 else 0.0
                if oos_fit <= 0 < is_fit:
                    console.print(f"  [bold red]⚠ OVERFIT[/]: {metric} collapses "
                                  f"{is_fit:+.3f} (IS) → {oos_fit:+.3f} (OOS). Params don't generalise.")
                elif retention < 0.5:
                    console.print(f"  [yellow]△ WEAK[/]: OOS keeps only {retention:.0%} of "
                                  f"in-sample {metric}. Treat with caution.")
                else:
                    console.print(f"  [bold green]✓ HOLDS UP[/]: OOS keeps {retention:.0%} of "
                                  f"in-sample {metric}.")
        else:
            console.print("  [yellow]No holdout evaluated — results are in-sample only.[/]")
