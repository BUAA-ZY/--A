# -*- coding: utf-8 -*-
"""Evaluate Problem 3 (FY1投放三弹) by optimizing [theta, u, t/d x3].

Usage examples:
  python scripts/eval_problem3.py --optimizer pso --steps 1200
  python scripts/eval_problem3.py --optimizer sa  --steps 1500
"""
import argparse

from training import TrainConfig, Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', type=str, default='pso')
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--max_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(scenario='problem3', optimizer=args.optimizer, steps=args.steps, seed=args.seed, max_workers=args.max_workers)
    trainer = Trainer(cfg)
    print(f"[eval_problem3] 并行评估进程数: {cfg.max_workers}")
    best_val, best_x = trainer.train()
    print(f"[Problem3] best total obscuration: {best_val:.3f}s")
    print(f"best params: {best_x}")


if __name__ == '__main__':
    main()


