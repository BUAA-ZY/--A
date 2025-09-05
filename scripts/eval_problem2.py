# -*- coding: utf-8 -*-
"""Evaluate Problem 2 by optimizing theta, u_speed, t_release, det_delay.

Usage:
    python scripts/eval_problem2.py --optimizer random_search --steps 400
"""
import argparse

from training import TrainConfig, Trainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--optimizer', type=str, default='random_search')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--max_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(scenario='problem2', optimizer=args.optimizer, steps=args.steps, seed=args.seed, max_workers=args.max_workers)
    trainer = Trainer(cfg)
    print(f"[eval_problem2] 并行评估进程数: {cfg.max_workers}")
    best_val, best_x = trainer.train()
    print(f"[Problem2] best obscuration: {best_val:.3f}s, params={best_x}")


if __name__ == '__main__':
    main()


