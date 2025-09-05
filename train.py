# -*- coding: utf-8 -*-
"""Command-line entry to run training/optimization with TensorBoard logging.

Usage:
    python train.py --scenario problem1 --optimizer random_search --steps 200 \
        --log_dir runs --seed 42 \
        --t_release_min 0.5 --t_release_max 5.0 \
        --det_delay_min 1.0 --det_delay_max 6.0 \
        --u_speed_min 70 --u_speed_max 140
"""
import argparse

from training import TrainConfig, Trainer


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', type=str, default='problem1')
    p.add_argument('--optimizer', type=str, default='random_search')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--log_dir', type=str, default='runs')
    p.add_argument('--seed', type=int, default=42)
    # Bounds
    p.add_argument('--t_release_min', type=float, default=0.5)
    p.add_argument('--t_release_max', type=float, default=5.0)
    p.add_argument('--det_delay_min', type=float, default=1.0)
    p.add_argument('--det_delay_max', type=float, default=6.0)
    p.add_argument('--u_speed_min', type=float, default=70.0)
    p.add_argument('--u_speed_max', type=float, default=140.0)
    return p


def main():
    args = build_argparser().parse_args()
    cfg = TrainConfig(
        scenario=args.scenario,
        optimizer=args.optimizer,
        steps=args.steps,
        log_dir=args.log_dir,
        seed=args.seed,
        t_release_min=args.t_release_min,
        t_release_max=args.t_release_max,
        det_delay_min=args.det_delay_min,
        det_delay_max=args.det_delay_max,
        u_speed_min=args.u_speed_min,
        u_speed_max=args.u_speed_max,
    )
    trainer = Trainer(cfg)
    best_val, best_x = trainer.train()
    print(f"Best obscuration time: {best_val:.3f}s")
    print(f"Best params: {best_x}")


if __name__ == '__main__':
    main()


