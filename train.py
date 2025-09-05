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
    p.add_argument('--scenario', type=str, default='problem3')
    p.add_argument('--optimizer', type=str, default='ga')
    p.add_argument('--steps', type=int, default=1024)
    p.add_argument('--log_dir', type=str, default='runs')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_workers', type=int, default=256, help='并行评估的进程数量（<=1 为关闭并行）')
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='使用GPU进行批评估')
    g.add_argument('--no_gpu', dest='use_gpu', action='store_false', help='禁用GPU批评估')
    p.set_defaults(use_gpu=None)
    p.add_argument('--gpu_device', type=str, default='cuda', help='GPU设备，如cuda或cuda:0')
    p.add_argument('--dt', type=float, default=0.02, help='仿真时间步长（用于GPU批评估）')
    # 批并行规模（GPU/CPU）
    p.add_argument('--rs_batch_size', type=int, default=32, help='RandomSearch批大小')
    p.add_argument('--pso_swarm_size', type=int, default=24, help='PSO粒子数量')
    p.add_argument('--ga_pop_size', type=int, default=256, help='GA种群大小')
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
        max_workers=args.max_workers,
        t_release_min=args.t_release_min,
        t_release_max=args.t_release_max,
        det_delay_min=args.det_delay_min,
        det_delay_max=args.det_delay_max,
        u_speed_min=args.u_speed_min,
        u_speed_max=args.u_speed_max,
    )
    if args.use_gpu is not None:
        cfg.use_gpu = bool(args.use_gpu)
    cfg.gpu_device = args.gpu_device
    cfg.dt = args.dt
    cfg.rs_batch_size = args.rs_batch_size
    cfg.pso_swarm_size = args.pso_swarm_size
    cfg.ga_pop_size = args.ga_pop_size
    trainer = Trainer(cfg)
    print(f"[train] 并行评估进程数: {cfg.max_workers}, use_gpu={cfg.use_gpu}, device={cfg.gpu_device}, dt={cfg.dt}")
    best_val, best_x = trainer.train()
    print(f"Best obscuration time: {best_val:.3f}s")
    print(f"Best params: {best_x}")


if __name__ == '__main__':
    main()


