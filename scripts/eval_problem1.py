# -*- coding: utf-8 -*-
"""Evaluate Problem 1 fixed configuration and print obscuration time.

Problem statement:
- FY1 speed = 120 m/s, heading to origin
- Release after 1.5 s
- Detonate 3.6 s after release
"""
from scenarios import Problem1
from physics.simulation import simulate_single_bomb


def main():
    sc = Problem1()
    st = sc.initial_states()
    res = simulate_single_bomb(
        m0=st['m0'], u0=st['u0'], heading_to_xy=st['heading_to_xy'], u_speed=st['u_speed'],
        bomb=st['bomb'], true_target=st['true_target'], t_max=st['t_max']
    )
    print(f"遮蔽总时长(题目1固定参数): {res.total_time:.3f} s")
    for iv in res.obscured_intervals:
        print(f"  区间: [{iv.t0:.3f}, {iv.t1:.3f}]  时长: {iv.t1-iv.t0:.3f} s")


if __name__ == '__main__':
    main()


