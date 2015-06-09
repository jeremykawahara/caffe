#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver.prototxt 2>&1 | tee cifar-start.log

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt \
    --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate 2>&1 | tee cifar-lr1.log

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt \
    --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate 2>&1 | tee cifar-lr2.log
