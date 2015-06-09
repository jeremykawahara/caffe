#!/usr/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver.prototxt 2>&1 | tee cifar-start.log

# reduce learning rate by factor of 10 after 8 epochs
$TOOLS/caffe train \
  --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt \
  --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate 2>&1 | tee cifar-lr1.log
