cd ../../

set TOOLS=./bin

"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_quick_solver.prototxt

# reduce learning rate by factor of 10 after 8 epochs
"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_quick_solver_lr1.prototxt --snapshot=examples/cifar10/cifar10_quick_iter_4000.solverstate

pause