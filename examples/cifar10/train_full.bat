cd ../../

set TOOLS=./bin

"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_full_solver.prototxt

# reduce learning rate by factor of 10
"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_full_solver_lr1.prototxt --snapshot=examples/cifar10/cifar10_full_iter_60000.solverstate

# reduce learning rate by factor of 10
"%TOOLS%/caffe.exe" train --solver=examples/cifar10/cifar10_full_solver_lr2.prototxt --snapshot=examples/cifar10/cifar10_full_iter_65000.solverstate

pause
