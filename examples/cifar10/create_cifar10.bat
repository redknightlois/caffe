cd ../../

set EXAMPLE=examples/cifar10
set DATA=data/cifar10/cifar-10-batches-bin
set BUILD=bin

REM set BACKEND=lmdb
set BACKEND=leveldb

echo "Creating %BACKEND%..."

rd /s /q "%EXAMPLE%/cifar10_train_%BACKEND%"
rd /s /q "%EXAMPLE%/cifar10_test_%BACKEND%"

"%BUILD%/convert_cifar10_data.exe" %DATA% %EXAMPLE% %BACKEND%

echo "Computing image mean..."

"%BUILD%/compute_image_mean.exe" -backend=%BACKEND% %EXAMPLE%/cifar10_train_%BACKEND% %EXAMPLE%/mean.binaryproto

echo "Done."

pause