echo "Downloading..."

set wget="../../3rdparty/bin/wget.exe"
set do_7za="../../3rdparty/bin/7za.exe"

%wget% --no-check-certificate http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "Unzipping..."

%do_7za% x cifar-10-binary.tar.gz
%do_7za% x cifar-10-binary.tar

echo "Done."
