echo "Downloading..."

set wget="../../3rdparty/bin/wget.exe"
set do_7za="../../3rdparty/bin/7za.exe"

%wget% --no-check-certificate http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz

echo "Unzipping..."

%do_7za% x caffe_ilsvrc12.tar.gz
%do_7za% x caffe_ilsvrc12.tar

echo "Done."
