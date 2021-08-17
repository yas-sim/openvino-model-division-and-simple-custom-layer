if not exist build ( mkdir build )
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
copy Release\myLayers.pyd ..
cd ..
