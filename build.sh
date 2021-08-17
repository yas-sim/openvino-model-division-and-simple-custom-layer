mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
cp myLayers.so ..
cd ..
