!bin/sh

pip3 install -r requirements.txt
cd docs
sphinx-build -b html source build
cd ..
