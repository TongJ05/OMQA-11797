# Test if ninja is working properly
echo "Testing ninja functionality..."
mkdir -p /tmp/ninja_test
cd /tmp/ninja_test
cat > build.ninja << 'EOF'
rule cc
  command = echo "Ninja is working"

build test: cc
EOF

ninja
NINJA_TEST_STATUS=$?

if [ $NINJA_TEST_STATUS -ne 0 ]; then
    echo "Ninja is not functioning correctly. Fixing installation..."
    pip uninstall -y ninja pybind11
    pip install -U setuptools wheel
    pip install -U pybind11
    pip install -U ninja
    
    # Test again
    ninja
    if [ $? -ne 0 ]; then
        echo "Ninja still not working. Aborting."
        exit 1
    fi
fi