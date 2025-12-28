#!/bin/bash
# Tennis Computer Vision - Setup and Test Script
# File: setup_and_test.sh

echo "============================================================"
echo "TENNIS CV - SETUP AND TEST SCRIPT"
echo "============================================================"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the TennisApp directory."
    exit 1
fi

echo "✓ Found requirements.txt"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Installing pip3..."
    echo "Please run: sudo apt install python3-pip"
    echo "Then run this script again."
    exit 1
fi

echo "✓ pip3 is available"

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Run structure validation
echo ""
echo "Running structure validation..."
python3 tests/validate_structure.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Running simple pipeline test..."
    python3 tests/run_simple_test.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "Running integration test with real video..."
        python3 tests/test_pipeline_integration.py
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "============================================================"
            echo "✅ ALL TESTS PASSED!"
            echo "============================================================"
            echo "Pipeline is working correctly!"
            echo ""
            echo "Next steps:"
            echo "1. Implement Court Detection & Homography"
            echo "2. Implement Ball & Player Detection"
            echo "3. Implement Ball Trajectory & Event Analysis"
        else
            echo "❌ Integration test failed"
            exit 1
        fi
    else
        echo "❌ Simple pipeline test failed"
        exit 1
    fi
else
    echo "❌ Structure validation failed"
    exit 1
fi


