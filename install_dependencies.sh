#!/bin/bash
# Install all dependencies for the adaptive router demo

echo "============================================================"
echo "Installing Adaptive Router Demo Dependencies"
echo "============================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if Node.js is available
if ! command -v npm &> /dev/null; then
    echo "✗ npm not found. Please install Node.js 16+"
    exit 1
fi

echo "Installing backend dependencies..."
cd backend
python3 -m pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Backend dependencies installed"
else
    echo "✗ Backend installation failed"
    exit 1
fi
cd ..

echo ""
echo "Installing frontend dependencies..."
cd frontend
npm install
if [ $? -eq 0 ]; then
    echo "✓ Frontend dependencies installed"
else
    echo "✗ Frontend installation failed"
    exit 1
fi
cd ..

echo ""
echo "============================================================"
echo "✓ All dependencies installed!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Export your models: python3 export_models_for_demo.py --checkpoint your_checkpoint.pt"
echo "  2. Verify setup: python3 verify_setup.py"
echo "  3. Start backend: cd backend && python3 run.py"
echo "  4. Start frontend: cd frontend && npm run dev"
