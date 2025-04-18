#!/bin/bash
# Setup script for Causal Blanket Analysis environment

# Print colored output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up environment for Causal Blanket Analysis...${NC}"

# Check if Python is installed
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    echo -e "${GREEN}Found Python: $(python3 --version)${NC}"
elif command -v python &>/dev/null; then
    PYTHON_CMD=python
    echo -e "${GREEN}Found Python: $(python --version)${NC}"
else
    echo -e "${RED}Error: Python not found. Please install Python 3.7 or higher.${NC}"
    exit 1
fi

# Recommend using virtual environment
echo -e "${BLUE}It is recommended to use a virtual environment.${NC}"
echo "Would you like to create a virtual environment? (y/n)"
read -r use_venv

if [[ "$use_venv" == "y" || "$use_venv" == "Y" ]]; then
    # Check if venv module is available
    if ! $PYTHON_CMD -c "import venv" &>/dev/null; then
        echo -e "${RED}Error: Python venv module not found. Please install it first.${NC}"
        exit 1
    fi
    
    # Create virtual environment
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv causal_env
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source causal_env/Scripts/activate
    else
        # Linux/Mac
        source causal_env/bin/activate
    fi
    
    echo -e "${GREEN}Virtual environment 'causal_env' created and activated.${NC}"
    PYTHON_CMD="python" # Inside venv, python command is correct
fi

# Install required packages
echo -e "${BLUE}Installing required packages...${NC}"
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Successfully installed required packages.${NC}"
else
    echo -e "${RED}Error installing packages. Please check your internet connection and try again.${NC}"
    exit 1
fi

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p data/particle_lenia_illumination
mkdir -p causal_analysis_plots
mkdir -p dashboard

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
MISSING_PACKAGES=0

for pkg in numpy scipy pandas matplotlib tqdm jax; do
    if ! $PYTHON_CMD -c "import $pkg" &>/dev/null; then
        echo -e "${RED}Package $pkg could not be imported!${NC}"
        MISSING_PACKAGES=$((MISSING_PACKAGES+1))
    fi
done

if [ $MISSING_PACKAGES -eq 0 ]; then
    echo -e "${GREEN}All core packages successfully installed!${NC}"
else
    echo -e "${RED}Warning: $MISSING_PACKAGES package(s) could not be imported.${NC}"
fi

# Display usage instructions
echo -e "${BLUE}Setup complete! Here's how to use the system:${NC}"
echo ""
echo "1. If using virtual environment, activate it with:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   source causal_env/Scripts/activate"
else
    echo "   source causal_env/bin/activate"
fi
echo ""
echo "2. Run the analysis:"
echo "   bash run_causal_blanket_analysis.sh"
echo ""
echo "3. View the dashboard by opening the following file in your browser:"
echo "   dashboard/index.html"

echo -e "${GREEN}Setup Complete!${NC}"
