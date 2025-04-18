#!/bin/bash
# Script to run causal blanket analysis on previously generated particle lenia data

# Color definitions
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if required packages are installed
echo -e "${BLUE}Checking required packages...${NC}"
if [ -f "requirements.txt" ]; then
    # Verify critical packages
    MISSING=0
    for pkg in numpy pandas matplotlib tqdm jax; do
        if ! python -c "import $pkg" &>/dev/null; then
            echo -e "${RED}Package $pkg is not installed!${NC}"
            MISSING=$((MISSING+1))
        fi
    done
    
    if [ $MISSING -gt 0 ]; then
        echo -e "${RED}Missing $MISSING required package(s). Please install using:${NC}"
        echo -e "pip install -r requirements.txt"
        echo -e "Or run the setup script: bash setup.sh"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}All required packages found!${NC}"
    fi
else
    echo -e "${RED}Warning: requirements.txt not found.${NC}"
    echo "You may need to install required packages manually if analysis fails."
fi

# Directory where the data was saved
DATA_DIR="./data/particle_lenia_illumination"

# Output CSV file name
OUTPUT_CSV="causal_blanket_results.csv"

# Output plot directory
PLOT_DIR="./causal_analysis_plots"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Warning: Data directory not found: $DATA_DIR${NC}"
    echo "Would you like to create it now? (y/n)"
    read -r create_dir
    if [[ "$create_dir" == "y" || "$create_dir" == "Y" ]]; then
        mkdir -p "$DATA_DIR"
        echo -e "${GREEN}Directory created. Please add your data files before running analysis.${NC}"
        exit 0
    else
        echo "Please ensure your data is in $DATA_DIR before running this script."
        exit 1
    fi
fi

# Run the analysis with improved partitioning
echo -e "${BLUE}Running causal blanket analysis with improved partitioning...${NC}"
python causal_blanket_analysis.py \
    --data_dir="$DATA_DIR" \
    --out_csv="$OUTPUT_CSV" \
    --bins=10 \
    --time_lag=1 \
    --plot \
    --n_analyze=10 \
    --plot_dir="$PLOT_DIR" \
    --min_timesteps=3 \
    --partition_method="velocity"

# Check if analysis completed successfully
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Causal blanket analysis failed. Exiting.${NC}"
    exit 1
fi

echo -e "${GREEN}Analysis complete. Results saved to $OUTPUT_CSV${NC}"

# Create a visualization dashboard using the dedicated script
echo -e "${BLUE}Creating visualization dashboard...${NC}"
python create_dashboard.py "$OUTPUT_CSV" "$PLOT_DIR"

# Check if dashboard creation was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Dashboard created. Open dashboard/index.html in a web browser to view results.${NC}"
else
    echo -e "${RED}Error: Dashboard creation failed.${NC}"
    exit 1
fi

# Print absolute path to the dashboard for easier access
DASHBOARD_PATH=$(pwd)/dashboard/index.html
echo -e "${GREEN}Dashboard available at: file://$DASHBOARD_PATH${NC}"

# Attempt to open dashboard automatically if possible
if command -v xdg-open &> /dev/null; then
    echo "Opening dashboard in default browser..."
    xdg-open "file://$DASHBOARD_PATH"
elif command -v open &> /dev/null; then
    echo "Opening dashboard in default browser..."
    open "file://$DASHBOARD_PATH"
else
    echo "To view the dashboard, open this file in your web browser:"
    echo "$DASHBOARD_PATH"
fi
