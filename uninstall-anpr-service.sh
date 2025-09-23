#!/bin/bash

# ANPR Streamlit Service Uninstaller
# This script removes the ANPR Streamlit systemd service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="anpr-streamlit"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo -e "${BLUE}🗑️  ANPR Streamlit Service Uninstaller${NC}"
echo "======================================"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root${NC}"
   echo "Please run: sudo $0"
   exit 1
fi

# Check if service exists
if [ ! -f "$SERVICE_PATH" ]; then
    echo -e "${YELLOW}⚠️  Service not found: $SERVICE_PATH${NC}"
    echo "Service may already be uninstalled."
    exit 0
fi

# Stop and disable service
echo -e "${YELLOW}🔄 Stopping and disabling service...${NC}"
systemctl stop $SERVICE_NAME 2>/dev/null || true
systemctl disable $SERVICE_NAME 2>/dev/null || true

# Remove service file
echo -e "${YELLOW}🗑️  Removing service file...${NC}"
rm -f "$SERVICE_PATH"

# Reload systemd
echo -e "${YELLOW}🔄 Reloading systemd daemon...${NC}"
systemctl daemon-reload
systemctl reset-failed

echo -e "${GREEN}✅ Service uninstalled successfully!${NC}"
echo ""
echo -e "${BLUE}📝 Manual cleanup (if needed):${NC}"
echo "  - Check for remaining processes: ps aux | grep streamlit"
echo "  - Remove logs: sudo journalctl --vacuum-time=1d"