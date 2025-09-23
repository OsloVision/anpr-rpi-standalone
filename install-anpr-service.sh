#!/bin/bash

# ANPR Streamlit Service Installation Script
# This script installs the ANPR Streamlit application as a systemd service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="anpr-streamlit"
SERVICE_FILE="anpr-streamlit.service"
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"

echo -e "${BLUE}🚀 ANPR Streamlit Service Installation${NC}"
echo "======================================"

# Check if running as root for port 80
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}❌ This script must be run as root to bind to port 80${NC}"
   echo "Please run: sudo $0"
   exit 1
fi

# Verify service file exists
if [ ! -f "$CURRENT_DIR/$SERVICE_FILE" ]; then
    echo -e "${RED}❌ Service file not found: $CURRENT_DIR/$SERVICE_FILE${NC}"
    exit 1
fi

# Stop existing service if running
echo -e "${YELLOW}🔄 Stopping existing service (if running)...${NC}"
systemctl stop $SERVICE_NAME 2>/dev/null || true
systemctl disable $SERVICE_NAME 2>/dev/null || true

# Copy service file
echo -e "${YELLOW}📁 Installing service file...${NC}"
cp "$CURRENT_DIR/$SERVICE_FILE" "$SERVICE_PATH"
chmod 644 "$SERVICE_PATH"

# Reload systemd
echo -e "${YELLOW}🔄 Reloading systemd daemon...${NC}"
systemctl daemon-reload

# Enable and start service
echo -e "${YELLOW}🚀 Enabling and starting service...${NC}"
systemctl enable $SERVICE_NAME
systemctl start $SERVICE_NAME

# Check service status
sleep 2
if systemctl is-active --quiet $SERVICE_NAME; then
    echo -e "${GREEN}✅ Service started successfully!${NC}"
    echo ""
    echo -e "${GREEN}📊 Service Status:${NC}"
    systemctl status $SERVICE_NAME --no-pager -l
    echo ""
    echo -e "${GREEN}🌐 Access the application at:${NC}"
    echo "  - Local: http://localhost"
    echo "  - Network: http://$(hostname -I | awk '{print $1}')"
    echo ""
    echo -e "${BLUE}📝 Useful commands:${NC}"
    echo "  - Check status: sudo systemctl status $SERVICE_NAME"
    echo "  - View logs: sudo journalctl -u $SERVICE_NAME -f"
    echo "  - Restart: sudo systemctl restart $SERVICE_NAME"
    echo "  - Stop: sudo systemctl stop $SERVICE_NAME"
    echo "  - Disable: sudo systemctl disable $SERVICE_NAME"
else
    echo -e "${RED}❌ Service failed to start!${NC}"
    echo ""
    echo -e "${YELLOW}📋 Service status:${NC}"
    systemctl status $SERVICE_NAME --no-pager -l
    echo ""
    echo -e "${YELLOW}📋 Recent logs:${NC}"
    journalctl -u $SERVICE_NAME --no-pager -l -n 20
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Installation complete!${NC}"