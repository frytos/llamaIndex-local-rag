#!/bin/bash
#
# Caddy Setup Script - Automatic HTTPS for RAG Streamlit App
#
# This script:
# - Detects OS (Ubuntu/Debian)
# - Installs Caddy from official repository
# - Deploys Caddyfile configuration
# - Enables and starts Caddy systemd service
# - Verifies ports 80 and 443 are accessible
#
# Usage:
# 1. Set DOMAIN in .env file: DOMAIN=rag.yourdomain.com
# 2. Run on your server: sudo ./config/caddy-setup.sh
#

set -e  # Exit on error

echo "=========================================="
echo "Caddy Setup for RAG Streamlit App"
echo "=========================================="
echo ""

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   echo "Error: This script must be run as root (use sudo)"
   exit 1
fi

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
else
    echo "Warning: .env file not found in current directory"
fi

# Check DOMAIN variable
if [ -z "$DOMAIN" ]; then
    echo "Error: DOMAIN environment variable not set"
    echo ""
    echo "Please set your domain in .env file:"
    echo "  DOMAIN=rag.yourdomain.com"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo "Domain: $DOMAIN"
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    OS_VERSION=$VERSION_ID
else
    echo "Error: Cannot detect OS. /etc/os-release not found."
    exit 1
fi

echo "Detected OS: $OS $OS_VERSION"

# Install Caddy based on OS
if [[ "$OS" == "ubuntu" ]] || [[ "$OS" == "debian" ]]; then
    echo ""
    echo "Installing Caddy from official repository..."

    # Install dependencies
    apt install -y debian-keyring debian-archive-keyring apt-transport-https curl

    # Add Caddy GPG key
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg

    # Add Caddy repository
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list

    # Update and install Caddy
    apt update
    apt install -y caddy

    echo "Caddy installed successfully!"
else
    echo "Error: Unsupported OS: $OS"
    echo "This script supports Ubuntu and Debian only."
    echo "For other distributions, install Caddy manually: https://caddyserver.com/docs/install"
    exit 1
fi

# Verify Caddy is installed
if ! command -v caddy &> /dev/null; then
    echo "Error: Caddy installation failed"
    exit 1
fi

CADDY_VERSION=$(caddy version | head -1)
echo "Installed: $CADDY_VERSION"
echo ""

# Deploy Caddyfile
echo "Deploying Caddyfile configuration..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CADDYFILE="$SCRIPT_DIR/Caddyfile"

if [ ! -f "$CADDYFILE" ]; then
    echo "Error: Caddyfile not found at $CADDYFILE"
    exit 1
fi

# Backup existing Caddyfile if it exists
if [ -f /etc/caddy/Caddyfile ]; then
    echo "Backing up existing Caddyfile to /etc/caddy/Caddyfile.backup"
    cp /etc/caddy/Caddyfile /etc/caddy/Caddyfile.backup
fi

# Copy new Caddyfile
cp "$CADDYFILE" /etc/caddy/Caddyfile
echo "Caddyfile deployed to /etc/caddy/Caddyfile"
echo ""

# Validate Caddyfile syntax
echo "Validating Caddyfile syntax..."
if caddy validate --config /etc/caddy/Caddyfile; then
    echo "Caddyfile syntax is valid!"
else
    echo "Error: Caddyfile syntax validation failed"
    exit 1
fi
echo ""

# Check ports 80 and 443
echo "Checking required ports..."
if ss -tuln | grep -q ":80 "; then
    echo "Warning: Port 80 is already in use"
    ss -tuln | grep ":80 "
    echo "You may need to stop the conflicting service."
fi

if ss -tuln | grep -q ":443 "; then
    echo "Warning: Port 443 is already in use"
    ss -tuln | grep ":443 "
    echo "You may need to stop the conflicting service."
fi
echo ""

# Enable and start Caddy
echo "Enabling Caddy service..."
systemctl enable caddy

echo "Starting Caddy service..."
systemctl restart caddy

# Wait a moment for Caddy to start
sleep 2

# Check Caddy status
if systemctl is-active --quiet caddy; then
    echo ""
    echo "=========================================="
    echo "SUCCESS! Caddy is running"
    echo "=========================================="
    echo ""
    echo "Caddy will automatically obtain a Let's Encrypt certificate for $DOMAIN"
    echo "This process takes 30-60 seconds on first run."
    echo ""
    echo "To check Caddy status:"
    echo "  sudo systemctl status caddy"
    echo ""
    echo "To view Caddy logs:"
    echo "  sudo journalctl -u caddy -f"
    echo ""
    echo "To view access logs:"
    echo "  sudo tail -f /var/log/caddy/access.log"
    echo ""
    echo "IMPORTANT: Ensure your DNS A record points to this server's IP"
    echo "IMPORTANT: Ensure ports 80 and 443 are open in your firewall"
    echo ""
    echo "Test HTTPS access:"
    echo "  curl -I https://$DOMAIN"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "ERROR: Caddy failed to start"
    echo "=========================================="
    echo ""
    echo "Check logs for details:"
    echo "  sudo journalctl -u caddy -n 50"
    echo ""
    exit 1
fi
