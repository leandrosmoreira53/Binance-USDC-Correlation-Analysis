#!/bin/bash
# VPS Deployment Script for Binance API Project

echo "🚀 Deploying Binance API Project to VPS..."

# Update system packages
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python and required system packages
echo "🐍 Installing Python and dependencies..."
sudo apt install -y python3.12 python3.12-venv python3-pip git curl

# Install additional packages for better performance
sudo apt install -y htop iotop nload

# Clone or copy project files
echo "📁 Setting up project directory..."
mkdir -p ~/binance-api
cd ~/binance-api

# Create virtual environment
echo "🔧 Creating virtual environment..."
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Set up VPS-optimized environment
echo "⚡ Applying VPS optimizations..."
python vps_config.py

# Create systemd service for auto-start
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/binance-api.service > /dev/null <<EOF
[Unit]
Description=Binance USDC Correlation Analysis App
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/binance-api
Environment=PATH=$HOME/binance-api/.venv/bin
ExecStart=$HOME/binance-api/.venv/bin/python app.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable binance-api.service

echo "✅ Deployment complete!"
echo ""
echo "🌐 Your app will be available at: http://YOUR_VPS_IP:8050"
echo "📊 To start the service: sudo systemctl start binance-api"
echo "📈 To check status: sudo systemctl status binance-api"
echo "📝 To view logs: sudo journalctl -u binance-api -f"
echo ""
echo "🔧 VPS Optimizations Applied:"
echo "   - 100 concurrent API requests (vs 10 local)"
echo "   - 500 symbols analyzed (vs 120 local)"
echo "   - 90 days of data (vs 30 local)"
echo "   - Auto-restart on failure"
echo "   - System monitoring tools installed"

