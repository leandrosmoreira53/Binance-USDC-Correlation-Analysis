# Binance USDC Correlation Analysis - VPS Deployment Guide

## 🚀 Performance Improvements on VPS

### Expected Speed Improvements:
- **10x faster data collection** (100 concurrent requests vs 10 local)
- **4x more symbols** (500 vs 120)
- **3x more historical data** (90 days vs 30)
- **Better network latency** to Binance API servers
- **Dedicated resources** (CPU, RAM, bandwidth)

### Current Performance Issues (Local):
- Slow API requests due to network latency
- Limited concurrent connections
- Resource sharing with other applications
- Potential ISP throttling

## 📋 VPS Requirements

### Minimum Specifications:
- **CPU**: 2+ cores
- **RAM**: 4GB+ 
- **Storage**: 20GB+ SSD
- **Bandwidth**: 1TB+ monthly
- **OS**: Ubuntu 20.04+ or Debian 11+

### Recommended Specifications:
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 50GB+ SSD
- **Bandwidth**: Unlimited
- **Location**: Close to Binance servers (Singapore, Tokyo, Frankfurt)

## 🔧 Quick Deployment

### Option 1: Automated Deployment
```bash
# Upload project files to VPS
scp -r /path/to/api_binance user@your-vps-ip:~/

# SSH into VPS
ssh user@your-vps-ip

# Run deployment script
cd ~/api_binance
./deploy_vps.sh
```

### Option 2: Manual Deployment
```bash
# 1. Install dependencies
sudo apt update && sudo apt install -y python3.12 python3.12-venv

# 2. Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install Python packages
pip install -r requirements.txt

# 4. Apply VPS optimizations
python vps_config.py

# 5. Run application
python app.py
```

## ⚡ VPS Optimizations Applied

### Environment Variables:
```bash
CONCURRENCY=100        # 10x more concurrent requests
SYMBOL_CAP=500         # 4x more symbols
DAYS=90               # 3x more historical data
VOL_THRESHOLD=100000  # Focus on liquid pairs
TIMEFRAME=4h          # More data points
```

### System Optimizations:
- **Auto-restart** on failure
- **System monitoring** tools (htop, iotop, nload)
- **Service management** with systemd
- **Log rotation** and monitoring

## 📊 Performance Monitoring

### Check Application Status:
```bash
sudo systemctl status binance-api
```

### View Real-time Logs:
```bash
sudo journalctl -u binance-api -f
```

### Monitor System Resources:
```bash
htop          # CPU and memory usage
iotop         # Disk I/O
nload         # Network usage
```

## 🌐 Accessing Your Application

Once deployed, your application will be available at:
```
http://YOUR_VPS_IP:8050
```

### Security Considerations:
- Configure firewall to allow port 8050
- Consider using reverse proxy (nginx)
- Enable HTTPS with Let's Encrypt
- Regular security updates

## 🔄 Data Refresh Strategy

### Automatic Refresh:
The application caches data and correlations to avoid repeated API calls. Data is refreshed when:
- Cache files are deleted
- "Refresh Data" button is clicked
- Application restarts (if cache is older than 24h)

### Manual Refresh:
```bash
# Stop service
sudo systemctl stop binance-api

# Clear cache
rm -rf data/*.parquet

# Restart service
sudo systemctl start binance-api
```

## 📈 Expected Performance Metrics

### Data Collection Time:
- **Local**: 5-10 minutes for 120 symbols
- **VPS**: 1-2 minutes for 500 symbols

### Memory Usage:
- **Local**: 2-4GB during data collection
- **VPS**: 4-8GB with optimizations

### API Rate Limits:
- Binance allows 1200 requests/minute
- VPS configuration uses ~100 concurrent requests
- Well within rate limits with proper delays

## 🛠️ Troubleshooting

### Common Issues:
1. **Port 8050 blocked**: Configure firewall
2. **Memory issues**: Increase VPS RAM or reduce SYMBOL_CAP
3. **API rate limits**: Reduce CONCURRENCY
4. **Slow performance**: Check network latency to Binance

### Performance Tuning:
```bash
# Reduce symbols if memory constrained
export SYMBOL_CAP=250

# Reduce concurrency if hitting rate limits
export CONCURRENCY=50

# Reduce historical data if storage limited
export DAYS=30
```

## 💡 Additional Optimizations

### For Even Better Performance:
1. **Use dedicated Binance API keys** (if available)
2. **Implement Redis caching** for frequently accessed data
3. **Use CDN** for static assets
4. **Database optimization** for large datasets
5. **Load balancing** for multiple instances

### Cost Optimization:
- Use spot instances for non-critical workloads
- Implement auto-scaling based on demand
- Monitor resource usage and optimize accordingly
