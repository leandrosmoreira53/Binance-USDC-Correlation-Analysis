#!/bin/bash
# Script para atualizar a aplicação no VPS

echo "🚀 Atualizando Binance API no VPS..."

# Parar a aplicação
echo "⏹️ Parando aplicação..."
sudo systemctl stop binance-api

# Navegar para o diretório
cd /opt/binance-api

# Fazer backup do cache atual
echo "💾 Fazendo backup do cache..."
cp -r data/ data_backup_$(date +%Y%m%d_%H%M%S)/

# Fazer pull das mudanças
echo "📥 Baixando atualizações..."
git pull origin main

# Atualizar dependências
echo "📦 Atualizando dependências..."
source .venv/bin/activate
pip install -r requirements.txt

# Aplicar configurações VPS
echo "⚡ Aplicando configurações VPS..."
python vps_config.py

# Reiniciar a aplicação
echo "🔄 Reiniciando aplicação..."
sudo systemctl start binance-api

# Verificar status
echo "✅ Verificando status..."
sudo systemctl status binance-api --no-pager

echo "🎉 Atualização concluída!"
echo "🌐 Aplicação disponível em: http://31.97.165.64:8050"
echo "📊 Para ver logs: sudo journalctl -u binance-api -f"
