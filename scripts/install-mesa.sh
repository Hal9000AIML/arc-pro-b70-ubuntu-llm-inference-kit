#!/usr/bin/env bash
# Install Mesa 26.0.5+ from kisak/kisak-mesa PPA.
# Enables VK_KHR_shader_bfloat16 + VK_KHR_shader_integer_dot_product on BMG (Arc Pro B70).
# Without this, the Vulkan backend runs without coopmat / bf16 paths and leaves 20-40% perf on the floor.
set -euo pipefail

if ! command -v add-apt-repository >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y software-properties-common
fi

sudo add-apt-repository -y ppa:kisak/kisak-mesa
sudo apt-get update
sudo apt-get install -y mesa-vulkan-drivers libvulkan1 vulkan-tools

echo
echo "Installed Mesa:"
apt-cache policy mesa-vulkan-drivers | grep -E 'Installed|Candidate'
echo
echo "Verifying BMG extensions (look for VK_KHR_shader_bfloat16):"
vulkaninfo --summary 2>/dev/null | grep -iE 'deviceName|driverName' || true
vulkaninfo 2>/dev/null | grep -cE 'VK_KHR_shader_bfloat16|VK_KHR_shader_integer_dot_product' \
  || echo "WARN: extensions not found; reboot may be required for new Mesa to take effect."
