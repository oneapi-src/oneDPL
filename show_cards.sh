# https://dgpu-docs.intel.com/driver/gpu-debugging.html
for card in /sys/class/drm/card*; do
  if [[ -e "${card}/prelim_enable_eu_debug" ]]; then
    link="$(readlink "${card}")"
    link="${link#.*devices/}"
    link="${link%/drm*}"
    vendor=$(cat ${card}/device/vendor)
    vendor=${vendor/0x} # Prune 0x
    device=$(cat ${card}/device/device)
    device=${device/0x} # Prune 0x
    value=$(cat ${card}/prelim_enable_eu_debug)
    echo "${card} (${vendor}:${device}) supports" \
      "prelim_enable_eu_debug. Current value: ${value}"
  fi
done
