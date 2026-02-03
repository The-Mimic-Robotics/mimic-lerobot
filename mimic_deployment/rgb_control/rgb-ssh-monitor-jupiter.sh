#!/bin/bash
# RGB LED SSH Monitor for Jupiter (ASRock B650M Pro RS WiFi)
# Controls ASRock LED Controller (USB 26ce:01a2)

# Configuration
CHECK_INTERVAL=5
LOG_FILE="/tmp/rgb-ssh-monitor-jupiter.log"
STATE_FILE="/tmp/rgb_ssh_state"
ASROCK_VENDOR="26ce"
ASROCK_PRODUCT="01a2"

# Color modes
MODE_SSH="red"
MODE_DEFAULT="rainbow"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Check if SSH sessions are active
check_ssh_active() {
    # Check for established SSH connections on port 22
    # Count lines but skip the header line from ss output
    local ssh_connections=$(ss -tn state established '( dport = :22 or sport = :22 )' 2>/dev/null | tail -n +2 | wc -l)
    
    # If there are any SSH connections, return success
    if [ "$ssh_connections" -gt 0 ]; then
        return 0
    else
        return 1
    fi
}

# Find ASRock LED Controller USB device
find_asrock_device() {
    local bus_dev=$(lsusb -d "${ASROCK_VENDOR}:${ASROCK_PRODUCT}" | grep -oP 'Bus \K\d+|Device \K\d+' | paste -sd ' ')
    
    if [ -z "$bus_dev" ]; then
        return 1
    fi
    
    read -r bus device <<< "$bus_dev"
    printf "/dev/bus/usb/%03d/%03d" "$bus" "$device"
    return 0
}

# Control ASRock RGB via OpenRGB
control_via_openrgb() {
    local mode="$1"
    
    if ! command -v openrgb &> /dev/null; then
        return 1
    fi
    
    case "$mode" in
        red)
            openrgb -d 0 -m static -c FF0000 2>/dev/null
            ;;
        rainbow)
            openrgb -d 0 -m rainbow 2>/dev/null || openrgb -d 0 -m spectrum-cycle 2>/dev/null
            ;;
        *)
            return 1
            ;;
    esac
    
    return $?
}

# Control ASRock RGB via Python script (fallback)
control_via_python() {
    local mode="$1"
    
    local python_script="/tmp/asrock_rgb_control.py"
    
    cat > "$python_script" << 'PYTHON_EOF'
import sys
import usb.core
import usb.util

def set_asrock_rgb(mode):
    # Find ASRock LED Controller
    dev = usb.core.find(idVendor=0x26ce, idProduct=0x01a2)
    
    if dev is None:
        print("ASRock LED Controller not found", file=sys.stderr)
        return False
    
    try:
        # Detach kernel driver if necessary
        if dev.is_kernel_driver_active(0):
            dev.detach_kernel_driver(0)
        
        # Set configuration
        dev.set_configuration()
        
        # RGB control packets for ASRock motherboards
        # These are device-specific and may need adjustment
        if mode == "red":
            # Static red color
            packet = [0x00, 0x41, 0x01, 0xFF, 0x00, 0x00, 0x00, 0x00]
        elif mode == "rainbow":
            # Rainbow/spectrum cycle mode
            packet = [0x00, 0x41, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00]
        else:
            return False
        
        # Send the control packet
        dev.ctrl_transfer(0x21, 0x09, 0x0300, 0x00, packet)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rainbow"
    success = set_asrock_rgb(mode)
    sys.exit(0 if success else 1)
PYTHON_EOF
    
    python3 "$python_script" "$mode" 2>/dev/null
    return $?
}

# Control ASRock RGB via liquidctl (some ASRock boards are supported)
control_via_liquidctl() {
    local mode="$1"
    
    if ! command -v liquidctl &> /dev/null; then
        return 1
    fi
    
    # Initialize
    liquidctl initialize all 2>/dev/null
    
    case "$mode" in
        red)
            liquidctl --match "ASRock" set led color fixed ff0000 2>/dev/null
            ;;
        rainbow)
            liquidctl --match "ASRock" set led color fading 350017 ff2608 2>/dev/null
            ;;
    esac
    
    return $?
}

# Direct HID control for ASRock (most reliable method)
control_via_hidapi() {
    local mode="$1"
    
    local device_path=$(find_asrock_device)
    if [ -z "$device_path" ]; then
        log_message "ASRock LED Controller not found"
        return 1
    fi
    
    # Install hidapi if not present
    if ! python3 -c "import hid" 2>/dev/null; then
        log_message "Installing python hidapi..."
        sudo apt install -y python3-hid >/dev/null 2>&1
    fi
    
    python3 - "$mode" << 'PYTHON_HID_EOF'
import sys
try:
    import hid
except ImportError:
    sys.exit(1)

def control_asrock(mode):
    # Open ASRock LED Controller
    h = hid.device()
    try:
        h.open(0x26ce, 0x01a2)
        h.set_nonblocking(1)
        
        if mode == "red":
            # Static red - ASRock Polychrome protocol
            # Format: [report_id, command, mode, R, G, B, speed, brightness, ...]
            data = [0x00, 0x41, 0x01, 0xFF, 0x00, 0x00, 0x00, 0xFF]
        elif mode == "rainbow":
            # Rainbow/spectrum cycle mode
            data = [0x00, 0x41, 0x02, 0x00, 0x00, 0x00, 0x20, 0xFF]
        else:
            return False
        
        # Pad to 64 bytes (HID report size)
        data.extend([0x00] * (64 - len(data)))
        
        # Send the report
        h.write(data)
        
        h.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "rainbow"
    success = control_asrock(mode)
    sys.exit(0 if success else 1)
PYTHON_HID_EOF
    
    return $?
}

# Main RGB control function
set_rgb_mode() {
    local mode="$1"
    local success=0
    
    log_message "Setting RGB to: $mode"
    
    # Try OpenRGB first (most compatible)
    if control_via_openrgb "$mode"; then
        log_message "Success via OpenRGB"
        return 0
    fi
    
    # Try hidapi direct control
    if control_via_hidapi "$mode"; then
        log_message "Success via HID API"
        return 0
    fi
    
    # Try liquidctl
    if control_via_liquidctl "$mode"; then
        log_message "Success via liquidctl"
        return 0
    fi
    
    # Try Python/USB
    if control_via_python "$mode"; then
        log_message "Success via Python USB"
        return 0
    fi
    
    log_message "Failed to control RGB - no working method found"
    return 1
}

# Install OpenRGB from AppImage
install_openrgb() {
    log_message "Installing OpenRGB..."
    
    local openrgb_url="https://gitlab.com/CalcProgrammer1/OpenRGB/-/releases/permalink/latest/downloads/OpenRGB_latest_amd64.deb"
    local temp_deb="/tmp/openrgb.deb"
    
    wget -q -O "$temp_deb" "$openrgb_url" 2>/dev/null
    
    if [ -f "$temp_deb" ]; then
        sudo dpkg -i "$temp_deb"
        sudo apt-get install -f -y
        rm "$temp_deb"
        log_message "OpenRGB installed successfully"
        return 0
    else
        log_message "Failed to download OpenRGB"
        return 1
    fi
}

# Setup dependencies for Jupiter
setup_jupiter() {
    log_message "Setting up RGB control for Jupiter (ASRock B650M Pro RS WiFi)"
    
    # Check if ASRock LED Controller is present
    if ! lsusb -d 26ce:01a2 &>/dev/null; then
        echo "ERROR: ASRock LED Controller not found!"
        echo "Make sure RGB headers are connected to the motherboard."
        return 1
    fi
    
    echo "Found: ASRock LED Controller (26ce:01a2)"
    
    # Install dependencies
    echo "Installing dependencies..."
    sudo apt update
    sudo apt install -y python3-hid python3-usb libhidapi-hidraw0 usbutils
    
    # Install OpenRGB
    if ! command -v openrgb &> /dev/null; then
        echo "Installing OpenRGB..."
        install_openrgb
    fi
    
    # Setup udev rules for non-root access
    echo "Setting up udev rules..."
    cat << 'UDEV_EOF' | sudo tee /etc/udev/rules.d/60-asrock-led.rules > /dev/null
# ASRock LED Controller
SUBSYSTEM=="usb", ATTRS{idVendor}=="26ce", ATTRS{idProduct}=="01a2", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="26ce", ATTRS{idProduct}=="01a2", MODE="0666", GROUP="plugdev"
UDEV_EOF
    
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    
    # Add user to plugdev group
    sudo usermod -a -G plugdev "$USER"
    
    log_message "Setup complete for Jupiter"
    echo ""
    echo "Setup complete! You may need to log out and back in for group changes."
    echo "Test with: sudo $0 --test-red"
}

# Monitoring loop
monitor_loop() {
    local last_state="none"
    
    log_message "RGB SSH Monitor started for Jupiter"
    
    # Set initial state
    if check_ssh_active; then
        set_rgb_mode "$MODE_SSH"
        last_state="ssh"
    else
        set_rgb_mode "$MODE_DEFAULT"
        last_state="default"
    fi
    
    while true; do
        if check_ssh_active; then
            if [ "$last_state" != "ssh" ]; then
                log_message "SSH detected"
                set_rgb_mode "$MODE_SSH"
                last_state="ssh"
                echo "ssh" > "$STATE_FILE"
            fi
        else
            if [ "$last_state" != "default" ]; then
                log_message "No SSH - restoring default"
                set_rgb_mode "$MODE_DEFAULT"
                last_state="default"
                echo "default" > "$STATE_FILE"
            fi
        fi
        
        sleep "$CHECK_INTERVAL"
    done
}

# Setup systemd service
setup_service() {
    local service_file="/etc/systemd/system/rgb-ssh-monitor-jupiter.service"
    local script_path="$(realpath "$0")"
    
    cat << EOF | sudo tee "$service_file" > /dev/null
[Unit]
Description=RGB LED SSH Monitor for Jupiter
After=network.target

[Service]
Type=simple
ExecStart=$script_path --daemon
Restart=always
RestartSec=10
User=root

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable rgb-ssh-monitor-jupiter.service
    sudo systemctl start rgb-ssh-monitor-jupiter.service
    
    log_message "Systemd service installed"
    echo "Service installed! Check status: sudo systemctl status rgb-ssh-monitor-jupiter"
}

# Show usage
show_usage() {
    cat << EOF
RGB LED SSH Monitor for Jupiter (ASRock B650M Pro RS WiFi)

Usage: $0 [OPTION]

Options:
    --daemon        Run continuous monitoring
    --once          Check once and set colors
    --setup         Install dependencies for Jupiter
    --install       Install as systemd service
    --status        Show current state
    --test-red      Test red color
    --test-rainbow  Test rainbow/default color
    --device-info   Show ASRock LED Controller info
    --help          Show this help

Examples:
    sudo $0 --setup         # First time setup
    sudo $0 --test-red      # Test red color
    sudo $0 --daemon        # Run monitoring
    sudo $0 --install       # Install as service

EOF
}

# Parse command line
case "${1:-}" in
    --daemon)
        monitor_loop
        ;;
    --once)
        if check_ssh_active; then
            echo "SSH active - setting red"
            set_rgb_mode "$MODE_SSH"
        else
            echo "No SSH - setting rainbow"
            set_rgb_mode "$MODE_DEFAULT"
        fi
        ;;
    --setup)
        setup_jupiter
        ;;
    --install)
        setup_service
        ;;
    --status)
        echo "Device: ASRock B650M Pro RS WiFi"
        if lsusb -d 26ce:01a2 &>/dev/null; then
            echo "LED Controller: Connected"
        else
            echo "LED Controller: NOT FOUND"
        fi
        
        if [ -f "$STATE_FILE" ]; then
            echo "Current mode: $(cat "$STATE_FILE")"
        fi
        
        check_ssh_active && echo "SSH: ACTIVE" || echo "SSH: None"
        ;;
    --test-red)
        echo "Testing red color on Jupiter..."
        set_rgb_mode "red"
        ;;
    --test-rainbow)
        echo "Testing rainbow color on Jupiter..."
        set_rgb_mode "rainbow"
        ;;
    --device-info)
        echo "=== Jupiter RGB Device Info ==="
        lsusb -v -d 26ce:01a2 2>/dev/null | head -20
        ;;
    --help|*)
        show_usage
        ;;
esac
