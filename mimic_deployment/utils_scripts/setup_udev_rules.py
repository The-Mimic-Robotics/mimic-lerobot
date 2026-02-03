#!/usr/bin/env python3
"""
Interactive UDEV Rules Setup for Mimic Robot
Sets up persistent device paths for cameras and robotic arms using vendor IDs.
Connect devices one by one and generate udev rules.
"""

import subprocess
import re
import time

def get_video_devices():
    """Get list of current video devices."""
    try:
        result = subprocess.run('ls /dev/video* 2>/dev/null', capture_output=True, text=True, shell=True)
        if result.returncode != 0 or not result.stdout.strip():
            return []
        devices = result.stdout.strip().split()
        # Only return capture devices (even numbered)
        return [d for d in devices if d and re.search(r'video[0-9]*[02468]$', d)]
    except Exception as e:
        print(f"Error getting video devices: {e}")
        return []

def get_serial_devices():
    """Get list of current serial devices (ttyUSB, ttyACM)."""
    try:
        devices = []
        for pattern in ['/dev/ttyUSB*', '/dev/ttyACM*']:
            result = subprocess.run(f'ls {pattern} 2>/dev/null', capture_output=True, text=True, shell=True)
            if result.returncode == 0 and result.stdout.strip():
                devices.extend(result.stdout.strip().split())
        return sorted(devices)
    except Exception as e:
        print(f"Error getting serial devices: {e}")
        return []

def get_device_details(device, device_type):
    """Get detailed info about a device (camera or serial)."""
    vendor_id = model_id = serial = name = None
    
    if device_type == 'camera':
        # Get camera name
        cmd = f"v4l2-ctl --device={device} --info 2>/dev/null | grep 'Card type' | cut -d: -f2"
        name_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        name = name_result.stdout.strip() if name_result.returncode == 0 else "Unknown Camera"
        
        # Get udev info
        cmd = f"udevadm info --query=all {device} | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT'"
        udev_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:  # serial device
        name = "Serial Device"
        # Get udev info for serial devices
        cmd = f"udevadm info --query=all {device} | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT|ID_SERIAL'"
        udev_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse udev info
    for line in udev_result.stdout.split('\n'):
        if 'ID_VENDOR_ID=' in line:
            vendor_id = line.split('=')[1].strip()
        elif 'ID_MODEL_ID=' in line:
            model_id = line.split('=')[1].strip()
        elif 'ID_SERIAL_SHORT=' in line:
            serial = line.split('=')[1].strip()
        elif 'ID_SERIAL=' in line and not serial:
            # Fallback to full serial if short not available
            serial = line.split('=')[1].strip()
    
    return {
        'device': device,
        'name': name,
        'vendor_id': vendor_id,
        'model_id': model_id,
        'serial': serial,
        'type': device_type
    }

def wait_for_new_device(initial_devices, device_type):
    """Wait for a new device to be connected."""
    print("\nWaiting for device to be connected...")
    print(f"   Initial devices: {initial_devices}")
    print("   (Press Ctrl+C to skip this device)")
    
    check_count = 0
    while True:
        time.sleep(1)
        if device_type == 'camera':
            current_devices = get_video_devices()
        else:
            current_devices = get_serial_devices()
        
        new_devices = [d for d in current_devices if d not in initial_devices]
        
        check_count += 1
        if check_count % 5 == 0:
            print(f"   Still waiting... (checked {check_count} times)")
            print(f"   Current devices: {current_devices}")
        
        if new_devices:
            print(f"\nDetected new device: {new_devices[0]}")
            return new_devices[0]

def setup_device_interactive(device_name, device_type, initial_devices):
    """Interactive setup for one device."""
    print("\n" + "=" * 60)
    print(f"  Setting up: {device_name}")
    print("=" * 60)
    print(f"\nPlease connect the {device_name} now...")
    input("Press ENTER after connecting the device...")
    
    new_device = wait_for_new_device(initial_devices, device_type)
    print(f"Detected new device at {new_device}")
    time.sleep(1)
    
    details = get_device_details(new_device, device_type)
    print(f"\nDevice Information:")
    print(f"    Device: {details['device']}")
    print(f"    Name: {details['name']}")
    print(f"    Vendor ID: {details['vendor_id']}")
    print(f"    Model ID: {details['model_id']}")
    print(f"    Serial: {details['serial'] or 'N/A'}")
    
    return details

def generate_udev_rules(devices, output_file='/etc/udev/rules.d/99-mimic-devices.rules'):
    """Generate complete udev rules file for cameras and serial devices."""
    rules = ["# Mimic Robot Device Rules\n", "# Auto-generated\n\n"]
    
    for dev in devices:
        symlink = dev['symlink']
        vendor_id = dev['vendor_id']
        model_id = dev['model_id']
        serial = dev['serial']
        device_type = dev['type']
        
        if device_type == 'camera':
            subsystem = 'video4linux'
            attr_index = 'ATTR{index}=="0"'
        else:  # serial device
            subsystem = 'tty'
            attr_index = ''
        
        if serial:
            if attr_index:
                rule = f'SUBSYSTEM=="{subsystem}", ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", ENV{{ID_SERIAL_SHORT}}=="{serial}", {attr_index}, SYMLINK+="{symlink}", MODE="0666"'
            else:
                rule = f'SUBSYSTEM=="{subsystem}", ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", ENV{{ID_SERIAL_SHORT}}=="{serial}", SYMLINK+="{symlink}", MODE="0666"'
        else:
            if attr_index:
                rule = f'SUBSYSTEM=="{subsystem}", ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", {attr_index}, SYMLINK+="{symlink}", MODE="0666"'
            else:
                rule = f'SUBSYSTEM=="{subsystem}", ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", SYMLINK+="{symlink}", MODE="0666"'
        rules.append(rule + '\n')
    
    temp_file = '/tmp/mimic_device_rules.txt'
    with open(temp_file, 'w') as f:
        f.writelines(rules)
    
    print(f"\nGenerated udev rules:\n")
    with open(temp_file, 'r') as f:
        print(f.read())
    
    response = input("\nInstall these rules now? (y/n): ").strip().lower()
    if response == 'y':
        subprocess.run(f'sudo cp {temp_file} {output_file}', shell=True)
        subprocess.run('sudo udevadm control --reload-rules', shell=True)
        subprocess.run('sudo udevadm trigger', shell=True)
        print(f"\nRules installed! Run: ls -la /dev/arm_* /dev/camera_* /dev/mecanum_base")
    else:
        print(f"\nRules saved to {temp_file}")

def main():
    print("=" * 70)
    print("  Interactive UDEV Rules Setup for Mimic Robot")
    print("=" * 70)
    print("\nIMPORTANT: Disconnect ALL devices (cameras and arms) before starting!")
    input("\nPress ENTER when all devices are disconnected...")
    
    initial_video_devices = get_video_devices()
    initial_serial_devices = get_serial_devices()
    
    if initial_video_devices or initial_serial_devices:
        print(f"\nWarning: Found {len(initial_video_devices)} camera(s) and {len(initial_serial_devices)} serial device(s)")
        print("   Please disconnect them and restart this script.")
        return
    
    print("\nNo devices detected. Ready to begin!")
    
    all_devices = []
    
    # Serial devices (arms and base)
    print("\n" + "=" * 70)
    print("  SERIAL DEVICES (Arms and Base)")
    print("=" * 70)
    
    serial_device_list = [
        ('arm_left_leader', 'SO-100 Left Leader Arm', 'serial'),
        ('arm_right_leader', 'SO-100 Right Leader Arm', 'serial'),
        ('arm_left_follower', 'SO-100 Left Follower Arm', 'serial'),
        ('arm_right_follower', 'SO-100 Right Follower Arm', 'serial'),
        ('mecanum_base', 'Mecanum Base Controller (ESP32)', 'serial'),
    ]
    
    for symlink, description, device_type in serial_device_list:
        response = input(f"\nSetup {description}? (y/n/quit): ").strip().lower()
        if response in ('quit', 'q'):
            break
        if response != 'y':
            continue
        
        dev_details = setup_device_interactive(description, device_type, initial_serial_devices)
        dev_details['symlink'] = symlink
        all_devices.append(dev_details)
        initial_serial_devices = get_serial_devices()
        print(f"\n{description} configured as /dev/{symlink}")
    
    # Cameras
    print("\n" + "=" * 70)
    print("  CAMERAS")
    print("=" * 70)
    
    camera_list = [
        ('camera_left_wrist', 'Left Wrist Camera', 'camera'),
        ('camera_right_wrist', 'Right Wrist Camera', 'camera'),
        ('camera_front', 'Front Camera', 'camera'),
        ('camera_top', 'Top Camera (ZED)', 'camera'),
    ]
    
    for symlink, description, device_type in camera_list:
        response = input(f"\nSetup {description}? (y/n/quit): ").strip().lower()
        if response in ('quit', 'q'):
            break
        if response != 'y':
            continue
        
        dev_details = setup_device_interactive(description, device_type, initial_video_devices)
        dev_details['symlink'] = symlink
        all_devices.append(dev_details)
        initial_video_devices = get_video_devices()
        print(f"\n{description} configured as /dev/{symlink}")
    
    if not all_devices:
        print("\nNo devices were configured!")
        return
    
    print("\n" + "=" * 70)
    print(f"  Generating udev rules for {len(all_devices)} device(s)...")
    print("=" * 70)
    generate_udev_rules(all_devices)

if __name__ == "__main__":
    main()
