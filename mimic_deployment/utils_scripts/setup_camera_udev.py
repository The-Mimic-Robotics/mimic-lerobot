#!/usr/bin/env python3
"""
Interactive Camera Setup - One at a Time
Connect cameras one by one and generate udev rules.
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

def get_camera_details(device):
    """Get detailed info about a camera."""
    cmd = f"v4l2-ctl --device={device} --info 2>/dev/null | grep 'Card type' | cut -d: -f2"
    name_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    camera_name = name_result.stdout.strip() if name_result.returncode == 0 else "Unknown"
    
    cmd = f"udevadm info --query=all {device} | grep -E 'ID_VENDOR_ID|ID_MODEL_ID|ID_SERIAL_SHORT'"
    udev_result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    vendor_id = model_id = serial = None
    for line in udev_result.stdout.split('\n'):
        if 'ID_VENDOR_ID=' in line:
            vendor_id = line.split('=')[1].strip()
        elif 'ID_MODEL_ID=' in line:
            model_id = line.split('=')[1].strip()
        elif 'ID_SERIAL_SHORT=' in line:
            serial = line.split('=')[1].strip()
    
    return {
        'device': device,
        'name': camera_name,
        'vendor_id': vendor_id,
        'model_id': model_id,
        'serial': serial
    }

def wait_for_new_camera(initial_devices):
    """Wait for a new camera to be connected."""
    print("\n⏳ Waiting for camera to be connected...")
    print(f"   Initial devices: {initial_devices}")
    print("   (Press Ctrl+C to skip this camera)")
    
    check_count = 0
    while True:
        time.sleep(1)
        current_devices = get_video_devices()
        new_devices = [d for d in current_devices if d not in initial_devices]
        
        check_count += 1
        if check_count % 5 == 0:
            print(f"   Still waiting... (checked {check_count} times)")
            print(f"   Current devices: {current_devices}")
        
        if new_devices:
            print(f"\n✅ Detected new camera: {new_devices[0]}")
            return new_devices[0]

def setup_camera_interactive(camera_name, initial_devices):
    """Interactive setup for one camera."""
    print("\n" + "=" * 60)
    print(f"  Setting up: {camera_name}")
    print("=" * 60)
    print(f"\n📌 Please connect the {camera_name} now...")
    
    new_device = wait_for_new_camera(initial_devices)
    print(f"✅ Detected new camera at {new_device}")
    time.sleep(1)
    
    details = get_camera_details(new_device)
    print(f"\n📷 Camera Information:")
    print(f"    Device: {details['device']}")
    print(f"    Name: {details['name']}")
    print(f"    Vendor ID: {details['vendor_id']}")
    print(f"    Model ID: {details['model_id']}")
    print(f"    Serial: {details['serial'] or 'N/A'}")
    
    return details

def generate_udev_rule(cameras, output_file='/etc/udev/rules.d/99-mimic-cameras.rules'):
    """Generate complete udev rules file."""
    rules = ["# Mimic Robot Camera Rules\n", "# Auto-generated\n\n"]
    
    for cam in cameras:
        symlink = cam['symlink']
        vendor_id = cam['vendor_id']
        model_id = cam['model_id']
        serial = cam['serial']
        
        if serial:
            rule = f'SUBSYSTEM=="video4linux", ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", ENV{{ID_SERIAL_SHORT}}=="{serial}", ATTR{{index}}=="0", SYMLINK+="{symlink}", MODE="0666"'
        else:
            rule = f'SUBSYSTEM=="video4linux", ATTRS{{idVendor}}=="{vendor_id}", ATTRS{{idProduct}}=="{model_id}", ATTR{{index}}=="0", SYMLINK+="{symlink}", MODE="0666"'
        rules.append(rule + '\n')
    
    temp_file = '/tmp/mimic_camera_rules.txt'
    with open(temp_file, 'w') as f:
        f.writelines(rules)
    
    print(f"\n✅ Generated udev rules:\n")
    with open(temp_file, 'r') as f:
        print(f.read())
    
    response = input("\n❓ Install these rules now? (y/n): ").strip().lower()
    if response == 'y':
        subprocess.run(f'sudo cp {temp_file} {output_file}', shell=True)
        subprocess.run('sudo udevadm control --reload-rules', shell=True)
        subprocess.run('sudo udevadm trigger', shell=True)
        print(f"\n✅ Rules installed! Run: ls -la /dev/camera_*")
    else:
        print(f"\n📝 Rules saved to {temp_file}")

def main():
    print("=" * 70)
    print("  🎥 Interactive Camera Setup - One at a Time")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Disconnect ALL cameras before starting!")
    input("\nPress ENTER when all cameras are disconnected...")
    
    initial_devices = get_video_devices()
    if initial_devices:
        print(f"\n⚠️  Warning: Found {len(initial_devices)} existing camera(s)")
        print("   Please disconnect them and restart this script.")
        return
    
    print("\n✅ No cameras detected. Ready to begin!")
    
    cameras = []
    camera_list = [
        ('camera_left_wrist', 'Left Wrist Camera'),
        ('camera_right_wrist', 'Right Wrist Camera'),
        ('camera_front', 'Front Camera'),
        ('camera_top', 'Top Camera (ZED)'),
    ]
    
    for symlink, description in camera_list:
        response = input(f"\n❓ Setup {description}? (y/n/quit): ").strip().lower()
        if response in ('quit', 'q'):
            break
        if response != 'y':
            continue
        
        cam_details = setup_camera_interactive(description, initial_devices)
        cam_details['symlink'] = symlink
        cameras.append(cam_details)
        initial_devices = get_video_devices()
        print(f"\n✅ {description} configured as /dev/{symlink}")
    
    if not cameras:
        print("\n❌ No cameras were configured!")
        return
    
    print("\n" + "=" * 70)
    print(f"  📝 Generating udev rules for {len(cameras)} camera(s)...")
    print("=" * 70)
    generate_udev_rule(cameras)

if __name__ == "__main__":
    main()
