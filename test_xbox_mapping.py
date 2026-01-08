#!/usr/bin/env python3
"""Test script to determine correct Xbox controller axis mappings"""

import pygame
import time

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("ERROR: No controller detected!")
    exit(1)

j = pygame.joystick.Joystick(0)
j.init()

print(f"Controller: {j.get_name()}")
print(f"Buttons: {j.get_numbuttons()}, Axes: {j.get_numaxes()}")
print("\n" + "="*70)
print("AXIS MAPPING TEST")
print("="*70)
print("\nMove each stick and observe which axis changes:\n")
print("LEFT STICK:")
print("  - Move LEFT/RIGHT -> This is axis for STRAFE (linear Y)")
print("  - Move UP/DOWN -> This is axis for FORWARD/BACK (linear X)")
print("\nRIGHT STICK:")
print("  - Move LEFT/RIGHT -> This is axis for ROTATION (angular yaw)")
print("\nLEFT/RIGHT TRIGGERS:")
print("  - Pull triggers -> These should be different axes (usually 2 and 5)")
print("\nPress Ctrl+C to exit\n")

print(f"{'Axis 0':>10} | {'Axis 1':>10} | {'Axis 2':>10} | {'Axis 3':>10} | {'Axis 4':>10} | {'Axis 5':>10} | Buttons")
print("-" * 90)

try:
    while True:
        pygame.event.pump()
        
        axes = [j.get_axis(i) for i in range(6)]
        buttons = [j.get_button(i) for i in range(12)]
        
        # Format axes with 2 decimals
        axis_str = " | ".join([f"{ax:>10.2f}" for ax in axes])
        
        # Show which buttons are pressed
        pressed = [str(i) for i, b in enumerate(buttons) if b]
        button_str = f"Pressed: {','.join(pressed)}" if pressed else ""
        
        print(f"{axis_str} | {button_str}", end='\r')
        time.sleep(0.1)
        
except KeyboardInterrupt:
    print("\n\nDone!")
    pygame.quit()
