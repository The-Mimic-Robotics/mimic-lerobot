import pygame
import sys
import os

# Configuration
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
os.environ["SDL_LINUX_JOYSTICK_DEADZONES"] = "1"

WIDTH, HEIGHT = 800, 400
BACKGROUND_COLOR = (30, 30, 30)
STICK_COLOR = (50, 50, 50)
DOT_COLOR = (0, 255, 100)       # Green for active elements
BUTTON_OFF_COLOR = (60, 60, 60) # Dark gray for inactive buttons
TEXT_COLOR = (200, 200, 200)

# Layout Positions
L_STICK_CENTER = (200, 200)
R_STICK_CENTER = (600, 200)
# Places buttons in the center of the screen between the sticks
BTN_0_CENTER = (360, 200) 
BTN_1_CENTER = (440, 200)
RADIUS = 80
BTN_RADIUS = 25

def map_axis(val):
    """Maps raw axis input to screen coordinates with a deadzone."""
    if abs(val) < 0.1: return 0
    return val * RADIUS

try:
    pygame.init()
    pygame.joystick.init()
    pygame.font.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ESP32 Wireless Gamepad Visualizer")
    clock = pygame.time.Clock()

    try:
        font = pygame.font.Font(None, 24)
        # Larger font for button labels
        btn_font = pygame.font.Font(None, 30) 
    except Exception as e:
        font = pygame.font.SysFont("arial", 24)
        btn_font = pygame.font.SysFont("arial", 30)

    if pygame.joystick.get_count() == 0:
        print("No Joystick Found")
        sys.exit()
    
    joy = pygame.joystick.Joystick(0)
    joy.init()

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        try:
            # --- Read Inputs ---
            lx = joy.get_axis(2)
            ly = joy.get_axis(3)
            
            num_axes = joy.get_numaxes()
            rx = joy.get_axis(0)
            ry = joy.get_axis(1)

            # Get button states (True/False)
            b0_state = joy.get_button(1)
            b1_state = joy.get_button(0)

            # Create a list of all pressed buttons for the text display
            pressed_buttons_list = []
            for i in range(joy.get_numbuttons()):
                if joy.get_button(i):
                    pressed_buttons_list.append(str(i))
            
        except Exception as e:
            print(f"Error reading axis: {e}")
            lx, ly, rx, ry = 0, 0, 0, 0
            b0_state, b1_state = False, False
            pressed_buttons_list = []

        # --- Draw ---
        screen.fill(BACKGROUND_COLOR)
        
        # 1. Left Stick
        pygame.draw.circle(screen, STICK_COLOR, L_STICK_CENTER, RADIUS, 5)
        pygame.draw.circle(screen, DOT_COLOR, (L_STICK_CENTER[0] + map_axis(lx), L_STICK_CENTER[1] + map_axis(ly)), 15)
        
        # 2. Right Stick
        pygame.draw.circle(screen, STICK_COLOR, R_STICK_CENTER, RADIUS, 5)
        pygame.draw.circle(screen, DOT_COLOR, (R_STICK_CENTER[0] + map_axis(rx), R_STICK_CENTER[1] + map_axis(ry)), 15)

        # 3. Buttons (New Addition)
        # Helper function to draw a button
        def draw_button(pos, is_pressed, label):
            color = DOT_COLOR if is_pressed else BUTTON_OFF_COLOR
            # Draw filled circle if pressed, outline if not
            width = 0 if is_pressed else 3 
            pygame.draw.circle(screen, color, pos, BTN_RADIUS, width)
            
            # Draw Button Label (0 or 1) inside
            lbl_color = (30, 30, 30) if is_pressed else TEXT_COLOR
            lbl_surf = btn_font.render(label, True, lbl_color)
            lbl_rect = lbl_surf.get_rect(center=pos)
            screen.blit(lbl_surf, lbl_rect)

        draw_button(BTN_0_CENTER, b0_state, "0")
        draw_button(BTN_1_CENTER, b1_state, "1")

        # 4. Text Info
        try:
            btns_str = ",".join(pressed_buttons_list)
            msg = f"L:({lx:.1f},{ly:.1f}) R:({rx:.1f},{ry:.1f})"
            txt = font.render(msg, True, TEXT_COLOR)
            screen.blit(txt, (20, 20))
        except Exception as e:
            print(f"Font Render Error: {e}")

        pygame.display.flip()
        clock.tick(60)

except Exception as e:
    print(f"\n\nCrashed with Error: {e}")
    import traceback
    traceback.print_exc()

pygame.quit()