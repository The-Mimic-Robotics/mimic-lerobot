from lerobot.motors.mecanum_base import MecanumBase
import time

def test_drive():
    # ADJUST YOUR PORT HERE
    base = MecanumBase(port="/dev/ttyUSB0") 
    
    print("Connecting...")
    base.connect()
    
    try:
        print("Driving Forward...")
        for _ in range(50): # Run for ~1 second (50 * 0.02s)
            base.send_twist(0.2, 0.0, 0.0) # 0.2 m/s forward
            
            odom, vel = base.read_odom()
            print(f"Odom: {odom} | Vel: {vel}")
            
            time.sleep(0.02) # 50Hz loop
            
        print("Stopping...")
        base.send_twist(0.0, 0.0, 0.0)
        time.sleep(0.5)
        
    except KeyboardInterrupt:
        print("Interrupted!")
        
    finally:
        base.disconnect()

if __name__ == "__main__":
    test_drive()