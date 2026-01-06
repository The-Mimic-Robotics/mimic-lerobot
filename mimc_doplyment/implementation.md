1. created the Mecanum base class
src/lerobot/motors/mecanum_base/mecanum.py
2. Test file for the class
python test_base.py
3. create the follower
-> based on src/lerobot/robots/bi_so100_follower

src/lerobot/robots/mimic_follower
important -> using the encoders to get the observation
need to verify the encoder accurancy to make suere viable
pi0 using encoders as secondary inputss
4. added it to all the scripts and the utils
5. Create leader arm 
same as before, added the choice of controller
keyboard for simple
TODO implement the xbox controller then joystick

5. testing 2 camera on the usb -> good 
6. add Zed camera support
only using left eye . no filtering for better accurary
classic pi0 setup
