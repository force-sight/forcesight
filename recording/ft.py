import numpy as np
import subprocess
import time

# when the gripper is -0.3 orientated downwards
EEF_PITCH_WEIGHT_OFFSET = np.array([-0.22010994, -0.02645493,  0.3561182, 0, 0, 0])

class FTCapture:
    def __init__(self, delay=0, ip="192.168.1.1"):
        print('Initializing force/torque sensor...')
        self.counts_per_force =  1e6
        self.counts_per_torque = 1e6
        self.first_frame_time = 0
        self.current_frame_time = 0
        self.frame_count = 0
        self.delay = delay # number of frames to delay before returning ft data to sync with camera in live model
        self.history = []
        self.ip = ip

    def get_ft(self):
        (stat, output) = subprocess.getstatusoutput(f"./netft {self.ip}")
        output_list = output.splitlines()
    
        # resetting ft data to zero
        ft_values = 6*[0]

        # parsing output strings from network call
        for i in range(6):
            if len(output_list[i + 1][4:]) > 1:
                if i <= 3:
                    ft_values[i] = int(output_list[i + 1][4:])/self.counts_per_force
                else:
                    ft_values[i] = int(output_list[i + 1][4:])/self.counts_per_torque
            else:
                ft_values[i] = 0

        self.current_frame_time = time.time()
        if self.first_frame_time == 0:
            self.first_frame_time = self.current_frame_time
        self.frame_count += 1

        if len(ft_values) != 6:
            print('Error: receiving invalid force/torque data')
            return None

        # shifting
        if self.delay > 0:
            self.history.append(ft_values)
            if len(self.history) > self.delay:
                self.history.pop(0)

            ft = self.history[0]
            ft = np.array(ft, dtype='float32')
            return ft
            
        else:
            ft = np.array(ft_values, dtype='float32')

        if np.all(np.abs(ft) < 1e-5):
            print('Error: receiving invalid force/torque data')
            exit()

        return ft

class MockFTCapture:
    """This is used to mock the FT sensor"""
    def __init__(self, delay=0):
        pass

    def get_ft(self):
        return EEF_PITCH_WEIGHT_OFFSET

if __name__ == "__main__":
    ft = FTCapture()
    start_time = time.time()

    while True:
        ft_data = ft.get_ft()
        current_time = time.time() - start_time
        print(np.round(ft_data, 4))
        print('Average FPS', ft.frame_count / (time.time() - ft.first_frame_time))
        print(ft.frame_count, ' frames captured')
