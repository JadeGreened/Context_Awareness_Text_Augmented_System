import threading
import time
from collections import deque
from viewer import hl2ss_lnm, hl2ss


class Eyetracking_Component:
    def __init__(self, mainSys, ip):
        self.mainSys = mainSys
        self.ip = ip
        self.fps = 30
        self.client = client = hl2ss_lnm.rx_eet(self.ip, hl2ss.StreamPort.EXTENDED_EYE_TRACKER, fps=self.fps)
        self.queue = deque(maxlen=5)


    def receive_data(self):
        while True:
            if len(self.queue) != 0:
                gaze_data = self.queue.pop()
                if gaze_data is not None:
                    result_to_be_written = self.determine_gaze_direction(gaze_data)
                    self.mainSys.read_then_write(result_to_be_written)
                    print(result_to_be_written)
            time.sleep(2)



    def determine_gaze_direction(self, gaze_data):
        if gaze_data is not None:
            vertical, horizontal, depth = gaze_data

            if vertical > 0.20:
                vertical_direction = 'down'
            elif vertical < 0.20:
                vertical_direction = 'up'
            else:
                vertical_direction = 'middle'

            if horizontal > 0:
                horizontal_direction = 'left'
            elif horizontal < 0:
                horizontal_direction = 'right'
            else:
                horizontal_direction = 'middle'

            if depth < 0.90:
                return f"The user's gaze is : {vertical_direction}-{horizontal_direction}"
            else:
                return "The user is looking front"
        else:
            print("No gaze detected")



    def eye_tracking_component_on_invoke(self):
        self.client.open()
        thread = threading.Thread(target=self.receive_data)
        thread.start()
        while True:
            data = self.client.get_next_packet()
            eet = hl2ss.unpack_eet(data.payload)
            self.queue.append(eet.combined_ray.direction)

            # print(eet.combined_ray.direction)