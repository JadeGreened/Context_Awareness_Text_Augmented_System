import threading
import time

from pynput import keyboard



from viewer import hl2ss
from viewer import hl2ss_lnm


class IMU_Component:
    def __init__(self, ip, mainSys):
        self.ip = ip
        self.mainSys = mainSys
        # Operating mode
        # 0: samples
        # 1: samples + rig pose
        # 2: query calibration (single transfer)
        self.mode = hl2ss.StreamMode.MODE_0

        if (self.mode == hl2ss.StreamMode.MODE_2):
            data = hl2ss_lnm.download_calibration_rm_imu(self.host, self.port)
            print('Calibration data')
            print('Extrinsics')
            print(data.extrinsics)
            quit()


    def get_client(self, port):
        client = hl2ss_lnm.rx_rm_imu(self.ip, port, mode=self.mode)
        return client






    def imu_component_on_invoke(self):
        thread_acc_created = False
        thread_gyo_created = False
        # Port
        # Options:
        # hl2ss.StreamPort.RM_IMU_ACCELEROMETER
        # hl2ss.StreamPort.RM_IMU_GYROSCOPE
        # hl2ss.StreamPort.RM_IMU_MAGNETOMETER
        client_Acc = self.get_client(hl2ss.StreamPort.RM_IMU_ACCELEROMETER)
        client_Gy = self.get_client(hl2ss.StreamPort.RM_IMU_GYROSCOPE)
        client_Acc.open()
        client_Gy.open()
        while True:
            acc_data = client_Acc.get_next_packet()
            gyro_data = client_Gy.get_next_packet()
            imu_data_acc = hl2ss.unpack_rm_imu(acc_data.payload)
            imu_data_gro = hl2ss.unpack_rm_imu(gyro_data.payload)
            sample_acc = imu_data_acc.get_frame(0)
            sample_gyro = imu_data_gro.get_frame(0)
            print('Accelerometer: x: ', sample_acc.x,", y: ", sample_acc.y,", z: ",sample_acc.z)
            print('Gyroscope: x: ', sample_gyro.x,", y: ", sample_gyro.y,", z: ",sample_gyro.z)

            time.sleep(0.02)






