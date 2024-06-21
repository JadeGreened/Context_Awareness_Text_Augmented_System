import threading
import time

from pynput import keyboard

import numpy as np
from sklearn.svm import SVC

from viewer import hl2ss
from viewer import hl2ss_lnm
from joblib import load

class IMU_Component:
    def __init__(self, ip, mainSys):
        self.ip = ip
        self.mainSys = mainSys
        # Operating mode
        # 0: samples
        # 1: samples + rig pose
        # 2: query calibration (single transfer)
        self.mode = hl2ss.StreamMode.MODE_0
        self.columns = np.array([0, 1, 2, 120, 121, 122])
        trainX, trainy, testX, testy = self.load_datasets()
        selected_columns = trainX[:, self.columns]
        # SVM algorithm
        self.model = SVC(probability=True, gamma='auto')
        self.model.fit(trainX, trainy)
        print("The model is trained successfully")
        self.data_mean = np.mean(trainX, axis=0)
        if (self.mode == hl2ss.StreamMode.MODE_2):
            data = hl2ss_lnm.download_calibration_rm_imu(self.host, self.port)
            print('Calibration data')
            print('Extrinsics')
            print(data.extrinsics)
            quit()

    def load_datasets(self):
        import numpy as np

        # 设置文件的路径
        file_path = r'D:\PythonProj\imu_har\Human-activity-recognition-using-IMU-main\Human-activity-recognition-using-IMU-main\dataset\UCI HAR Dataset\UCI HAR Dataset'

        # 使用 numpy 的 loadtxt 函数加载数据
        try:
            trainX = np.loadtxt(file_path + "/train" + "/X_train.txt")
            trainy = np.loadtxt(file_path + "/train" + "/y_train.txt")
            testX = np.loadtxt(file_path + "/test" + "/X_test.txt")
            testy = np.loadtxt(file_path + "/test" + "/y_test.txt")
            print("The training file is loaded")
        except Exception as e:
            print(f"error:{e}")
        # 如果需要查看加载的数据的形状可以使用
        print("The dimension is: ", trainX.shape)
        return trainX, trainy, testX, testy


    def get_client(self, port):
        client = hl2ss_lnm.rx_rm_imu(self.ip, port, mode=self.mode)
        return client

    def make_prediction(self, data):
        label = np.array(["Walking", "Upstairs", "Downstairs", "Sitting", "Standing", "Laying"])
        prediction_index = int(self.model.predict(data)[0])  # 只调用一次预测
        print(prediction_index)
        motion = label[prediction_index]  # 直接使用预测索引
        print(motion)
        text = "The user is currently " + motion
        self.mainSys.read_then_write(text)

    def replace_with_combined_data(self, combined_data):

        # 创建索引映射，根据 columns 将 combined_data 映射到相应的列平均值
        for i, col_index in enumerate(self.columns):
            if col_index < len(self.data_mean):  # 确保索引不超出范围
                self.data_mean[col_index] = combined_data[i]

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
        columns = [0, 1, 2, 120, 121, 122]
        while True:
            acc_data = client_Acc.get_next_packet()
            gyro_data = client_Gy.get_next_packet()
            imu_data_acc = hl2ss.unpack_rm_imu(acc_data.payload)
            imu_data_gro = hl2ss.unpack_rm_imu(gyro_data.payload)
            sample_acc = imu_data_acc.get_frame(0)
            sample_gyro = imu_data_gro.get_frame(0)
            print('Accelerometer: x: ', sample_acc.x,", y: ", sample_acc.y,", z: ",sample_acc.z)
            print('Gyroscope: x: ', sample_gyro.x,", y: ", sample_gyro.y,", z: ",sample_gyro.z)
            combined_data = np.array([sample_acc.x,sample_acc.y,sample_acc.z, sample_gyro.x, sample_gyro.y, sample_gyro.z])
            self.replace_with_combined_data(combined_data)
            data_for_inference = self.data_mean
            data_for_inference = data_for_inference.reshape(1, -1)
            self.make_prediction(data_for_inference)
            time.sleep(2)






