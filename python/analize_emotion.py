import sys
from os import path

import cv2
import numpy as np

from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QSizePolicy
import requests
import json
import pyaudio
import wave

class RecordVideo(QtCore.QObject):
    image_data = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera = cv2.VideoCapture(camera_port)
        self.camera.set(3, 240)  # Width
        self.camera.set(4, 240)  # Heigh

        self.timer = QtCore.QBasicTimer()

    def start_recording(self):
        self.timer.start(0, self)

    def timerEvent(self, event):
        if (event.timerId() != self.timer.timerId()):
            return

        read, data = self.camera.read()
        if read:
            self.image_data.emit(data)
            # TODO 画像書き込みまくるのでどうなのか
            cv2.imwrite("output.jpg", data)

    def stop_recording(self):
        self.timer.stop()

class FaceDetectionWidget(QtWidgets.QWidget):
    def __init__(self, haar_cascade_filepath, parent=None):
        super().__init__(parent)
        self.classifier = cv2.CascadeClassifier(haar_cascade_filepath)
        self.image = QtGui.QImage()
        self._red = (0, 0, 255)
        self._width = 2
        self._min_size = (100, 100)

    def detect_faces(self, image: np.ndarray):
        # haarclassifiers work better in black and white
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.equalizeHist(gray_image)

        faces = self.classifier.detectMultiScale(gray_image,
                                                 scaleFactor=1.3,
                                                 minNeighbors=4,
                                                 flags=cv2.CASCADE_SCALE_IMAGE,
                                                 minSize=self._min_size)

        return faces

    def image_data_slot(self, image_data):
        faces = self.detect_faces(image_data)
        # TODO 顔検出OFF
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(image_data,
        #                   (x, y),
        #                   (x+w, y+h),
        #                   self._red,
        #                   self._width)

        self.image = self.get_qimage(image_data)
        if self.image.size() != self.size():
            self.setFixedSize(self.image.size())

        self.update()

    def get_qimage(self, image: np.ndarray):
        height, width, colors = image.shape
        bytesPerLine = 3 * width
        QImage = QtGui.QImage

        image = QImage(image.data,
                       width,
                       height,
                       bytesPerLine,
                       QImage.Format_RGB888)

        image = image.rgbSwapped()
        return image

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

class FaceRequest():
    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'
    headers = {
        'Ocp-Apim-Subscription-Key': '01091e39d9c348d2b2076f5fd5b0ed5c',
        'Content-Type': 'application/octet-stream'
    }
    
    params = {
        'returnFaceId': 'true',
        'returnFaceLandmarks': 'false',
        'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise'
    }

    def get_face_data(self):
        data = open('output.jpg', 'rb')
        response = requests.post(self.face_api_url, params=self.params, headers=self.headers, data=data)
        print('request: ' + '\r\n' + str(response.content) + '\r\n')
        emotion = json.loads(response.content)[0]['faceAttributes']['emotion']
        print('emotion!!!: ' + '\r\n' + str(emotion))
    
        return json.loads(response.content)

class RecodeVoice():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16 # int16型
    CHANNELS = 1             # モノラル
    RATE = 11025             # 11.025.kHz
    RECORD_SECONDS = 4       # 4秒録音
    WAVE_OUTPUT_FILENAME = 'output.wav'

    def start_recording(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK)

        print("* 録音中（4秒）")

        frames = []

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("* 録音終了")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

class VoiceRequest():
    url = 'https://api.webempath.net/v2/analyzeWav'
    file_name = 'output.wav'
    params = {
        'apikey' : 'ZZPkCjup8qk9_Fogwj_j9chO7PB9inIGcqfljwxgtcI'
    }

    def get_voice_data(self):
        files = {
            'wav' : (self.file_name, open(self.file_name, 'rb').read(), 'audio/x-wav')
        }
        response = requests.post(self.url, params=self.params, files=files)
        print(response.content)
        return json.loads(response.content)


class MainWidget(QtWidgets.QWidget):
    
    def __init__(self, haarcascade_filepath, parent=None):
        super().__init__(parent)

        # 表情キャプチャ
        fp = haarcascade_filepath
        self.face_detection_widget = FaceDetectionWidget(fp)

        self.record_video = RecordVideo()

        image_data_slot = self.face_detection_widget.image_data_slot
        self.record_video.image_data.connect(image_data_slot)

        layout = QtWidgets.QVBoxLayout()
        self.face_detection_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout.addWidget(self.face_detection_widget, stretch=0, alignment=(QtCore.Qt.AlignTop))
        self.setLayout(layout)

        self.face_request = FaceRequest()

        self.face_button = QtWidgets.QPushButton('画面キャプチャ ＆ 表情感情取得', self)
        self.face_button.setGeometry(QtCore.QRect(50, 260, 250, 35))
        self.face_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogApplyButton')))
        self.face_button.clicked.connect(self.capture_image_and_draw_face_data)

        self.face_status_text = QtWidgets.QLabel('', self)
        self.face_status_text.setGeometry(QtCore.QRect(55, 290, 200, 20))
        self.face_status_text.setStyleSheet('color: red')

        self.result_face_json = None
        
        # 音声キャプチャ
        self.recode_voice = RecodeVoice()
        self.voice_request = VoiceRequest()

        self.voice_guide_text = QtWidgets.QLabel('※ボタンを押してから4秒間音声を取得し感情を取得', self)
        self.voice_guide_text.setGeometry(QtCore.QRect(0, 450, 380, 20))
        self.voice_guide_text.setAlignment(QtCore.Qt.AlignCenter)

        self.voice_button = QtWidgets.QPushButton('音声キャプチャ ＆ 音声感情取得', self)
        self.voice_button.setGeometry(QtCore.QRect(50, 520, 250, 35))
        self.voice_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogApplyButton')))
        self.voice_button.clicked.connect(self.capture_voice_and_draw_voice_data)

        self.voice_status_text = QtWidgets.QLabel('', self)
        self.voice_status_text.setGeometry(QtCore.QRect(55, 550, 200, 20))
        self.voice_status_text.setStyleSheet('color: red')

        self.result_voice_json = None

        # データ入力欄
        self.set_data_input_group()

        # データ送信ボタン
        self.send_button = QtWidgets.QPushButton('データ送信', self)
        self.send_button.setGeometry(QtCore.QRect(325, 775, 150, 35))
        self.send_status_text = QtWidgets.QLabel('', self)
        self.send_status_text.setGeometry(QtCore.QRect(10, 810, 780, 20))
        self.send_status_text.setStyleSheet('color: red')
        self.send_button.clicked.connect(self.send)
        self.send_status_text.setAlignment(QtCore.Qt.AlignCenter)
    
    def capture_image_and_draw_face_data(self):
        # 動画をストップ
        # TODO リクエストを別スレッドの処理にして状況を表示したいが難しいのでとりあえずコンソールに出力
        print('表情感情の解析中...（MicrosoftのAPIを実行中）')
        self.record_video.stop_recording()
        # 表情感情取得APIを実行、グラフを書き込み
        # TODO エラー処理（顔が認識できなかった時）は未実装
        self.result_face_json = self.face_request.get_face_data()
        # TODO グラフへの書き込み
        self.face_button.setText('クリア')
        self.face_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogCancelButton')))
        self.face_button.clicked.disconnect()
        self.face_button.clicked.connect(self.restart_video)
        self.face_status_text.setText('表情感情の解析が完了しました！')
        
    def restart_video(self):
        self.record_video.start_recording()
        self.face_button.setText('画面キャプチャ ＆ 表情感情取得')
        self.face_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogApplyButton')))
        self.face_button.clicked.disconnect()
        self.face_button.clicked.connect(self.capture_image_and_draw_face_data)
        self.face_status_text.setText('')
        self.result_face_json = None

    def capture_voice_and_draw_voice_data(self):
        # self.voice_status_text.setText('音声を取得しています')
        self.recode_voice.start_recording()
        print('音声感情の解析中...（EmpathのAPIを実行中）')
        # TODO Empath APIの実行のエラー処理は未実装
        self.result_voice_json = self.voice_request.get_voice_data()
        # TODO グラフへの書き込み
        self.voice_button.setText('クリア')
        self.voice_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogCancelButton')))
        self.voice_button.clicked.disconnect()
        self.voice_button.clicked.connect(self.restart_voice)
        self.voice_status_text.setText('音声感情の解析が完了しました！')

    def restart_voice(self):
        self.voice_button.setText('音声キャプチャ ＆ 音声感情取得')
        self.voice_button.setIcon(self.style().standardIcon(getattr(QtWidgets.QStyle, 'SP_DialogApplyButton')))
        self.voice_button.clicked.disconnect()
        self.voice_button.clicked.connect(self.capture_voice_and_draw_voice_data)
        self.voice_status_text.setText('')
        self.result_voice_json = None


    def set_data_input_group(self):
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setGeometry(QtCore.QRect(10, 600, 780, 150))
        self.groupBox.setTitle('データ入力欄', )
        
        # ユーザID
        self.user_label = QtWidgets.QLabel('ユーザID（社員番号）', self.groupBox)
        self.user_label.setGeometry(QtCore.QRect(30, 45, 130, 20))
        self.user_label.raise_()
        self.user_text_edit = QtWidgets.QPlainTextEdit(self.groupBox) 
        self.user_text_edit.setGeometry(QtCore.QRect(160, 45, 100, 25))
        self.user_text_edit.raise_()

        # 残業時間
        self.overtime_label = QtWidgets.QLabel('残業時間（h）', self.groupBox)
        self.overtime_label.setGeometry(QtCore.QRect(30, 100, 130, 20))
        self.overtime_label.raise_()
        self.overtime_text_edit = QtWidgets.QPlainTextEdit(self.groupBox) 
        self.overtime_text_edit.setGeometry(QtCore.QRect(160, 100, 100, 25))
        self.overtime_text_edit.raise_()

        # 精神疲労度
        self.mental_stress_label = QtWidgets.QLabel('精神疲労度（0〜100）', self.groupBox)
        self.mental_stress_label.setGeometry(QtCore.QRect(390, 45, 130, 20))
        self.mental_stress_label.raise_()
        self.mental_stress_slider = QtWidgets.QSlider(self.groupBox)
        self.mental_stress_slider.setOrientation(QtCore.Qt.Horizontal)
        self.mental_stress_slider.setGeometry(QtCore.QRect(525, 45, 160, 25))
        self.mental_stress_slider.raise_()
        self.mental_stress_text_edit = QtWidgets.QPlainTextEdit(self.groupBox) 
        self.mental_stress_text_edit.setGeometry(QtCore.QRect(695, 45, 35, 25))
        self.mental_stress_text_edit.raise_()
        # 精神疲労度初期値
        self.mental_stress_slider.setValue(50)
        self.mental_stress_slider.setMaximum(100)
        self.mental_stress_text_edit.setPlainText('50')
        # 精神疲労度のスライダーとテキストボックスの連動
        self.mental_stress_slider.valueChanged.connect(lambda value: self.mental_stress_text_edit.setPlainText(str(value)))

        # 肉体疲労度
        self.physical_stress_label = QtWidgets.QLabel('肉体疲労度（0〜100）', self.groupBox)
        self.physical_stress_label.setGeometry(QtCore.QRect(390, 100, 130, 20))
        self.physical_stress_label.raise_()
        self.physical_stress_slider = QtWidgets.QSlider(self.groupBox)
        self.physical_stress_slider.setOrientation(QtCore.Qt.Horizontal)
        self.physical_stress_slider.setGeometry(QtCore.QRect(525, 100, 160, 25))
        self.physical_stress_slider.raise_()
        self.physical_stress_text_edit = QtWidgets.QPlainTextEdit(self.groupBox) 
        self.physical_stress_text_edit.setGeometry(QtCore.QRect(695, 100, 35, 25))
        self.physical_stress_text_edit.raise_()
        # 肉体疲労度初期値
        self.physical_stress_slider.setValue(50)
        self.physical_stress_slider.setMaximum(100)
        self.physical_stress_text_edit.setPlainText('50')
        # 肉体疲労度のスライダーとテキストボックスの連動
        self.physical_stress_slider.valueChanged.connect(lambda value: self.physical_stress_text_edit.setPlainText(str(value)))

    def send(self):
        face = self.result_face_json
        voice = self.result_voice_json
        user_id = self.user_text_edit.toPlainText()
        overtime = self.overtime_text_edit.toPlainText()
        mental_stress = self.mental_stress_text_edit.toPlainText()
        physical_stress = self.physical_stress_text_edit.toPlainText()

        if face is None or voice is None or not user_id or not overtime or not mental_stress or not physical_stress:
            self.send_status_text.setText('感情を未取得もしくは未入力の項目が存在します。')
            return
        
        # TODO JSONを作成して、データ送信
        self.send_status_text.setText('データが正常に送信されました。といきたいのですがデータ送信処理は未実装です。。。')


def main(haar_cascade_filepath):
    app = QtWidgets.QApplication(sys.argv)

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(haar_cascade_filepath)
    
    main_window.setCentralWidget(main_widget)
    main_window.resize(800, 900)
    main_window.show()
    main_widget.record_video.start_recording()
    sys.exit(app.exec_())


if __name__ == '__main__':
    script_dir = path.dirname(path.realpath(__file__))
    cascade_filepath = 'haarcascade_frontalface_default.xml'
    cascade_filepath = path.abspath(cascade_filepath)
    main(cascade_filepath)