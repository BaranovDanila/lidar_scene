import sys
import os
# import win32gui

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QWidget
from PyQt5 import QtGui, QtWidgets
from design1 import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from directions import Lidar
from signal_lidar import SignalGauss
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pptk
import time
import BVH
import math
import vtkplotlib as vpl
import pandas as pd
from utills3d import StlMesh, JsonMesh
import json
from datetime import datetime


os.environ["PYOPENCL_CTX"] = '0'
os.environ["PYOPENCL_COMPILER_OUTPUT"] = '1'


class Gui(QMainWindow):
    def __init__(self):
        super(Gui, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.openfileButton.clicked.connect(self.browse_files)
        self.ui.single_chanelButton.clicked.connect(self.get_dynamic_scene)
        self.ui.modelingButton.clicked.connect(self.modeling)
        self.ui.pushButton_move_position.clicked.connect(self.move_position)
        self.ui.pushButton_rotate.clicked.connect(self.rotate)
        self.ui.radioButton_dynamic_scene.clicked.connect(self.check_scene)
        self.ui.radioButton_static_scene.clicked.connect(self.check_scene)
        self.ui.single_chanelButton.clicked.connect(self.single_modeling)
        self.ui.getallout_Button.clicked.connect(self.get_all_out)
        self.ui.pluschanel_Button.clicked.connect(self.swipe_chanel_plus)
        self.ui.minuschanel_Button.clicked.connect(self.swipe_chanel_minus)
        self.ui.chanelR_Button.clicked.connect(self.set_chanel)
        self.ui.treesButton.clicked.connect(self.build_tree)
        self.ui.pushButton_dyn_scene.clicked.connect(self.scene_step)
        self.figure_dp = plt.figure()
        self.canvas_dp = FigureCanvas(self.figure_dp)
        self.toolbar_dp = NavigationToolbar(self.canvas_dp, self)
        layout_dp = self.ui.verticalLayout_4
        layout_dp.addWidget(self.toolbar_dp)
        layout_dp.addWidget(self.canvas_dp)
        self.figure_in = plt.figure()
        self.canvas_in = FigureCanvas(self.figure_in)
        self.toolbar_in = NavigationToolbar(self.canvas_in, self)
        layout_in = self.ui.verticalLayout_7
        layout_in.addWidget(self.toolbar_in)
        layout_in.addWidget(self.canvas_in)
        self.figure_ih = plt.figure()
        self.canvas_ih = FigureCanvas(self.figure_ih)
        self.toolbar_ih = NavigationToolbar(self.canvas_ih, self)
        layout_ih = self.ui.verticalLayout_8
        layout_ih.addWidget(self.toolbar_ih)
        layout_ih.addWidget(self.canvas_ih)
        self.figure_out = plt.figure()
        self.canvas_out = FigureCanvas(self.figure_out)
        self.toolbar_out = NavigationToolbar(self.canvas_out, self)
        layout_out = self.ui.verticalLayout_9
        layout_out.addWidget(self.toolbar_out)
        layout_out.addWidget(self.canvas_out)
        self.figure_dps = plt.figure()
        self.canvas_dps = FigureCanvas(self.figure_dps)
        self.toolbar_dps = NavigationToolbar(self.canvas_dps, self)
        layout_dps = self.ui.verticalLayout_10
        layout_dps.addWidget(self.toolbar_dps)
        layout_dps.addWidget(self.canvas_dps)
        self.figure_ao = plt.figure()
        self.canvas_ao = FigureCanvas(self.figure_ao)
        self.toolbar_ao = NavigationToolbar(self.canvas_ao, self)
        layout_ao = self.ui.verticalLayout_11
        layout_ao.addWidget(self.toolbar_ao)
        layout_ao.addWidget(self.canvas_ao)
        layout_cp = self.ui.verticalLayout_5
        self.fig_scene = vpl.QtFigure()
        self.ui.verticalLayout_6.addWidget(self.fig_scene)
        self.v = pptk.viewer(pptk.rand(1, 3))
        self.scene = None
        self.file_path = None
        self.aabb = None
        self.tri = None
        self.tri_start = None
        self.n_tri = None
        self.children = None
        self.parents = None
        self.scene_status = 'static'

        
        # hwnd = win32gui.FindWindowEx(0, 0, None, "viewer")  # retrieve the window ID of the viewer
        # self.window = QtGui.QWindow.fromWinId(hwnd)  # get the viewer inside a window
        # widget = QtWidgets.QWidget()
        # self.windowcontainer = self.createWindowContainer(self.window, widget)
        # layout_cp.addWidget(self.windowcontainer)

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.v.close()

    def check_scene(self):
        if self.ui.radioButton_static_scene.isChecked():
            self.scene_status = 'static'
        if self.ui.radioButton_dynamic_scene.isChecked():
            self.scene_status = 'dynamic'
        print(self.scene_status)

    def browse_files(self):
        scene = QFileDialog.getOpenFileName(self, 'Open file', os.getcwd(),
                                            'STL files (*.stl);;JSON files (*.json)')
        self.ui.lineEdit.setText(scene[0])
        self.file_path = self.ui.lineEdit.text()
        try:
            if self.file_path.split('/')[-1].split('.')[-1] == 'stl':
                self.scene = StlMesh(self.file_path)
                self.ui.radioButton_static_scene.setChecked(True)
                print(self.scene.vectors)
                print(self.scene.vectors.dtype)
            if self.file_path.split('/')[-1].split('.')[-1] == 'json':
                self.ui.radioButton_dynamic_scene.setChecked(True)
                with open(self.file_path) as f:
                    scene = json.load(f)
                for key in scene.keys():
                    self.scene = JsonMesh(scene, key)
                    print(self.scene.vectors[1, 1, :])
                    print(self.scene.vectors.dtype)
                    # self.meshes_list.append(mesh)
        except FileNotFoundError:
            print('Выберите файл')
        # self.show_scene()

    def build_tree(self):
        start_time = time.time()
        root = BVH.build_bvh(self.scene.vectors, 200, mode='median')
        self.aabb, self.tri, self.tri_start, self.n_tri, self.children, self.parents = BVH.bvh_serializer(root)
        print('Дерево построено за', (time.time() - start_time))

    def get_dynamic_scene(self) -> None:
        points = pd.read_csv('points.csv')
        # x_coord = points['X'].values
        # z_coord = points['Y'].values

        for x, z in points.values:
            self.scene.move(x, 0, z)

            self.modeling()

    def modeling(self):
        cp = np.array([0, 0, 0, 0])
        self.signal = SignalGauss(5e-6, 1e6, float(self.ui.lineEdit_2.text()))
        start_time = time.time()
        # i = 6
        for i in range(1):
            # lidar.rotate_window('z', 60)
            lidar.scan_bvh(self.aabb, self.tri, self.tri_start, self.n_tri, self.children)
            # lidar.scan(self.scene.vectors)
            cp = np.vstack((cp, lidar.cloud_points))
        cp = cp[1:]

        print("Время сканирования", (time.time() - start_time))
        lidar.t_d = float(self.ui.lineEdit_2.text())
        lidar.create_plot_dp(lidar.dp, figure=self.figure_dp, canvas=self.canvas_dp)
        
        # save the image with name
        utc_now = datetime.utcnow()
        date_time_str = utc_now.strftime('%Y%m%d_%H%M%S')
        np.savez(f'./result_dp/{date_time_str}.npz', np.flipud(lidar.dp))   # save range image to the file 

        # print(lidar.dp.shape, end='\n\n')
        # print(lidar.dp)
        self.signal.get_plot(figure=self.figure_in, canvas=self.canvas_in)
	
        a = cp[:, 3]
        a = (a-np.min(a))/(np.max(a)-np.min(a))
        viridis = cm.get_cmap('viridis', 256)
        colors = viridis(a)
        # self.v.close()
        self.v.clear()
        self.v.load(cp[:, :3])
        self.v.attributes(colors[:, :3])
        self.v.set(point_size=0.01)
        self.v.set(lookat=np.array(lidar.position, dtype=np.float32))
        # self.v.set(lookat=(lidar.position[0], lidar.position[1], lidar.position[2]))
        self.v.set(phi=math.radians(270))
        self.v.set(theta=math.radians(0))
        self.v.set(r=10)
        self.v.set(show_axis=True)
        self.show_scene()

    def show_scene(self):
        camera = self.fig_scene.camera
        self.fig_scene.close()
        self.fig_scene = vpl.QtFigure()
        self.ui.verticalLayout_6.addWidget(self.fig_scene)
        self.fig_scene.camera = camera
        self.scene.show_mesh(self.fig_scene, axes=False)
        x_ax = np.array([lidar.position, [np.max(self.scene.x) * 1.2, lidar.position[1], lidar.position[2]]])
        y_ax = np.array([lidar.position, [lidar.position[0], np.max(self.scene.y) * 1.2, lidar.position[2]]])
        z_ax = np.array([lidar.position, [lidar.position[0], lidar.position[1], np.max(self.scene.z) * 1.2]])
        m1 = np.array([lidar.position, lidar.dir_data[0]])
        m2 = np.array([lidar.position, lidar.dir_data[-1]])
        vpl.plot(x_ax, color='red', fig=self.fig_scene, line_width=5.0)
        vpl.plot(y_ax, color='green', fig=self.fig_scene, line_width=5.0)
        vpl.plot(z_ax, color='blue', fig=self.fig_scene, line_width=5.0)
        self.fig_scene.show()

    def single_modeling(self):
        directions = lidar.get_single_dir(int(self.ui.spinBox_chanel_v.value()), int(self.ui.spinBox_chanel_h.value()))
        print('Матрица лучей сформирована')
        # lidar.scan(scene_lidar.triangles_data, 'single', directions)
        lidar.scan_bvh(self.aabblist, self.trilist, self.tristartlist, self.ntrilist, self.childlist, mode='single', directions=directions)
        print('Сканирование завершено')
        # lidar.scan(sphere.triangles, 'single', directions)
        lidar.get_ih(lidar.single_dp)
        print('Импульсная характеристика рассчитана')
        lidar.create_plot_dp(lidar.single_dp, mode='single', figure=self.figure_dps, canvas=self.canvas_dps)
        lidar.create_plot_ih(figure=self.figure_ih, canvas=self.canvas_ih)
        lidar.create_plot_out_signal(self.signal.signal, figure=self.figure_out, canvas=self.canvas_out)
        print('Графики построены')

    @staticmethod
    def position():
        lidar.set_position(10, 20, 30)

    def move_position(self):
        delta = np.array([float(obj) for obj in (self.ui.lineEdit_set_position.text().split(';'))], dtype=np.float32)
        lidar.move_position(delta)
        self.modeling()

    def rotate(self):
        angles = np.array([float(obj) for obj in (self.ui.lineEdit_rotate.text().split(';'))], dtype=np.float32)
        lidar.rotate_window('x', angles[0])
        lidar.rotate_window('z', angles[1])
        self.modeling()

    def scene_step(self):
        t = float(self.ui.lineEdit_time_scene.text())
        vel = self.scene.vel
        acc = self.scene.acc
        self.scene.vectors[:, :, :] += vel*t + acc/2*t**2
        self.scene.vel += acc*t
        root = BVH.build_bvh(self.scene.vectors, 200, mode='median')
        self.aabb, self.tri, self.tri_start, self.n_tri, self.children, self.parents = BVH.bvh_serializer(root)
        self.modeling()

    def get_all_out(self):
        self.all_out = lidar.get_all_out_signal(self.aabblist, self.trilist, self.tristartlist, self.ntrilist, self.childlist, self.signal.signal)
        # self.all_out = self.all_out/(lidar.single_qr_h*lidar.single_qr_v)
        print(np.amax(self.all_out))
        print('расчёт окончен')
        self.figure_ao.clear()
        print(1)
        ax = self.figure_ao.add_subplot(111)
        print(2)
        plot = ax.imshow(self.all_out[self.i, :, :], vmax=np.amax(self.all_out))
        print(3)
        self.figure_ao.colorbar(plot)
        print(4)
        self.canvas_ao.draw()

    def swipe_chanel_plus(self):
        self.i += 1
        self.figure_ao.clear()
        ax = self.figure_ao.add_subplot(111)
        plot = ax.imshow(self.all_out[self.i], vmax=np.amax(self.all_out))
        self.figure_ao.colorbar(plot)
        self.canvas_ao.draw()
        self.ui.chanelR_spinBox.setValue(int(self.i))

    def swipe_chanel_minus(self):
        self.i -= 1
        self.figure_ao.clear()
        ax = self.figure_ao.add_subplot(111)
        plot = ax.imshow(self.all_out[self.i], vmax=np.amax(self.all_out))
        self.figure_ao.colorbar(plot)
        self.canvas_ao.draw()
        self.ui.chanelR_spinBox.setValue(int(self.i))

    def set_chanel(self):
        self.i = int(self.ui.chanelR_spinBox.value())
        self.figure_ao.clear()
        ax = self.figure_ao.add_subplot(111)
        plot = ax.imshow(self.all_out[self.i], vmax=np.amax(self.all_out))
        self.figure_ao.colorbar(plot)
        self.canvas_ao.draw()

    def plot_min_triangles(self):
        # n_tri = [i*1000+100 for i in range(51)]
        n_tri = [100, 200, 300, 500, 800, 1000, 2000, 3000, 5000, 8000, 10000, 20000, 30000, 50000, 80000, 100000, 200000, 300000]
        time_tree = []
        time_scan = []
        time_sum = []
        for n in n_tri:
            # print(n)
            start_time = time.time()
            root = BVH.build_bvh(self.scene.triangles_data, mode='median', min_tri=n)
            aabblist, trilist, tristartlist, ntrilist, childlist, parentslist = BVH.bvh_serializer(root)
            tree = time.time()-start_time
            time_tree.append(tree)
            start_time = time.time()
            lidar.scan_bvh(aabblist, trilist, tristartlist, ntrilist, childlist)
            scan = time.time() - start_time
            time_scan.append(scan)
            time_sum.append(tree+scan)
            print('N poly', n, 'Time tree', tree, 'Time scan', scan, 'Time sum', tree+scan)
        print(time_tree, time_scan, time_sum)
        print(time_scan)
        print(time_sum)
        plt.figure()
        plt.plot(n_tri, time_tree, label='Время построения дерева')
        plt.plot(n_tri, time_scan, label='Время сканирования')
        plt.plot(n_tri, time_sum, lw=2, label='Суммарное время')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Минимальное число полигонов')
        plt.ylabel('Время, с')
        plt.xscale('log')
        plt.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    lidar = Lidar(60, 40, 0.1, 0.1)
    # lidar.set_position(np.array([0, 45, 0], dtype=np.float32))
    # lidar.rotate_window('z', 60)
    window = Gui()
    window.show()
    sys.exit(app.exec_())




