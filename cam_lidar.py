#!/usr/bin/env python3
# pylint: disable=line-too-long

from pathlib import Path
from os import listdir
import time
import tkinter as tk
from enum import Enum
import numpy as np
from PIL import Image, ImageTk
import cv2 as cv2
import yaml
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
import os
from tqdm import trange
import pandas as pd


class States(Enum):
    CAMERA_SELECTION = 1
    LIDAR_SELECTION = 2

def get_points_cloud(path):
    return np.genfromtxt(path, delimiter=',')



def get_frame(video_name,N):
    # get a frame from ADVIO video stream at a given time.
    cap = cv2.VideoCapture(video_name)
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if N>totalFrames:
        print('Not a valid frame index')
        return False
    cap.set(cv2.CAP_PROP_POS_FRAMES,N)
    ret, frame = cap.read()
    # frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if np.shape(frame)[0]==720:
        frame=frame.swapaxes(1,0)
        frame=np.flip(frame,1)
    cap.release()
    return frame


def read_pose(path):
    # read the given pose file into a numpy array
    data=pd.read_csv(path,names=list('tabcdefg'))
    v=[]
    for c in 'tabcdefg':
        v.append(np.array(list(map(float,data[list(c)].values))))
    M=np.column_stack((v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]))
    #M=M[np.sum(M[:,4:8]**2,1)>0.01,:]
    return M


def read_frames(path):
    # read the given frame times into a numpy array
    data=pd.read_csv(path,names=list('ta'))    
    v=[]
    for c in 'ta':
        v.append(np.array(list(map(float,data[list(c)].values))))
    M=np.column_stack((v[0],v[1]))
    return M


def read_fix(path):
    # read fixpoint file into numpy array
    data=pd.read_csv(path,names=list('tabcdef'))
    v=[]
    for c in 'tabcdef':
        v.append(np.array(list(map(float,data[list(c)].values))))
    M=np.column_stack((v[0],v[1],v[3],v[2],v[4],v[5],v[6]))
    return M


class CamLidarTool(object):
    def __init__(self, data_path, data_id, extract):

        data_dir = Path(data_path)
        print("data_dir", data_dir)

        # initialise arrays that store the loaded data
        self.pointclouds = []

        # Set data directories. The arguments are sanitized by Click, so we don't need to check for an unknown sensor

        pointcloud_data_directory = data_dir.joinpath(Path('tango'))
        self.pointcloud_data_files = sorted(list(pointcloud_data_directory.glob('point-cloud-*.csv')))

        camera_data_directory = data_dir.joinpath(Path('iphone')) # 
        # extract
        extract_path = str(camera_data_directory.joinpath(Path('frames')).as_posix())
        if extract:
            os.makedirs(extract_path, exist_ok=True)
            print("extract_path", extract_path)
            cap = cv2.VideoCapture(camera_data_directory.joinpath(Path('frames.mov')).as_posix() )
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Create a loop to extract and save each frame
            for frame_num in trange(total_frames):
                # Read the frame from the video
                ret, frame = cap.read()

                # Check if the frame was read successfully
                if not ret:
                    print("Error: Unable to read the frame.")
                    break
                # Save the frame as an image (e.g., PNG or JPEG) in the desired output folder
                # Replace 'output_folder' with the path to your output folder
                frame_num = str(frame_num).zfill(6)
                output_file = os.path.join(extract_path, f'frame_{frame_num}.png')
                cv2.imwrite(output_file, frame)
                # Display progress (optional)

            cap.release()
        camera_image_paths = sorted(list(Path(extract_path).glob('*.png'))) 

        # align the frames, point  cloud and gt paths

        path= os.path.join(data_path,'iphone','frames.csv')
        self.frame_timestamps = read_frames(path)
        path= os.path.join(data_path,'ground-truth','pose.csv')
        self.gt_path = read_pose(path)
        self.point_cloud_timestamps_path = os.path.join(data_path,'tango','point-cloud.csv')
        self.point_cloud_timestamps = read_frames(self.point_cloud_timestamps_path)

        frame_t, frame_index, pose_index, pointcloud_index = self.align_frame_pose_pointcloud(data_id)

        # load the image and pcl
        self.image = np.array(Image.open(camera_image_paths[frame_index]))
        self.pointcloud = get_points_cloud(self.pointcloud_data_files[pointcloud_index])
        self.imgpoints = []  # 2d correspondence points
        self.lidar_correspondences = []  # 3d correspondence points

        self.root = tk.Tk()
        self.canvas: tk.Canvas = None
        self.main_button: tk.Button = None
        self.button_pressed: bool = False
        self.img_ref = None  # image reference of the camera image for the canvas

        self.nearest_neighbor: NearestNeighbors = None  # nearest neighbor data structure for LiDAR points

        self.state: States = None  # Stores tue current state from the enum States. Either CAMERA or LIDAR selection

        self.state_label: tk.Label = None  # tkinter label which displays the current state
        self.status_label: tk.Label = None  # tkinter label which displays the current status e.g. "0/10 pairs selected"
        self.selected_pairs: int = 0  # count the number of selected lidar/camera correspondences

        # holds the selected correspondences
        self.selected_camera_points = []
        self.selected_lidar_indices = []
        self.tk_clicked_points = []  # stores the displayed rectangles when a point is selected

        self.lidar_on_canvas = []  # Stores the reference to the LiDAR points that are drawn on the canvas

        # calibration data
        self.distortion = np.array([])
        self.K: np.array = np.array([])  # camera projection matrix

        # our current best guess of the transformation from lidar to camera
        self.translation: np.array = np.array([])  # translation from lidar to camera
        self.rotation: np.array = np.array([])  # rotation from lidar to camera. This is NOT a rotation matrix,
        # but a rotation vector. Use cv2.Rodrigues() to transform between rotation vector and matrix (and vice versa)

        self.init_calibration()


    def align_frame_pose_pointcloud(self,data_id):
        frame_t = self.frame_timestamps[data_id][0]
        frame_index = int(self.frame_timestamps[data_id][1])
        pose_index = int(np.argmin((self.gt_path[:,0]-frame_t)**2))
        pointcloud_index = int(np.argmin((self.point_cloud_timestamps[:,0]-frame_t)**2))
        print('t difference of points and frame', self.point_cloud_timestamps[pointcloud_index,0] - frame_t) # TODO transform point cloud based on interpolation 
        print('t difference of gt pose and frame', self.gt_path[pose_index,0] - frame_t) 

        return frame_t, frame_index, pose_index, pointcloud_index


    def init_calibration(self) -> None:
        """
        Initialised the calibration matrices. This is our current best guess for LiDAR->Camera calibration,
        and the camera projection and distortion data
        @return: None
        """

        # camera distortion
        self.distortion = np.array([-0.165483, 0.0966005, 0.00094785, 0.00101802])

        # transformation mrh to fw camera
        mrh_fs = cv2.FileStorage("data/calibration_data/extrinsics_mrh_forward.yaml", cv2.FILE_STORAGE_READ)
        mrh_R = mrh_fs.getNode("R_mtx").mat()
        mrh_t = mrh_fs.getNode("t_mtx").mat()
        transformation_mrh_fw_cam = np.hstack((mrh_R, mrh_t))
        transformation_mrh_fw_cam = np.vstack((transformation_mrh_fw_cam, np.array([0.0, 0.0, 0.0, 1.0])))

        # camera calibration matrix
        with open('data/calibration_data/forward.yaml', 'r') as f:
            fw_cam_trans = yaml.safe_load(f)
        self.K = np.array(fw_cam_trans['camera_matrix']['data']).reshape(3, 3)

        # FW to MRH transformation
        with open('data/calibration_data/static_transformations.yaml') as f:
            fw_to_mrh_yaml = yaml.safe_load(f)
        trans_data = fw_to_mrh_yaml['2020-07-05_tuggen']['fw_lidar_to_mrh_lidar']['translation']
        fw_to_mrh_t = np.array([trans_data['x'], trans_data['y'], trans_data['z']])
        rot_data = fw_to_mrh_yaml['2020-07-05_tuggen']['fw_lidar_to_mrh_lidar']['rotation']
        r = R.from_euler('zyx', np.array([rot_data['yaw'], rot_data['pitch'], rot_data['roll']]), degrees=False)
        fw_to_mrh_R = r.as_matrix()

        transformation_fw_mrh = np.column_stack((fw_to_mrh_R, fw_to_mrh_t))  # create 3x4 matrix
        transformation_fw_mrh = np.vstack((transformation_fw_mrh, np.array([0.0, 0.0, 0.0, 1.0])))  # create 4x4 matrix

        # combine both transformations together
        transformation = np.matmul(transformation_mrh_fw_cam, transformation_fw_mrh)

        # extract rotation, translation from transformation matrix
        self.translation = transformation[0:-1, 3]
        rotation = transformation[0:-1, 0:3]
        rotation, _ = cv2.Rodrigues(rotation)
        self.rotation = rotation

    def draw_lidar_to_canvas(self, projected_points: np.ndarray, color: str = 'blue') -> None:
        """
        Draws the projected LiDAR points to the canvas with a tkinter rectangle
        @param projected_points: nx2 array of LiDAR points projected to the camera
        @param color: tkinter string of the fill color.
                      Check: http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter for colors
        """

        for point in projected_points:
            x = point[0]
            y = point[1]
            point = self.canvas.create_rectangle(x - 1, y - 1, x + 1, y + 1, fill=color, width=0)
            self.lidar_on_canvas.append(point)  # store a reference to the rectangle so that it can be deleted

    def display_image(self) -> None:
        """
        Display the loaded image at full size on the canvas
        """

        # we have to store this variable in self.img_ref because otherwise it's cleared after this function ends
        # and for some reason, tkinter doesn't have access to the data anymore
        self.img_ref = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, anchor='nw', image=self.img_ref)
        self.canvas.pack()

    def _button_press(self, event):
        self.button_pressed = True

        # disable button as data is now calibrated
        self.main_button.config(state=tk.DISABLED)

        # remove the previously drawn LiDAR points
        for point in self.lidar_on_canvas:
            self.canvas.delete(point)

        # indices are packed inside a bunch of arrays. Use list and np.squeeze to get a list of lists for the indices
        indices = np.squeeze(np.array(list(index for index in self.selected_lidar_indices)))

        _, r, t = cv2.solvePnP(np.array(self.lidar_points_3d[indices], dtype=np.float32),
                               np.array(self.selected_camera_points, dtype=np.float32),
                               self.K, self.distortion)

        """
    [[0.18934124]
     [0.68308064]
     [1.89624156]]
    [[-0.99982693 -0.00769254  0.01693887]
     [-0.01674675 -0.02438831 -0.99956228]
     [ 0.00810229 -0.99967296  0.02425526]]
        """

        projected_points, _ = cv2.projectPoints(self.lidar_points_3d, r, t, self.K, self.distortion)
        projected_points = np.squeeze(projected_points)  # result has a strange structure. Put back into nx2 matrix

        self.draw_lidar_to_canvas(projected_points)

        rot_mat, _ = cv2.Rodrigues(r)
        print(t)
        print(rot_mat)

        # reset for 2nd iteration

        # remove the pressed points on the canvas
        for r in self.tk_clicked_points:
            self.canvas.delete(r)

        self.rotation = r
        self.translation = t
        self.selected_pairs = 0
        self.selected_lidar_indices = []
        self.selected_camera_points = []
        self.nearest_neighbor = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(projected_points)
        self.projected_points = projected_points
        self.status_label.config(text=f'{self.selected_pairs} point pairs selected (minimum 10)')

    def _button_release(self, event):  # pylint: disable = unused-argument
        """
        Callback function when the tkinter button is released. Only change button state, this is required for
        the _canvas_press function.
        """
        self.button_pressed = False

    def _canvas_press(self, eventorigin):
        """
        Callback function when the tkinter window gets pressed anywhere. This function handles two
        different cases, whether we are in camera point selection mode, or LiDAR point selection mode.
        For both cases, it stores the selected point in an array, switches the state, and updates the text.

        If the user clicks outside the image or clicks the button, the function doesn't register this as a
        point selection. The LiDAR points are are selected by nearest neighbor.
        """
        x = eventorigin.x
        y = eventorigin.y

        if self.button_pressed or y > 640:
            return

        if self.state == States.CAMERA_SELECTION:
            self.state = States.LIDAR_SELECTION
            self.state_label.config(text="Select a LiDAR point")
            rec = self.canvas.create_rectangle(x - 2, y - 2, x + 2, y + 2, fill="green yellow", width=1)
            self.tk_clicked_points.append(rec)
            self.selected_camera_points.append([x, y])

            # disable because a corresponding LiDAR point has to be selected first
            self.main_button.config(state=tk.DISABLED)

        elif self.state == States.LIDAR_SELECTION:
            self.state = States.CAMERA_SELECTION
            self.state_label.config(text="Select a camera point")
            indices = self.nearest_neighbor.kneighbors(np.array([[x, y]]), n_neighbors=1, return_distance=False)

            x = np.squeeze(self.projected_points[indices])[0]
            y = np.squeeze(self.projected_points[indices])[1]
            rec = self.canvas.create_rectangle(x - 2, y - 2, x + 2, y + 2, fill="white", width=1)
            self.tk_clicked_points.append(rec)
            self.selected_pairs += 1
            self.status_label.config(text=f'{self.selected_pairs} point pairs selected (minimum 10)')
            self.selected_lidar_indices.append(indices)

            if self.selected_pairs >= 10:
                self.main_button.config(state=tk.NORMAL)

    def run(self):
        img_shape = self.image.shape
        self.canvas = tk.Canvas(self.root,
                                width=img_shape[1],
                                height=img_shape[0] + 80)

        self.root.bind(
            '<Button 1>', self._canvas_press
        )  # register callback to handle left mouse clicks on the canvas

        # init state
        self.state_label = tk.Label(self.root, text="Select a camera point")
        self.status_label = tk.Label(self.root, text="0 point pairs selected (minimum 10)")
        self.state_label.place(x=40, y=img_shape[0] + 5)
        self.status_label.place(x=300, y=img_shape[0] + 5)
        self.state = States.CAMERA_SELECTION

        self.main_button = tk.Button(self.canvas, text="Continue")
        self.main_button.bind("<ButtonPress>", self._button_press)
        self.main_button.bind("<ButtonRelease>", self._button_release)
        self.main_button.place(x=600, y=img_shape[0] + 5)
        self.main_button.config(state=tk.DISABLED)

        self.display_image()

        points = np.array(self.pointcloud[:, :3])
        mask_behind = points[:, 1] < 0
        points = points[mask_behind]
        self.lidar_points_3d = points

        projected_points, _ = cv2.projectPoints(points, self.rotation, self.translation, self.K, self.distortion)
        projected_points = np.squeeze(projected_points)
        self.projected_points = projected_points

        self.nearest_neighbor = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(projected_points)

        self.draw_lidar_to_canvas(projected_points, 'red')

        self.root.mainloop()