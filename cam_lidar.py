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


# 1550
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

def make_rigid_matrix(rot, trans):
    rot = rot.reshape(3,3)
    trans = trans.reshape(3,1)
    rigid = np.concatenate([rot,trans],axis=1)
    rigid = np.concatenate([rigid, np.array([0,0,0,1]).reshape(1,4) ],axis=0)
    return rigid


def read_fix(path):
    # read fixpoint file into numpy array
    data=pd.read_csv(path,names=list('tabcdef'))
    v=[]
    for c in 'tabcdef':
        v.append(np.array(list(map(float,data[list(c)].values))))
    M=np.column_stack((v[0],v[1],v[3],v[2],v[4],v[5],v[6]))
    return M


def as_intrinsics_matrix(intrinsics):
    """
    Get matrix representation of intrinsics.

    """
    K = np.eye(3)
    K[0, 0] = intrinsics[0]
    K[1, 1] = intrinsics[1]
    K[0, 2] = intrinsics[2]
    K[1, 2] = intrinsics[3]
    return K

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

        # presee the point clouds

        frame_t, frame_index, pose_index, pointcloud_index = self.align_frame_pose_pointcloud(data_id)

        # accumulate point cloud 
        pointcloud_acc = self.accumulate_point_cloud(data_id)


        # load the image and pcl
        self.image = np.array(Image.open(camera_image_paths[frame_index]))
        self.pointcloud = pointcloud_acc #get_points_cloud(self.pointcloud_data_files[pointcloud_index])
       
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

    def accumulate_point_cloud(self, data_id):
        frame_t = self.frame_timestamps[data_id][0]
        frame_index = int(self.frame_timestamps[data_id][1])
        pointcloud_index = int(np.argmin((self.point_cloud_timestamps[:,0]-frame_t)**2))
        pointcloud_index_previous_t = self.point_cloud_timestamps[pointcloud_index - 2,0]
        pointcloud_index_future_t = self.point_cloud_timestamps[pointcloud_index + 2,0]
        pose_index_previous = int(np.argmin((self.gt_path[:,0]-pointcloud_index_previous_t)**2)) - 1
        pose_index_future = int(np.argmin((self.gt_path[:,0]-pointcloud_index_future_t)**2)) + 1
        # pose interpolation
        input_t = self.gt_path[pose_index_previous: pose_index_future,0]
        input_trans = self.gt_path[pose_index_previous: pose_index_future,1:4]
        input_rots = self.gt_path[pose_index_previous: pose_index_future,4:8]
        input_rots = np.hstack([input_rots[:,1:4], input_rots[:,:1]]) # to x y z w
        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp
        from scipy.interpolate import CubicSpline
        input_rots = R.from_quat(input_rots)
        slerp = Slerp(input_t, input_rots)
        spline = CubicSpline(input_t, input_trans)
        
        target_pose_rot = slerp(frame_t).as_matrix()
        target_pose_trans = spline(frame_t)
        T_target = make_rigid_matrix(target_pose_rot, target_pose_trans)
        
        # accumulate lidar points to target
        pointcloud_acc = []
        for lidar_points_index in range(pointcloud_index - 1, pointcloud_index +2):
            pointcloud_i = get_points_cloud(self.pointcloud_data_files[lidar_points_index])
            rot_i = slerp(self.point_cloud_timestamps[lidar_points_index,0]).as_matrix()
            trans_i = spline(self.point_cloud_timestamps[lidar_points_index,0])
            Ti = make_rigid_matrix(rot_i,trans_i )
            relative_transform = np.linalg.inv(T_target) @ Ti
            pointcloud_i = np.concatenate([pointcloud_i, np.ones_like(pointcloud_i)[:,:1]], axis = -1)
            pointcloud_i = (pointcloud_i @ relative_transform.transpose())[:,:3]
            pointcloud_acc.append(pointcloud_i)

        pointcloud_acc = np.concatenate((pointcloud_acc[0], pointcloud_acc[1], pointcloud_acc[2]), axis=0)

        pointcloud_acc = pointcloud_acc.reshape(-1,3)
        print("pointcloud_acc", np.shape(pointcloud_acc))
        return pointcloud_acc
            

        

    def align_frame_pose_pointcloud(self, data_id):
        frame_t = self.frame_timestamps[data_id][0]
        frame_index = int(self.frame_timestamps[data_id][1])
        pose_index = int(np.argmin((self.gt_path[:,0]-frame_t)**2))
        pointcloud_index = int(np.argmin((self.point_cloud_timestamps[:,0]-frame_t)**2))
        print('t difference of points and frame', self.point_cloud_timestamps[pointcloud_index,0] - frame_t) # TODO transform point cloud based on interpolation 
        # assert np.abs(self.point_cloud_timestamps[pointcloud_index,0] - frame_t) < 0.1, f'{np.abs(self.point_cloud_timestamps[pointcloud_index,0] - frame_t)}'
        print('t difference of gt pose and frame', self.gt_path[pose_index,0] - frame_t) 

        return frame_t, frame_index, pose_index, pointcloud_index


    def init_calibration(self) -> None:
        """
        Initialised the calibration matrices. This is our current best guess for LiDAR->Camera calibration,
        and the camera projection and distortion data
        @return: None
        """

        # camera distortion and calibration matrix
        self.distortion = np.array([0.0478, 0.0339, -0.00033, -0.00091])        
        self.K = as_intrinsics_matrix([1077.2, 1079.3,362.145, 636.3873])
        self.BODY_TO_CAM0 = np.array(
        [[0.9999763379093255, -0.004079205042965442, -0.005539287650170447, -0.008977668364731128],
            [-0.004066386342107199, -0.9999890330121858, 0.0023234365646622014, 0.07557012320238939],
            [-0.00554870467502187, -0.0023008567036498766, -0.9999819588046867, -0.005545773942541918],
            [0.0, 0.0, 0.0, 1.0]])

        # camera 

        # init transformation
        transformation =  np.array(
            [[1.,  0.0, 0., -0.2],
            [0.,0.9553365, -0.2955202, 0.2],
            [0.,  0.2955202,  0.9553365, 0.0],
            [0.0, 0.0, 0.0, 1.0]])
        

        # np.array(
        # [[1., -0.0000000,  0.0, -0.],
        #     [0.0,  1., 0.0, 0.],
        #     [0.0,  0.0,  1., -0.0],
        #     [0.0, 0.0, 0.0, 1.0]])
        
        # np.array(
        # [[1.,  0.0, 0., -0.2],
        #     [0.,0.9553365, -0.2955202, -0.1],
        #     [0.,  0.2955202,  0.9553365, -0.1],
        #     [0.0, 0.0, 0.0, 1.0]])
        
        # np.array(
        # [[1., -0.0000000,  0.0, -0.],
        #     [0.0,  1., 0.0, 0.],
        #     [0.0,  0.0,  1., -0.0],
        #     [0.0, 0.0, 0.0, 1.0]])
        
        # np.array(
        # [[1.,  0.0, 0., -0.2],
        #     [0.,0.9553365, -0.2955202, -0.1],
        #     [0.,  0.2955202,  0.9553365, -0.1],
        #     [0.0, 0.0, 0.0, 1.0]])
        
        # np.array(
        # [[-0.99859172,  0.00570201, -0.05274519, 0.36],
        #     [0.02287402, -0.85077482, -0.52503237, 0.],
        #     [-0.04786802, -0.52549948,  0.84944626, 0.],
        #     [0.0, 0.0, 0.0, 1.0]])
# [  1.0000000,  0.0000000,  0.0000000;
#    0.0000000,  0.9610555, -0.2763557;
#    0.0000000,  0.2763557,  0.9610555 ]      
        # extract rotation, translation from transformation matrix
        self.translation = transformation[0:-1, 3]
        rotation = np.array([  0.9999500, -0.0000000,  0.0099998,
   0.0036161,  0.9323273, -0.3615974,
  -0.0093231,  0.3616154,  0.9322807 ]).reshape(3,3)
        
        
        
        # transformation[0:-1, 0:3]
        rotation, _ = cv2.Rodrigues(rotation)
        self.rotation = rotation


    def draw_lidar_to_canvas(self, projected_points: np.ndarray, z_values = None) -> None:
        def rgb_to_four_bits_per_color_string(red, green, blue):
            # Convert the RGB values to hexadecimal strings and remove the '0x' prefix
            red_hex = format(red, '02x')
            green_hex = format(green, '02x')
            blue_hex = format(blue, '02x')

            # Combine the hexadecimal values to form the four bits per color string
            color_string = red_hex.upper() + green_hex.upper() + blue_hex.upper()
            return color_string

        """
        Draws the projected LiDAR points to the canvas with a tkinter rectangle
        @param projected_points: nx2 array of LiDAR points projected to the camera
        @param color: tkinter string of the fill color.
                      Check: http://www.science.smith.edu/dftwiki/index.php/Color_Charts_for_TKinter for colors
        """
        if isinstance(z_values, np.ndarray):
            z_values = z_values.reshape(-1,1)
            z_values = (255* (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-5) ).astype(np.uint8)
            # (255* (z_values ) / (np.max(z_values)+ 1e-5) ).astype(np.uint8)
            # (255* (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-5) ).astype(np.uint8)
            z_values = cv2.applyColorMap(z_values, cv2.COLORMAP_JET)
        else:
            z_values = np.ones(len(projected_points)).reshape(-1,1)
            z_values = (255* (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-5) ).astype(np.uint8)
            z_values = cv2.applyColorMap(z_values, cv2.COLORMAP_JET)
        max_depth = np.max(z_values)
        min_depth = np.min(z_values)
        print("draw pointcloud length", len(projected_points))

        for count, point in enumerate(projected_points):
            x = point[0]
            y = point[1]
            z = z_values[count][0] # use to decide color
            color_string = rgb_to_four_bits_per_color_string(z[0],z[1],z[2])
            color_string = '#'+color_string
            point = self.canvas.create_rectangle(x - 1, y - 1, x + 1, y + 1, fill=color_string, width=0)
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
                               cameraMatrix = self.K, distCoeffs = self.distortion)
                            #    rvec = self.rotation,
                            #    tvec = cv2.UMat(self.translation),
                            #    useExtrinsicGuess = True)

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


        if self.button_pressed:
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
                                width=img_shape[1]+180,
                                height=img_shape[0] + 80)

        self.root.bind(
            '<Button 1>', self._canvas_press
        )  # register callback to handle left mouse clicks on the canvas

        # init state
        self.state_label = tk.Label(self.root, text="Select a camera point")
        self.status_label = tk.Label(self.root, text="0 point pairs selected (minimum 10)")
        self.state_label.place(x=40, y=50)
        self.status_label.place(x=300, y=50)
        self.state = States.CAMERA_SELECTION

        self.main_button = tk.Button(self.canvas, text="Continue")
        self.main_button.bind("<ButtonPress>", self._button_press)
        self.main_button.bind("<ButtonRelease>", self._button_release)
        self.main_button.place(x=600, y=50)
        self.main_button.config(state=tk.DISABLED)

        self.display_image()

        points = np.array(self.pointcloud[:, :3])
        mask_behind = points[:, -1] > 0
        points = points[mask_behind]
        print('forwad points',np.shape(points))
        self.lidar_points_3d = points
        z_value = self.lidar_points_3d[:,-1]
        projected_points, _ = cv2.projectPoints(points, self.rotation, self.translation, self.K, self.distortion)

        projected_points = np.squeeze(projected_points)
        self.projected_points = projected_points

        self.nearest_neighbor = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(projected_points)

        self.draw_lidar_to_canvas(projected_points, z_value)

        self.root.mainloop()