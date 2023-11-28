import numpy as np
import open3d as o3d
import cv2
import os
import sys
from sys import platform
from harvesters.core import Harvester
import threading
import time

def freerun():
    # PhotoneoTL_DEV_<ID>
    device_id = "PhotoneoTL_DEV_IDV-016"
    print("--> device_id: ", device_id)

    if platform == "win32":
        cti_file_path_suffix = "/API/bin/photoneo.cti"
        save_last_scan_path_prefix = "C:/Users/Public"
    else:
        cti_file_path_suffix = "/API/lib/photoneo.cti"
        save_last_scan_path_prefix = "~"
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + cti_file_path_suffix
    print("--> cti_file_path: ", cti_file_path)

    with Harvester() as h:
        h.add_file(cti_file_path, True, True)
        h.update()

        # Print out available devices
        print()
        print("Name : ID")
        print("---------")
        for item in h.device_info_list:
            print(item.property_dict['serial_number'], ' : ', item.property_dict['id_'])
        print()

        with h.create({'id_': device_id}) as ia:
            features = ia.remote_device.node_map

            #print(dir(features))
            print("TriggerMode BEFORE: ", features.PhotoneoTriggerMode.value)
            features.PhotoneoTriggerMode.value = "Freerun"
            print("TriggerMode AFTER: ", features.PhotoneoTriggerMode.value)
            
            # Order is fixed on the selected output structure. Disabled fields are shown as empty components.
            # Individual structures can enabled/disabled by the following features:
            # SendTexture, SendPointCloud, SendNormalMap, SendDepthMap, SendConfidenceMap, SendEventMap, SendColorCameraImage
            # payload.components[#]
            # [0] Texture
            # [1] TextureRGB
            # [2] PointCloud [X,Y,Z,...]
            # [3] NormalMap [X,Y,Z,...]
            # [4] DepthMap
            # [5] ConfidenceMap
            # [6] EventMap
            # [7] ColorCameraImage

            features.SendTexture.value = True
            features.SendPointCloud.value = True
            features.SendNormalMap.value = False
            features.SendDepthMap.value = True
            features.SendConfidenceMap.value = False

            ia.start()
            # pcd = o3d.geometry.PointCloud()
            
            while True:
                with ia.fetch(timeout=10.0) as buffer:
                    # grab newest frame
                    # do something with second frame
                    payload = buffer.payload
                    
                    point_cloud_component = payload.components[2]
                    pointcloud = np.zeros((point_cloud_component.height * point_cloud_component.width, 3))
                    if point_cloud_component.width > 0 and point_cloud_component.height > 0:
                        pointcloud = point_cloud_component.data.reshape(point_cloud_component.height * point_cloud_component.width, 3).copy()
                        # print(pointcloud)
                    else:
                        print("PointCloud is empty!")
                    # pcd.points = o3d.utility.Vector3dVector(pointcloud)
                    
                    texture_grey_component = payload.components[0]
                    texture_rgb_component = payload.components[1]
                    texture = np.zeros((texture_grey_component.height, texture_grey_component.width))
                    texture_rgb = np.zeros((point_cloud_component.height * point_cloud_component.width, 3))
                    if texture_grey_component.width > 0 and texture_grey_component.height > 0:
                        texture = texture_grey_component.data.reshape(texture_grey_component.height, texture_grey_component.width, 1).copy()
                        # texture_rgb[:, 0] = np.reshape(1/65536 * texture, -1)
                        # texture_rgb[:, 1] = np.reshape(1/65536 * texture, -1)
                        # texture_rgb[:, 2] = np.reshape(1/65536 * texture, -1)
                        
                        texture_rgb[:, 0] = np.reshape(texture, -1)
                        texture_rgb[:, 1] = np.reshape(texture, -1)
                        texture_rgb[:, 2] = np.reshape(texture, -1)
                    elif texture_rgb_component.width > 0 and texture_rgb_component.height > 0:
                        texture = texture_rgb_component.data.reshape(texture_rgb_component.height, texture_rgb_component.width, 3).copy()
                        texture_rgb[:, 0] = np.reshape(1/65536 * texture[:, :, 0], -1)
                        texture_rgb[:, 1] = np.reshape(1/65536 * texture[:, :, 1], -1)
                        texture_rgb[:, 2] = np.reshape(1/65536 * texture[:, :, 2], -1)
                    else:
                        print("TextureRGB is empty!")
                        
                    #First filtration using Point Cloud Data
                    pointcloud_1st_edition = pointcloud
                    pointcloud_1st_edition[np.logical_or(pointcloud_1st_edition[: , 0] < 45 , pointcloud_1st_edition[: , 0] > 65)] = 0 #Filt with X position data
                    pointcloud_1st_edition[np.logical_or(pointcloud_1st_edition[: , 1] < 130 , pointcloud_1st_edition[: , 1] > 140)] = 0 #Filt with Y position data
                    pointcloud_1st_edition[np.logical_or(pointcloud_1st_edition[: , 2] < 85, pointcloud_1st_edition[: , 2] > 110)] = 0 #Filt with Z position data
                    
                    # filtered_points = pointcloud[np.any(pointcloud != 0, axis=1)]
                    first_filtered_points = np.sum(np.any(pointcloud_1st_edition != 0, axis=1))
                    print("There are:", first_filtered_points, "points after 1st filtration")
                    
                    #2nd filtration using Greyscale Channel Information
                    texture_greyscale_1st = texture_rgb
                    texture_greyscale_1st[np.all(pointcloud_1st_edition == 0, axis = 1)] = 0
                    
                    texture_greyscale_2nd = texture_greyscale_1st
                    texture_greyscale_2nd[(texture_greyscale_2nd[: , 0] > 300)] = 0 #Filt with greyscale channel information
                    second_filtered_points = np.sum(np.any(texture_greyscale_2nd != 0, axis=1))
                    print("There are:", second_filtered_points , "points after 2nd filtration")
                    
                    #Visualize the filtered points
                    pointcloud_2nd_edition = pointcloud_1st_edition
                    pointcloud_2nd_edition[np.all(texture_greyscale_2nd == 0, axis=1)] = 0
                    
                    # pcd1 = o3d.geometry.PointCloud()
                    # pcd1.points = o3d.utility.Vector3dVector(pointcloud_1st_edition)
                    # texture_visual = cv2.normalize(1/65536 * texture_greyscale_1st, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    # pcd1.colors = o3d.utility.Vector3dVector(texture_visual)
                    # o3d.visualization.draw_geometries([pcd1], width=800,height=600)
                    
                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector(pointcloud_2nd_edition)
                    texture_visual = cv2.normalize(1/65536 * texture_greyscale_2nd, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    pcd2.colors = o3d.utility.Vector3dVector(texture_visual)
                    o3d.visualization.draw_geometries([pcd2], width=800,height=600)
                    
                    #Select 4 vertexes
                    x_max = np.max(pointcloud_2nd_edition[: , 0])
                    z_max = np.max(pointcloud_2nd_edition[: , 2])
                    
                    pointcloud_non_zero_x = pointcloud_2nd_edition[pointcloud_2nd_edition[: , 0] != 0][: , 0]
                    x_min = np.min(pointcloud_non_zero_x)
                    pointcloud_non_zero_z = pointcloud_2nd_edition[pointcloud_2nd_edition[: , 2] != 0][: , 2]
                    z_min = np.min(pointcloud_non_zero_z)
                    print("x_max:", x_max, "\nx_min:", x_min, "\nz_max:", z_max, "\nz_min", z_min)
                    
                    point_x_max = pointcloud_2nd_edition[pointcloud_2nd_edition[: , 0] == x_max]
                    point_x_min = pointcloud_2nd_edition[pointcloud_2nd_edition[: , 0] == x_min]
                    point_z_max = pointcloud_2nd_edition[pointcloud_2nd_edition[: , 2] == z_max]
                    point_z_min = pointcloud_2nd_edition[pointcloud_2nd_edition[: , 2] == z_min]
                    print("Point_x_max:", point_x_max, "\nPoint_x_min:", point_x_min, "\nPoint_z_max:", point_z_max, "\nPoint_z_min:", point_z_min)
                    
                    
                    # max_value = np.max(texture_rgb)
                    # print(max_value)
                    
                    # index_of_max = np.argmax(texture_rgb)
                    # row = index_of_max // texture_rgb.shape[1]
                    # col = index_of_max % texture_rgb.shape[1]
                    
                    # print("Position at row:", row, "col:", col)
                    # print(texture.dtype)
                    
                    
                    time.sleep(100)


if __name__ == "__main__":
    freerun()