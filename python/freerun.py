import numpy as np
import open3d as o3d
import cv2
import os
import sys
from sys import platform
from harvesters.core import Harvester
import time

def display_pointcloud(pointcloud_comp):
    if pointcloud_comp.width == 0 or pointcloud_comp.height == 0:
        print("PointCloud is empty!")
        return
    
    # Reshape for Open3D visualization to N x 3 arrays
    pointcloud = pointcloud_comp.data.reshape(pointcloud_comp.height * pointcloud_comp.width, 3).copy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.visualization.draw_geometries([pcd], width=800,height=600)
    return

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
            pcd = o3d.geometry.PointCloud()
            
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
                    pointcloud_1st_edition[np.logical_or(pointcloud_1st_edition[: , 0] < 48.5 , pointcloud_1st_edition[: , 0] > 59.5)] = 0 #Filt with X position data
                    pointcloud_1st_edition[np.logical_or(pointcloud_1st_edition[: , 1] < 131.5 , pointcloud_1st_edition[: , 1] > 136.5)] = 0 #Filt with Y position data
                    pointcloud_1st_edition[np.logical_or(pointcloud_1st_edition[: , 2] < 88.5 , pointcloud_1st_edition[: , 2] > 104.5)] = 0 #Filt with Z position data
                    
                    # filtered_points = pointcloud[np.any(pointcloud != 0, axis=1)]
                    first_filtered_points = np.sum(np.any(pointcloud_1st_edition != 0, axis=1))
                    print("There are:", first_filtered_points, "points after 1st filtration")
                    
                    #2nd filtration using Greyscale Channel Information
                    texture_greyscale = texture_rgb
                    texture_greyscale[np.all(pointcloud_1st_edition == 0, axis = 1)] = 0
                    texture_greyscale[(texture_greyscale[: , 0] > 400)] = 0 #Filt with greyscale channel information
                    second_filtered_points = np.sum(np.any(texture_greyscale != 0, axis=1))
                    print("There are:", second_filtered_points , "points after 2nd filtration")
                    
                    #Display the selected points
                    pointcloud_2nd_edition = pointcloud_1st_edition
                    pointcloud_2nd_edition[np.all(texture_grey_component == 0, axis=1)] = 0
                    
                    
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