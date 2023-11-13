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
            features.SendDepthMap.value = False
            features.SendConfidenceMap.value = False

            ia.start()
            
            # vis = o3d.visualization.Visualizer()
            # vis.create_window(width=800,height=600)
            pcd = o3d.geometry.PointCloud()
            # vis.add_geometry(pcd)
            
            while True:
                with ia.fetch(timeout=10.0) as buffer:
                    # grab newest frame
                    # do something with second frame
                    payload = buffer.payload
                    
                    point_cloud_component = payload.components[2]
                    if point_cloud_component.width > 0 and point_cloud_component.height > 0:
                        pointcloud = point_cloud_component.data.reshape(point_cloud_component.height * point_cloud_component.width, 3).copy()
                        print(pointcloud)
                    else:
                        print("PointCloud is empty!")
                    # pcd.points = o3d.utility.Vector3dVector(pointcloud)
                    
                    texture_rgb_component = payload.components[1]
                    if texture_rgb_component.width > 0 and texture_rgb_component.height > 0:
                        texture_rgb = np.zeros((point_cloud_component.height * point_cloud_component.width, 3))
                        texture = texture_rgb_component.data.reshape(texture_rgb_component.height, texture_rgb_component.width, 3).copy()
                        
                        texture_rgb[:, 0] = np.reshape(1/65536 * texture[:, :, 0], -1)
                        texture_rgb[:, 1] = np.reshape(1/65536 * texture[:, :, 1], -1)
                        texture_rgb[:, 2] = np.reshape(1/65536 * texture[:, :, 2], -1)
                        
                        texture_rgb = cv2.normalize(texture_rgb, dst=None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        print(texture_rgb)
                    else:
                        print("TextureRGB is empty!")
                    # pcd.colors = o3d.utility.Vector3dVector(texture_rgb)
                    
                    # vis.update_geometry(pcd)
                    # vis.poll_events()
                    # vis.update_renderer()
                    time.sleep(1/9)
                    # vis.clear_geometries
                    # time.sleep(0.5)

if __name__ == "__main__":
    freerun()