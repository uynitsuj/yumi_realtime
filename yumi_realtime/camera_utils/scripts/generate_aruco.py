import cv2
import numpy as np
import os

def generate_aruco_tags(dictionary=cv2.aruco.DICT_6X6_250, tag_ids=range(10), size=200):
    """
    Generate and save ArUco tags as PNG files
    
    Args:
        dictionary: ArUco dictionary type
        tag_ids: List of tag IDs to generate
        size: Size of output image in pixels
    """
    # Create output directory
    os.makedirs('aruco_tags', exist_ok=True)
    
    # Get ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    
    # Generate each tag
    for tag_id in tag_ids:
        # Create tag
        tag = np.zeros((size, size, 1), dtype=np.uint8)
        cv2.aruco.generateImageMarker(aruco_dict, tag_id, size, tag, 1)
        
        # Save tag
        filename = f'aruco_tags/aruco_tag_{tag_id}.png'
        cv2.imwrite(filename, tag)
        
if __name__ == '__main__':
    # Generate 10 tags using 6x6 dictionary
    generate_aruco_tags()
    
    # Optional: Generate with different parameters
    # generate_aruco_tags(dictionary=cv2.aruco.DICT_5X5_250, tag_ids=range(5), size=300)