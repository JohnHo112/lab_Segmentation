import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from Watershed import watershed
from MultiStage import MultiStageMerge
import Tool
from config import Filters, Thresholds, L
from FastSegment_RGB import fast_segmentation


def main():
    # ground truth and resize to target
    path = "Groundtruth_label_image/lena_label.jpeg"
    path = "Groundtruth_label_image/peppers_label.jpeg"
    #path = "Groundtruth_label_image/baboon_label.jpeg"
    groundTruthImage = cv2.imread(path)
    groundTruthImage = cv2.resize(groundTruthImage, (128, 128))
    groundTruthImage = cv2.cvtColor(groundTruthImage, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(groundTruthImage)
    plt.title("ground truth image")
    R, groundTruthRegions = fast_segmentation(groundTruthImage, 5, 50)
    groundTruthRegions = Tool.convertRegions(groundTruthRegions)
    print(f"ground truth regions num: {len(groundTruthRegions)}")
    sortedRegions, r = Tool.sort_region(groundTruthRegions)
    Tool.show_segamented_image(groundTruthImage, sortedRegions, r[0:16])

    # read target image
    path = "Pic/Lena_256bmp.bmp"
    path = "Pic/peppers_s.bmp"
    #path = "Pic/baboon2.jpg"
    image = cv2.imread(path)
    image = cv2.resize(image, (128, 128))
    image = cv2.flip(image, 1)
    # image = image[:10, :10, :10]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    M, N, O = image.shape
    # plt.figure()
    # plt.imshow(image)
    # plt.title("target image")
    # convert rgb to ycbcr
    ycbcr = Tool.RGB_to_ycbcr(image)

    # watershed algorithm
    start = time.time()
    regions = watershed(ycbcr["y"], 30).run()
    end = time.time()
    print(f"Watershed time: {end-start}")
    print(f"Watershed regions num: {len(regions)}")

    # convert the results
    R = Tool.regions_to_R(regions, M, N)
    regions, r = Tool.sort_region(regions)

    # show results
    
    Tool.show_segamented_image(image, regions, r[0:16])
    # plt.figure()
    # plt.imshow(R)

    # multi-stage merge
    start = time.time()
    regions1 = MultiStageMerge(image, regions.copy(), ycbcr, L, Thresholds, Filters).run()
    end = time.time()
    print(f"merge time: {end-start}")
    sortedRegions, r = Tool.sort_region(regions1)
    # Tool.show_segamented_image(image, sortedRegions, r[0:16])

    # convert the results
    R1 = Tool.regions_to_R(regions1, M, N)
    plt.figure()
    plt.imshow(R1)

    print(f"IOU: {Tool.IOU(regions1, groundTruthRegions, 128*128)}")
    plt.show()
    
if __name__ == "__main__":
    main()