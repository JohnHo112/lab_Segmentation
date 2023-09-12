import matplotlib.pyplot as plt
import time
import Tool, OverSegmentation, Watershed

def main():
    # read image
    path = "Pic/Lena_256bmp.bmp"
    image = plt.imread(path)
    image = image[:128, 128:]
    M, N, O = image.shape

    # rgb image to y image
    ycbcr = Tool.RGB_to_ycbcr(image)
    plt.imshow(image)
    plt.title("Image")
    plt.figure()

    # watershed
    start = time.time()
    w = Watershed.Watershed(ycbcr["y"], 3)
    regions = w.run()
    end = time.time()
    watershedTime = end-start
    watershedRegionsNum = len(regions)
    # convert result
    regions, r = Tool.sort_region(regions)
    R = Tool.regions_to_R(regions, M, N)
    # show the result plot
    Tool.show_segamented_image(image, regions, r[0:16])

    # # processing watershed over segmentation
    # start = time.time()
    # R, regions = OverSegmentation.process_over_segmentation(R, regions, ycbcr, 100, 1)
    # end = time.time()
    # processOverSegmentationTime1 = end-start
    # processOverSegmentationRegionsNum1 = len(regions)
    # # # convert result
    # regions, r = Tool.sort_region(regions)
    # R = Tool.regions_to_R(regions, M, N)
    # # show the result plot
    # Tool.show_segamented_image(image, regions, r[0:16])

    # processing watershed over segmentation border
    start = time.time()
    R, regions = OverSegmentation.process_over_segmentation_adv(R, regions, ycbcr, 100, 1)
    end = time.time()
    processOverSegmentationTime2 = end-start
    processOverSegmentationRegionsNum2 = len(regions)
    # convert result
    regions, r = Tool.sort_region(regions)
    R = Tool.regions_to_R(regions, M, N)
    # show the result plot
    Tool.show_segamented_image(image, regions, r[0:16])


    # show the efficiency
    print(f"watershed time: {watershedTime}")
    # print(f"process over segmetation time: {processOverSegmentationTime1}")
    print(f"process over segmetation time: {processOverSegmentationTime2}")
    print(f"watershed regions num: {watershedRegionsNum}")
    # print(f"process over segmetation regions num: {processOverSegmentationRegionsNum1}")
    print(f"process over segmetation regions num: {processOverSegmentationRegionsNum2}")
    plt.show()
    

if __name__ == '__main__':
    main()