import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from MultiStage import MultiStageMerge
import Tool
from config import Filters, Thresholds, L_multi
from FastSegment_RGB import fast_segmentation
import time

def main():
  # get gound truth
  path = "Groundtruth_label_image/lena_label.jpeg"
  path = "Groundtruth_label_image/baboon_label.jpeg"
  path = "Groundtruth_label_image/peppers_label.jpeg"
  groundTruthImage = cv2.imread(path)
  groundTruthImage = cv2.resize(groundTruthImage, (256, 256))
  groundTruthImage = cv2.cvtColor(groundTruthImage, cv2.COLOR_BGR2RGB)
  plt.figure()
  plt.imshow(groundTruthImage)
  plt.title("ground truth image")
  R, groundTruthRegions = fast_segmentation(groundTruthImage, 5, 50)
  groundTruthRegions = Tool.convertRegions(groundTruthRegions)
  print(f"ground truth regions num: {len(groundTruthRegions)}")
  sortedRegions, r = Tool.sort_region(groundTruthRegions)
  Tool.show_segamented_image(groundTruthImage, sortedRegions, r[0:16])

  path = "Pic/Lena256c.jpg"
  path = "Pic/baboon2.jpg"
  path = "Pic/peppers_s.bmp"
  image = cv2.imread(path)
  image = cv2.resize(image, (256, 256))
  image = cv2.flip(image, 1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  M, N, O = image.shape

  # convert rgb to ycbcr
  ycbcr = Tool.RGB_to_ycbcr(image)

  # get ers segment
  ers_segment = pd.read_csv("ERS\ers_segments\lena_ers.csv", header=None).to_numpy()
  ers_segment = pd.read_csv("ERS/ers_segments/baboon_ers.csv", header=None).to_numpy()
  ers_segment = pd.read_csv("ERS\ers_segments\peppers_ers.csv", header=None).to_numpy()
  ers_segment = np.flip(ers_segment, 1)
  regions = Tool.R_to_regions(ers_segment)

  # convert the results
  R = Tool.regions_to_R(regions, M, N)
  regions, r = Tool.sort_region(regions)

  # show results 
  Tool.show_segamented_image(image, regions, r[0:16])

  L = L_multi
  # print(L)

  # multi-stage merge
  start = time.time()
  regions1 = MultiStageMerge(image, regions.copy(), ycbcr, L, Thresholds, Filters).run()
  end = time.time()
  print(f"merge time: {end-start}")
  sortedRegions, r = Tool.sort_region(regions1)

  # convert the results
  R1 = Tool.regions_to_R(regions1, M, N)
  plt.figure()
  plt.imshow(R1)

  print(f"IOU: {Tool.IOU(regions1, groundTruthRegions, 256*256)}")
  plt.show()
  

if __name__ == "__main__":
  main()