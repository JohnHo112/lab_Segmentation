import Tool
import matplotlib.pyplot as plt


class MultiStageMerge:
    def __init__(self, image, regions, ycbcr, L, Thresholds, Filters):
        self.image = image
        self.regions = regions
        self.ycbcr = ycbcr
        self.M, self.N = ycbcr["y"].shape
        self.R = Tool.regions_to_R(regions, self.M, self.N)
        self.L = L
        self.Thresholds = Thresholds
        self.Filters = Filters
    
    def process_small_regions(self, L, delta):
        adjacent = Tool.adjacent_regions(self.R, self.regions)
        meanycbcr = Tool.compute_ycbcr_mean(self.regions, self.ycbcr)
        regions_to_merge = []
        for r, pixels in self.regions.items():
            if len(pixels) < delta:
                A = {"y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"]}
                minDiff = 10000000
                minAdj = -1
                for adj in adjacent[r]["adj_regions"]:
                    B = {"y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"]}
                    dist = Tool.process_small_regions_distance(A, B, L)
                    if dist < minDiff:
                        minDiff = dist
                        minAdj = adj
                if minAdj != -1:
                    regions_to_merge.append((r, minAdj))

        for r, minAdj in regions_to_merge:
            if r in self.regions and minAdj in self.regions:
                if r == minAdj:
                    continue
                Tool.merge(self.R, self.regions, r, minAdj)
                for i in range(len(regions_to_merge)):
                    if regions_to_merge[i][0] == minAdj:
                        regions_to_merge[i] = (r, regions_to_merge[i][1])
                    if regions_to_merge[i][1] == minAdj:
                        regions_to_merge[i] = (regions_to_merge[i][0], r)
        
    def merge_adjacent_regions(self, L, distance, Filter):
        adjacent = Tool.adjacent_regions(self.R, self.regions)
        border = Tool.find_border(self.R, self.regions)
        sobelgx = Tool.gradient(self.ycbcr["y"], Filter["sobelx"])
        sobelgy = Tool.gradient(self.ycbcr["y"], Filter["sobely"])
        sobelg = (sobelgx**2+sobelgy**2)*(0.5)
        laplaceg = Tool.gradient(self.ycbcr["y"], Filter["laplace"])
        meanycbcr = Tool.compute_ycbcr_mean(self.regions, self.ycbcr)
        regions_to_merge = []

        distRecord = []

        for r, pixels in self.regions.items():
            A = {"y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"], "tx": Tool.compute_texture(self.regions, sobelgx, r, 0.5), "ty": Tool.compute_texture(self.regions, sobelgy, r, 0.5)} 
            t = distance
            for adj in adjacent[r]["adj_regions"]:
                B = {"y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"], "tx": Tool.compute_texture(self.regions, sobelgx, adj, 0.5), "ty": Tool.compute_texture(self.regions, sobelgy, adj, 0.5)} 
                meanSobel = Tool.border_gradient(border[r][adj], sobelg)
                meanLaplace = Tool.border_gradient(border[r][adj], laplaceg)
                dist = Tool.merge_adjacent_regions_distance(A, B, L, meanSobel, meanLaplace)
                distRecord.append(dist)
                if dist < t:
                    regions_to_merge.append((r, adj))
        # plt.figure()
        # plt.plot(distRecord)

        for r, adj in regions_to_merge:
            if r in self.regions and adj in self.regions:
                if r == adj:
                    continue
                Tool.merge(self.R, self.regions, r, adj)
                for i in range(len(regions_to_merge)):
                    if regions_to_merge[i][0] == adj:
                        regions_to_merge[i] = (r, regions_to_merge[i][1])
                    if regions_to_merge[i][1] == adj:
                        regions_to_merge[i] = (regions_to_merge[i][0], r)

    def stage(self, L, Threshold, Filter):
        for _ in range(10):
            self.process_small_regions(L, Threshold["delta"])
        for _ in range(10):
            self.merge_adjacent_regions(L, Threshold["distance"], Filter)
        

    def run(self):
        for i in range(1, len(self.L)+1):
            self.stage(self.L[i], self.Thresholds[i], self.Filters[i])
            print(f"merge regions num1: {len(self.regions)}")
            # sortedRegions, r = Tool.sort_region(self.regions)
            # Tool.show_segamented_image(self.image, sortedRegions, r[0:16])
        return self.regions
        
        