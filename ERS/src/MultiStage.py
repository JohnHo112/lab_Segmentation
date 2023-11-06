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
        
    def merge_adjacent_regions(self, L, Thresholds, Filter):
        adjacent = Tool.adjacent_regions(self.R, self.regions)
        border = Tool.find_border(self.R, self.regions)
        meanycbcr = Tool.compute_ycbcr_mean(self.regions, self.ycbcr)
        centers = Tool.compute_center(self.regions)

        sobelgx = Tool.gradient(self.ycbcr["y"], Filter["sobelx"])
        sobelgy = Tool.gradient(self.ycbcr["y"], Filter["sobely"]) 
        sobelg = (sobelgx**2+sobelgy**2)*(0.5)   
        laplaceg = Tool.gradient(self.ycbcr["y"], Filter["laplace"])

        regions_to_merge = []

        for r, pixels in self.regions.items():
            Atx = Tool.compute_texture(self.regions, sobelgx, r, 0.5)
            Aty = Tool.compute_texture(self.regions, sobelgy, r, 0.5)
            A = {"m": centers[r][0], "n": centers[r][1], "y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"], "tx": Atx, "ty": Aty} 
            t = Thresholds["distance"]
            
            for adj in adjacent[r]["adj_regions"]:
                Btx = Tool.compute_texture(self.regions, sobelgx, adj, 0.5)
                Bty = Tool.compute_texture(self.regions, sobelgy, adj, 0.5)
                B = {"m": centers[adj][0], "n": centers[adj][1], "y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"], "tx": Btx, "ty": Bty} 
                meanSobel = Tool.border_gradient(border[r][adj], sobelg, True)
                meanLaplace = Tool.border_gradient(border[r][adj], laplaceg, True)

                dist = Tool.merge_adjacent_regions_distance(A, B, L, meanSobel, meanLaplace)

                if (min(len(pixels), len(self.regions[adj])) > Thresholds["area"]) or (min(Atx, Btx) > Thresholds["tx"]) or (min(Aty, Bty) > Thresholds["ty"]):
                    t = t*1.5
                if dist < t:
                    regions_to_merge.append((r, adj))

        Tool.merge_regions(regions_to_merge, self.R, self.regions)

    def stage(self, L, Threshold, Filter):
        for _ in range(1):
            self.merge_adjacent_regions(L, Threshold, Filter)
        
    def run(self):
        for i in range(1, len(self.L)+1):
            self.stage(self.L[i], self.Thresholds[i], self.Filters[i])
            print(f"merge regions num1: {len(self.regions)}")
            Tool.min_pixels(self.regions)
            sortedRegions, r = Tool.sort_region(self.regions)
            Tool.show_segamented_image(self.image, sortedRegions, r[0:16])

        return self.regions
        
        