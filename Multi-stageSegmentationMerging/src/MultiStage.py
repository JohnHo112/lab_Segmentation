import Tool
import matplotlib.pyplot as plt
import cv2


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
        meanRGB = Tool.compute_RGB_mean(self.regions, self.image)

        regions_to_merge = [] 
        for r, pixels in self.regions.items():
            # using ycbcr to merge
            # if len(pixels) < delta:
            #     A = {"y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"]}
            #     minDiff = 10000000
            #     minAdj = -1
            #     for adj in adjacent[r]["adj_regions"]:
            #         B = {"y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"]}
            #         dist = Tool.process_small_regions_distance(A, B, L)
            #         if dist < minDiff:
            #             minDiff = dist
            #             minAdj = adj
            #     if minAdj != -1:
            #         regions_to_merge.append((r, minAdj))
            
            # using RGB to merge
            if len(pixels) < delta:
                A = {"R": meanRGB[r]["R"], "G": meanRGB[r]["G"], "B": meanRGB[r]["B"]}
                minDiff = 10000000
                minAdj = -1
                for adj in adjacent[r]["adj_regions"]:
                    B = {"R": meanRGB[adj]["R"], "G": meanRGB[adj]["G"], "B": meanRGB[adj]["B"]}
                    dist = Tool.color_distance(A, B)
                    if dist < minDiff:
                        minDiff = dist
                        minAdj = adj
                if minAdj != -1:
                    regions_to_merge.append((r, minAdj))

        Tool.merge_regions(regions_to_merge, self.R, self.regions)
        
    def merge_adjacent_regions(self, L, Thresholds, Filter):
        adjacent = Tool.adjacent_regions(self.R, self.regions)
        border = Tool.find_border(self.R, self.regions)
        meanycbcr = Tool.compute_ycbcr_mean(self.regions, self.ycbcr)

        sobelgx = Tool.gradient(self.ycbcr["y"], Filter["sobelx"])
        sobelgy = Tool.gradient(self.ycbcr["y"], Filter["sobely"])
        # sobelg = (sobelgx+sobelgy)   
        sobelg = (sobelgx**2+sobelgy**2)*(0.5)     

           
        laplaceg = Tool.gradient(self.ycbcr["y"], Filter["laplace"])

        regions_to_merge = []

        distRecord = []
        areaRecord = []
        texturexRecord = []
        textureyRecord = []

        sobelmeanRecord = []
        laplacemeanRecord = []
        yRecord = []
        cbRecord = []
        crRecord = []
        textureRecord = []

        for r, pixels in self.regions.items():
            Atx = Tool.compute_texture(self.regions, sobelgx, r, 0.5)
            Aty = Tool.compute_texture(self.regions, sobelgy, r, 0.5)
            A = {"y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"], "tx": Atx, "ty": Aty} 
            t = Thresholds["distance"]
            for adj in adjacent[r]["adj_regions"]:
                Btx = Tool.compute_texture(self.regions, sobelgx, adj, 0.5)
                Bty = Tool.compute_texture(self.regions, sobelgy, adj, 0.5)
                B = {"y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"], "tx": Btx, "ty": Bty} 
                meanSobel = Tool.border_gradient(border[r][adj], sobelg, True)
                meanLaplace = Tool.border_gradient(border[r][adj], laplaceg, True)

                sobelmeanRecord.append(meanSobel)
                laplacemeanRecord.append(meanLaplace)
                yRecord.append((A["y"]-B["y"])**2)
                cbRecord.append((A["cb"]-B["cb"])**2)
                crRecord.append((A["cr"]-B["cr"])**2)
                textureRecord.append((A["tx"]-B["tx"])**2+(A["ty"]-B["ty"])**2)
                

                dist = Tool.merge_adjacent_regions_distance(A, B, L, meanSobel, meanLaplace)
                distRecord.append(dist)
                areaRecord.append(min(len(pixels), len(self.regions[adj])))
                texturexRecord.append(min(Atx, Btx))
                textureyRecord.append(min(Aty, Bty))
                if (min(len(pixels), len(self.regions[adj])) > Thresholds["area"]) or (min(Atx, Btx) > Thresholds["tx"]) or (min(Aty, Bty) > Thresholds["ty"]):
                    t = t*1.5

                if dist < t:
                    regions_to_merge.append((r, adj))
        
        # plt.figure()
        # plt.title("sobelmean")
        # plt.plot(sobelmeanRecord)
        # plt.figure()
        # plt.title("laplacemean")
        # plt.plot(laplacemeanRecord)
        # plt.figure()
        # plt.title("y")
        # plt.plot(yRecord)
        # plt.figure()
        # plt.title("cb")
        # plt.plot(cbRecord)
        # plt.figure()
        # plt.title("cr")
        # plt.plot(crRecord)
        # plt.figure()
        # plt.title("textureRecord")
        # plt.plot(textureRecord)

        plt.figure()
        plt.title("dist")
        plt.plot(distRecord)
        plt.figure()
        plt.title("area")
        plt.plot(areaRecord)

        Tool.merge_regions(regions_to_merge, self.R, self.regions)

    def stage(self, L, Threshold, Filter):
        for _ in range(3):
            self.process_small_regions(L, Threshold["delta"])
        # for _ in range(5):
        self.merge_adjacent_regions(L, Threshold, Filter)
        
        

    def run(self):
        for i in range(1, len(self.L)+1):
            self.stage(self.L[i], self.Thresholds[i], self.Filters[i])
            print(f"merge regions num1: {len(self.regions)}")
            Tool.min_pixels(self.regions)
            sortedRegions, r = Tool.sort_region(self.regions)
            Tool.show_segamented_image(self.image, sortedRegions, r[0:16])

        return self.regions
        
        