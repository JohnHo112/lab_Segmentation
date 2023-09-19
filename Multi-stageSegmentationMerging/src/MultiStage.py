from scipy import ndimage
import Tool
import matplotlib.pyplot as plt

def process_over_segmentation(R, regions, ycbcr, L, delta, threshold, sxfilter, syfilter, lfilter):
    def distance1(A, B, L):
        return (L["l"]*(A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2)**(1/2)
    
    def distance2(A, B, L, sobelMean, laplaceMean):
        return (L["l1"]*(A["y"]-B["y"])**2+(A["cb"]-B["cb"])**2+(A["cr"]-B["cr"])**2+L["l2"]*sobelMean+L["l3"]*laplaceMean+L["lt"]*((A["tx"]-B["tx"])**2+(A["ty"]-B["ty"])**2))*(0.5)
    
    def border_gradient(border, gradient):
        g = 0
        for m, n in border:
            g += abs(gradient[m, n])
        g = g/len(border)
        return g
    
    def compute_texture(regions, g, r, a):
        texture = 0
        l = len(regions[r])
        for m, n in regions[r]:
            texture += abs(g[m, n])
        texture = (texture/l)**a
        return texture

    def process_small_regions(R, regions, meanycbcr, delta):
        adjacent = Tool.adjacent_regions(R, regions)
        regions_to_merge = []
        for r, pixels in regions.items():
            if len(pixels) < delta:
                A = {"y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"]}
                minDiff = 10000000
                minAdj = -1
                for adj in adjacent[r]["adj_regions"]:
                    B = {"y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"]}
                    dist = distance1(A, B, L)
                    if dist < minDiff:
                        minDiff = dist
                        minAdj = adj
                if minAdj != -1:
                    regions_to_merge.append((r, minAdj))

        for r, minAdj in regions_to_merge:
            if r in regions and minAdj in regions:
                if r == minAdj:
                    continue
                Tool.merge(R, regions, r, minAdj)
                for i in range(len(regions_to_merge)):
                    if regions_to_merge[i][0] == minAdj:
                        regions_to_merge[i] = (r, regions_to_merge[i][1])
                    if regions_to_merge[i][1] == minAdj:
                        regions_to_merge[i] = (regions_to_merge[i][0], r)
        
    def merge_adjacent_regions(R, regions, ycbcr, meanycbcr, threshold, sxfilter, syfilter, lfilter):
        adjacent = Tool.adjacent_regions(R, regions)
        border = Tool.find_border(R, regions)
        sobelgx = Tool.gradient(ycbcr["y"], sxfilter)
        sobelgy = Tool.gradient(ycbcr["y"], syfilter)
        sobelg = (sobelgx**2+sobelgy**2)*(0.5)
        laplaceg = Tool.gradient(ycbcr["y"], lfilter)
        regions_to_merge = []
        
        arearecord = []
        textrecordx = []
        textrecordy = []
        distrecord = []

        for r, pixels in regions.items():
            A = {"y": meanycbcr[r]["y"], "cb": meanycbcr[r]["cb"], "cr": meanycbcr[r]["cr"], "tx": compute_texture(regions, sobelgx, r, 0.5), "ty": compute_texture(regions, sobelgy, r, 0.5)} 
            t = threshold
            for adj in adjacent[r]["adj_regions"]:
                B = {"y": meanycbcr[adj]["y"], "cb": meanycbcr[adj]["cb"], "cr": meanycbcr[adj]["cr"], "tx": compute_texture(regions, sobelgx, adj, 0.5), "ty": compute_texture(regions, sobelgy, adj, 0.5)} 
                arearecord.append(min(len(regions[r]), len(regions[r])))
                textrecordx.append(min(A["tx"], B["tx"]))
                textrecordy.append(min(A["ty"], B["ty"]))
                # if min(len(regions[r]), len(regions[r])) > 300 or min(A["tx"], B["tx"]) > 8 or min(A["ty"], B["ty"]) > 8:
                #     t = 1.5*t
                meanSobel = border_gradient(border[r][adj], sobelg)
                meanLaplace = border_gradient(border[r][adj], laplaceg)
                dist = distance2(A, B, L, meanSobel, meanLaplace)
                distrecord.append(dist)
                if dist < t:
                    regions_to_merge.append((r, adj))
        # plt.figure()
        # plt.plot(arearecord)
        # plt.title("record area")
        # plt.figure()
        # plt.plot(textrecordx)
        # plt.title("recordx")
        # plt.figure()
        # plt.plot(textrecordy)
        # plt.title("recordy")
        # plt.figure()
        # plt.plot(distrecord)
        # plt.title("recorddist")
        # plt.figure()


        for r, adj in regions_to_merge:
            if r in regions and adj in regions:
                if r == adj:
                    continue
                Tool.merge(R, regions, r, adj)
                for i in range(len(regions_to_merge)):
                    if regions_to_merge[i][0] == adj:
                        regions_to_merge[i] = (r, regions_to_merge[i][1])
                    if regions_to_merge[i][1] == adj:
                        regions_to_merge[i] = (regions_to_merge[i][0], r)

    meanycbcr = Tool.compute_ycbcr_mean(regions, ycbcr)
    for _ in range(10):
        process_small_regions(R, regions, meanycbcr, delta)
        meanycbcr = Tool.compute_ycbcr_mean(regions, ycbcr)
    merge_adjacent_regions(R, regions, ycbcr, meanycbcr, threshold, sxfilter, syfilter, lfilter)

    return R, regions   
