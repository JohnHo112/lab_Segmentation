from scipy import ndimage
import Tool

def process_over_segmentation(R, regions, ycbcr, delta, threshold):
    def distance(Ay, Acb, Acr, By, Bcb, Bcr, l):
        return (l*(Ay-By)**2+(Acb-Bcb)**2+(Acr-Bcr)**2)**(1/2)

    def process_small_regions(R, regions, meanycbcr, delta):
        adjacent = Tool.adjacent_regions(R, regions)
        regions_to_merge = []
        for r, pixels in regions.items():
            if len(pixels) < delta:
                Ay, Acb, Acr = meanycbcr[r]["y"], meanycbcr[r]["cb"], meanycbcr[r]["cr"]
                minDiff = 10000000
                minAdj = -1
                for adj in adjacent[r]["adj_regions"]:
                    By, Bcb, Bcr = meanycbcr[adj]["y"], meanycbcr[adj]["cb"], meanycbcr[adj]["cr"]
                    dist = distance(Ay, Acb, Acr, By, Bcb, Bcr, 1)
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
                
    def merge_adjacent_regions(R, regions, meanycbcr, threshold):
        adjacent = Tool.adjacent_regions(R, regions)
        regions_to_merge = []
        for r, pixels in regions.items():
            Ay, Acb, Acr = meanycbcr[r]["y"], meanycbcr[r]["cb"], meanycbcr[r]["cr"]
            for adj in adjacent[r]["adj_regions"]:
                By, Bcb, Bcr = meanycbcr[adj]["y"], meanycbcr[adj]["cb"], meanycbcr[adj]["cr"]
                dist = distance(Ay, Acb, Acr, By, Bcb, Bcr, 0.8)
                if dist < threshold:
                    regions_to_merge.append((r, adj))

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
    merge_adjacent_regions(R, regions, meanycbcr, threshold)

    return R, regions

def process_over_segmentation_adv(R, regions, ycbcr, delta, threshold):
    def distance1(Ay, Acb, Acr, By, Bcb, Bcr, l):
        return (l*(Ay-By)**2+(Acb-Bcb)**2+(Acr-Bcr)**2)**(1/2)
    
    def distance2(Ay, Acb, Acr, By, Bcb, Bcr, l1, l2, l3, sobelMean, laplaceMean):
        return (l1*(Ay-By)**2+(Acb-Bcb)**2+(Acr-Bcr)**2+l2*sobelMean+l3*laplaceMean)*(0.5)
    
    def border_gradient(border, gradient):
        g = 0
        for m, n in border:
            g += gradient[m, n]
        g = g/len(border)
        return g

    def process_small_regions(R, regions, meanycbcr, delta):
        adjacent = Tool.adjacent_regions(R, regions)
        regions_to_merge = []
        for r, pixels in regions.items():
            if len(pixels) < delta:
                Ay, Acb, Acr = meanycbcr[r]["y"], meanycbcr[r]["cb"], meanycbcr[r]["cr"]
                minDiff = 10000000
                minAdj = -1
                for adj in adjacent[r]["adj_regions"]:
                    By, Bcb, Bcr = meanycbcr[adj]["y"], meanycbcr[adj]["cb"], meanycbcr[adj]["cr"]
                    dist = distance1(Ay, Acb, Acr, By, Bcb, Bcr, 1)
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
        
    def merge_adjacent_regions(R, regions, ycbcr, meanycbcr, threshold):
        adjacent = Tool.adjacent_regions(R, regions)
        border = Tool.find_border(R, regions)
        sobelg = Tool.image_gradient(ycbcr["y"], ndimage.sobel)
        laplaceg = ndimage.laplace(ycbcr["y"])
        regions_to_merge = []
        for r, pixels in regions.items():
            Ay, Acb, Acr = meanycbcr[r]["y"], meanycbcr[r]["cb"], meanycbcr[r]["cr"]
            for adj in adjacent[r]["adj_regions"]:
                By, Bcb, Bcr = meanycbcr[adj]["y"], meanycbcr[adj]["cb"], meanycbcr[adj]["cr"]
                meanSobel = border_gradient(border[r][adj], sobelg)
                meanLaplace = border_gradient(border[r][adj], laplaceg)
                dist = distance2(Ay, Acb, Acr, By, Bcb, Bcr, 0.8, 0.1, 0.1, meanSobel, meanLaplace)
                if dist < threshold:
                    regions_to_merge.append((r, adj))

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
    merge_adjacent_regions(R, regions, ycbcr, meanycbcr, threshold)

    return R, regions