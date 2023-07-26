import matplotlib.pyplot as plt
import numpy as np

# set parameters
threshold = 25
delta = 4

# read image
path = "Pic/Lena_256bmp.bmp"
image = plt.imread(path)
imageINT = np.array(image, dtype=int)
image = imageINT
print(f"shape: {image.shape}")


# segmentation
def fast_segmentation(img, threshold, delta):
    M, N, O = img.shape
    C = img
    R = np.ones((M, N), dtype=int) * -1

    def merge_region(i, j):
        for m, n in region[j]["B"]:
            R[m, n] = i

    def process_over_segmentation(R, region):
        overSegList = []
        for i in region:
            if len(region[i]["B"]) < delta:
                overSegList.append(i)

        for i in overSegList:
            adjacent_regions = set()
            for m, n in region[i]["B"]:
                if m > 0 and R[m - 1, n] != i:
                    adjacent_regions.add(R[m - 1, n])
                if m < M - 1 and R[m + 1, n] != i:
                    adjacent_regions.add(R[m + 1, n])
                if n > 0 and R[m, n - 1] != i:
                    adjacent_regions.add(R[m, n - 1])
                if n < N - 1 and R[m, n + 1] != i:
                    adjacent_regions.add(R[m, n + 1])

            replace_region = -1
            min_diff = 256
            for j in adjacent_regions:
                if (
                    abs(region[i]["AR"] - region[j]["AR"])
                    + abs(region[i]["AG"] - region[j]["AG"])
                    + abs(region[i]["AB"] - region[j]["AB"])
                ) / 3 < min_diff:
                    replace_region = j
                    min_diff = (
                        abs(region[i]["AR"] - region[j]["AR"])
                        + abs(region[i]["AG"] - region[j]["AG"])
                        + abs(region[i]["AB"] - region[j]["AB"])
                    ) / 3

            for m, n in region[i]["B"]:
                R[m, n] = replace_region

            region[replace_region]["AR"] = (
                region[replace_region]["AR"] * len(region[replace_region]["B"])
                + region[i]["AR"] * len(region[i]["B"])
            ) / (len(region[replace_region]["B"]) + len(region[i]["B"]))
            region[replace_region]["AG"] = (
                region[replace_region]["AG"] * len(region[replace_region]["B"])
                + region[i]["AG"] * len(region[i]["B"])
            ) / (len(region[replace_region]["B"]) + len(region[i]["B"]))
            region[replace_region]["AB"] = (
                region[replace_region]["AB"] * len(region[replace_region]["B"])
                + region[i]["AB"] * len(region[i]["B"])
            ) / (len(region[replace_region]["B"]) + len(region[i]["B"]))
            region[replace_region]["B"].extend(region[i]["B"])
            del region[i]

    j = 1
    R[0, 0] = 1
    region = {1: {"B": [(0, 0)], "AR": C[0, 0, 0], "AG": C[0, 0, 1], "AB": C[0, 0, 2]}}

    for m in range(M):
        for n in range(N):
            if m == 0:
                if n == 0 and m == 0:
                    continue
                if (
                    R[m, n - 1] == j
                    and (
                        abs(C[m, n, 0] - region[j]["AR"])
                        + abs(C[m, n, 1] - region[j]["AG"])
                        + abs(C[m, n, 2] - region[j]["AB"])
                    )
                    / 3
                    <= threshold
                ):
                    R[m, n] = j
                    region[j]["AR"] = (
                        region[j]["AR"] * len(region[j]["B"]) + C[m, n, 0]
                    ) / (len(region[j]["B"]) + 1)

                    region[j]["AG"] = (
                        region[j]["AG"] * len(region[j]["B"]) + C[m, n, 1]
                    ) / (len(region[j]["B"]) + 1)
                    region[j]["AB"] = (
                        region[j]["AB"] * len(region[j]["B"]) + C[m, n, 2]
                    ) / (len(region[j]["B"]) + 1)
                    region[j]["B"].append((m, n))
                elif (
                    R[m, n - 1] == j
                    and (
                        abs(C[m, n, 0] - region[j]["AR"])
                        + abs(C[m, n, 1] - region[j]["AG"])
                        + abs(C[m, n, 2] - region[j]["AB"])
                    )
                    / 3
                    > threshold
                ):
                    R[m, n] = j + 1
                    region[j + 1] = {
                        "AR": C[m, n, 0],
                        "AG": C[m, n, 1],
                        "AB": C[m, n, 2],
                        "B": [(m, n)],
                    }

                    j += 1
            elif m >= 1 and n == 0:
                i = R[m - 1, n]
                if (
                    abs(C[m, n, 0] - region[i]["AR"])
                    + abs(C[m, n, 1] - region[i]["AG"])
                    + abs(C[m, n, 2] - region[i]["AB"])
                ) / 3 <= threshold:
                    R[m, n] = i
                    region[i]["AR"] = (
                        region[i]["AR"] * len(region[i]["B"]) + C[m, n, 0]
                    ) / (len(region[i]["B"]) + 1)

                    region[i]["AG"] = (
                        region[i]["AG"] * len(region[i]["B"]) + C[m, n, 1]
                    ) / (len(region[i]["B"]) + 1)
                    region[i]["AB"] = (
                        region[i]["AB"] * len(region[i]["B"]) + C[m, n, 2]
                    ) / (len(region[i]["B"]) + 1)
                    region[i]["B"].append((m, n))
                elif (
                    abs(C[m, n, 0] - region[i]["AR"])
                    + abs(C[m, n, 1] - region[i]["AG"])
                    + abs(C[m, n, 2] - region[i]["AB"])
                ) / 3 > threshold:
                    R[m, n] = j + 1
                    region[j + 1] = {
                        "AR": C[m, n, 0],
                        "AG": C[m, n, 1],
                        "AB": C[m, n, 2],
                        "B": [(m, n)],
                    }

                    j += 1
            else:
                i = R[m - 1, n]
                k = R[m, n - 1]

                if (
                    abs(C[m, n, 0] - region[i]["AR"])
                    + abs(C[m, n, 1] - region[i]["AG"])
                    + abs(C[m, n, 2] - region[i]["AB"])
                ) / 3 <= threshold and (
                    abs(C[m, n, 0] - region[k]["AR"])
                    + abs(C[m, n, 1] - region[k]["AG"])
                    + abs(C[m, n, 2] - region[k]["AB"])
                ) / 3 > threshold:
                    R[m, n] = i
                    region[i]["AR"] = (
                        region[i]["AR"] * len(region[i]["B"]) + C[m, n, 0]
                    ) / (len(region[i]["B"]) + 1)

                    region[i]["AG"] = (
                        region[i]["AG"] * len(region[i]["B"]) + C[m, n, 1]
                    ) / (len(region[i]["B"]) + 1)
                    region[i]["AB"] = (
                        region[i]["AB"] * len(region[i]["B"]) + C[m, n, 2]
                    ) / (len(region[i]["B"]) + 1)
                    region[i]["B"].append((m, n))
                elif (
                    abs(C[m, n, 0] - region[i]["AR"])
                    + abs(C[m, n, 1] - region[i]["AG"])
                    + abs(C[m, n, 2] - region[i]["AB"])
                ) / 3 > threshold and (
                    abs(C[m, n, 0] - region[k]["AR"])
                    + abs(C[m, n, 1] - region[k]["AG"])
                    + abs(C[m, n, 2] - region[k]["AB"])
                ) / 3 <= threshold:
                    R[m, n] = k
                    region[i]["AR"] = (
                        region[i]["AR"] * len(region[i]["B"]) + C[m, n, 0]
                    ) / (len(region[i]["B"]) + 1)
                    region[i]["AG"] = (
                        region[i]["AG"] * len(region[i]["B"]) + C[m, n, 1]
                    ) / (len(region[i]["B"]) + 1)
                    region[i]["AB"] = (
                        region[i]["AB"] * len(region[i]["B"]) + C[m, n, 2]
                    ) / (len(region[i]["B"]) + 1)
                    region[k]["B"].append((m, n))
                elif (
                    abs(C[m, n, 0] - region[i]["AR"])
                    + abs(C[m, n, 1] - region[i]["AG"])
                    + abs(C[m, n, 2] - region[i]["AB"])
                ) / 3 > threshold and (
                    abs(C[m, n, 0] - region[k]["AR"])
                    + abs(C[m, n, 1] - region[k]["AG"])
                    + abs(C[m, n, 2] - region[k]["AB"])
                ) / 3 > threshold:
                    R[m, n] = j + 1
                    region[j + 1] = {
                        "AR": C[m, n, 0],
                        "AG": C[m, n, 1],
                        "AB": C[m, n, 2],
                        "B": [(m, n)],
                    }

                    j += 1
                elif (
                    abs(C[m, n, 0] - region[i]["AR"])
                    + abs(C[m, n, 1] - region[i]["AG"])
                    + abs(C[m, n, 2] - region[i]["AB"])
                ) / 3 <= threshold and (
                    abs(C[m, n, 0] - region[k]["AR"])
                    + abs(C[m, n, 1] - region[k]["AG"])
                    + abs(C[m, n, 2] - region[k]["AB"])
                ) / 3 <= threshold:
                    if i == k:
                        R[m, n] = i
                        region[i]["AR"] = (
                            region[i]["AR"] * len(region[i]["B"]) + C[m, n, 0]
                        ) / (len(region[i]["B"]) + 1)

                        region[i]["AG"] = (
                            region[i]["AG"] * len(region[i]["B"]) + C[m, n, 1]
                        ) / (len(region[i]["B"]) + 1)
                        region[i]["AB"] = (
                            region[i]["AB"] * len(region[i]["B"]) + C[m, n, 2]
                        ) / (len(region[i]["B"]) + 1)
                        region[i]["B"].append((m, n))

                    else:
                        R[m, n] = i
                        merge_region(i, k)

                        region[i]["AR"] = (
                            region[i]["AR"] * len(region[i]["B"])
                            + region[k]["AR"] * len(region[k]["B"])
                            + C[m, n, 0]
                        ) / (len(region[i]["B"]) + len(region[k]["B"]) + 1)

                        region[i]["AG"] = (
                            region[i]["AG"] * len(region[i]["B"])
                            + region[k]["AG"] * len(region[k]["B"])
                            + C[m, n, 1]
                        ) / (len(region[i]["B"]) + len(region[k]["B"]) + 1)
                        region[i]["AB"] = (
                            region[i]["AB"] * len(region[i]["B"])
                            + region[k]["AB"] * len(region[k]["B"])
                            + C[m, n, 2]
                        ) / (len(region[i]["B"]) + len(region[k]["B"]) + 1)

                        region[i]["B"].extend(region[k]["B"])
                        region[i]["B"].append((m, n))
                        del region[k]

    process_over_segmentation(R, region)
    return R, region


def sort_region(region):
    a = []
    for i in region:
        a.append((len(region[i]["B"]), i))
    a = sorted(a, reverse=True)
    sorted_region = {}
    for l, i in a:
        sorted_region[i] = region[i]
    return sorted_region


def show_region(region):
    r = []
    for i in region:
        r.append(i)
    # print(r)
    return r


def show_segamented_image(image, region, l):
    fig, ax = plt.subplots(3, 3)
    i = 0
    j = 0
    for k in range(9):
        if i // 3 == 1:
            i = 0
            j += 1
        temp = image.copy()
        for m, n in region[l[k]]["B"]:
            temp[m, n] = 255
        ax[i][j].imshow(temp)
        i += 1


# main function
R, region = fast_segmentation(image, threshold, delta)
region = sort_region(region)
r = show_region(region)
show_segamented_image(image, region, r[0:9])
plt.figure()
plt.imshow(image)
plt.show()
