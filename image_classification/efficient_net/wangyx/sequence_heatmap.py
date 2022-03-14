import cv2
import numpy as np


def conf_heatmap(confs, clses, labels, name_file):
    h = 64
    interval = 5
    colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
    write_confs = np.ones((h, len(confs) * interval, 3), dtype=np.uint8) * 255
    write_clses = np.ones((h // 8, len(clses) * interval, 3), dtype=np.uint8) * 255
    write_labels = np.ones((h // 8, len(labels) * interval, 3), dtype=np.uint8) * 255

    write_confs[h // 2 : h // 2 + 1] = 0
    write_confs[h // 4 : h // 4 + 1] = 0
    for i in range(len(confs)):
        color = (0, 255, 0) if clses[i] == labels[i] else (0, 0, 255)
        # cv2.line(write_confs,
        #          (i * interval, h - int(round(h * confs[i]))), ((i + 1) * interval, h - int(round(h * confs[i + 1]))),
        #          colors[clses[i]], 2)
        # if i != 0 and i != len(confs)-1:
        #     cv2.line(write_confs,
        #              (i * interval, h - int(round(h * confs[i - 1]))),
        #              (i * interval + 1, h - int(round(h * confs[i]))),
        #              (0, 255, 0), 2)
        #     cv2.line(write_confs,
        #              ((i + 1) * interval - 1, h - int(round(h * confs[i]))),
        #              ((i + 1) * interval, h - int(round(h * confs[i + 1]))),
        #              (0, 255, 0), 2)
        # cv2.line(write_confs,
        #          (i * interval, h - int(round(h * confs[i]))),
        #          ((i + 1) * interval, h - int(round(h * confs[i]))),
        #          color, 2)
        write_confs[
            max(0, h - int(round(h * confs[i])) - 1) : h - int(round(h * confs[i])) + 2,
            i * interval : (i + 1) * interval,
        ] = color
    for i in range(len(clses)):
        write_clses[1:-1, i * interval : (i + 1) * interval] = colors[clses[i]]
        write_labels[1:-1, i * interval : (i + 1) * interval] = colors[labels[i]]

    out = np.concatenate((write_confs, write_clses, write_labels), 0)
    cv2.imwrite(f"{name_file}.png", out)
    cv2.waitKey()


def softmax(xs, cls_ind):
    return np.exp(xs[cls_ind]) / sum([np.exp(x) for x in xs])


def sort_win(name, list_name, ind):
    postfix = list_name[0].split(".")[-1]
    nums = [int(x.split("/")[-1].replace(name, "").split(".")[0]) for x in list_name]
    sort_need = [[nums[i], ind[i]] for i in range(len(nums))]
    sort_out = sorted(sort_need)
    return [x[1] for x in sort_out]


if __name__ == "__main__":
    name_head = ["LC-CSR-31-", "LC-CSR-4VID001_", "LC-CSR-87-", "LC-CSR-9-", "TBL-5-"]

    file = open("file_bag/msg.txt")
    list_msg = file.readlines()
    file.close()
    file = open("file_bag/name_img.txt")
    list_name = file.readlines()
    file.close()
    confs = [softmax(eval(x.split("*")[0]), int(x.split("*")[-2])) for x in list_msg]
    clses = [int(x.split("*")[-2]) for x in list_msg]
    labels = [int(x.split("*")[-1]) for x in list_msg]
    names = [eval(x)[0] for x in list_name]
    check = [eval(x)[1] for x in list_name]
    cc = [False for x in range(len(check)) if labels[x] != check[x]]
    if len(cc) == 0:
        for name in name_head:
            ind = [i for i, x in enumerate(names) if name in x]
            ind_out = sort_win(name, [x for x in names if name in x], ind)
            conf_in = [confs[x] for x in ind_out]
            clses_in = [clses[x] for x in ind_out]
            labels_in = [labels[x] for x in ind_out]
            conf_heatmap(conf_in, clses_in, labels_in, name[:-1])
