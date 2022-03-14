import json

pathJson = r"/home/withai/wangyx/CVS/EfficientCVSi/wangyx/file_bag/DataSetS.json"
pathS = r"/home/withai/wangyx/CVS/EfficientCVSi/wangyx/file_bag/CVSDataSetV1.json"
listAllOperation = []
classes = [
    "cvsi,0",
    "cvsi,1",
    "cvsi,2",
    "cvsi,3",
    "cvsii,0",
    "cvsii,2",
    "cvsii,3",
    "cvsiii,0",
    "cvsiii,1",
    "cvsiii,2",
    "cvsiii,3",
]
train = []
valid = []
test = []
trainCount = [0] * len(classes)
validCount = [0] * len(classes)
testCount = [0] * len(classes)
dictS = {}

with open(pathJson) as f:
    dictMsg = json.load(f)
    f.close()

dictCount = {}
for rule in dictMsg:
    if rule not in dictCount.keys():
        dictCount[rule] = {}
    for score in dictMsg[rule]:
        if score not in dictCount[rule].keys():
            dictCount[rule][score] = {}

        listJpgName = dictMsg[rule][score]
        listOperationName = list(set([x[: x.find("_")] for x in listJpgName]))
        for operationName in listOperationName:
            dictCount[rule][score][operationName] = len(
                [x for x in listJpgName if x[: x.find("_")] == operationName]
            )
print(dictCount)

# 8:1:1划分数据集
for score in ["2", "1", "3", "0"]:
    for rule in ["cvsi", "cvsii", "cvsiii"]:
        cls = f"{rule},{score}"
        if cls == "cvsii,1":
            continue
        ind = classes.index(cls)

        while dictCount[rule][score] != {}:
            operationName = list(dictCount[rule][score].keys())[0]

            if (
                trainCount[ind]
                / (trainCount[ind] + validCount[ind] + testCount[ind] + 0.00001)
                < 0.8
            ):
                train.append(operationName)
                for scoreTemp in ["2", "1", "3", "0"]:
                    for ruleTemp in ["cvsi", "cvsii", "cvsiii"]:
                        if f"{ruleTemp},{scoreTemp}" == "cvsii,1":
                            continue
                        indTemp = classes.index(f"{ruleTemp},{scoreTemp}")
                        if operationName in dictCount[ruleTemp][scoreTemp]:
                            trainCount[indTemp] += dictCount[ruleTemp][scoreTemp][
                                operationName
                            ]
                            dictCount[ruleTemp][scoreTemp].pop(operationName)
            elif (
                testCount[ind]
                / (trainCount[ind] + validCount[ind] + testCount[ind] + 0.00001)
                < 0.1
            ):
                test.append(operationName)
                for scoreTemp in ["2", "1", "3", "0"]:
                    for ruleTemp in ["cvsi", "cvsii", "cvsiii"]:
                        if f"{ruleTemp},{scoreTemp}" == "cvsii,1":
                            continue
                        indTemp = classes.index(f"{ruleTemp},{scoreTemp}")
                        if operationName in dictCount[ruleTemp][scoreTemp]:
                            testCount[indTemp] += dictCount[ruleTemp][scoreTemp][
                                operationName
                            ]
                            dictCount[ruleTemp][scoreTemp].pop(operationName)
            else:
                valid.append(operationName)
                for scoreTemp in ["2", "1", "3", "0"]:
                    for ruleTemp in ["cvsi", "cvsii", "cvsiii"]:
                        if f"{ruleTemp},{scoreTemp}" == "cvsii,1":
                            continue
                        indTemp = classes.index(f"{ruleTemp},{scoreTemp}")
                        if operationName in dictCount[ruleTemp][scoreTemp]:
                            validCount[indTemp] += dictCount[ruleTemp][scoreTemp][
                                operationName
                            ]
                            dictCount[ruleTemp][scoreTemp].pop(operationName)
print(trainCount)
print(validCount)
print(testCount)

# 处理好的数据存入字典
dictS["train"] = {}
dictS["test"] = {}
dictS["valid"] = {}
for rule in dictMsg:
    if rule not in dictS["train"].keys():
        dictS["train"][rule] = {}
    for score in dictMsg[rule]:
        if score not in dictS["train"][rule].keys():
            dictS["train"][rule][score] = []
        dictS["train"][rule][score].extend(
            sorted([x for x in dictMsg[rule][score] if x[: x.find("_")] in train])
        )
for rule in dictMsg:
    if rule not in dictS["test"].keys():
        dictS["test"][rule] = {}
    for score in dictMsg[rule]:
        if score not in dictS["test"][rule].keys():
            dictS["test"][rule][score] = []
        dictS["test"][rule][score].extend(
            sorted([x for x in dictMsg[rule][score] if x[: x.find("_")] in test])
        )
for rule in dictMsg:
    if rule not in dictS["valid"].keys():
        dictS["valid"][rule] = {}
    for score in dictMsg[rule]:
        if score not in dictS["valid"][rule].keys():
            dictS["valid"][rule][score] = []
        dictS["valid"][rule][score].extend(
            sorted([x for x in dictMsg[rule][score] if x[: x.find("_")] in valid])
        )

with open(pathS, mode="w", encoding="utf-8") as f:
    json.dump(dictS, f, indent=2, ensure_ascii=False)
    f.close()
