baseModel = "resnet50"
arch = "resnet50"
modality = "RGB"

if baseModel == "resnet50" or baseModel == "vgg":
    inputSize = 224
elif baseModel == "BNInception":
    inputSize = 224
elif baseModel == "InceptionV3":
    inputSize = 299
elif baseModel == "inception":
    inputSize = 299
else:
    raise ValueError("Unknown base model: {}".format(baseModel))

scaleSize = inputSize * 256 // 224
cropSize = inputSize
