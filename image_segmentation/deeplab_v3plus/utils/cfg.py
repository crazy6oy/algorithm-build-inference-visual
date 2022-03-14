# TRAIN_CATEGORY = ["bg",
#                   "cystic artery",
#                   "cystic bed",
#                   "cystic duct",
#                   "gallbladder"]

TRAIN_CATEGORY = {
    "cystic artery": 1,
    "cystic duct": 2,
    "cystic plate": 3,
    "dissected windows in the hepatocystic triangle": 4,
    "gallbladder": 5,
    "liver": 6,
}

COLORS = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 0],
]

DIVIDE = {
    "train": [
        "LC-DY-791147",
        "LC-LSH-20389901",
        "LC-HX-0015604597",
        "LC-PZHH-0000109665",
        "LC-CHZH-638752",
        "LC-CHZH-639489",
        "LC-HX-0021127687",
        "LC-HX-0034026072",
        "LC-CHZH-637712",
        "LC-HX-0034112942",
        "LC-ZG-1035575",
        "LC-YR-WCH-TBL-N5",
        "LC-HX-0033662243",
        "LC-HX-0014649990",
        "LC-HX-0000398396",
        "LC-HX-0033254233",
        "LC-LSH-20258908",
        "LC-CHZH-632135",
        "LC-HX-0017790888",
        "LC-ZG-1050810",
        "LC-CHZH-638175",
        "LC-LSH-20965512",
        "LC-HX-0032941367",
        "LC-CHZH-631184",
        "LC-LSH-20267095",
        "LC-CHZH-634906",
        "LC-CHZH-628308",
        "LC-CHZH-634965",
        "LC-HX-0003447434",
    ],
    "val": [
        "LC-YR-CSR-23",
        "LC-HX-0021011325",
        "LC-ZG-1053445",
        "LC-CHZH-635177",
        "LC-CHZH-635321",
        "LC-CHZH-637924",
        "LC-LSH-20267397",
    ],
    "test": ["LC-CHZH-634774", "LC-CHZH-638387", "LC-HX-0014760885", "LC-MY-A01204531"],
}
