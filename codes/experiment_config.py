EXPERIMENTS = [
    {
        "name": "sg_3_1",
        "USE_SG_FILTER": True,
        "WINDOW_LENGTH": 3,
        "POLYORDER": 1,
        "USE_SMOOTHED_FOR_FEATURES": True,
        "USE_NORMALIZATION": True,
    },
    {
        "name": "sg_5_2",
        "USE_SG_FILTER": True,
        "WINDOW_LENGTH": 5,
        "POLYORDER": 2,
        "USE_SMOOTHED_FOR_FEATURES": True,
        "USE_NORMALIZATION": True,
    },
    {
        "name": "sg_11_2",
        "USE_SG_FILTER": True,
        "WINDOW_LENGTH": 11,
        "POLYORDER": 2,
        "USE_SMOOTHED_FOR_FEATURES": True,
        "USE_NORMALIZATION": True,
    },
    {
        "name": "no_sg",
        "USE_SG_FILTER": False,
        "WINDOW_LENGTH": 0,
        "POLYORDER": 0,
        "USE_SMOOTHED_FOR_FEATURES": False,
        "USE_NORMALIZATION": True,
    }
]