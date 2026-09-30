"""ShapeNetCore v2 synset ids -> category names (used as the text prompt of the semantic path)."""

SYNSET_TO_NAME = {
    "02691156": "airplane",
    "02747177": "ashcan",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellular telephone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "computer keyboard",
    "03207941": "dishwasher",
    "03211117": "display",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "loudspeaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}


def normalized_mesh(obj_path):
    """Load a ShapeNetCore v2 `model_normalized.obj` in the TRELLIS data convention.

    Same as the TRELLIS Blender toolkit: y-up -> z-up (Blender OBJ import), longest bbox side = 1, bbox centred at
    the origin. Returns a trimesh.Trimesh (geometry only).
    """
    import numpy as np
    import trimesh
    s = trimesh.load(obj_path, process=False)
    m = trimesh.util.concatenate([g for g in s.geometry.values()]) if isinstance(s, trimesh.Scene) else s
    v = m.vertices[:, [0, 2, 1]] * np.array([1.0, -1.0, 1.0])
    v = v / (v.max(0) - v.min(0)).max()
    v = v - (v.min(0) + v.max(0)) / 2
    return trimesh.Trimesh(v, m.faces, process=False)
