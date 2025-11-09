import pyarrow as pa

link_features = [
    "distance_m",
    "implied_speed_knots",
    "delta_speed",
    "delta_course",
    "bearing_diff",
    "kinematic_error",
    "delta_time",
    "y1", 
    "y2", 
    "x1",
    "x2",
    "t1",
    "t2",
    "speed1",
    "speed2",
    "course1",
    "course2",
    "dx1",
    "dy1",
    "dx2",
    "dy2",
]

n_norm_classes = 17
perms = 0

currpt_features = ["dy2", "dx2", "course2", "speed2", "t2", "x2", "y2"]
normal_features = [x for x in link_features if x not in currpt_features]
getNormFeatures = lambda i: [x + f"_{i}" for x in normal_features]

colnames = sum([getNormFeatures(i) for i in range(n_norm_classes)], start=[]) + currpt_features
ynames = [f"y_{i}" for i in range(n_norm_classes + 1)]

full_names = colnames + ynames

schema = pa.schema([pa.field(x, pa.float32()) for x in full_names])