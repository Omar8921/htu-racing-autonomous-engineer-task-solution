import numpy as np

def remove_ground(
    points: np.ndarray,
    num_iters: int,
    dist_thresh: float    
) -> np.ndarray:
    # Initialize the generator
    rng = np.random.default_rng()

    best_score = -np.inf
    non_ground = np.empty((0, 3), dtype=np.float32)

    for _ in range(num_iters):
        # Take 3 random points
        indices = rng.choice(points.shape[0], size=3, replace=False)
        sample = points[indices]

        # ============================== 
        # Find the plane that passes through
        # these points, step by step.
        # ============================== 
        point1, point2, point3 = sample

        # Find two vectors on the plane
        v1 = point2 - point1
        v2 = point3 - point1

        # Find the normal vector
        normal_vector = np.cross(v1, v2)
        denominator = np.linalg.norm(normal_vector)

        if denominator < 1e-8:
            continue

        D = -np.dot(point1, normal_vector)

        numerator = np.abs(np.dot(points, normal_vector) + D)
        distances = numerator / denominator

        score = np.sum(distances <= dist_thresh) / points.shape[0]

        if score >= best_score:
            best_score = score
            non_ground = points[distances > dist_thresh]

    return non_ground