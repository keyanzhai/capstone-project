import numpy as np

def rom_right_elbow(right_wrist_pos, right_elbow_pos, right_shoulder_pos):
    # RIGHT_WRIST, RIGHT_ELBOW, RIGHT_SHOULDER

    vector_1 = right_wrist_pos - right_elbow_pos # vector from elbow to wrist
    vector_2 = right_shoulder_pos - right_elbow_pos # vector from elbow to shoulder

    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    rom_right_elbow = np.arccos(dot_product)

    return np.degrees(rom_right_elbow)