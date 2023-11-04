import json
import os
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "303142897"
ID2 = "201271095"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y



def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    mu = 0
    sigma_x = 1
    sigma_y = 1
    sigma_xv = 0.5
    sigma_yv = 0.5

    ## add noise
    new_state_matrix = s_prior
    new_state_matrix[:1, :] = new_state_matrix[:1, :] + \
                              np.round(np.random.normal(mu, sigma_x, size=(1, 100)))
    new_state_matrix[1:2, :] = new_state_matrix[1:2, :] + \
                               np.round(np.random.normal(mu, sigma_y, size=(1, 100)))
    new_state_matrix[4:5, :] = new_state_matrix[4:5, :] + \
                               np.round(np.random.normal(mu, sigma_xv, size=(1, 100)))
    new_state_matrix[5:6, :] = new_state_matrix[5:6, :] + \
                               np.round(np.random.normal(mu, sigma_yv, size=(1, 100)))

    state_drifted = new_state_matrix.astype(int)

    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    x, y, w, h, xc, yc = state
    sub_image = image[int(y-h):int(y+h), int(x-w):int(x+w), :]

    b, g, r = cv2.split(sub_image)

    b = b // 16
    g = g // 16
    r = r // 16

    hist = np.zeros((16, 16, 16))

    h_sub = int(len(sub_image))
    w_sub = int(len(sub_image[0]))

    for i in range(h_sub):
        for j in range(w_sub):
            hist[b[i, j]][g[i, j]][r[i, j]] += 1

    hist = hist.reshape((4096, 1))
    hist /= np.sum(hist)

    return hist


def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    S_next = []

    for i in range(len(previous_state[1])):
        r = np.random.uniform(0, 1)
        j = np.argmax( cdf >= r )
        S_next.append(previous_state[:, j])

    S_next = np.array(S_next)

    return S_next.T

def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = np.exp(20 * np.sum(np.sqrt(p * q)))

    return distance

def get_avg_particle(S, W):
    """ Additional function we've added for calculating average particle values.

    Args:
        param S: State vector
        param W: Weights vector

    Return:
        x_avg, y_avg, w_avg, h_avg
    """
    (x_avg, y_avg, w_avg, h_avg) = (0, 0, 0, 0)
    for index, particle in enumerate(S.T):
        x_avg += particle[0] * W[index]
        y_avg += particle[1] * W[index]
        w_avg += particle[2] * W[index]
        h_avg += particle[3] * W[index]

    return x_avg, y_avg, w_avg, h_avg


def get_max_match_particle(S, W):
    """ Additional function we've added for calculating maximum particle values.

    Args:
        param S: State vector
        param W: Weights vector

    Return:
        x_max, y_max, w_max, h_max
    """
    max_particle = S.T[np.argmax(W)]
    return max_particle[0], max_particle[1], max_particle[2], max_particle[3]

def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:

    fig, ax = plt.subplots(1)
    image = image[:, :, ::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))

    # Avg particle box
    (x_avg, y_avg, w_avg, h_avg) = get_avg_particle(state, W)

    rect = patches.Rectangle((x_avg - w_avg, y_avg - h_avg), 2*w_avg, 2*h_avg, linewidth=3, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    (x_max, y_max, w_max, h_max) = get_max_match_particle(state, W)

    rect = patches.Rectangle((x_max - w_avg, y_max - h_avg), 2*w_max, 2*h_max, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state

def get_weights(image, S, q):
    """ Additional function we've added for calculating weights.

    Args:
        param image: Input image
        param S: State vector
        param q: Normalized histogram for s_intiial vector

    Return:
        weights
    """
    weights = []
    for col in S.T:
        p = compute_normalized_histogram(image, col)
        weights.append(bhattacharyya_distance(p, q))

    weights = np.array(weights)
    weights /= np.sum(weights)

    return weights

def get_cdf(weights):
    """
    Additional function we've added for calculating CDF.

    Args:
        param weights: weights

    Return:
        CDF
    """
    cdf = np.zeros_like(weights)
    cdf[0] = weights[0]
    for i in range(1, len(weights)):
        cdf[i] = weights[i] + cdf[i-1]

    return cdf

def main():
    # Initialization
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    W = get_weights(image, S, q)
    C = get_cdf(W)

    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:
        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS -> S = AS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE) -> Adding noise
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        W = get_weights(current_image, S, q)
        C = get_cdf(W)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)

if __name__ == "__main__":
    main()
