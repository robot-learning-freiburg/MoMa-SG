import base64
import os

import cv2
import hydra
import loguru
import numpy as np
from omegaconf import DictConfig
from openai import OpenAI


# Function to encode the image
def encode_image(image_path):
    """
    Read an image file from disk and encode its bytes as a base64 string.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64-encoded image content.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function to encode image from already loaded cv2 rgb image
def encode_image_from_cv2(rgb_image: np.ndarray) -> str:
    """
    Encode an in-memory RGB image as a base64-encoded JPEG string.

    Args:
        rgb_image (numpy.ndarray): The input image in RGB format.

    Returns:
        str: Base64-encoded JPEG content.
    """
    _, buffer = cv2.imencode(".jpg", rgb_image)
    return base64.b64encode(buffer).decode("utf-8")


def query_articulation_state(cfg: DictConfig, img: np.ndarray, question: str) -> str:
    """
    DEPRECATED: only queries for a single image instead of a series of images as in query_articulation_mode().
    Uses a vision-language model to label the state of an object in an image.

    Args:
        cfg (DictConfig): Configuration containing the OpenAI API key.
        img (numpy.ndarray): The input image in RGB format.
        question (str): The question to ask the model about the image.

    Returns:
        str: The model's response to the question.
    """
    os.environ["OPENAI_API_KEY"] = cfg.openai_key
    client = OpenAI()

    # Getting the Base64 string
    base64_image = encode_image_from_cv2(img)

    response = client.responses.create(
        model=cfg.articulation.vlm,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{question}"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    return response.output_text.strip().lower()


def query_articulation_mode(cfg: DictConfig, imgs: np.ndarray, question: str) -> str:
    """
    Uses a VLM to label a series of images depicting an object being articulated.

    Args:
        cfg (DictConfig): Configuration containing the OpenAI API key.
        img (numpy.ndarray): The input image in RGB format.
        question (str): The question to ask the model about the image.

    Returns:
        str: The model's response to the question.
    """
    os.environ["OPENAI_API_KEY"] = cfg.openai_key
    client = OpenAI()

    content = [{"type": "input_text", "text": f"{question}"}]
    for i in range(len(imgs)):
        content.append(
            {
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{encode_image_from_cv2(imgs[i])}",
            }
        )

    response = client.responses.create(
        model=cfg.articulation.vlm,
        input=[{"role": "user", "content": content}],
    )

    return response.output_text.strip().lower()


def plot_articulation_mode(cfg, rgb_frames, prompt_kf_idcs, parsed_trend, motion_trend, i, save_dir):
    """
    Plot the keyframes fed to the VLM and the parsed articulation modes

    :param cfg: parameters
    :param rgb_frames: rgb frames
    :param first_frame: first frame of segment
    :param first_pair_frame: frame to which minimal motion is observed and estimated
    :param max_theta_frame: frame of maximum articulation
    :param last_frame: last frame of segment
    :param parsed_trend: VLM motion vote
    :param motion_trend: articulation motion profile
    :param i: segment index
    :param save_dir: directory to save the plot
    """
    concat_img = np.concatenate([cv2.cvtColor(rgb_frames[kf_idx], cv2.COLOR_RGB2BGR) for kf_idx in prompt_kf_idcs], axis=1)
    concat_img = cv2.putText(concat_img, f"({prompt_kf_idcs[0]})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    concat_img = cv2.putText(concat_img, f"({prompt_kf_idcs[1]})", (rgb_frames[0].shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    concat_img = cv2.putText(concat_img, f"({prompt_kf_idcs[2]})", (2 * rgb_frames[0].shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    concat_img = cv2.putText(concat_img, f"({prompt_kf_idcs[3]})", (3 * rgb_frames[0].shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(concat_img, f"VLM Type: {cfg.articulation.vlm}, ", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(concat_img, f"Vote: {parsed_trend.value}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(concat_img, f"Motion Profile: {motion_trend}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if not os.path.exists(os.path.join(save_dir, "articulation")):
        os.makedirs(os.path.join(save_dir, "articulation"), exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "articulation", f"segment_{i:04d}_mode.png"), concat_img)
        loguru.logger.info(f"Saved articulation mode plot to {os.path.join(save_dir, 'articulation', f'segment_{i:04d}_mode.png')}")


@hydra.main(version_base=None, config_path="../configs", config_name="momasg")
def main(cfg: DictConfig) -> None:
    """
    Standalone demo entry point: sends a single hardcoded example image to the
    configured VLM and prints whether it judges the articulated object open or closed.

    Args:
        cfg (DictConfig): Hydra configuration containing the OpenAI API key.
    """
    os.environ["OPENAI_API_KEY"] = cfg.openai_key

    client = OpenAI()
    # Path to your image
    image_path = "/path/to/arti4d/raw/rh201/scene_2025-04-25-15-16-29/rgb/rgb_image_1745587055003430935.jpg"

    # Getting the Base64 string
    base64_image = encode_image(image_path)

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "is the articulated object open or closed? please just answer with either 'opened' or 'closed'"},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    print(response.output_text)


if __name__ == "__main__":
    main()
