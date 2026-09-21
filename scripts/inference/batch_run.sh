#!/bin/bash

# Set base directories
METHOD="momasg"
USE_GT_SEGMENTS="False" ############################################## CHECK THIS!

MOMASG_DIR="/path/to/MoMa-SG/"
DATASET_DIR="/path/to/arti4d/raw/"
PRED_DIR="/path/to/output_${METHOD}/arti4d/"
RUN_SCRIPT="${MOMASG_DIR}/moma_sg/graph/engine.py"

LOAD_DIR="/path/to/output_${METHOD}/arti4d" # Directory to load results from
OUTPUT_DIR="/path/to/output_${METHOD}/arti4d" # Directory to save results

# if [ "$STURM" = "True" ]; then
#     # OUTPUT_DIR="./output_data/bbox_trajectory/sturm"
#     OUTPUT_DIR="./output_data/arti4d_sturm_original"
# else
#     OUTPUT_DIR="./output_data/arti4d"
# fi

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory $DATASET_DIR does not exist."
    exit 1
fi

CONSIDERED_SCENES=("rh078" "rr080" "rh201" "din080") 


source ~/.bashrc
conda activate momasg

# Find all scene directories and evaluate them
echo "Searching for scene directories in ${DATASET_DIR}..."
for ROOM in ${DATASET_DIR}/*/; do
    for SCENE_DIR in ${ROOM}/scene_*; do
        if [ -d "$SCENE_DIR" ]; then
            SCENE_NAME=$(basename "$SCENE_DIR")
            ROOM_NAME=$(basename "$ROOM")
            echo "Evaluating scene: $ROOM_NAME / $SCENE_NAME"

            # check if ROOM_NAME is in CONSIDERED_SCENES
            if [[ ! " ${CONSIDERED_SCENES[@]} " =~ " ${ROOM_NAME} " ]]; then
                echo "Warning: Room $ROOM_NAME not in considered scenes list. Skipping scene."
                continue
            fi


            
            if [ ! -d "$PRED_DIR" ]; then
                echo "Warning: Prediction directory $PRED_DIR not found. Skipping scene."
                continue
            fi

            # Create output directory if it doesn't exist
            RES_OUT="${PRED_DIR}/$ROOM_NAME/${SCENE_NAME}/intermediate_results/"
            mkdir -p "$RES_OUT"
            
            # Run evaluation script
            # echo "Running: python $EVAL_SCRIPT --gt $GT_JSON_FILE --pred $PRED_DIR"
            
            # parse hydra config
            python "$RUN_SCRIPT" dataset.root_path="${DATASET_DIR}/${ROOM_NAME}/${SCENE_NAME}" interaction.use_gt_segments=${USE_GT_SEGMENTS} cache.load_dir=${LOAD_DIR} cache.output_dir=${OUTPUT_DIR} # > /dev/null 2>&1
            
            echo "Finished inference on ${ROOM_NAME}/${SCENE_NAME}"
            echo "----------------------------------------"
            # exit 0
        fi
    done
done

echo "Processed all scenes!"