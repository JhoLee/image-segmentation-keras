import tqdm

for _ in tqdm.tqdm(range(10), desc="training"):
    ## Training FCN8 With CelebA Dataset
    ### By 15 epochs, 3200steps.
    import os
    import sys
    import datetime

    import PIL.Image as Image
    import matplotlib.pyplot as plt


    ROOT_DIR = os.path.join(os.getcwd(), '../..')
    sys.path.append(ROOT_DIR)

    from keras_segmentation.models.fcn import fcn_32 as M

    now = datetime.datetime.now()
    NOW = "{:%Y%m%dT%H%M}".format(now)

    OUT_DIR = os.path.join(ROOT_DIR, 'out')
    CHECKPOINTS_DIR = os.path.join(OUT_DIR, 'checkpoints')
    PREDICTIONS_DIR = os.path.join(OUT_DIR, 'predictions')
    LOGS_DIR = os.path.join(OUT_DIR, 'logs')

    DATASET_DIR = os.path.join(ROOT_DIR, 'dataset/celeba')

    TRAIN_IMAGES = os.path.join(DATASET_DIR, 'train/original')
    TRAIN_ANNOTATIONS = os.path.join(DATASET_DIR, 'train/mask_')
    VAL_IMAGES = os.path.join(DATASET_DIR, 'val/original')
    VAL_ANNOTATIONS = os.path.join(DATASET_DIR, 'val/mask_')

    # Configurations
    MODEL_NAME = 'fcn_32'
    N_CLASSES = 3
    # I_HEIGHT = 416
    I_HEIGHT = 218
    # I_WIDTH = 608
    I_WIDTH = 178

    STEPS_PER_EPOCH = 3200

    EPOCHS = 10

    tag = 'celeba/{name}/ep{epochs}/st{steps}'.format(
        name=MODEL_NAME,
        epochs=EPOCHS,
        steps=STEPS_PER_EPOCH
    )

    CHECKPOINTS_DIR = os.path.join(CHECKPOINTS_DIR, tag)
    PREDICTIONS_DIR = os.path.join(PREDICTIONS_DIR, tag)
    LOGS_DIR = os.path.join(LOGS_DIR, tag)
    if not os.path.exists(CHECKPOINTS_DIR):
        os.makedirs(CHECKPOINTS_DIR)
    if not os.path.exists(PREDICTIONS_DIR):
        os.makedirs(PREDICTIONS_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)

    dir_configuration = """
    NOW = {now}
    tag = {tag}
    out_dir : {out_dir}
    checkpoints_dir : {checkpoints_dir}
    logs_dir : {logs_dir}
    predictions_dir : {predictions_dir}

    dataset_dir : {dataset_dir}
    train_images : {train_images}
    train_annotations : {train_annotations}
    val_images : {val_images}
    val_annotations : {val_annotations}
    """.format(
        now=NOW,
        tag=tag,
        out_dir=OUT_DIR,
        checkpoints_dir=CHECKPOINTS_DIR,
        logs_dir=LOGS_DIR,
        predictions_dir=PREDICTIONS_DIR,

        dataset_dir=DATASET_DIR,
        train_images=TRAIN_IMAGES,
        train_annotations=TRAIN_ANNOTATIONS,
        val_images=VAL_IMAGES,
        val_annotations=VAL_ANNOTATIONS,
    )
    print(dir_configuration)

    # Define model
    model = M(n_classes=N_CLASSES, input_height=I_HEIGHT, input_width=I_WIDTH)

    # Train
    checkpoints_path = os.path.join(CHECKPOINTS_DIR, NOW)
    logs_path = os.path.join(LOGS_DIR, NOW)

    model.train(
        train_images = TRAIN_IMAGES,
        train_annotations = TRAIN_ANNOTATIONS,
        checkpoints_path=checkpoints_path,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        log_dir=logs_path,
        validate=True,
        val_images=VAL_IMAGES,
        val_annotations = VAL_ANNOTATIONS,
        val_steps=355,
        save_best_only=True # Save model if 'val_loss' had been improved.
    )


    # evaluating the model
    evaluation = model.evaluate_segmentation(inp_images_dir=VAL_IMAGES, annotations_dir=VAL_ANNOTATIONS)
    print(evaluation)
