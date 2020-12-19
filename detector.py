from yolo_mask import YoloMask


def init_detector():
    detector_path = input('Enter detector path: ')

    return YoloMask(detector_path)

def detect_from_image(mask_detector):
    image_path = input('Enter image path: ')
    mask_detector.detect_from_image(image_path)


def detect_from_video(mask_detector):
    video_path = input('Enter Video path: ')
    mask_detector.detect_from_video(video_path)


def detect_from_cam(mask_detector):
    src = int(input('Enter cam source: '))
    mask_detector.detect_from_video(src)


def set_show_masks(mask_detector):
    show_masks = input('Do you want to show masks? [yes|no]: ')
    show_masks = True if show_masks == 'yes' else False
    mask_detector.show_masks = show_masks


def set_show_fps(mask_detector):
    show_fps = input('Do you want to show FPS? [yes|no]: ')
    show_fps = True if show_fps == 'yes' else False
    mask_detector.show_fps = show_fps


def set_show_scores(mask_detector):
    show_scores = input('Do you want to show scores? [yes|no]: ')
    show_scores = True if show_scores == 'yes' else False
    mask_detector.show_scores = show_scores


def set_write_detection(mask_detector):
    write_detection = input(
        'Do you want to write the output image/video? [yes|no]: '
    )
    write_detection = True if write_detection == 'yes' else False
    mask_detector.write_detection = write_detection


def set_score_threshold(mask_detector):
    score_threshold = float(input('Choose a value: '))
    mask_detector.score_threshold = score_threshold


def set_iou_threshold(mask_detector):
    iou_threshold = float(input('Choose a value: '))
    mask_detector.iou_threshold = iou_threshold


def display_config(mask_detector):
    print(
        '\n[INFO]:\n'
        f'\tShow masks: {mask_detector.show_masks}\n'
        f'\tShow FPS: {mask_detector.show_fps}\n'
        f'\tShow scores: {mask_detector.show_scores}\n'
        f'\tWrite detection: {mask_detector.write_detection}\n'
        f'\tScore threshold value: {mask_detector.score_threshold}\n'
        f'\tIOU threshold value: {mask_detector.iou_threshold}'
    )


def switch(value):
    return {
        1: detect_from_image,
        2: detect_from_video,
        3: detect_from_cam,
        4: set_show_masks,
        5: set_show_fps,
        6: set_show_scores,
        7: set_write_detection,
        8: set_score_threshold,
        9: set_iou_threshold,
        10: display_config
    }.get(value)


if __name__ == '__main__':
    mask_detector = init_detector()

    while True:
        print(
            '\nOperations:\n'
            '\t1.Detect from image.\n'
            '\t2.Detect from video.\n'
            '\t3.Detect from live cam.\n'
            '\t4.Set show masks.\n'
            '\t5.Set show FPS.\n'
            '\t6.Set show scores.\n'
            '\t7.Set write detection.\n'
            '\t8.Set score threshold.\n'
            '\t9.set IOU threshold.\n'
            '\t10.See current configurations.\n'
            '\t11.Exit.'
        )

        choice = int(input('\nChoose an operation: '))

        if choice == 11:
            break

        if choice > 10 or choice < 1:
            print('\n[WARNING]: Please choose a valid operation.')
            continue

        print('*********************************************************')

        operation = switch(choice)
        operation(mask_detector)

