from yolo_mask import YoloMask


def detect_from_image(mask_detector):
    image_path = input('Enter image path: ')
    write_image = input('Do you want to write the image? [Yes|No]: ')
    write_image = True if write_image == 'Yes' else False
    mask_detector.detect_from_image(image_path, write_image)


def detect_from_video(mask_detector):
    video_path = input('Enter Video path: ')
    write_video = input('Do you want to write the video? [Yes|No]: ')
    write_video = True if write_video == 'Yes' else False
    mask_detector.detect_from_video(video_path, write_video)


def detect_from_cam(mask_detector):
    src = int(input('Enter cam source: '))
    write_video = input('Do you want to write the video? [Yes|No]: ')
    write_video = True if write_video == 'Yes' else False
    mask_detector.detect_from_video(src, write_video)


def set_show_masks(mask_detector):
    show_masks = input('Do you want to show masks? [Yes|No]: ')
    show_masks = True if show_masks == 'Yes' else False
    mask_detector.show_masks = show_masks


def set_show_fps(mask_detector):
    show_fps = input('Do you want to show FPS? [Yes|No]: ')
    show_fps = True if show_fps == 'Yes' else False
    mask_detector.show_fps = show_fps


def set_show_scores(mask_detector):
    show_scores = input('Do you want to show FPS? [Yes|No]: ')
    show_scores = True if show_scores == 'Yes' else False
    mask_detector.show_scores = show_scores


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
        7: set_score_threshold,
        8: set_iou_threshold,
        9: display_config
    }.get(value)


if __name__ == '__main__':
    path = input('Enter detector path:')
    mask_detector = YoloMask(path)

    while True:
        print(
            '\nOperations:\n'
            '\t1.Detect from image.\n'
            '\t2.Detect from video.\n'
            '\t3.Detect from live cam.\n'
            '\t4.Set show masks.\n'
            '\t5.Set show FPS.\n'
            '\t6.Set show scores.\n'
            '\t7.Set score threshold.\n'
            '\t8.set IOU threshold.\n'
            '\t9.See current configurations.\n'
            '\t10.Exit.'
        )

        choice = int(input('\nChoose an operation: '))

        if choice == 10:
            break

        if choice > 9 or choice < 1:
            print('\n[WARNING]: Please choose a valid operation.')
            continue

        print('*********************************************************')

        operation = switch(choice)
        operation(mask_detector)

