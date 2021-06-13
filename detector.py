import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='System options')
    parser.add_argument(
        'source',
        metavar='source type',
        choices=['demo', 'cam', 'detect'],
        type=str,
        help='options: demo (video), cam (live cam), detect (image)',
    )
    parser.add_argument(
        'system',
        metavar='system type',
        choices=['mask', 'distance'],
        type=str,
        help='options: mask, distance (social distancing)',
    )
    parser.add_argument(
        'path',
        metavar='source path',
        type=str,
        help='image path, video path, or cam source',
    )

    parser.add_argument(
        '--tiny',
        help='Set true to detect using YOLOv4-tiny',
        default=False,
        action='store_true'
    )
        
    args = vars(parser.parse_args())

    if args['system'] == 'mask':
        from yolo.yolo_mask import YoloMask
        detector = YoloMask(args['tiny'])
        
    if args['system'] == 'distance':
        from yolo.yolo_social_distance import YoloSocialDistance
        detector = YoloSocialDistance(args['tiny'])
    
    if args['source'] == 'detect':
        detector.detect_from_image(args['path'])
        
    elif args['source'] == 'demo':
        detector.detect_from_video(args['path'])
        
    elif args['source'] == 'cam':
        args['path'] = int(args['path'])
        detector.detect_from_video(args['path'])
