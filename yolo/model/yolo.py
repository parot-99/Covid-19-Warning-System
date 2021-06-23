import tensorflow as tf
from .backbone import cspdarknet53, cspdarknet53_tiny
from .neck import yolo_neck, yolo_tiny_neck
from .dense_prediction import dense_prediction
from .config import cfg
from .utils import parse_layers, load_weights


class Yolo:
    def __init__(self, classes, tiny=False):
        self.tiny = tiny

        input_layer = tf.keras.layers.Input(
            [cfg.INPUT_SIZE, cfg.INPUT_SIZE, 3]
        )
        backbone = (
            cspdarknet53(input_layer)
            if not tiny
            else cspdarknet53_tiny(input_layer)
        )
        neck = (
            yolo_neck(backbone, classes)
            if not tiny
            else yolo_tiny_neck(backbone, classes)
        )
        head = dense_prediction(neck, classes, tiny)
        self.model = tf.keras.Model(input_layer, head)

    def load_weights(self, weights_path):
        conv_layers = (
            cfg.CONV_LAYERS if not self.tiny else cfg.CONV_LAYERS_TINY
        )
        residual_layers = (
            cfg.RESIDUAL_LAYERS if not self.tiny else cfg.RESIDUAL_LAYERS_TINY
        )
        output_layers = (
            cfg.OUTPUT_LAYERS if not self.tiny else cfg.OUTPUT_LAYERS_TINY
        )

        layers_names = parse_layers(
            self.model, conv_layers, residual_layers
        )
        load_weights(
            self.model, weights_path, layers_names, output_layers
        )

    def get_graph(self):
        return tf.function(self.model)
