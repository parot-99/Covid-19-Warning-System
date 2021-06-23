import numpy as np

def parse_layers(model, conv_layers, residual_layers):
    model_layers = []

    for layer in conv_layers:
        if layer in residual_layers:
            residual_layer = model.layers[layer].name
            model_layers.append(f"{residual_layer}$1")
            model_layers.append(f"{residual_layer}$2")

        else:
            conv_layer = model.layers[layer].name
            model_layers.append(conv_layer)

    return model_layers


def load_weights(model, weights_file, model_layers, output_layers):
    flag = True

    with open(weights_file, "rb") as file:
        major, minor, revision, seen, _ = np.fromfile(
            file, dtype=np.int32, count=5
        )

        for i, layer in enumerate(model_layers):
            layer_name = layer.split("$")[0]
            layer_type = layer_name.split("_")[0]

            if layer_type == "conv":
                conv_layer = model.get_layer(layer)
                in_dim = conv_layer.input_shape[-1]

            if layer_type == "residual":
                residual_layer = model.get_layer(layer_name)
                in_dim = residual_layer.input_shape[-1]

                if layer.split("$")[1] == "1":
                    conv_layer = residual_layer.conv1

                if layer.split("$")[1] == "2":
                    conv_layer = residual_layer.conv2

                    if flag:
                        in_dim //= 2
                        flag = False

            filters = conv_layer.conv.filters
            k_size = conv_layer.conv.kernel_size[0]

            if i not in output_layers:
                bn_weights = np.fromfile(
                    file, dtype=np.float32, count=4 * filters
                )
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = conv_layer.batch_normalization

            else:
                conv_bias = np.fromfile(file, dtype=np.float32, count=filters)

            conv_shape = (filters, in_dim, k_size, k_size)
            conv_weights = np.fromfile(
                file, dtype=np.float32, count=np.product(conv_shape)
            )

            conv_weights = conv_weights.reshape(conv_shape).transpose(
                [2, 3, 1, 0]
            )

            if i not in output_layers:
                conv_layer.conv.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)

            else:
                conv_layer.conv.set_weights([conv_weights, conv_bias])
