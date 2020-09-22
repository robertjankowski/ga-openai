from tensorflow import keras
from tensorflow.keras import layers
from dataclasses import dataclass


@dataclass
class Shape:
    input: int
    output: int


class MultiTaskModel:
    def __init__(self, first_shape: Shape, second_shape: Shape, *hidden_layers):
        self.model = self._build_model(first_shape, second_shape, *hidden_layers)
        self._compile_model()

    @staticmethod
    def _build_model(first_shape: Shape, second_shape: Shape, *hidden_layers):
        first_input = keras.Input(shape=first_shape.input, name="first_input")
        second_input = keras.Input(shape=second_shape.input, name="second_input")

        x = layers.concatenate([first_input, second_input])
        for hl in hidden_layers:
            x = layers.Dense(hl)(x)

        first_output = layers.Dense(first_shape.output, name="first_output")(x)
        second_output = layers.Dense(second_shape.output, name="second_output")(x)

        return keras.Model(inputs=[first_input, second_input], outputs=[first_output, second_output])

    def _compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss={
                "first_output": keras.losses.MeanSquaredError(),
                "second_output": keras.losses.MeanSquaredError()
            },
            loss_weights=[1.0, 1.0]
        )

    def fit(self, data, epochs: int, batch_size: int):
        self.model.fit(data, epochs=epochs, batch_size=batch_size)

    def plot_model(self, output_path, show_shapes=False):
        keras.utils.plot_model(self.model, output_path, show_shapes=show_shapes)


if __name__ == '__main__':
    cartpole_shape = Shape(4, 1)
    bipedalwalker_shape = Shape(24, 4)

    hidden_sizes = [12, 20, 8]
    model = MultiTaskModel(cartpole_shape, bipedalwalker_shape, *hidden_sizes)
    model.plot_model("../../docs/multitask_learning_example_architecture.png", show_shapes=True)

    # TODO:
    # 1. load data
    # 2. feed model with data
    # 3. plot loss vs epochs
    # 4. check how the model performs in two environments
