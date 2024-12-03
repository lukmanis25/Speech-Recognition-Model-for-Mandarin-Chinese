import threading
from tf_keras.src.callbacks import Callback

class InterruptTraining(Callback):
    def __init__(self):
        super().__init__()
        self.stop_training = False
        self.check_thread = threading.Thread(target=self._check_stop_signal, daemon=True)
        self.check_thread.start()

    def _check_stop_signal(self):
        print("Naciśnij 'q' i Enter, aby zatrzymać trening.")
        while not self.stop_training:
            user_input = input()  # Czeka na wpisanie czegoś w terminalu
            if user_input.lower() == 'q':
                self.stop_training = True
                break

    def on_batch_end(self, batch, logs=None):
        if self.stop_training:
            print("Trening przerwany przez użytkownika.")
            self.model.stop_training = True