import ctypes

class Model:
    def __init__(self, checkpoint_path, tokenizer_path, vocab_size, temperature, topp, rng_seed):
        # Load the shared library
        self._lib = ctypes.CDLL('build/libllama2.so')  # Adjust the path based on your system

        # Specify the argument types and return type for init_model
        self._lib.init_model.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_ulonglong]
        self._lib.init_model.restype = ctypes.POINTER(ctypes.c_void_p)

        # Specify the argument types and return type for free_model
        self._lib.free_model.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._lib.free_model.restype = None

        # Specify the argument types and return type for free_model
        self._lib.generate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_int]
        self._lib.generate.restype = None

        # Initialize the Model
        self._model_ptr = self._lib.init_model(
            ctypes.create_string_buffer(checkpoint_path.encode('utf-8')),
            ctypes.create_string_buffer(tokenizer_path.encode('utf-8')),
            ctypes.c_int(vocab_size),
            ctypes.c_float(temperature),
            ctypes.c_float(topp),
            ctypes.c_ulonglong(rng_seed)
        )
        if not self._model_ptr:
            raise RuntimeError("Failed to initialize the model")

    def generate(self, prompt, steps):
        self._lib.generate(
            self._model_ptr,
            ctypes.create_string_buffer(prompt.encode('utf-8')),
            ctypes.c_int(steps)
        )

    def __del__(self):
        # Free the Model when the object is deleted
        if hasattr(self, '_model_ptr') and self._model_ptr:
            self._lib.free_model(self._model_ptr)

# Example usage:
model = Model("/path/to/model.bin", "tokenizer.bin", 32000, 0.5, 0.6, 1337)
model.generate("[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>> Hello how are you today [INST]", 500);
