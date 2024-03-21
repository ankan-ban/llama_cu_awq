using System;
using System.Runtime.InteropServices;
using System.Text;

class Model
{
    private IntPtr _modelPtr;
    private IntPtr _libHandle;
    private delegate void HandlerDelegate(string piece);

    [DllImport("build/libllama2.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern IntPtr init_model(
        [MarshalAs(UnmanagedType.LPStr)] string checkpointPath,
        [MarshalAs(UnmanagedType.LPStr)] string tokenizerPath,
        int vocabSize,
        float temperature,
        float topp,
        ulong rngSeed
    );

    [DllImport("build/libllama2.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void free_model(IntPtr model);

    [DllImport("build/libllama2.so", CallingConvention = CallingConvention.Cdecl)]
    private static extern void generate(
        IntPtr model,
        [MarshalAs(UnmanagedType.LPStr)] string prompt,
        int steps,
        HandlerDelegate handler
    );

    public Model(string checkpointPath, string tokenizerPath, int vocabSize, float temperature, float topp, ulong rngSeed)
    {
        _libHandle = NativeLibrary.Load("build/libllama2.so");
        if (_libHandle == IntPtr.Zero)
            throw new Exception("Failed to load library");

        _modelPtr = init_model(checkpointPath, tokenizerPath, vocabSize, temperature, topp, rngSeed);
        if (_modelPtr == IntPtr.Zero)
            throw new Exception("Failed to initialize the model");
    }

    public string Generate(string prompt, int steps)
    {
        var pieces = new StringBuilder();

        void Handler(string piece)
        {
            pieces.Append(piece);
        }

        var handlerDelegate = new HandlerDelegate(Handler);

        generate(_modelPtr, prompt, steps, handlerDelegate);

        return pieces.ToString();
    }

    ~Model()
    {
        if (_modelPtr != IntPtr.Zero)
        {
            free_model(_modelPtr);
            _modelPtr = IntPtr.Zero;
        }

        if (_libHandle != IntPtr.Zero)
        {
            NativeLibrary.Free(_libHandle);
            _libHandle = IntPtr.Zero;
        }
    }
}

class Program
{
    static void Main()
    {
        Model model = new Model("/path/to/model.bin", "tokenizer.bin", 32000, 0f, 0.6f, 1337);
        string result = model.Generate("[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\n Hello how are you today [INST]", 500);
        Console.WriteLine(result);
    }
}

