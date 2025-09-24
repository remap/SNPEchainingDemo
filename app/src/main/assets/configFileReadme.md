The config file is a json file that holds information about where to model names, locations,
runtime (which should be "D" for NPU), and input-output bindings.
All strings should be within double quotes "", not ''.
No comments are allowed.
It should follow this format:

{
    "baseDir": "/sdcard/Android/data/com.example.snpechainingdemo/files/dlc", <-- replace by your app package name
    "models": [ <-- can accommodate a list of models
        {
            "name": "model1",
            "asset": "my_model_1_prepared.dlc",
            "runtime": "D",
            "inputs": {
                "model_input_tensor_name_1": "workspace_tensor_name_1",
                "model_input_tensor_name_n": "workspace_tensor_name_n"
            },
            "outputs": {
                "model_output_tensor_name_1": "workspace_tensor_name_k",
                "model_output_tensor_name_n": "workspace_tensor_name_j"
            }
        },
        {
            "name": "model2",
            "asset": "my_model_2_prepared.dlc",
            "runtime": "D",
            "inputs": {
                "model_input_tensor_name_1": "workspace_tensor_name_k",
                "model_input_tensor_name_n": "workspace_tensor_name_p"
            },
            "outputs": {
                "model_output_tensor_name_1": "workspace_tensor_name_a",
                "model_output_tensor_name_n": "workspace_tensor_name_b"
        }
    ],
    "init": {
        "sample": { "kind": "random", "mean": 0.0, "std": 1.0, "seed": 123 },
        "temb":   { "kind": "file",   "path": "/sdcard/inputs/temb.bin" },
        "encoder_hidden_states": { "kind": "asset", "path": "warmup/ehs.bin" }
    }
}

Left hand side tensor names must match exactly what the model expects.
Right hand side tensor names are your choice. If the outputs of one model are the inputs
of another model, use the same workspace name -- model chaining relies on name-based input-output bindings.
Similarly, if some models share common tensor inputs, use the same workspace tensor name, (a) to
find the correct tensor, and (b) to avoid unnecessary memory allocation.