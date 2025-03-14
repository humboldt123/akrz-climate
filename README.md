# AKRZ Climate Data Parser


## Initial Setup for Running Llama Parser on a Server

1. SSH to the server:
   ```bash
   ssh [abc123]@[server_name].cci.drexel.edu
   ```

2. Git clone this repository:
    ```bash
    git clone https://github.com/humboldt123/akrz-climate
    ```

3. Add your `.mbox` file to the `akrz-climate` directory
    ```bash
    mv path/to/input.mbox ./akrz-climate
    cd akrz-climate
    ```

4. Create a virtual environment:
   ```bash
   python3 -m venv ~/venv
   source ~/venv/bin/activate
   ```

5. Set HF cache environment variable:
   ```bash
   export HF_HOME=/local-ssd/hf_cache/
   ```
   
   Add this to your ~/.bashrc to make it permanent:
   ```bash
   echo 'export HF_HOME=/local-ssd/hf_cache/' >> ~/.bashrc
   ```

6. Install required packages:
   ```bash
   pip install transformers torch accelerate requests
   pip install datasets
   pip install sentencepiece
   ```


7. Create a new screen session:
   ```bash
   screen -S akrz
   ```

8. Inside the screen, activate the environment and test the script:
   ```bash
   source ~/venv/bin/activate
   export CUDA_VISIBLE_DEVICES=0,1  # pick whatever gpus are available
   python3 parse.py akrz.mbox output.csv junk.txt --limit 2 --gpus 5,6
   ```

9. Detach, without stopping it:
   - Press `Ctrl+A` then `D`
   - You'll return to your regular shell, but the process continues running


## Run the Parser

1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Select available GPUs (modify as needed):
   ```bash
   export CUDA_VISIBLE_DEVICES=0,1
   ```

3. Run the parser within your screen session:
   ```bash
   python3 parse.py akrz.mbox output.csv junk.txt --gpus 5,6
   ```

## Troubleshooting

- If you get disconnected, your screen session will continue running
- If you encounter CUDA out-of-memory errors, try using more GPUs or reduce batch size
- To force HF_HOME to be used correctly, you can also use:
  ```bash
  HF_HOME=/local-ssd/hf_cache/ python3 parse.py ...
  ```