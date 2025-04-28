
# # Modal Cookbook: Recipe for Inference Throughput Maximization
# In certain applications, the bottom line comes to throughput: process a set of inputs as fast as possible.
# Let's explore how to maximize throughput by using Modal on an embedding example, and see just how fast
# we can encode the [wildflow sweet-coral dataset](https://huggingface.co/datasets/wildflow/sweet-corals "huggingface/wildflow/sweet-coral")
# using the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity").

# Import everything we need for the locally-run Python (everything in our local_entrypoint function at the bottom).
import time
from pathlib import Path
from more_itertools import chunked
from modal.volume import FileEntry
import concurrent.futures

import modal

# We will be running this example several times, and logging speeds to a local file specified here.
local_logfile = Path("~/results/rt5.txt").expanduser()
CSV_FILE = Path(local_logfile).with_suffix(".csv")
local_logfile.parent.mkdir(parents=True, exist_ok=True)


# ## Key Parameters
# Key factors impacting throughput include batchsize, the amount of concurrency we allow for our app.
# * `batch_size` is a parameter passed to the [Infinity inference engine](https://github.com/michaelfeil/infinity "github/michaelfeil/infinity"), and it means the usual thing for machine learning inference: a group of images are processed through the neural network together.
# * `max_concurrency` sets the [@modal.concurrent(max_inputs:int) ](https://modal.com/docs/guide/concurrent-inputs#input-concurrency "Modal: input concurrency") argument for the inference app. 
# This takes advantage of the asynchronous nature of the Infinity embedding inference app.
# * `gpu` is a string specifying the GPU to be used. 
batch_size: int = 100
max_concurrency: int = 10
gpu: str = "H100"
max_containers: int = 1
# This `max_ims` parameter simply caps the total number of images that are parsed (for testing/debugging).
# Set to -1 to parse all.
max_ims: int = 1000

# This should point to a model on huggingface that is supported by Infinity.
# Note that your specifically chosen model might require specialized imports when
# designing the image!
MODEL_NAME = "openai/clip-vit-base-patch16" # 599 MB

# ## Define the image and data volume
# Setup the image
simple_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(["infinity_emb[all]==0.0.76",   # for Infinity inference lib
                  "sentencepiece",               # for this particular chosen model
                  "more-itertools"])             # for elegant list batching
    .env({"INFINITY_MODEL_ID": MODEL_NAME, "HF_HOME": "/data"})      # for Infinity inference lib
)

# Setup a [Modal Volume](https://modal.com/docs/guide/volumes#volumes "Modal.Volume") containing all of the images we want to encode.
vol_name = "sweet-coral-db-20k"
vol_mnt = Path("/data")
vol = modal.Volume.from_name(vol_name, environment_name="ben-dev")

# Initialize the app
app = modal.App('vol-simple-infinity', image=simple_image, volumes={vol_mnt: vol})

# Imports inside the container
with simple_image.imports():
    from infinity_emb import AsyncEmbeddingEngine, EngineArgs
    from infinity_emb.primitives import Dtype, InferenceEngine
    from PIL import Image

# Here we define an app.cls that wraps Infinity's AsyncEmbeddingEngine.
@app.cls(gpu=gpu, image=simple_image, volumes={vol_mnt: vol}, 
         timeout=24*60*60,
         min_containers=max_containers, max_containers=max_containers)
@modal.concurrent(max_inputs=max_concurrency)
class InfinityModel:

    # The enter* decorator
    @modal.enter()
    async def enter(self):
        # self.model = AsyncEmbeddingEngine.from_args(EngineArgs(
        #         model_name_or_path=MODEL_NAME,
        #         batch_size=batch_size,
        #         model_warmup=False,
        #         engine=InferenceEngine.torch,
        #         dtype=Dtype.float16,
        #     ))
        # await self.model.astart()
        pass
    # TODO: get the ecit funcvtion

    @modal.method()
    async def embed(self, images: list[FileEntry])->list[int]:
        start_engine = time.perf_counter()
        model = AsyncEmbeddingEngine.from_args(EngineArgs(
                model_name_or_path=MODEL_NAME,
                batch_size=batch_size,
                model_warmup=False,
                engine=InferenceEngine.torch,
                dtype=Dtype.float16,
            ))
        await model.astart()
        stop_engine = time.perf_counter()
        elapsed_engine = stop_engine - start_engine
        print(f"starting engine took => {elapsed_engine:.3f}s")

        start_total = time.perf_counter()

        # Read images from disk    
        start_read = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            full_paths = [vol_mnt / impath.path for impath in images]
            images = list(executor.map(Image.open, full_paths))
        
        stop_read = time.perf_counter()
        elapsed_read = stop_read - start_read
        print(f"read took => {elapsed_read:.3f}s")

        # Embed images
        start_embed = time.perf_counter()
        _, _ = await model.image_embed(images=images)
    
        stop_embed = time.perf_counter()
        elapsed_embed = stop_embed - start_embed
        print(f"engine perf batch={len(images)} elapsed={elapsed_embed:.3f}s")

        stop_total = time.perf_counter()
        elapsed_total = stop_total - start_total
        print(f"embed() call perf batch={len(images)} elapsed={elapsed_embed:.3f}s")

        await model.astop()
        return elapsed_total


@app.local_entrypoint()
def backbone(expname:str=''):
    st=time.time()

    # Init the model inference app
    embedder = InfinityModel()
    startup_dur = time.time() - st
    print(f"Took {startup_dur}s to start Infinity Engine.")
    
    # Catalog data
    im_path_list = list(filter(lambda x: x.path.endswith(".jpg"),
                         vol.listdir('/data', recursive=True)))
    print(f"Found {len(im_path_list)} JPEGs, ", end='')

    # Optional: cutoff number of images for testing (set to -1 to encode all)
    if max_ims > 0:
        im_path_list = im_path_list[:min(len(im_path_list), max_ims)]
    n_ims = len(im_path_list)
    print(f"using {n_ims}.")

    # Embed batches via remote `map` call
    throughputs=[]
    for thru_put in embedder.embed.map(chunked(im_path_list, batch_size)):
        throughputs.append(thru_put)
    print(throughputs[0])
    # Time it!
    total_duration = time.time() - st

    total_embed_time = sum(throughputs)

    # Log
    if n_ims>0:
        log_msg = (
            f"simple_volume.py::{expname}::batch_size={batch_size}::n_ims={n_ims}::concurrency={max_concurrency}\n"
            f"\tTotal time:\t{total_duration/60:.2f} min\n"
            f"\tOverall throughput:\t{n_ims/total_duration:.2f} im/s\n"
            f"\tEmbedding-only throughput (avg):\t{total_embed_time:.2f} im/s\n"
        )

        print(log_msg)

