# ComfyUI-LTX2-TRAINER

LoRA trainer for LTX 2.X video generation models in ComfyUI

**LTX2-TRAINER BATMAN EDITION**
<img width="1920" height="512" alt="Flux2-Klein_00015_" src="https://github.com/user-attachments/assets/3d80e48b-8893-4f60-9f21-be2123baf87e" />

<a href="https://buymeacoffee.com/JaimitoEs" target="\_blank">

&#x20; <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174">

</a>

**Adaptation of the LTX2-Training lora for ComfyUI**.
<img width="1812" height="761" alt="Captura de pantalla 2026-03-17 212625" src="https://github.com/user-attachments/assets/ad7b7a10-4d64-4346-9730-7c997dd397b7" />

How to update:
* open terminal in custom_nodes\ComfyUI-LTX2-TRAINER\:
```
git pull
```

* About updates: if you clone this repository by using github desktop it will be easier to track updates and pulling, choose your desired way to do this. If it's not working for you probally you are missing a dependency module instalation. Pay attention to this.

* New info note : If you are on Windows and using Blackwell, uninstall all the attention modules (xformers, flash 2.8.3, i don't know if sageattention can affect but LTX2 don't use it). In our environment with ComfyUI and pytorch 2.10 and uninstalling all attentions makes Blackwell with native attention and performing much better. If you can find a flash attention 3 wheel compatible with your environment then yes, you will see a performance boost (there is plenty of them for linux). Flash 2.8.3 seems to work much slower (uninstall it for Blackwell envrionments).

* About versions : The code has many changes giving extra optimizations compared to the oficial trainer, improving Dynamo Inductor and memory management. I noticed this days fewer updates from Lightricks in their packages. As the current version is working great, i will give a time to add the newest Lightricks implementations (rebuilding the code takes time and the LightTricks commits does not stop during the early stages). Whatever, this build works perfectly creating amazing LoRAs in single gpu (not tested in multi-gpu, but you can always post an issue in case).

Version 1.05:
* Added CUDA_VISIBLE_DEVICES into nodes for multi/single gpu compatibility. For multi-gpu, it seems you can't use the shared memory (sytem ram).
* This means if you don't have enought VRAM in your Gpu devices to load the model, you will get an OMM. Hidding extra devices and making visible only one Gpu solve the issue for low Vram.
* To make it more clear, if you have 2 Gpus with 12GB, you will get an OMM while using distributed. Hidding extra devices activates again the shared memory (a restart of the terminal is required each time you change this value). Of course, if you have 2 gpus with 32GB you will not have this issue.

  
Version 1.04:
* Updated requirements adding optimum-quanto.

Version 1.03:
* Blackwell fix detaching text encoder after quantization.
* Added requirements text file.
* Updated workflow.
* Added optimum module in the requirements package.
* Added new instalation instructions.

Version 1.02:
* Added a new workflow to split your long video into the desired bucket frame-len.

Version 1.01:
* Instructions updated.
* Link to the proper text encoder for training fixed.
* Added the compatible xformers wheel for training in pytorch 2.10 cuda 130.
* Tested and working. (if you see any troubles pin your issue).


**Features**:

* LTX 2.0 and LTX 2.3 training.
* All LTX features implemented.
* Train TI2V or video to video (Ic-LoRA).
* Added toggles per modules.
* Automatic generation of captioning with QwenOmni \[24GB-VRAM].
* Automatic generation of captioning with Qwen3VL and Transcribe (Faster and better than Qwen Omni) \[12GB-VRAM]
* Modular groups to toggle between captioning and training (these are separated to inspect your Json file to make you desired changes if needed).
* You can customize your workflow with your prefered nodes, being able to click and run all in a single shot.
* You just need to pass your videos folder and set up your trigger lora. Everything is packed in the same folder, the json is generated with the same name of your lora.
* Compatible with 12GB of VRAM (need minimum 96GB of Ram to be able to quantize text encoder and checkpoints in real time (You can always download an already quantized compatible checkpoint, probally also the text encoder and be able to run it with 64GB of RAM).
* Path injections for FFmpeg and LTX2 scripts to make it compatible with comfyUI. (Voices a are perfect with 2.3 version)



**Comparision with LTX2 github repository**:

* Crash fix quantizing a model in 8 bit.
* Crash fix getting you an OMM caused to send everything into cpu. 
* Now the text encoder is sent to the cpu on preparing models, then the trasformer is placed first in GPU for fast quantization (before audio models).
* Audio models are loaded after the quantization process of the model if requiered, ensuring better performance.
* Added strategic cuda cleanups during the preparation of models and also in each validation.
* removed unnecesary model moves between cpu-cuda.
* Decorated and disabled all the functions that makes unnecesary Dynamo crash-graphs making a recompilation hit in the very beggining of validations. Also for those functions that handle context managment and affects to inductor.
* Now the train make only a few recompiles instead of 77 for the full process (Very important to make it fly).
* Memory cleaning in video generations detaching  and sending it to the cpu and then to the garbage collector.
* Now validations does not break the performance if there is more than 1 sampling to generate (there could be a decreasing if you are running in low Vram, but the train is stable).
* Now the training is very stable per step, even more in the subsequent validations.
* Captining clean Vram in every batch.
* Now The preprocessing inspect each files to see if there is missing lantents from the full batch or discard preprocessing it again if exists (good if you make a mistake and you need to cancel or preserve the folder jumping directly to training from nodes).
* Crash fix decoding preprocesses (option to inspect your encoded latents).
* Removed some progress bars and replaced per logs in the main process for comfyUI.
* Checkpoint loading fix by placing step 0 in the init of the train object instead of defining it in the train definition (Avoiding to reset a checkpoint lora and make it learn from scratch).

**Benchmark**:

*Environment*:

* ComfyUI windows portable (latest version 3-17-2026).
* Pytorch 2.10.
* xformers.
* Cuda 130.
* Triton.
* Torch Inductor.
* Dynamo to 64 cache size.
* Accelerate.

*Hardware*:

* NVIDIA RTX 4070 SUPER 12GB VRAM
* 128GB RAM

*DATASET*:

* 53 videos 640x320x49f.
* With audio.
* Text to video.
* RANK 32.

*MODULES*:

* Video self attention.
* Audio self attention.
* video attends to audio.
* Audio attends to video.
* This is a total of 138M of parameters in rank 32. (higher ranks and modules add more parameters making it harder to compute, better quality but much slower to train in low Vram)

* Final FP8 Lora : Best time around 5h (without taking into account validations) to generate a 2000 steps lora. So, i can't imagine with a 5090. Don't be rat and buy me a coffe.

**FP8 LORA EXAMPLE (90's vhs style)**:

https://github.com/user-attachments/assets/cd2edde8-271b-48f5-b11d-3c1575a4cc67


* Best time per step 8.66. 

Some screen shots to take into consideration the graph stability:
<img width="775" height="828" alt="Captura de pantalla 2026-03-17 041354" src="https://github.com/user-attachments/assets/25f43b76-548d-4d92-a58d-0ffa211389a3" />

After the first validation loop everything is stable, a proof of the ending steps :


<img width="791" height="993" alt="Captura de pantalla 2026-03-17 140618" src="https://github.com/user-attachments/assets/277d3eb4-ba82-4d87-977f-2f4e2f58c9bb" />

* After code changes, graphs nevermore recompiles mantaining the stability until the end of training.
* There is the need of one loop Interval between train/eval to starting watching the performance and stability. Probally this is not happening in big Gpus but take into account that, you are training with 12GB and this is already a success so, start the train, go to sleep and enjoy it in the morning.

* Conclusion : With this customized LTX2 library, you are able to train a Lora in Low Vram, but be realistic, take care about your configs to not fullfiling your memory making an infinite bottleneck during training. And those that handle better gpus will notice a big boost of performance.

**Things to take into consideration**: 

* Don't care about loss, it seems to not show proper data.
* To get a precise character with the default learning rate, you will need more than 2000 steps, but, with 2000 steps and increasing the lora strenght > 1 Will give you it. The best practice is retain this low learning rate and train  between 4000-6000. You can alway play with the learning rate an make your experiments.

* Trained lora cause noised results in inference using the default settings of ComfyUI (not sure in new versions), as a starting point, use the basic scheduler to see the results (don't make the mistake to throw into the trash your lora).

* Datasets must be symetric in len, join your best videos and cut it in exact pieces of X frames (depending of how you can handle with your gpu). Videos that contains lower frames than the setted up in the config will be ignored and those with higher frames will be trimmed, avoid my initial mistakes about it. There is also the LTX Split scene node that can be useful to capture the best scenes of a video (seems in beta but it Works). My recomendation is, créate your own workfow with a loop to cut scenes in a shot. Probally you can take inspiration on how the custom captioning is made in the training workflow.

* About updates, in previous beta versions, a reach to clamp to 10 seconds without any kind of fluctiantion with Inductor during the full training. Very precise but causing multi validations extremely slow (validations are importants and at least one must be provided, but more is better to eval). Is a game of finetuning dynamo and the internal core to reach a Good balance. So, not sure if i can improve it more, but if is the case, i will update it.

* A i said before, a single validation Will give you the fastest inference in the full training, but multiple validations Will give you more quality making the model evaluating it. But tested with a single validation with Good results, perfect voice and getting the character. Test and make your choice.

* MultiGpu should work, i take care to retain all the logic of the core but removed some progress stadistic like bars (printing steps instead). to place all the training data in the main process, keeping just the training in the multiprocess (in search of performance). Unfortunately, i don't have a MultiGpu system to test it. But, if you have any issue let me know and i will fix it.


**CREATES YOUR VIDEO BUCKET**

* Joint your desired scenes to train in a long video, then you can use this workflow to split your video to the same length as the frames you're going to fit in the bucket:

<img width="1858" height="476" alt="image" src="https://github.com/user-attachments/assets/4301e64e-4a9c-4de2-bd1c-4ad2d58e039f" />



**CAPTIONING**

* You can use the Qwen3VL subgraph or the oficially QwenOmni 2.5 to caption description and speech. The subgraph is much faster being able to select your desired models to describe and transcribe in many languages. QwenOmni is much heavy and a bit less accurate with prompts. if you ask me, choose Qwen3 subgraph.
<img width="1065" height="503" alt="Captura de pantalla 2026-03-18 010957" src="https://github.com/user-attachments/assets/ceb7ef42-d64c-4af3-b4f5-df1cff9bf58c" />

* Important note for Qwen3:  Custom nodes changes between updates, in the moment of pushing this, new options as setted up qwen vl nodes (Also there is different repositories with the same and can create conflicts), use the 2.1.0 QwenVL version or manage to fix it in newer versions. Make sure to enter into the subgraph and check everything again (subgraphs are a bit dirty in that aspect).
<img width="1296" height="566" alt="Captura de pantalla 2026-03-17 220825" src="https://github.com/user-attachments/assets/17fb5fe2-6d61-46ad-98ef-291a114e2d79" />
* The Qwen subgraph could be broken of what is showing depending of the comfyUI version. It must be like this picture (this could change between versions but you will be able to set up it):
<img width="1888" height="730" alt="Captura de pantalla 2026-03-17 222702" src="https://github.com/user-attachments/assets/c832a22c-2eda-4f67-8167-293f2facb25b" />

**Troubleshootings**:

* You must to Run in Xformers or Flash attention 3 depending if you are using Blackwell or Ada (Xformers for ada is provided in the package [pytorch 2.10 cu130 compatible]. Find your compatible wheel if you want to use flash attention 3 for Blackwell depending of your environment).

* Be sure to make a restart after creating you JSON caption (Clear Vram because transcribe nodes does not free the vram memory).

* To download automatically caption models you need HF online but, if you are getting issues with hf or Transformers starting the training [after getting models] go to the file "ltx_environment.py" and uncomment the --TRAIN OFFLINE-- section by removing the hashtags. (a restart of the console is required).

<img width="695" height="325" alt="Captura de pantalla 2026-03-17 224636" src="https://github.com/user-attachments/assets/46297862-c0f2-42e0-b793-4e3e27db78db" />

* If the subgraph presets are loosed (caused by comfy updates) reset it and activate the following in the picture :
  
<img width="2047" height="714" alt="image" src="https://github.com/user-attachments/assets/c17da168-db96-4a03-8c3c-6ceeff8cea76" />

**training that appears blocked or too slow**:

* Qwen3 subgraph does not clean the Vram after the Json generation, be sure to restart comfy after captioning.
* try restart your computer, turn off any app, clean torch, cuda, comfy caches, try it offline, the custom qwen3 captioner graph and train are compatible offline, Qwen Omni needs internet, you will see a noticeable difference.

**Instalation instruction (TESTED FOR WINDOWS)**:

* Downlolad the text encoder for training in https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized/tree/main
* You can use any of the LTX.2X .safetensors models.
* open terminal in custom_nodes\ComfyUI-LTX2-TRAINER\:
```  
..\..\..\python_embeded\python -m pip install -r requirements.txt
```
* open terminal in custom_nodes\ComfyUI-LTX2-TRAINER\LTX-2:
```  
..\..\..\..\python_embeded\python -m pip install -e packages/ltx-core
```
```
..\..\..\..\python_embeded\python -m pip install -e packages/ltx-pipelines
```
```
..\..\..\..\python_embeded\python -m pip install -e packages/ltx-trainer
```
* 40x0 series (Ada) IF you want to use xformers with pytorch 2.10 cuda 130, install the wheel provided in the package:
* Be sure to clean your previous xformers instalation.
* Place the xformers file in the root of the python_embeded folder.
* In python_embeded folder:
```
.\python -m pip install xformers-0.0.34%2Btorch2.10cu130-cp39-abi3-win_amd64.whl".
```

* For Blackwell, you must to search a flash attention 3 wheel compatible with your environment or not using any attention. flash 2.8.3 seems to work slower, uninstall it.

* (If you use linux there is wheels for Flash 3 on internet)

**Other custom nodes dependencies for the alternative captioner x100 faster (kudos for the creators)**
* TTS audio suite
* comfyui-qwenvl
* ComfyUI-Custom-Scripts
* ComfyUI-Easy-Use
* ComfyUI-VideoHelperSuite
* comfy-mtb
* custom-scripts

<img width="960" height="256" alt="Flux2-Klein_00014_" src="https://github.com/user-attachments/assets/f171b752-6fcf-4e37-b98c-d162056700d0" />

**Unfortunatelly, i can't commit my changes directly into LTX-2 repository, there is so many changes and paths inyections to make it compatible with Comfy Portable. But if LTX-2 want to take a look into it, they are welcome here**.
Visit https://github.com/Lightricks/LTX-2 to have the full documentation to mastering your loras.









