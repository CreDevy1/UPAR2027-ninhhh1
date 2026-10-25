# Same image as the Codabench grader: Python 3.11, torch 2.4.1, torchvision, numpy, Pillow.
# The submitted packages need nothing else.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
WORKDIR /work
COPY tools /work/tools
ENTRYPOINT ["python", "/work/tools/run_package.py"]
