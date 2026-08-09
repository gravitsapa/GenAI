from gen_ai.data.datasets import AnimeFaces256
from gen_ai.data.image_sample import show_sample
from matplotlib import pyplot as plt

def main():
    anime_faces = AnimeFaces256(128, 128)
    sample = anime_faces[1]
    fig, axs = plt.subplots()
    show_sample(sample, axs)
    plt.show()

    print("Done")
