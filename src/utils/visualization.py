import matplotlib.pyplot as plt


def visualize_kernel_groups(channel_map: dict[int, int]):
    kernels = list(channel_map.keys())
    channels = list(channel_map.values())

    plt.bar(
        [str(k) for k in kernels],
        channels
    )
    plt.xlabel("Kernel Size")
    plt.ylabel("Number of Channels")
    plt.title("MixConv Kernel-Channel Assignment")
    plt.show()
