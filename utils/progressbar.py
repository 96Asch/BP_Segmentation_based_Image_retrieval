from IPython.display import clear_output


def update_progress(text, it, size):
    bar_length = 20

    if it == 0:
        progress = 0.0
    else:
        progress = float(it / size)

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    res = text + ": [{0}/{1}] [{2}] {3:.1f}%".format(it, size, "#" * block + "-" * (bar_length - block), progress * 100)
    print(res)
