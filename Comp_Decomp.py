import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from scipy.fftpack import dct, idct
from collections import Counter
import heapq

# quantization tables
Y_QT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=np.float64)

C_QT = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=np.float64)

# zigzag order
_ZIGZAG_INDEX = sorted(
    ((a, b) for a in range(8) for b in range(8)),
    key=lambda p: (p[0] + p[1], -p[0] if (p[0] + p[1]) % 2 else p[0]),
)


# Huffman node
class HuffNode:
    __slots__ = ("symbol", "freq", "left", "right")

    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


# build tree
def build_huffman_tree(freq_table):
    heap = [HuffNode(sym, f) for sym, f in freq_table.items()]
    heapq.heapify(heap)
    if len(heap) == 1:
        # single symbol
        return heap[0]
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        parent = HuffNode(freq=a.freq + b.freq)
        parent.left, parent.right = a, b
        heapq.heappush(heap, parent)
    return heap[0]


# assign codes
def build_codes(root):
    codes = {}
    if root.symbol is not None and root.left is None and root.right is None:
        codes[root.symbol] = "0"  # single symbol fix
        return codes

    def walk(node, prefix):
        if node is None:
            return
        if node.symbol is not None:
            codes[node.symbol] = prefix
            return
        walk(node.left, prefix + "0")
        walk(node.right, prefix + "1")

    walk(root, "")
    return codes


# decode tree node
class DecodeNode:
    __slots__ = ("symbol", "left", "right")

    def __init__(self):
        self.symbol = None
        self.left = None
        self.right = None


# rebuild tree from codes
def rebuild_tree(code_table):
    root = DecodeNode()
    for symbol, code in code_table.items():
        node = root
        for bit in code:
            if bit == "0":
                node.left = node.left or DecodeNode()
                node = node.left
            else:
                node.right = node.right or DecodeNode()
                node = node.right
        node.symbol = symbol
    return root


# decode bitstring
def decode_bitstring(bits, tree):
    out = []
    node = tree
    if tree.symbol is not None and tree.left is None and tree.right is None:
        return [tree.symbol] * len(bits)  # single symbol case
    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node is None:
            raise ValueError("corrupt bitstream: invalid Huffman code")
        if node.symbol is not None:
            out.append(node.symbol)
            node = tree
    return out


# pack bits to bytes
def pack_bits(bitstring):
    if not bitstring:
        return np.array([], dtype=np.uint8), 0
    bit_arr = np.frombuffer(bitstring.encode("ascii"), dtype=np.uint8) - ord("0")
    packed = np.packbits(bit_arr)
    return packed, len(bitstring)


# unpack bytes to bits
def unpack_bits(packed, bit_length):
    if bit_length == 0:
        return ""
    bits = np.unpackbits(packed)[:bit_length]
    return "".join("1" if b else "0" for b in bits)


# RGB to YCbCr
def rgb_to_ycbcr(img):
    img = img.astype(np.float32)
    y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    cb = -0.1687 * img[:, :, 0] - 0.3313 * img[:, :, 1] + 0.5 * img[:, :, 2] + 128
    cr = 0.5 * img[:, :, 0] - 0.4187 * img[:, :, 1] - 0.0813 * img[:, :, 2] + 128
    return y, cb, cr


# YCbCr to RGB
def ycbcr_to_rgb(y, cb, cr):
    y = y.astype(np.float32)
    cb = cb.astype(np.float32) - 128
    cr = cr.astype(np.float32) - 128
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


# zigzag scan
def zigzag(block):
    return np.array([block[i, j] for i, j in _ZIGZAG_INDEX])


# reverse zigzag
def unzigzag(arr):
    block = np.zeros((8, 8))
    for i, (a, b) in enumerate(_ZIGZAG_INDEX):
        block[a, b] = arr[i]
    return block


# run-length encode
def run_length_encode(arr):
    pairs, zero_run = [], 0
    for v in arr:
        if v == 0:
            zero_run += 1
        else:
            pairs.append((zero_run, v))
            zero_run = 0
    pairs.append((0, 0))  # end marker
    return pairs


# run-length decode
def run_length_decode(pairs):
    out = []
    for zero_run, val in pairs:
        if (zero_run, val) == (0, 0):
            break
        out.extend([0] * zero_run)
        out.append(val)
    if len(out) < 64:
        out.extend([0] * (64 - len(out)))
    return out[:64]


# DCT per block
def channel_to_blocks(channel, qtable):
    h, w = channel.shape
    hp = h + (8 - h % 8) % 8
    wp = w + (8 - w % 8) % 8
    padded = np.zeros((hp, wp))
    padded[:h, :w] = channel - 128
    blocks = []
    for y0 in range(0, hp, 8):
        for x0 in range(0, wp, 8):
            block = padded[y0:y0 + 8, x0:x0 + 8]
            coeffs = dct(dct(block.T, norm="ortho").T, norm="ortho")
            quantized = np.round(coeffs / qtable)
            blocks.append(run_length_encode(zigzag(quantized)))
    return blocks, (hp // 8) * (wp // 8)


# inverse DCT per block
def blocks_to_channel(rle_blocks, shape, qtable):
    h, w = shape
    hp = h + (8 - h % 8) % 8
    wp = w + (8 - w % 8) % 8
    out = np.zeros((hp, wp))
    idx = 0
    for y0 in range(0, hp, 8):
        for x0 in range(0, wp, 8):
            coeffs = unzigzag(run_length_decode(rle_blocks[idx])) * qtable
            pixels = idct(idct(coeffs.T, norm="ortho").T, norm="ortho") + 128
            out[y0:y0 + 8, x0:x0 + 8] = np.clip(pixels, 0, 255)
            idx += 1
    return out[:h, :w]


# full channel encode
def encode_channel(channel, qtable):
    blocks, n_blocks = channel_to_blocks(channel, qtable)
    symbols = [s for block in blocks for s in block]
    freq = Counter(symbols)
    tree = build_huffman_tree(freq)
    codes = build_codes(tree)
    bitstring = "".join(codes[s] for s in symbols)
    packed, bit_len = pack_bits(bitstring)
    return {
        "packed_bits": packed,
        "bit_length": bit_len,
        "codes": codes,
        "n_blocks": n_blocks,
        "shape": channel.shape,
    }


# full channel decode
def decode_channel(payload, qtable):
    tree = rebuild_tree(payload["codes"])
    bitstring = unpack_bits(payload["packed_bits"], payload["bit_length"])
    symbols = decode_bitstring(bitstring, tree)

    blocks, current = [], []
    for sym in symbols:
        current.append(sym)
        if sym == (0, 0):
            blocks.append(current)
            current = []
    if len(blocks) != payload["n_blocks"]:
        raise ValueError(
            f"expected {payload['n_blocks']} blocks, decoded {len(blocks)} "
            "- the file is corrupt or truncated"
        )
    return blocks_to_channel(blocks, payload["shape"], qtable)


# compress full image
def compress_image(img):
    if img.ndim == 2 or img.shape[2] == 1:
        return {"mode": "grayscale", "Y": encode_channel(img, Y_QT)}
    y, cb, cr = rgb_to_ycbcr(img)
    return {
        "mode": "color",
        "Y": encode_channel(y, Y_QT),
        "Cb": encode_channel(cb, C_QT),
        "Cr": encode_channel(cr, C_QT),
    }


# decompress full image
def decompress_image(payload):
    if payload["mode"] == "grayscale":
        return np.clip(decode_channel(payload["Y"], Y_QT), 0, 255).astype(np.uint8)
    y = decode_channel(payload["Y"], Y_QT)
    cb = decode_channel(payload["Cb"], C_QT)
    cr = decode_channel(payload["Cr"], C_QT)
    return ycbcr_to_rgb(y, cb, cr)


# GUI class
class JpegToolzGUI:
    PREVIEW_W, PREVIEW_H = 450, 320

    def __init__(self, master):
        self.win = master
        master.title("JPEG Toolz")
        master.geometry("520x640")
        master.resizable(False, False)

        self.canvas = tk.Canvas(master, width=self.PREVIEW_W, height=self.PREVIEW_H, bg="lightgray")
        self.canvas.pack(pady=8)

        btn_frame = tk.Frame(master)
        btn_frame.pack(pady=5)

        tk.Button(btn_frame, text="Load Image", width=16, command=self.load_image).grid(row=0, column=0, padx=4, pady=2)
        tk.Button(btn_frame, text="Compress + Save", width=16, command=self.compress_and_save).grid(row=0, column=1, padx=4, pady=2)
        tk.Button(btn_frame, text="Decompress", width=16, command=self.decompress_file).grid(row=1, column=0, padx=4, pady=2)
        tk.Button(btn_frame, text="Save As Image", width=16, command=self.save_image).grid(row=1, column=1, padx=4, pady=2)

        self.status = tk.Label(master, text="Status: ready", font=("Arial", 10), wraplength=480, justify="left")
        self.status.pack(pady=6)

        self.current_image = None
        self._preview_ref = None
        self._source_path = None
        self._busy = False

    # update status text
    def set_status(self, text):
        self.status.config(text=f"Status: {text}")

    # show preview image
    def show_preview(self, rgb_array):
        img = Image.fromarray(rgb_array)
        img.thumbnail((self.PREVIEW_W, self.PREVIEW_H))  # keep aspect
        self._preview_ref = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        x = (self.PREVIEW_W - img.width) // 2
        y = (self.PREVIEW_H - img.height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self._preview_ref)

    # run worker thread
    def run_in_background(self, worker, on_done):
        if self._busy:
            messagebox.showinfo("Busy", "Please wait for the current operation to finish.")
            return
        self._busy = True

        def target():
            try:
                result = worker()
                error = None
            except Exception as exc:
                result, error = None, exc
            self.win.after(0, lambda: self._finish(on_done, result, error))

        threading.Thread(target=target, daemon=True).start()

    # finish background task
    def _finish(self, on_done, result, error):
        self._busy = False
        if error is not None:
            messagebox.showerror("Error", str(error))
            self.set_status("failed - see error dialog")
            return
        on_done(result)

    # load image file
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        try:
            raw = cv2.imread(path)
            if raw is None:
                raise ValueError("could not read that file as an image")
            self.current_image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            self._source_path = path
            self.show_preview(self.current_image)
            self.set_status(f"loaded {os.path.basename(path)} ({self.current_image.shape[1]}x{self.current_image.shape[0]})")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # compress and save
    def compress_and_save(self):
        if self.current_image is None:
            messagebox.showwarning("No image", "Load an image first.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("Compressed", "*.npz")])
        if not save_path:
            return
        original_size = os.path.getsize(self._source_path) if self._source_path else None
        self.set_status("compressing...")

        def worker():
            payload = compress_image(self.current_image)
            np.savez_compressed(save_path, compressed=payload, allow_pickle=True)
            return os.path.getsize(save_path)

        def done(compressed_size):
            if original_size:
                ratio = original_size / max(compressed_size, 1)
                self.set_status(
                    f"saved {os.path.basename(save_path)} - "
                    f"{original_size:,}B -> {compressed_size:,}B ({ratio:.2f}x)"
                )
            else:
                self.set_status(f"saved {os.path.basename(save_path)} ({compressed_size:,} bytes)")

        self.run_in_background(worker, done)

    # decompress file
    def decompress_file(self):
        path = filedialog.askopenfilename(filetypes=[("Compressed", "*.npz")])
        if not path:
            return
        self.set_status("decompressing...")

        def worker():
            npz = np.load(path, allow_pickle=True)
            payload = npz["compressed"].item()
            return decompress_image(payload)

        def done(rgb_array):
            self.current_image = rgb_array
            self._source_path = None
            self.show_preview(rgb_array)
            self.set_status(f"decompressed {os.path.basename(path)}")

        self.run_in_background(worker, done)

    # save as image
    def save_image(self):
        if self.current_image is None:
            messagebox.showwarning("Nothing to save", "There is no image in memory yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")])
        if not path:
            return
        try:
            Image.fromarray(self.current_image).save(path)
            self.set_status(f"saved image to {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    JpegToolzGUI(root)
    root.mainloop()
