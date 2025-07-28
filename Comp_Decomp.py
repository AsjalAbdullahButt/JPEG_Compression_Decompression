import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from scipy.fftpack import dct, idct
from collections import Counter
import heapq
import os

# --------------- constants here ----------------
Y_QT = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62], [
                18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92], [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])
C_QT = np.array([[17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99], [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99], [
                99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]])


class Huff:
    def __init__(self, sym=None, f=0):
        self.symbol = sym
        self.freq = f
        self.l = None
        self.r = None

    def __lt__(self, oth): return self.freq < oth.freq


def make_huff_tree(frq):
    pq = [Huff(s, f) for s, f in frq.items()]
    heapq.heapify(pq)
    while len(pq) > 1:
        n1 = heapq.heappop(pq)
        n2 = heapq.heappop(pq)
        m = Huff(f=n1.freq+n2.freq)
        m.l = n1
        m.r = n2
        heapq.heappush(pq, m)
    return pq[0]


def gen_codes(root):
    dct = {}

    def recur(n, p=""):
        if n is None:
            return
        if n.symbol is not None:
            dct[n.symbol] = p
        recur(n.l, p+"0")
        recur(n.r, p+"1")
    recur(root)
    return dct


def rgb2ycc(im):
    im = im.astype(np.float32)
    y = 0.299*im[:, :, 0] + 0.587*im[:, :, 1] + 0.114*im[:, :, 2]
    cb = -0.1687*im[:, :, 0] - 0.3313*im[:, :, 1] + 0.5*im[:, :, 2] + 128
    cr = 0.5*im[:, :, 0] - 0.4187*im[:, :, 1] - 0.0813*im[:, :, 2] + 128
    return y, cb, cr


def zzi():
    return sorted(((a, b) for a in range(8) for b in range(8)), key=lambda p: (p[0]+p[1], -p[0] if (p[0]+p[1]) % 2 else p[0]))


def zig(b): return np.array([b[i, j] for i, j in zzi()])


def rle(ar):
    r, zc = [], 0
    for x in ar:
        if x == 0:
            zc += 1
        else:
            r.append((zc, x))
            zc = 0
    r.append((0, 0))
    return r


def blk_dct(img, q):
    h, w = img.shape
    hp = h + (8 - h % 8) % 8
    wp = w + (8 - w % 8) % 8
    pimg = np.zeros((hp, wp))
    pimg[:h, :w] = img - 128
    r = []
    for a in range(0, hp, 8):
        for b in range(0, wp, 8):
            blk = pimg[a:a+8, b:b+8]
            d = dct(dct(blk.T, norm='ortho').T, norm='ortho')
            qnt = np.round(d/q)
            r.append(rle(zig(qnt)))
    return r


def do_compr(chan, q):
    blks = blk_dct(chan, q)
    syms = [s for b in blks for s in b]
    freq = Counter(syms)
    tr = make_huff_tree(freq)
    tbl = gen_codes(tr)
    bits = ''.join(tbl[s] for s in syms)
    return {"blocks": blks, "bitstream": bits, "huffman_table": tbl}


def jpeg_compr(im):
    if len(im.shape) == 2 or im.shape[2] == 1:
        return {"mode": "grayscale", "Y": do_compr(im, Y_QT), "shape": im.shape}
    y, cb, cr = rgb2ycc(im)
    return {"mode": "color", "Y": do_compr(y, Y_QT), "Cb": do_compr(cb, C_QT), "Cr": do_compr(cr, C_QT), "shapes": {"Y": y.shape, "Cb": cb.shape, "Cr": cr.shape}}


class Tree:
    def __init__(self): self.symbol = None; self.l = None; self.r = None


def rebld(cb):
    root = Tree()
    for sym, code in cb.items():
        n = root
        for b in code:
            if b == '0':
                if n.l is None:
                    n.l = Tree()
                n = n.l
            else:
                if n.r is None:
                    n.r = Tree()
                n = n.r
        n.symbol = sym
    return root


def decode_bits(bits, tree):
    out, n = [], tree
    for b in bits:
        n = n.l if b == '0' else n.r
        if n.symbol is not None:
            out.append(n.symbol)
            n = tree
    return out


def rle_dec(rle):
    x, z = [], 0
    for a, b in rle:
        if (a, b) == (0, 0):
            break
        x.extend([0]*a)
        x.append(b)
    while len(x) < 64:
        x.append(0)
    return x


def zigrev(ar):
    ind = zzi()
    blk = np.zeros((8, 8))
    for i, (a, b) in enumerate(ind):
        blk[a, b] = ar[i]
    return blk


def idct_blk(b): return idct(idct(b.T, norm='ortho').T, norm='ortho')


def blks_to_img(rleblks, shape, q):
    h, w = shape
    hp = h + (8-h % 8) % 8
    wp = w + (8-w % 8) % 8
    img = np.zeros((hp, wp))
    idx = 0
    for a in range(0, hp, 8):
        for b in range(0, wp, 8):
            blk = zigrev(rle_dec(rleblks[idx])) * q
            pix = idct_blk(blk) + 128
            img[a:a+8, b:b+8] = np.clip(pix, 0, 255)
            idx += 1
    return img[:h, :w]


def ycc2rgb(y, cb, cr):
    y = y.astype(np.float32)
    cb -= 128
    cr -= 128
    r = y + 1.402*cr
    g = y - 0.344136*cb - 0.714136*cr
    b = y + 1.772*cb
    return np.clip(np.stack([r, g, b], axis=2), 0, 255).astype(np.uint8)


def decompress_stuff(npz):
    d = npz['compressed'].item()
    if d['mode'] == 'grayscale':
        y = blks_to_img(d['Y']['blocks'], d['shape'], Y_QT)
        return y.astype(np.uint8)
    yt = rebld(d['Y']['huffman_table'])
    cbt = rebld(d['Cb']['huffman_table'])
    crt = rebld(d['Cr']['huffman_table'])
    ysy = decode_bits(d['Y']['bitstream'], yt)
    cbs = decode_bits(d['Cb']['bitstream'], cbt)
    crs = decode_bits(d['Cr']['bitstream'], crt)

    def splitblks(s):
        r, b = [], []
        for sym in s:
            b.append(sym)
            if sym == (0, 0):
                r.append(b)
                b = []
        return r
    y = blks_to_img(splitblks(ysy), d['shapes']['Y'], Y_QT)
    cb = blks_to_img(splitblks(cbs), d['shapes']['Cb'], C_QT)
    cr = blks_to_img(splitblks(crs), d['shapes']['Cr'], C_QT)
    return ycc2rgb(y, cb, cr)

# ---------------- GUI Code -----------------


class GUI:
    def __init__(self, master):
        self.win = master
        master.title("JPEG Toolz")
        master.geometry("520x620")

        self.cnvs = tk.Canvas(master, width=450, height=320, bg='lightgray')
        self.cnvs.pack(pady=8)

        frm = tk.Frame(master)
        frm.pack(pady=5)

        self.btn1 = tk.Button(frm, text="Load Img", command=self.loadimg)
        self.btn1.grid(row=0, column=0, padx=4)

        self.btn2 = tk.Button(frm, text="Compress+Save",
                              command=self.comp_save)
        self.btn2.grid(row=0, column=1, padx=4)

        self.btn3 = tk.Button(frm, text="Decompress", command=self.decompfile)
        self.btn3.grid(row=1, column=0, padx=4)

        self.btn4 = tk.Button(frm, text="Save Output", command=self.saveimg)
        self.btn4.grid(row=1, column=1, padx=4)

        self.lbl = tk.Label(master, text="Status: ok", font=("Arial", 10))
        self.lbl.pack()

        self.img_now = None
        self.ph = None

    def loadimg(self):
        fp = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")])
        if not fp:
            return
        try:
            i = cv2.imread(fp)
            if i is None:
                raise Exception("couldn't load")
            self.img_now = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(self.img_now).resize((450, 320))
            self.ph = ImageTk.PhotoImage(im)
            self.cnvs.create_image(0, 0, anchor=tk.NW, image=self.ph)
            self.lbl.config(text=f"Loaded: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def comp_save(self):
        if self.img_now is None:
            messagebox.showwarning("Error", "Load image first")
            return
        try:
            c = jpeg_compr(self.img_now)
            fp = filedialog.asksaveasfilename(
                defaultextension=".npz", filetypes=[("NPZ", "*.npz")])
            if fp:
                np.savez_compressed(fp, compressed=c)
                messagebox.showinfo("Saved", f"Compressed at:\n{fp}")
        except Exception as e:
            messagebox.showerror("Oops", f"Compression failed:\n{e}")

    def decompfile(self):
        fp = filedialog.askopenfilename(filetypes=[("NPZ", "*.npz")])
        if not fp:
            return
        try:
            d = np.load(fp, allow_pickle=True)
            self.img_now = decompress_stuff(d)
            im = Image.fromarray(self.img_now).resize((450, 320))
            self.ph = ImageTk.PhotoImage(im)
            self.cnvs.create_image(0, 0, anchor=tk.NW, image=self.ph)
            self.lbl.config(text=f"Decompressed: {os.path.basename(fp)}")
        except Exception as e:
            messagebox.showerror("Error", f"Decompression error:\n{e}")

    def saveimg(self):
        if self.img_now is None:
            messagebox.showwarning("Wait", "No image to save")
            return
        fp = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[
                                          ("JPEG", "*.jpg *.jpeg")])
        if fp:
            try:
                Image.fromarray(self.img_now).save(fp)
                messagebox.showinfo("Yay", f"Saved at:\n{fp}")
            except Exception as e:
                messagebox.showerror("Fail", f"Couldn't save:\n{e}")


if __name__ == "__main__":
    r = tk.Tk()
    a = GUI(r)
    r.mainloop()
