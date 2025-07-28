# JPEG_Compression_Decompression

A Python-based desktop GUI application for JPEG-like image compression and decompression, using Discrete Cosine Transform (DCT), quantization, Run-Length Encoding (RLE), and Huffman Coding. Built with **Tkinter**, **OpenCV**, and **NumPy**, this tool visualizes the entire pipeline from loading an image to compressing it and saving/decompressing results.

---

## 📦 Features

- ✅ Load and preview JPEG and PNG images
- 📉 Compress images using:
  - RGB to YCbCr color conversion
  - 8×8 DCT block transformation
  - Quantization with standard JPEG matrices
  - Zigzag scan and RLE encoding
  - Huffman coding for entropy compression
- 💾 Save compressed file (`.npz`) with metadata
- 🔄 Decompress and restore image from `.npz`
- 🖼 Save decompressed image to `.jpg`

---

## 🖥 GUI Preview

The GUI includes:
- Canvas to display original and decompressed images
- Buttons to:
  - Load image
  - Compress & Save
  - Decompress file
  - Save final image

---

## 🧠 Compression Workflow

1. **RGB to YCbCr** conversion for separating luminance and chrominance.
2. **Block-wise DCT** is applied on 8×8 blocks.
3. **Quantization** using JPEG quantization matrices (`Y_QT`, `C_QT`).
4. **Zigzag traversal** to convert 2D blocks to 1D.
5. **Run-Length Encoding (RLE)** of coefficients.
6. **Huffman Tree** is built and used to encode the RLE symbols.
7. Data is saved in `.npz` format with bitstream and tree data.

---

## 🔁 Decompression Workflow

1. Load compressed `.npz`.
2. Decode bitstream using Huffman Tree.
3. Apply RLE decoding and inverse zigzag.
4. Apply **Inverse DCT** to reconstruct each 8×8 block.
5. Recombine channels and convert back to **RGB**.

---

## 📦 Requirements

Install dependencies using pip:

```bash
pip install numpy opencv-python Pillow scipy
