# PDF Compression for Research Papers Using Ghostscript

This document records a lightweight and high-quality PDF compression workflow using Ghostscript. The method is particularly useful for research papers, LaTeX-generated PDFs, and figure-heavy manuscripts where file size must be reduced without noticeable quality degradation.

---

# Recommended Command

```bash
gs -sDEVICE=pdfwrite \
   -dCompatibilityLevel=1.4 \
   -dNOPAUSE \
   -dQUIET \
   -dBATCH \
   -dDetectDuplicateImages=true \
   -dCompressFonts=true \
   -dDownsampleColorImages=false \
   -dDownsampleGrayImages=false \
   -dDownsampleMonoImages=false \
   -sOutputFile=compressed.pdf \
   input.pdf
```

---

# What This Command Does

Unlike aggressive PDF compression methods, this workflow mainly performs:

* structural optimization
* duplicate object removal
* stream recompression
* font optimization

while preserving the original image resolution.

As a result, it often reduces PDF size significantly with little or no visible quality loss.

---

# Key Optimizations

## 1. Duplicate Image Detection

```bash
-dDetectDuplicateImages=true
```

Ghostscript detects identical embedded images and stores only one copy internally.

This is especially effective for:

* repeated figures
* logos
* repeated rasterized panels
* duplicated transparency layers
* reused heatmaps or plots

In many scientific PDFs, duplicated assets account for a surprisingly large fraction of file size.

---

## 2. Stream Recompression

Even without image downsampling, Ghostscript rewrites image and object streams using more efficient encodings.

This may include:

* recompressing image streams
* consolidating fragmented objects
* improving ZIP/Flate compression
* cleaning inefficient PDF structures

Image resolution is preserved.

---

## 3. Removal of Redundant PDF Objects

Ghostscript reconstructs the PDF from scratch and removes:

* unused objects
* stale metadata
* incremental update history
* dead cross-references
* unnecessary padding
* hidden editor artifacts

This can dramatically reduce the size of PDFs that have undergone multiple edits.

---

## 4. Font Compression and Subsetting

```bash
-dCompressFonts=true
```

Ghostscript recompresses embedded fonts and may subset them to include only used glyphs.

This is particularly useful for:

* large Unicode fonts
* partially used font families
* mixed Illustrator/LaTeX workflows

---

# Why Image Quality Is Preserved

The following options disable image downsampling:

```bash
-dDownsampleColorImages=false
-dDownsampleGrayImages=false
-dDownsampleMonoImages=false
```

Therefore:

* image dimensions remain unchanged
* DPI is preserved
* no aggressive JPEG recompression is applied

This makes the workflow suitable for:

* journal submissions
* conference papers
* camera-ready manuscripts
* figure-sensitive documents

---

# Typical Results

For research PDFs, reductions such as:

* 40 MB to 12 MB
* 100 MB to 20 MB

are common, especially when the original PDF contains:

* duplicated images
* fragmented vector objects
* excessive metadata
* repeated transparency groups

---

# Installation

## Ubuntu / Debian

```bash
sudo apt install ghostscript
```

## macOS (Homebrew)

```bash
brew install ghostscript
```

---

# Notes

* This workflow is generally safer than using aggressive presets like `/screen`.
* It is preferable when only moderate size reduction is needed.
* Particularly effective for LaTeX + matplotlib + Illustrator generated PDFs.

---
